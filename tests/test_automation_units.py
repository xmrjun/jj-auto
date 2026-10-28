"""自动化各组件：选牌规划、按钮匹配、角色与动作序列推算、受保护输入、稳定门。"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from automation.bidding import BidPolicy, bid_level, choose_bid_button, hand_strength
from automation.buttons import ButtonDetector, ButtonTemplate, load_button_detector, write_width
from automation.card_selection import UnplannableAction, plan_card_clicks, raised_flags
from automation.emergency_stop import AutomationStopped, EmergencyStop
from automation.game_record import RoundRecord
from automation.input_backend import GuardedInputBackend, UnsafeClick
from automation.safety import StableObservationGate
from automation.types import AutomationMode, CardBox, ClickPlan, ClickPoint, TableObservation
from decision.protocol import InconsistentState

RECT = (100, 50, 1380, 850)


def boxes(cards, raised=()):
    """从左到右生成手牌框，raised 中的下标上移 20 像素。"""
    return [CardBox(card, (100+30*i, 400-20*(i in raised), 125+30*i, 440-20*(i in raised)))
            for i, card in enumerate(cards)]


class FakeNative:
    """记录点击的假输入。"""

    def __init__(self, hwnd=7, rect=RECT, cover=None, on_click=None):
        """创建假输入。"""
        self.hwnd, self.rect, self.cover, self.on_click = hwnd, rect, cover, on_click
        self.clicks = []
        self.cursor = (0, 0)

    def window_rect(self, hwnd):
        """返回窗口矩形。"""
        return self.rect

    def root_window_at(self, x, y):
        """返回点下方窗口。"""
        return self.cover if self.cover is not None else self.hwnd

    def cursor_pos(self):
        """返回鼠标位置。"""
        return self.cursor

    def click(self, x, y):
        """记录点击。"""
        self.clicks.append((x, y))
        self.cursor = (x, y)
        if self.on_click:
            self.on_click(self)


def two_clicks():
    """两点计划。"""
    return ClickPlan((ClickPoint(10, 10, "a"), ClickPoint(20, 20, "b")), "test")


class CardSelectionTests(unittest.TestCase):
    """基于真实牌框规划选牌。"""

    def test_duplicate_rank_selects_distinct_boxes(self) -> None:
        """两张 A 选择两个不同牌框。"""
        plan = plan_card_clicks(["A", "A"], boxes(["A", "A", "A", "3"]))
        self.assertEqual(len({click.x for click in plan.clicks}), 2)

    def test_missing_rank_rejects_entire_plan(self) -> None:
        """缺牌时整体拒绝，不做部分点击。"""
        with self.assertRaises(UnplannableAction):
            plan_card_clicks(["大", "小"], boxes(["小", "3"]))

    def test_extra_selected_card_is_clicked_back(self) -> None:
        """已经选中但不需要的牌会被点回，已选中的目标牌不再点击。"""
        hand = boxes(["3", "4", "5"], raised={0, 1})
        flags = raised_flags(hand)
        self.assertEqual(flags, (True, True, False))
        plan = plan_card_clicks(["4", "5"], hand, flags)
        self.assertEqual([click.label for click in plan.clicks], ["3", "5"])

    def test_plan_uses_box_centers_and_rejects_out_of_image(self) -> None:
        """点击点为牌框中心，越界拒绝。"""
        plan = plan_card_clicks(["3"], [CardBox("3", (100, 400, 140, 500))])
        self.assertEqual((plan.clicks[0].x, plan.clicks[0].y), (120, 450))
        with self.assertRaises(UnplannableAction):
            plan_card_clicks(["3"], [CardBox("3", (100, 400, 140, 500))], image_size=(110, 600))


class ButtonDetectorTests(unittest.TestCase):
    """合成图上的模板匹配。"""

    def template(self):
        """生成有纹理的模板。"""
        rng = np.random.default_rng(1)
        return (rng.random((20, 40))*255).astype(np.uint8)

    def screen(self, positions, scale=1.0):
        """把模板按比例贴到噪声背景上。"""
        import cv2

        rng = np.random.default_rng(2)
        image = (rng.random((400, 640))*50).astype(np.uint8)
        tpl = cv2.resize(self.template(), (int(40*scale), int(20*scale)))
        for x, y in positions:
            image[y:y+tpl.shape[0], x:x+tpl.shape[1]] = tpl
        return image

    def test_detects_scaled_template_in_window_coordinates(self) -> None:
        """窗口比截取时宽 1.5 倍时仍能定位。"""
        detector = ButtonDetector([ButtonTemplate("play", self.template(), 640/1.5)], match_scale=1.0)
        match = detector.detect(self.screen([(200, 300)], 1.5))["play"]
        self.assertTrue(match.actionable)
        self.assertAlmostEqual(match.center[0], 230, delta=3)
        self.assertAlmostEqual(match.center[1], 315, delta=3)

    def test_two_equal_matches_are_not_actionable(self) -> None:
        """出现两个同样好的位置时不点击。"""
        detector = ButtonDetector([ButtonTemplate("play", self.template(), 640)], match_scale=1.0)
        self.assertFalse(detector.detect(self.screen([(100, 100), (400, 300)]))["play"].actionable)

    def test_absent_button_and_missing_file(self) -> None:
        """没有按钮时不返回；缺失模板列出名称，宽度记录生效。"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            import cv2

            (root/"t").mkdir()
            cv2.imwrite(str(root/"t"/"play.png"), self.template())
            write_width(root/"t", "play.png", 320)
            detector, missing = load_button_detector(
                {"template_dir": "t", "buttons": {"play": {"file": "play.png"}, "pass": {"file": "pass.png"}}}, root)
            self.assertEqual(missing, ["pass"])
            self.assertEqual(detector.templates["play"].reference_width, 320)
            self.assertEqual(detector.detect(self.screen([])), {})


class BiddingTests(unittest.TestCase):
    """叫分规则。"""

    def test_strong_hand_bids_three_and_weak_hand_passes(self) -> None:
        """强牌叫 3 分，弱牌不叫。"""
        strong = ["大", "小", "2", "2", "A"]
        self.assertEqual(choose_bid_button(bid_level(hand_strength(strong), BidPolicy()),
                                           {"bid_1", "bid_2", "bid_3", "bid_pass"}), "bid_3")
        self.assertEqual(choose_bid_button(bid_level(hand_strength(["3", "4"]), BidPolicy()),
                                           {"bid_1", "bid_pass"}), "bid_pass")

    def test_thresholds_must_increase(self) -> None:
        """阈值必须递增。"""
        with self.assertRaises(ValueError):
            BidPolicy(bid_1=9, bid_2=8)


class RoundRecordTests(unittest.TestCase):
    """角色推断与动作序列。"""

    hand17 = "4 5 6 7 8 9 10 J Q K A 2 3 4 5 6 7".split()

    def test_landlord_by_twenty_cards(self) -> None:
        """20 张且无人出牌为地主。"""
        record = RoundRecord(["3", "8", "9"], {})
        preview = record.preview(self.hand17+["8", "9", "10"], {})
        self.assertEqual((preview.position, preview.sequence), ("landlord", ()))

    def test_right_played_first_means_we_are_landlord_up(self) -> None:
        """下家先出说明下家是地主，上家随后不出。"""
        record = RoundRecord(["3", "8", "9"], {"right": [["K"]]})
        preview = record.preview(self.hand17, {"right": [["K"], ["3"]], "left": []})
        self.assertEqual(preview.position, "landlord_up")
        self.assertEqual(preview.sequence, (("landlord", ("3",)), ("landlord_down", ())))

    def test_left_played_first_means_we_are_landlord_down(self) -> None:
        """只有上家出牌说明上家是地主。"""
        preview = RoundRecord(["3", "8", "9"], {}).preview(self.hand17, {"left": [["3"]]})
        self.assertEqual((preview.position, preview.sequence), ("landlord_down", (("landlord", ("3",)),)))

    def test_preview_does_not_mutate_and_commit_advances(self) -> None:
        """预览不改记录；提交后下一轮按 本家-下家-上家 推进。"""
        record = RoundRecord(["3", "8", "9"], {})
        first = record.preview(self.hand17, {"left": [["3"]]})
        self.assertIsNone(record.position)
        record.commit(first)
        later = record.preview(self.hand17[1:], {"left": [["3"], ["6"]], "self": [["5"]], "right": []})
        self.assertEqual(later.sequence[1:], (("landlord_down", ("5",)), ("landlord_up", ()), ("landlord", ("6",))))

    def test_auto_action_mismatch_is_inconsistent(self) -> None:
        """记牌器记录与自动执行的动作不一致时拒绝。"""
        record = RoundRecord(["3", "8", "9"], {})
        record.commit(record.preview(self.hand17, {"left": [["3"]]}))
        record.expected_own = ("5",)
        with self.assertRaises(InconsistentState):
            record.preview(self.hand17, {"left": [["3"]], "self": [["6"]]})

    def test_unknown_role_is_inconsistent(self) -> None:
        """17 张但没人出过牌无法判断。"""
        with self.assertRaises(InconsistentState):
            RoundRecord(["3", "8", "9"], {}).preview(self.hand17, {})


class InputBackendTests(unittest.TestCase):
    """受保护点击。"""

    def test_observe_mode_records_but_never_clicks(self) -> None:
        """观察模式零点击。"""
        native = FakeNative()
        backend = GuardedInputBackend(native, EmergencyStop(), sleep=lambda _: None)
        self.assertEqual(backend.execute(two_clicks(), AutomationMode.OBSERVE, 7, RECT), 0)
        self.assertEqual(native.clicks, [])

    def test_emergency_stop_blocks_remaining_clicks(self) -> None:
        """第一下点击后触发停止，第二下不会发送。"""
        stop = EmergencyStop()
        native = FakeNative(on_click=lambda _: stop.trigger("测试"))
        with self.assertRaises(AutomationStopped):
            GuardedInputBackend(native, stop, sleep=lambda _: None).execute(two_clicks(), AutomationMode.AUTO, 7, RECT)
        self.assertEqual(native.clicks, [(110, 60)])

    def test_moved_or_covered_window_is_refused(self) -> None:
        """窗口移动或被遮挡时不点击。"""
        for native in (FakeNative(rect=(0, 0, 10, 10)), FakeNative(cover=99)):
            with self.subTest(native=native.rect), self.assertRaises(UnsafeClick):
                GuardedInputBackend(native, EmergencyStop(), sleep=lambda _: None).execute(
                    two_clicks(), AutomationMode.AUTO, 7, RECT)
            self.assertEqual(native.clicks, [])

    def test_user_moving_mouse_stops_automation(self) -> None:
        """两次点击之间鼠标被人移动视为接管。"""
        def move(native):
            """模拟用户移动鼠标。"""
            native.cursor = (900, 900)

        stop = EmergencyStop()
        native = FakeNative(on_click=move)
        with self.assertRaises(AutomationStopped):
            GuardedInputBackend(native, stop, sleep=lambda _: None).execute(two_clicks(), AutomationMode.AUTO, 7, RECT)
        self.assertTrue(stop.is_set())
        self.assertEqual(len(native.clicks), 1)


class GateTests(unittest.TestCase):
    """稳定门。"""

    def frame(self, cards, valid=True, rect=RECT):
        """生成观测。"""
        return TableObservation(0.0, valid, (1280, 800), rect, tuple(boxes(cards)))

    def test_disagreeing_frames_never_unlock(self) -> None:
        """不一致的帧不放行，无效帧清零。"""
        gate = StableObservationGate(3)
        results = [gate.observe(self.frame(c)) for c in (["3", "4"], ["3", "6"], ["3", "4"], ["3", "4"])]
        self.assertEqual(results, [False]*4)
        self.assertTrue(gate.observe(self.frame(["3", "4"])))
        self.assertFalse(gate.observe(self.frame(["3", "4"], valid=False)))

    def test_window_change_resets(self) -> None:
        """窗口矩形变化视为不同画面。"""
        gate = StableObservationGate(2)
        gate.observe(self.frame(["3"]))
        self.assertFalse(gate.observe(self.frame(["3"], rect=(0, 0, 1600, 900))))


if __name__ == "__main__":
    unittest.main()
