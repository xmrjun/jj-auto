"""闭环协调器：观察模式零点击、自动模式先选牌再出牌、验证、超时、停止与局间状态。"""

import unittest

from automation.coordinator import AutomationConfig, AutomationCoordinator
from automation.emergency_stop import EmergencyStop
from automation.input_backend import GuardedInputBackend
from automation.types import AutomationMode, AutomationPhase, ButtonMatch, TableObservation
from decision.douzero_adapter import DouZeroAdapter
from tests.test_automation_units import RECT, FakeNative, boxes

HAND20 = "3 4 5 6 7 8 9 10 J Q K A 2 2 2 9 9 6 6 小".split()
HAND17 = "3 4 5 6 7 8 9 10 J Q K A 2 2 2 9 9".split()
BOTTOM = ("6", "6", "小")


class SinglesFirst:
    """偏好牌面最小的单张，便于断言。"""

    def score(self, infoset, legal):
        """单张得分最高，越小越好；不出最低。"""
        return [(-100.0 if not action else (100.0-action[0] if len(action) == 1 else 0.0)) for action in legal]


class PassAlways:
    """总是不出（如果合法）。"""

    def score(self, infoset, legal):
        """不出得分最高。"""
        return [1.0 if not action else 0.0 for action in legal]


def button(name, x=700, y=600):
    """生成按钮匹配。"""
    return ButtonMatch(name, (x-40, y-15, x+40, y+15), 0.95)


class Clock:
    """手动推进的时钟。"""

    def __init__(self):
        """从零开始。"""
        self.now = 0.0

    def __call__(self):
        """返回当前时间。"""
        return self.now


class CoordinatorTests(unittest.TestCase):
    """使用假输入与假打分后端，不加载模型、不产生真实点击。"""

    def make(self, mode, scorer=None, **config):
        """创建协调器并切换模式。"""
        self.native = FakeNative()
        self.stop = EmergencyStop()
        self.clock = Clock()
        backend = GuardedInputBackend(self.native, self.stop, sleep=lambda _: None)
        coordinator = AutomationCoordinator(DouZeroAdapter(scorer or SinglesFirst()), backend,
                                            AutomationConfig(**config), self.clock)
        coordinator.set_mode(mode)
        return coordinator

    def feed(self, coordinator, frames=3, hand=HAND20, raised=(), buttons=("play",), bottom=BOTTOM,
             history=None, extra_buttons=None):
        """连续送入相同帧。"""
        status = None
        for _ in range(frames):
            self.clock.now += 0.2
            obs = TableObservation(
                self.clock.now, True, (1280, 800), RECT, tuple(boxes(hand, raised)),
                {"self": (), "left": (), "right": ()}, bottom,
                {name: button(name) for name in buttons} | (extra_buttons or {}), history or {})
            status = coordinator.handle(obs, 7)
        return status

    def test_disabled_never_acts(self) -> None:
        """关闭模式不处理任何帧。"""
        coordinator = self.make("disabled")
        status = self.feed(coordinator, 5)
        self.assertEqual((status.phase, self.native.clicks), (AutomationPhase.WAITING, []))

    def test_observe_suggests_but_never_clicks(self) -> None:
        """观察模式给出建议和计划点击，但不点击。"""
        coordinator = self.make("observe")
        status = self.feed(coordinator)
        self.assertEqual(status.role, "地主")
        self.assertEqual(status.suggestion, "建议：3")
        self.assertEqual(len(status.plan), 2)
        self.assertEqual(self.native.clicks, [])

    def test_auto_selects_then_plays_then_verifies(self) -> None:
        """自动模式：选牌 → 画面确认选中 → 点击出牌 → 张数减少才算完成。"""
        coordinator = self.make("auto")
        self.feed(coordinator)
        self.assertEqual(len(self.native.clicks), 1)
        self.assertEqual(self.native.clicks[0], (RECT[0]+112, RECT[1]+420))
        self.assertEqual(coordinator.status.phase, AutomationPhase.VERIFYING)
        self.feed(coordinator, raised={0})
        self.assertEqual(len(self.native.clicks), 2, "确认选中后点击出牌按钮")
        self.assertEqual(self.native.clicks[1], (RECT[0]+700, RECT[1]+600))
        self.feed(coordinator, hand=HAND20[1:], buttons=())
        self.assertEqual(coordinator.status.phase, AutomationPhase.PLAYING)
        self.assertEqual(coordinator.status.suggestion, "已出 3")
        self.assertEqual(coordinator.record.expected_own, ("3",))
        self.assertEqual(len(self.native.clicks), 2)

    def test_selection_mismatch_pauses_without_retry_storm(self) -> None:
        """选牌没有生效时只纠正一次，然后暂停，不会连续点击。"""
        coordinator = self.make("auto", max_corrections=1)
        self.feed(coordinator)
        self.feed(coordinator)
        self.feed(coordinator, 6)
        self.assertEqual(coordinator.status.phase, AutomationPhase.PAUSED)
        self.assertEqual(len(self.native.clicks), 2)
        self.feed(coordinator, 6)
        self.assertEqual(len(self.native.clicks), 2)

    def test_verification_timeout_pauses(self) -> None:
        """点击出牌后画面一直不变化，超时暂停。"""
        coordinator = self.make("auto", verify_timeout=1.0)
        self.feed(coordinator)
        self.feed(coordinator, raised={0})
        self.feed(coordinator, 10, raised={0})
        self.assertEqual(coordinator.status.phase, AutomationPhase.PAUSED)
        self.assertIn("超时", coordinator.status.reason)
        self.assertEqual(len(self.native.clicks), 2)

    def test_pass_clicks_pass_button(self) -> None:
        """模型选择不出时点击“不出”，手牌不变且按钮消失即完成。"""
        coordinator = self.make("auto", scorer=PassAlways())
        self.feed(coordinator, hand=HAND17, buttons=())
        self.feed(coordinator, hand=HAND17, buttons=("play", "pass"), history={"left": [["3"]]})
        self.assertEqual(coordinator.status.suggestion, "建议：不出")
        self.assertEqual(len(self.native.clicks), 1)
        self.feed(coordinator, hand=HAND17, buttons=(), history={"left": [["3"]]})
        self.assertEqual(coordinator.status.suggestion, "已不出")

    def test_emergency_stop_disables(self) -> None:
        """紧急停止后模式变为关闭且不再点击。"""
        coordinator = self.make("auto")
        self.stop.trigger("测试停止")
        status = self.feed(coordinator)
        self.assertEqual((status.mode, status.reason), (AutomationMode.DISABLED, "测试停止"))
        self.assertEqual(self.native.clicks, [])

    def test_inconsistent_history_waits_without_clicking(self) -> None:
        """17 张手牌但对手都没出过牌时无法判断角色，不操作。"""
        coordinator = self.make("auto")
        status = self.feed(coordinator, hand=HAND17)
        self.assertIn("不一致", status.reason)
        self.assertEqual(self.native.clicks, [])

    def test_blocking_dialog_mid_round_pauses_and_round_end_resumes(self) -> None:
        """局中出现托管/弹窗即暂停；出现继续按钮后结束本局并恢复等待。"""
        coordinator = self.make("auto")
        self.feed(coordinator, buttons=())
        self.assertIsNotNone(coordinator.record)
        self.feed(coordinator, 1, buttons=("trusteeship",))
        self.assertEqual(coordinator.status.phase, AutomationPhase.PAUSED)
        self.feed(coordinator, buttons=("continue",))
        self.assertEqual(coordinator.status.phase, AutomationPhase.WAITING)
        self.assertIsNone(coordinator.record)
        self.assertEqual(self.native.clicks, [])

    def test_dialog_between_rounds_does_not_lock(self) -> None:
        """局间弹窗只提示，不锁定暂停。"""
        coordinator = self.make("auto")
        self.feed(coordinator, 1, buttons=("confirm",), bottom=())
        self.assertNotEqual(coordinator.status.phase, AutomationPhase.PAUSED)

    def test_bid_uses_strength(self) -> None:
        """叫分按钮出现时按手牌强度叫分，按钮消失即确认。"""
        coordinator = self.make("auto")
        self.feed(coordinator, hand=HAND17, buttons=("bid_1", "bid_2", "bid_3", "bid_pass"), bottom=())
        self.assertEqual(coordinator.status.suggestion, "手牌强度 7，叫1分")
        self.assertEqual(len(self.native.clicks), 1)
        self.feed(coordinator, hand=HAND17, buttons=(), bottom=())
        self.assertEqual(coordinator.status.phase, AutomationPhase.BIDDING)
        self.assertEqual(len(self.native.clicks), 1)

    def test_bottom_must_stay_empty_before_round_ends(self) -> None:
        """底牌区短暂为空不结束本局。"""
        coordinator = self.make("observe", round_end_seconds=3.0)
        self.feed(coordinator, buttons=())
        self.feed(coordinator, 3, buttons=(), bottom=())
        self.assertIsNotNone(coordinator.record)
        self.feed(coordinator, 20, buttons=(), bottom=())
        self.assertIsNone(coordinator.record)


if __name__ == "__main__":
    unittest.main()
