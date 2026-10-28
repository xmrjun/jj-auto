"""出牌决策层：牌码、合法动作、历史校验、适配器与权重加载策略。"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from core.model_assets import (ModelIntegrityError, UnapprovedModelError, UnsafeModelFormatError,
                               resolve_decision_model)
from decision.card_codec import decode_cards, encode_cards, format_cards
from decision.douzero_adapter import DouZeroAdapter, MODEL_FILES, TorchDouZeroScorer
from decision.protocol import DecisionState, IllegalDecision, InconsistentState
from decision.rules import build_infoset, legal_actions


def state(position, hand, seq=(), three=("3", "4", "5")):
    """用 Han 牌点构造决策状态。"""
    return DecisionState(position, tuple(encode_cards(hand)), tuple(encode_cards(three)),
                         tuple((role, tuple(encode_cards(cards))) for role, cards in seq))


FARMER_HAND = "4 5 6 7 8 9 10 J Q K A 2 3 4 5 6 7".split()
LANDLORD_HAND = FARMER_HAND+["8", "9", "10"]


class FakeScorer:
    """按动作张数打分，越多越好；可返回错误长度。"""

    def __init__(self, broken=False):
        """创建假后端。"""
        self.broken = broken
        self.calls = 0

    def score(self, infoset, legal):
        """返回估值。"""
        self.calls += 1
        values = [float(len(action)) for action in legal]
        return values[:-1] if self.broken else values


class CodecAndRulesTests(unittest.TestCase):
    """牌码转换与合法动作。"""

    def test_codec_round_trips_jokers_and_ten(self) -> None:
        """10、J 与大小王往返一致。"""
        cards = ["10", "J", "小", "大"]
        self.assertEqual(decode_cards(encode_cards(cards)), cards)
        self.assertEqual(format_cards(["3", "大"]), "大 3")
        with self.assertRaises(ValueError):
            encode_cards(["T"])

    def test_following_includes_pass_and_bombs(self) -> None:
        """跟单张时可以不出，也可以用炸弹。"""
        moves = legal_actions(encode_cards(["5", "5", "5", "5", "4"]), [[10]])
        self.assertIn([], moves)
        self.assertIn([5, 5, 5, 5], moves)
        self.assertNotIn([4], moves)

    def test_rocket_cannot_be_beaten(self) -> None:
        """王炸之后只能不出。"""
        self.assertEqual(legal_actions(encode_cards(["2", "2", "2", "2"]), [[20, 30]]), [[]])


class InfoSetTests(unittest.TestCase):
    """识别历史必须符合规则才交给模型。"""

    def test_landlord_first_turn_has_no_pass(self) -> None:
        """地主首出不能不出，底牌和其他人手牌正确。"""
        info = build_infoset(state("landlord", LANDLORD_HAND), "lstm")
        self.assertNotIn([], info.legal_actions)
        self.assertEqual(len(info.other_hand_cards), 34)
        self.assertEqual(info.num_cards_left_dict, {"landlord": 20, "landlord_down": 17, "landlord_up": 17})

    def test_wrong_turn_order_is_rejected(self) -> None:
        """第一手必须是地主。"""
        with self.assertRaises(InconsistentState):
            build_infoset(state("landlord_up", FARMER_HAND, [("landlord_down", ["3"])]))

    def test_pass_on_free_lead_is_rejected(self) -> None:
        """自由出牌时不能记为不出。"""
        with self.assertRaises(InconsistentState):
            build_infoset(state("landlord_up", FARMER_HAND, [("landlord", [])]))

    def test_weaker_follow_is_rejected(self) -> None:
        """压不过上一手的记录说明识别有误。"""
        seq = [("landlord", ["K"]), ("landlord_down", ["3"])]
        with self.assertRaises(InconsistentState):
            build_infoset(state("landlord_up", FARMER_HAND, seq))

    def test_hand_count_must_match_history(self) -> None:
        """本家张数与记录推算不符时拒绝决策。"""
        seq = [("landlord", ["3"]), ("landlord_down", ["4"]), ("landlord_up", ["5"]),
               ("landlord", []), ("landlord_down", [])]
        with self.assertRaises(InconsistentState):
            build_infoset(state("landlord_up", FARMER_HAND, seq))

    def test_landlord_played_bottom_card_leaves_three(self) -> None:
        """地主打出的底牌从剩余底牌特征中移除。"""
        seq = [("landlord", ["3"]), ("landlord_down", ["4"])]
        info = build_infoset(state("landlord_up", FARMER_HAND, seq, three=("3", "大", "A")), "resnet")
        self.assertEqual(sorted(info.three_landlord_cards), [14, 30])
        self.assertEqual(info.last_move, [4])
        self.assertEqual(info.card_play_action_seq[0], ("landlord", [3]))


class AdapterTests(unittest.TestCase):
    """适配器只返回合法动作。"""

    def test_picks_highest_scored_legal_action(self) -> None:
        """估值最高者被选中，候选按估值排序。"""
        decision = DouZeroAdapter(FakeScorer()).decide(state("landlord", LANDLORD_HAND))
        self.assertGreater(len(decision.action), 1)
        self.assertEqual(decision.value, decision.candidates[0][1])

    def test_bad_backend_output_is_rejected(self) -> None:
        """估值数量不符不能被当作决策。"""
        with self.assertRaises(IllegalDecision):
            DouZeroAdapter(FakeScorer(broken=True)).decide(state("landlord", LANDLORD_HAND))

    def test_single_legal_action_skips_backend(self) -> None:
        """只能不出时不调用模型。"""
        scorer = FakeScorer()
        seq = [("landlord", ["大", "小"])]
        decision = DouZeroAdapter(scorer).decide(state("landlord_down", FARMER_HAND, seq, three=("3", "4", "8")))
        self.assertEqual((decision.action, decision.value, scorer.calls), ((), None, 0))


class DecisionModelPolicyTests(unittest.TestCase):
    """决策权重只接受清单批准且哈希一致的 .ckpt 状态字典。"""

    def setUp(self) -> None:
        """创建临时目录。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def manifest_for(self, paths):
        """为文件生成批准清单。"""
        return {"schema_version": 1, "artifacts": [
            {"path": f"{p.parent.name}/{p.name}", "format": "torch_state_dict", "approved_for_runtime": True,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}

    def test_rejects_pickle_formats_and_unapproved_and_tampered(self) -> None:
        """.pkl/.pt 拒绝；未登记拒绝；内容被改动拒绝。"""
        path = self.root/"m"/"resnet_landlord.ckpt"
        path.parent.mkdir()
        path.write_bytes(b"original")
        manifest = self.manifest_for([path])
        self.assertEqual(resolve_decision_model(path, manifest=manifest).format, "torch_state_dict")
        with self.assertRaises(UnsafeModelFormatError):
            resolve_decision_model(self.root/"m"/"x.pkl", manifest=manifest)
        with self.assertRaises(UnapprovedModelError):
            resolve_decision_model(self.root/"m"/"other.ckpt", manifest=manifest)
        path.write_bytes(b"tampered")
        with self.assertRaises(ModelIntegrityError):
            resolve_decision_model(path, manifest=manifest)

    def test_repository_manifest_pins_every_approved_artifact(self) -> None:
        """仓库清单中每个批准制品都有哈希和固定提交来源。"""
        manifest = json.loads(Path("security/model-manifest.json").read_text(encoding="utf-8"))
        approved = [entry for entry in manifest["artifacts"] if entry.get("approved_for_runtime")]
        self.assertEqual({entry["format"] for entry in approved}, {"onnx", "torch_state_dict"})
        for entry in approved:
            self.assertEqual(len(entry["sha256"]), 64)
            commit = entry["source"].split("/blob/")[1].split("/")[0]
            self.assertEqual(len(commit), 40)


class TorchScorerTests(unittest.TestCase):
    """随机初始化权重（非真实模型）验证编码与网络形状匹配，并拒绝任意 pickle 对象。"""

    @classmethod
    def setUpClass(cls) -> None:
        """缺少 torch 时跳过。"""
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("未安装 torch")

    def build(self, model_type):
        """保存随机权重并返回目录与清单。"""
        import torch

        from third_party.douzero import models

        folder = Path(tempfile.mkdtemp(dir=self.root))/model_type
        folder.mkdir()
        paths = []
        for position, name in MODEL_FILES[model_type].items():
            if model_type == "resnet":
                network = models.ResnetModel()
            else:
                network = models.LandlordLstmModel() if position == "landlord" else models.FarmerLstmModel()
            torch.save(network.state_dict(), folder/name)
            paths.append(folder/name)
        return folder, DecisionModelPolicyTests.manifest_for(self, paths)

    def setUp(self) -> None:
        """创建临时目录。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_both_model_types_score_every_position(self) -> None:
        """LSTM 与 ResNet 的三个角色都能对合法动作打分。"""
        cases = [state("landlord", LANDLORD_HAND),
                 state("landlord_down", FARMER_HAND, [("landlord", ["3"])], three=("3", "8", "8")),
                 state("landlord_up", FARMER_HAND, [("landlord", ["3"]), ("landlord_down", [])],
                       three=("3", "8", "8"))]
        for model_type in ("lstm", "resnet"):
            folder, manifest = self.build(model_type)
            scorer = TorchDouZeroScorer(folder, manifest=manifest)
            self.assertEqual(scorer.model_type, model_type)
            adapter = DouZeroAdapter(scorer, model_type)
            for case in cases:
                with self.subTest(model_type=model_type, position=case.position):
                    info = adapter.infoset(case)
                    self.assertEqual(len(scorer.score(info, info.legal_actions)), len(info.legal_actions))

    def test_checkpoint_with_arbitrary_object_is_refused(self) -> None:
        """含自定义对象的 pickle 不会被 weights_only 执行。"""
        import pickle

        folder, manifest = self.build("resnet")
        target = folder/"resnet_landlord.ckpt"
        import torch

        torch.save({"payload": Path("x")}, target)
        manifest = DecisionModelPolicyTests.manifest_for(self, sorted(folder.iterdir()))
        with self.assertRaises(pickle.UnpicklingError):
            TorchDouZeroScorer(folder, manifest=manifest)


if __name__ == "__main__":
    unittest.main()
