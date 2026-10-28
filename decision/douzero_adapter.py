"""DouZero 出牌决策：合法动作校验、本地权重安全加载与打分。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

from decision.protocol import ActionScorer, Decision, DecisionState, IllegalDecision, PLAY_ORDER, Position
from decision.rules import InfoSet, build_infoset

if TYPE_CHECKING:
    import torch

MODEL_FILES: dict[str, dict[Position, str]] = {
    "resnet": {position: f"resnet_{position}.ckpt" for position in PLAY_ORDER},
    "lstm": {position: f"{position}.ckpt" for position in PLAY_ORDER},
}


class DouZeroAdapter:
    """把识别得到的牌局状态交给打分后端，并只返回合法动作。

    Attributes:
        scorer: 合法动作打分后端。
        model_type: 后端使用的特征编码类型。
    """

    def __init__(self, scorer: ActionScorer, model_type: str = "lstm") -> None:
        """创建适配器。

        Args:
            scorer: 合法动作打分后端。
            model_type: ``lstm`` 或 ``resnet``。

        Raises:
            ValueError: 模型类型未知。
        """
        if model_type not in MODEL_FILES:
            raise ValueError(f"未知模型类型: {model_type}")
        self.scorer = scorer
        self.model_type = model_type

    def infoset(self, state: DecisionState) -> InfoSet:
        """校验并构造信息集。

        Args:
            state: 当前牌局。

        Returns:
            DouZero 信息集。

        Raises:
            InconsistentState: 历史与规则或张数矛盾。
        """
        return build_infoset(state, self.model_type)

    def decide(self, state: DecisionState) -> Decision:
        """返回估值最高的合法动作。

        Args:
            state: 当前牌局。

        Returns:
            选中的动作、估值与前三候选。

        Raises:
            InconsistentState: 历史与规则或张数矛盾。
            IllegalDecision: 后端估值数量不符或含非有限数。
        """
        infoset = self.infoset(state)
        legal = infoset.legal_actions
        if len(legal) == 1:
            return Decision(tuple(legal[0]), None)
        values = list(self.scorer.score(infoset, legal))
        if len(values) != len(legal) or not all(isinstance(v, float) and isfinite(v) for v in values):
            raise IllegalDecision("决策后端返回的估值数量或数值无效")
        ranked = sorted(range(len(legal)), key=lambda index: values[index], reverse=True)
        best = tuple(legal[ranked[0]])
        if list(best) not in legal:
            raise IllegalDecision(f"后端选择了非法动作: {best}")
        candidates = tuple((tuple(legal[index]), values[index]) for index in ranked[:3])
        return Decision(best, values[ranked[0]], candidates)


def detect_model_type(directory: str | Path) -> str:
    """根据目录中的文件名判断权重类型。

    Args:
        directory: 权重目录。

    Returns:
        ``resnet`` 或 ``lstm``。

    Raises:
        FileNotFoundError: 目录中缺少完整的三份权重。
    """
    folder = Path(directory)
    for model_type, names in MODEL_FILES.items():
        if all((folder/name).is_file() for name in names.values()):
            return model_type
    raise FileNotFoundError(
        f"{folder} 中没有完整的 DouZero 权重（需要 resnet_landlord/landlord_up/landlord_down.ckpt "
        "或 landlord/landlord_up/landlord_down.ckpt）")


class TorchDouZeroScorer:
    """使用本地 DouZero 权重为合法动作打分。

    权重先经 security/model-manifest.json 的格式、批准与 SHA-256 校验，
    再以 ``torch.load(weights_only=True)`` 读取，只接受张量状态字典。

    Attributes:
        directory: 权重目录。
        model_type: ``resnet`` 或 ``lstm``。
        device: 推理设备。
        models: 已加载的三个角色模型。
    """

    def __init__(self, directory: str | Path, device: str = "cpu", model_type: str | None = None,
                 manifest: Mapping[str, object] | None = None) -> None:
        """校验并加载三个角色的模型。

        Args:
            directory: 权重目录。
            device: ``cpu`` 或 ``cuda``；CUDA 不可用时回退 CPU。
            model_type: 省略时按文件名判断。
            manifest: 测试注入的模型清单；省略时读取 security/model-manifest.json。

        Raises:
            FileNotFoundError: 权重缺失。
            ModelAssetError: 权重未批准、格式不符或哈希不一致。
            RuntimeError: 权重与网络结构不匹配。
        """
        import torch

        from core.model_assets import resolve_decision_model

        self.directory = Path(directory)
        self.model_type = model_type or detect_model_type(self.directory)
        if self.model_type not in MODEL_FILES:
            raise ValueError(f"未知模型类型: {self.model_type}")
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.models: dict[str, torch.nn.Module] = {}
        for position, name in MODEL_FILES[self.model_type].items():
            selected = resolve_decision_model(self.directory/name, manifest=manifest)
            self.models[position] = self._load(position, selected.path)

    def _load(self, position: Position, path: Path) -> "torch.nn.Module":
        """以只读张量方式载入单个角色模型。

        Args:
            position: 角色。
            path: 已校验的权重路径。

        Returns:
            eval 模式的网络。

        Raises:
            RuntimeError: 权重缺少网络需要的参数或形状不符。
        """
        import torch

        from third_party.douzero import models

        network: torch.nn.Module
        if self.model_type == "resnet":
            network = models.ResnetModel()
        elif position == "landlord":
            network = models.LandlordLstmModel()
        else:
            network = models.FarmerLstmModel()
        state = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(state, dict):
            raise RuntimeError(f"{path.name} 不是状态字典")
        expected = network.state_dict()
        missing = [key for key in expected if key not in state]
        if missing:
            raise RuntimeError(f"{path.name} 缺少参数: {', '.join(missing[:5])}")
        network.load_state_dict({key: state[key] for key in expected})
        network.to(self.device)
        network.eval()
        return network

    def score(self, infoset: object, legal_actions: Sequence[Sequence[int]]) -> list[float]:
        """为每个合法动作计算模型估值。

        Args:
            infoset: decision.rules.InfoSet。
            legal_actions: 与信息集一致的合法动作。

        Returns:
            每个动作的估值。

        Raises:
            TypeError: infoset 类型错误。
        """
        import torch

        from third_party.douzero import encoding

        if not isinstance(infoset, InfoSet):
            raise TypeError("infoset 必须是 decision.rules.InfoSet")
        if self.model_type == "resnet":
            obs = encoding._get_obs_resnet(infoset, infoset.player_position)
        else:
            obs = encoding.get_obs(infoset)
        z = torch.from_numpy(obs["z_batch"]).float().to(self.device)
        x = torch.from_numpy(obs["x_batch"]).float().to(self.device)
        with torch.no_grad():
            values = self.models[infoset.player_position].forward(z, x, return_value=True)["values"]
        return [float(v) for v in values.detach().cpu().reshape(-1).tolist()]
