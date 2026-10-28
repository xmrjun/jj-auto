"""决策层的纯数据契约与异常。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Position = Literal["landlord", "landlord_up", "landlord_down"]
PLAY_ORDER: tuple[Position, Position, Position] = ("landlord", "landlord_down", "landlord_up")


class DecisionError(RuntimeError):
    """决策层错误基类。"""


class InconsistentState(DecisionError, ValueError):
    """识别得到的牌局历史与规则或张数矛盾。"""


class IllegalDecision(DecisionError):
    """后端返回的动作不在当前合法动作集合中。"""


@dataclass(frozen=True)
class DecisionState:
    """轮到本家出牌时的完整公开信息。

    Attributes:
        position: 本家角色。
        hand: 当前识别并已稳定确认的本家手牌牌码。
        three_landlord_cards: 三张底牌牌码。
        action_seq: 自地主首出起按顺序的全部动作，空元组表示不出。
    """

    position: Position
    hand: tuple[int, ...]
    three_landlord_cards: tuple[int, ...]
    action_seq: tuple[tuple[Position, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        """校验字段类型，避免把可变列表或未知角色带入模型。

        Raises:
            TypeError: 字段类型错误。
            ValueError: 角色未知。
        """
        if self.position not in PLAY_ORDER:
            raise ValueError(f"未知角色: {self.position}")
        if not isinstance(self.hand, tuple) or not isinstance(self.three_landlord_cards, tuple):
            raise TypeError("手牌与底牌必须是元组")
        if not isinstance(self.action_seq, tuple):
            raise TypeError("动作序列必须是元组")
        for item in self.action_seq:
            if (not isinstance(item, tuple) or len(item) != 2 or item[0] not in PLAY_ORDER
                    or not isinstance(item[1], tuple)):
                raise TypeError("动作序列元素必须是 (角色, 牌码元组)")


@dataclass(frozen=True)
class Decision:
    """一次出牌决策。

    Attributes:
        action: 选中的动作牌码，空元组表示不出。
        value: 模型对该动作的估值；只有一个合法动作时为 None。
        candidates: 估值最高的若干动作及估值，供界面展示。
    """

    action: tuple[int, ...]
    value: float | None
    candidates: tuple[tuple[tuple[int, ...], float], ...] = ()


class ActionScorer(Protocol):
    """为合法动作逐一打分的后端接口。"""

    def score(self, infoset: object, legal_actions: Sequence[Sequence[int]]) -> list[float]:
        """返回与 legal_actions 等长的估值。

        Args:
            infoset: decision.rules.InfoSet。
            legal_actions: 当前合法动作。

        Returns:
            每个合法动作的估值。
        """
        ...
