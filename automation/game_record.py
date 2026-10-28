"""把记牌器已确认的各座位出牌整理为 DouZero 动作序列，并推断本家角色。

每次轮到本家时，自上次同步后每个座位恰好行动一次；没有新出牌即视为不出。
所有推算先在副本上完成，只有决策层校验通过后才提交，失败时下一帧可重试。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from decision.card_codec import encode_cards
from decision.protocol import DecisionState, InconsistentState, Position

SEAT_POSITIONS: dict[Position, dict[str, Position]] = {
    "landlord": {"self": "landlord", "right": "landlord_down", "left": "landlord_up"},
    "landlord_down": {"self": "landlord_down", "right": "landlord_up", "left": "landlord"},
    "landlord_up": {"self": "landlord_up", "right": "landlord", "left": "landlord_down"},
}
ROLE_TEXT: dict[Position, str] = {"landlord": "地主", "landlord_down": "地主下家", "landlord_up": "地主上家"}
CYCLE: tuple[str, str, str] = ("self", "right", "left")


@dataclass(frozen=True)
class TurnPreview:
    """尚未提交的一次本家回合推算。

    Attributes:
        state: 交给决策层的牌局。
        position: 本家角色。
        sequence: 推算后的完整动作序列（Han 牌点）。
        synced: 推算后各座位已消费的历史条数。
    """

    state: DecisionState
    position: Position
    sequence: tuple[tuple[Position, tuple[str, ...]], ...]
    synced: Mapping[str, int]


class RoundRecord:
    """一局的底牌、角色和动作序列。

    Attributes:
        three: 三张底牌。
        position: 本家角色；首个本家回合前为 None。
        sequence: 已提交的动作序列。
        synced: 各座位已消费的记牌器历史条数。
        expected_own: 自动模式最近一次已执行并验证的本家动作，用于交叉核对。
    """

    def __init__(self, bottom: Sequence[str], history: Mapping[str, Sequence[Sequence[str]]]) -> None:
        """在底牌亮出时开始记录。

        Args:
            bottom: 底牌牌点，必须三张。
            history: 记牌器当前历史；已有条目属于开局前，不计入本局。

        Raises:
            ValueError: 底牌不是三张。
        """
        if len(bottom) != 3:
            raise ValueError("底牌必须是三张")
        self.three: tuple[str, ...] = tuple(bottom)
        self.position: Position | None = None
        self.sequence: list[tuple[Position, tuple[str, ...]]] = []
        self.synced: dict[str, int] = {seat: len(history.get(seat, ())) for seat in CYCLE}
        self.expected_own: tuple[str, ...] | None = None

    def preview(self, hand: Sequence[str], history: Mapping[str, Sequence[Sequence[str]]]) -> TurnPreview:
        """推算当前本家回合的完整牌局，不修改记录。

        Args:
            hand: 当前稳定识别的本家手牌。
            history: 记牌器已确认的各座位历史。

        Returns:
            待提交的推算结果。

        Raises:
            InconsistentState: 角色无法确定、历史缩短或与自动执行的动作不一致。
        """
        fresh: dict[str, list[tuple[str, ...]]] = {}
        for seat in CYCLE:
            items = list(history.get(seat, ()))
            if len(items) < self.synced[seat]:
                raise InconsistentState(f"{seat} 历史条数减少，记牌器可能已重置")
            fresh[seat] = [tuple(item) for item in items[self.synced[seat]:]]
        events: list[tuple[str, tuple[str, ...]]] = []
        if self.position is None:
            position = self._first_position(len(hand), fresh)
            start = {"landlord": None, "landlord_down": "left", "landlord_up": "right"}[position]
            if start is not None:
                for seat in CYCLE[CYCLE.index(start):]:
                    queue = fresh[seat]
                    if len(queue) > 1:
                        raise InconsistentState("首个本家回合前同一座位出现多手牌")
                    events.append((seat, queue[0] if queue else ()))
        else:
            position = self.position
            own = fresh["self"][0] if fresh["self"] else ()
            if self.expected_own is not None and sorted(own) != sorted(self.expected_own):
                raise InconsistentState("记牌器记录的本家出牌与自动执行的动作不一致")
            while True:
                for seat in CYCLE:
                    queue = fresh[seat]
                    events.append((seat, queue.pop(0) if queue else ()))
                if not any(fresh.values()):
                    break
        mapping = SEAT_POSITIONS[position]
        sequence = tuple(self.sequence)+tuple((mapping[seat], cards) for seat, cards in events)
        state = DecisionState(
            position, tuple(encode_cards(hand)), tuple(encode_cards(self.three)),
            tuple((role, tuple(encode_cards(cards))) for role, cards in sequence))
        synced = {seat: len(history.get(seat, ())) for seat in CYCLE}
        return TurnPreview(state, position, sequence, synced)

    @staticmethod
    def _first_position(hand_count: int, fresh: Mapping[str, Sequence[tuple[str, ...]]]) -> Position:
        """在第一个本家回合按手牌张数和对手是否已出牌推断角色。

        地主先出：本家 20 张且无人出过牌为地主；17 张时下家已出说明下家是地主，
        否则上家已出说明上家是地主。

        Args:
            hand_count: 当前手牌张数。
            fresh: 本局各座位新出牌。

        Returns:
            本家角色。

        Raises:
            InconsistentState: 无法唯一确定。
        """
        if fresh["self"]:
            raise InconsistentState("第一个本家回合之前已记录本家出牌")
        if hand_count == 20 and not fresh["right"] and not fresh["left"]:
            return "landlord"
        if hand_count == 17:
            if fresh["right"]:
                return "landlord_up"
            if fresh["left"]:
                return "landlord_down"
            raise InconsistentState("本家 17 张但对手都未出牌，无法判断地主")
        raise InconsistentState(f"首个回合手牌 {hand_count} 张，无法判断角色")

    def commit(self, preview: TurnPreview) -> None:
        """提交已通过决策层校验的推算。

        Args:
            preview: preview 的返回值。
        """
        self.position = preview.position
        self.sequence = list(preview.sequence)
        self.synced = dict(preview.synced)
        self.expected_own = None

    def remaining(self, history: Mapping[str, Sequence[Sequence[str]]] | None = None) -> dict[str, int] | None:
        """返回各座位剩余张数，可叠加尚未提交的记牌器新出牌。

        Args:
            history: 记牌器当前历史；None 只按已提交序列计算。

        Returns:
            座位到张数；角色未定时为 None。
        """
        if self.position is None:
            return None
        counts: dict[Position, int] = {"landlord": 20, "landlord_down": 17, "landlord_up": 17}
        for role, cards in self.sequence:
            counts[role] -= len(cards)
        mapping = SEAT_POSITIONS[self.position]
        result = {seat: counts[mapping[seat]] for seat in CYCLE}
        if history is not None:
            for seat in CYCLE:
                for cards in list(history.get(seat, ()))[self.synced[seat]:]:
                    result[seat] -= len(cards)
        return result
