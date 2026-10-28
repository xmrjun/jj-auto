"""斗地主合法动作、历史一致性校验和 DouZero 信息集构造。

合法动作生成沿用 DouZero（Apache-2.0）的牌型生成与比较规则，
见 third_party/douzero 与 THIRD_PARTY.md。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from decision.card_codec import FULL_DECK_ENV, within_deck
from decision.protocol import PLAY_ORDER, DecisionState, InconsistentState, Position
from third_party.douzero import move_detector as md
from third_party.douzero import move_selector as ms
from third_party.douzero.move_generator import MovesGener

INITIAL_COUNTS: dict[Position, int] = {"landlord": 20, "landlord_up": 17, "landlord_down": 17}
# ResNet 权重推理时使用的固定叫分/加倍特征，取自其公开推理代码的默认值。
RESNET_BID_INFO: tuple[tuple[float, float, float], ...] = ((1, 0.5, 1), (1, 1, 1), (1, 1, -4), (1, 1, 1))
RESNET_MULTIPLY_INFO: tuple[float, float, float] = (1, 1, 1)


def move_type(cards: Sequence[int]) -> int:
    """返回 DouZero 牌型编号。

    Args:
        cards: 升序或任意顺序的牌码。

    Returns:
        move_detector 中的 TYPE_* 编号；非法牌型为 TYPE_15_WRONG。

    Examples:
        >>> move_type([3, 3]) == md.TYPE_2_PAIR
        True
    """
    return int(md.get_move_type(sorted(cards))["type"])


def is_bomb(cards: Sequence[int]) -> bool:
    """判断动作是否为炸弹或王炸。

    Args:
        cards: 牌码。

    Returns:
        四张相同或大小王时为 True。

    Examples:
        >>> is_bomb([5, 5, 5, 5]), is_bomb([20, 30]), is_bomb([5, 5])
        (True, True, False)
    """
    return move_type(cards) in (md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB)


def rival_move(action_seq: Sequence[Sequence[int]]) -> list[int]:
    """按 DouZero 规则取当前需要压过的动作。

    最后一手不出时看倒数第二手；两手都不出则自由出牌。

    Args:
        action_seq: 仅包含牌码的历史动作。

    Returns:
        需要压过的动作；空列表表示自由出牌。

    Examples:
        >>> rival_move([[3], []])
        [3]
        >>> rival_move([[3], [], []])
        []
    """
    if not action_seq:
        return []
    if len(action_seq[-1]) == 0:
        return list(action_seq[-2]) if len(action_seq) >= 2 else []
    return list(action_seq[-1])


def legal_actions(hand: Sequence[int], action_seq: Sequence[Sequence[int]]) -> list[list[int]]:
    """生成当前手牌对上一手的全部合法动作。

    Args:
        hand: 本家手牌牌码。
        action_seq: 仅包含牌码的历史动作。

    Returns:
        升序牌码动作列表；需要跟牌时包含不出 ``[]``。

    Examples:
        >>> legal_actions([3, 4], [[5]])
        [[]]
        >>> sorted(legal_actions([3, 3], []))
        [[3], [3, 3]]
    """
    generator = MovesGener(sorted(hand))
    rival = rival_move(action_seq)
    rival_type = md.get_move_type(rival)
    kind = rival_type["type"]
    length = rival_type.get("len", 1)
    moves: list[list[int]]
    if kind == md.TYPE_0_PASS:
        moves = generator.gen_moves()
    elif kind == md.TYPE_1_SINGLE:
        moves = ms.filter_type_1_single(generator.gen_type_1_single(), rival)
    elif kind == md.TYPE_2_PAIR:
        moves = ms.filter_type_2_pair(generator.gen_type_2_pair(), rival)
    elif kind == md.TYPE_3_TRIPLE:
        moves = ms.filter_type_3_triple(generator.gen_type_3_triple(), rival)
    elif kind == md.TYPE_4_BOMB:
        moves = ms.filter_type_4_bomb(generator.gen_type_4_bomb()+generator.gen_type_5_king_bomb(), rival)
    elif kind == md.TYPE_5_KING_BOMB:
        moves = []
    elif kind == md.TYPE_6_3_1:
        moves = ms.filter_type_6_3_1(generator.gen_type_6_3_1(), rival)
    elif kind == md.TYPE_7_3_2:
        moves = ms.filter_type_7_3_2(generator.gen_type_7_3_2(), rival)
    elif kind == md.TYPE_8_SERIAL_SINGLE:
        moves = ms.filter_type_8_serial_single(generator.gen_type_8_serial_single(repeat_num=length), rival)
    elif kind == md.TYPE_9_SERIAL_PAIR:
        moves = ms.filter_type_9_serial_pair(generator.gen_type_9_serial_pair(repeat_num=length), rival)
    elif kind == md.TYPE_10_SERIAL_TRIPLE:
        moves = ms.filter_type_10_serial_triple(generator.gen_type_10_serial_triple(repeat_num=length), rival)
    elif kind == md.TYPE_11_SERIAL_3_1:
        moves = ms.filter_type_11_serial_3_1(generator.gen_type_11_serial_3_1(repeat_num=length), rival)
    elif kind == md.TYPE_12_SERIAL_3_2:
        moves = ms.filter_type_12_serial_3_2(generator.gen_type_12_serial_3_2(repeat_num=length), rival)
    elif kind == md.TYPE_13_4_2:
        moves = ms.filter_type_13_4_2(generator.gen_type_13_4_2(), rival)
    elif kind == md.TYPE_14_4_22:
        moves = ms.filter_type_14_4_22(generator.gen_type_14_4_22(), rival)
    else:
        raise InconsistentState(f"上一手不是合法牌型: {rival}")
    if kind not in (md.TYPE_0_PASS, md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB):
        moves = moves+generator.gen_type_4_bomb()+generator.gen_type_5_king_bomb()
    if rival:
        moves = moves+[[]]
    return [sorted(move) for move in moves]


def _subtract(pool: Counter[int], cards: Sequence[int], message: str) -> None:
    """从多重集中扣除牌，不足时报告矛盾。

    Args:
        pool: 被修改的多重集。
        cards: 要扣除的牌码。
        message: 失败说明。
    Raises:
        InconsistentState: 牌数不足。
    """
    for card in cards:
        if pool[card] <= 0:
            raise InconsistentState(message)
        pool[card] -= 1


@dataclass
class InfoSet:
    """DouZero 编码函数读取的信息集字段。

    Attributes:
        player_position: 本家角色。
        player_hand_cards: 本家手牌。
        legal_actions: 合法动作。
        card_play_action_seq: 历史动作；LSTM 为牌码列表，ResNet 为 (角色, 牌码) 列表。
        other_hand_cards: 另外两家手牌之和。
        last_move: 需要压过的动作。
        last_two_moves: 最近两手。
        last_move_dict: 各角色最近一次动作。
        played_cards: 各角色已出牌。
        num_cards_left_dict: 各角色剩余张数。
        three_landlord_cards: 地主尚未打出的底牌。
        bomb_num: 已出炸弹和王炸次数。
        bid_info: ResNet 叫分特征。
        multiply_info: ResNet 加倍特征。
        all_handcards: ResNet 编码函数读取但不参与输出的占位字段。
    """

    player_position: Position
    player_hand_cards: list[int]
    legal_actions: list[list[int]]
    card_play_action_seq: list[object]
    other_hand_cards: list[int]
    last_move: list[int]
    last_two_moves: list[list[int]]
    last_move_dict: dict[str, list[int]]
    played_cards: dict[str, list[int]]
    num_cards_left_dict: dict[str, int]
    three_landlord_cards: list[int]
    bomb_num: int
    bid_info: list[list[float]] = field(default_factory=lambda: [list(row) for row in RESNET_BID_INFO])
    multiply_info: list[float] = field(default_factory=lambda: list(RESNET_MULTIPLY_INFO))
    all_handcards: dict[str, list[int]] = field(
        default_factory=lambda: {position: [] for position in PLAY_ORDER})


def build_infoset(state: DecisionState, model_type: str = "lstm") -> InfoSet:
    """校验历史后构造 DouZero 信息集。

    Args:
        state: 轮到本家时的公开信息。
        model_type: ``lstm``（DouZero 原版）或 ``resnet``。

    Returns:
        可直接交给编码函数的信息集。

    Raises:
        ValueError: model_type 未知。
        InconsistentState: 行动顺序、牌型、张数或底牌与规则矛盾。
    """
    if model_type not in ("lstm", "resnet"):
        raise ValueError(f"未知模型类型: {model_type}")
    if len(state.three_landlord_cards) != 3 or not within_deck(state.three_landlord_cards):
        raise InconsistentState("底牌必须是三张合法牌")
    if not state.hand:
        raise InconsistentState("本家手牌为空，无需决策")
    played: dict[Position, list[int]] = {position: [] for position in PLAY_ORDER}
    last_moves: dict[str, list[int]] = {position: [] for position in PLAY_ORDER}
    pool: Counter[int] = Counter(FULL_DECK_ENV)
    _subtract(pool, state.hand, "本家手牌超过整副牌数量")
    remaining_three = list(state.three_landlord_cards)
    plain: list[list[int]] = []
    bombs = 0
    for index, (position, action) in enumerate(state.action_seq):
        if position != PLAY_ORDER[index % 3]:
            raise InconsistentState(f"第 {index+1} 手应由 {PLAY_ORDER[index % 3]} 出牌，实际为 {position}")
        cards = sorted(action)
        if cards:
            if move_type(cards) == md.TYPE_15_WRONG:
                raise InconsistentState(f"识别到非法牌型: {cards}")
            rival = rival_move(plain)
            if rival and cards not in legal_actions(cards, plain):
                raise InconsistentState(f"{position} 的 {cards} 压不过 {rival}")
        elif index == 0 or not rival_move(plain):
            raise InconsistentState(f"{position} 在自由出牌时不出")
        _subtract(pool, cards, f"{position} 的出牌超过剩余牌数")
        played[position].extend(cards)
        last_moves[position] = list(cards)
        if len(played[position]) > INITIAL_COUNTS[position]:
            raise InconsistentState(f"{position} 出牌张数超过初始手牌")
        if position == "landlord":
            for card in cards:
                if not remaining_three:
                    break
                if card in remaining_three:
                    remaining_three.remove(card)
        bombs += int(bool(cards) and is_bomb(cards))
        plain.append(cards)
    if PLAY_ORDER[len(state.action_seq) % 3] != state.position:
        raise InconsistentState("当前不是本家行动顺序")
    left = {position: INITIAL_COUNTS[position]-len(played[position]) for position in PLAY_ORDER}
    if left[state.position] != len(state.hand):
        raise InconsistentState(
            f"手牌 {len(state.hand)} 张与记录推算的 {left[state.position]} 张不一致")
    if any(count <= 0 for position, count in left.items() if position != state.position):
        raise InconsistentState("已有玩家出完牌，本局应已结束")
    legal = legal_actions(state.hand, plain)
    if not legal:
        raise InconsistentState("没有合法动作")
    seq: list[object] = (list(plain) if model_type == "lstm"
                         else [(position, list(action)) for position, action in state.action_seq])
    last_two = [[], []]
    for move in plain[-2:]:
        last_two.insert(0, move)
        last_two = last_two[:2]
    return InfoSet(
        player_position=state.position,
        player_hand_cards=sorted(state.hand),
        legal_actions=legal,
        card_play_action_seq=seq,
        other_hand_cards=sorted(pool.elements()),
        last_move=rival_move(plain),
        last_two_moves=last_two,
        last_move_dict=last_moves,
        played_cards={key: list(value) for key, value in played.items()},
        num_cards_left_dict=left,
        three_landlord_cards=remaining_three,
        bomb_num=min(bombs, 14),
    )
