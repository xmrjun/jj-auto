"""Han 牌点名称与 DouZero 整数牌码之间的双向转换。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

HAN_TO_ENV: dict[str, int] = {
    "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14, "2": 17, "小": 20, "大": 30,
}
ENV_TO_HAN: dict[int, str] = {value: key for key, value in HAN_TO_ENV.items()}
FULL_DECK_ENV: tuple[int, ...] = tuple(
    sorted([card for card in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]))
DECK_LIMITS: dict[int, int] = dict(Counter(FULL_DECK_ENV))


def encode_cards(cards: Iterable[str]) -> list[int]:
    """把 Han 牌点转换为升序的 DouZero 牌码。

    Args:
        cards: Han 牌点，如 ``["10", "J", "小"]``。

    Returns:
        升序整数牌码。

    Raises:
        TypeError: 牌点不是字符串。
        ValueError: 出现未知牌点。

    Examples:
        >>> encode_cards(["大", "3", "10"])
        [3, 10, 30]
    """
    result: list[int] = []
    for card in cards:
        if not isinstance(card, str):
            raise TypeError("牌点必须是字符串")
        if card not in HAN_TO_ENV:
            raise ValueError(f"未知牌点: {card}")
        result.append(HAN_TO_ENV[card])
    return sorted(result)


def decode_cards(cards: Iterable[int]) -> list[str]:
    """把 DouZero 牌码转换为按牌码升序的 Han 牌点。

    Args:
        cards: DouZero 整数牌码。

    Returns:
        Han 牌点列表。

    Raises:
        TypeError: 牌码不是整数。
        ValueError: 出现未知牌码。

    Examples:
        >>> decode_cards([30, 10, 20])
        ['10', '小', '大']
    """
    result: list[int] = []
    for card in cards:
        if isinstance(card, bool) or not isinstance(card, int):
            raise TypeError("牌码必须是整数")
        if card not in ENV_TO_HAN:
            raise ValueError(f"未知牌码: {card}")
        result.append(card)
    return [ENV_TO_HAN[card] for card in sorted(result)]


def format_cards(cards: Sequence[str]) -> str:
    """把牌点格式化为界面文本，空出牌显示为“不出”。

    Args:
        cards: Han 牌点。

    Returns:
        以空格分隔、按牌力降序的文本。

    Examples:
        >>> format_cards(["3", "大", "3"])
        '大 3 3'
        >>> format_cards([])
        '不出'
    """
    if not cards:
        return "不出"
    return " ".join(reversed(decode_cards(encode_cards(cards))))


def within_deck(cards: Iterable[int]) -> bool:
    """判断牌码多重集是否不超过一副牌的张数。

    Args:
        cards: DouZero 牌码。

    Returns:
        每种牌都不超过整副牌数量时为 True。

    Examples:
        >>> within_deck([3, 3, 3, 3])
        True
        >>> within_deck([30, 30])
        False
    """
    return all(count <= DECK_LIMITS.get(card, 0) for card, count in Counter(cards).items())
