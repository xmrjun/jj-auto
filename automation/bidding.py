"""叫分、抢地主与加倍的透明手牌强度规则，阈值全部可配置。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

BID_BUTTONS: tuple[str, ...] = ("bid_1", "bid_2", "bid_3", "bid_pass", "call_landlord", "no_call",
                                "rob_landlord", "no_rob")
DOUBLE_BUTTONS: tuple[str, ...] = ("double", "no_double")


@dataclass(frozen=True)
class BidPolicy:
    """叫分阈值：手牌强度不低于阈值即叫对应分数。

    Attributes:
        bid_1: 叫 1 分的最低强度。
        bid_2: 叫 2 分的最低强度。
        bid_3: 叫 3 分（或抢地主）的最低强度。
        double: 加倍的最低强度。
    """

    bid_1: float = 6.0
    bid_2: float = 8.0
    bid_3: float = 10.0
    double: float = 11.0

    def __post_init__(self) -> None:
        """校验阈值递增。

        Raises:
            ValueError: 阈值不是递增序列。
        """
        if not self.bid_1 <= self.bid_2 <= self.bid_3:
            raise ValueError("叫分阈值必须满足 bid_1 ≤ bid_2 ≤ bid_3")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "BidPolicy":
        """从配置字典创建策略。

        Args:
            values: 含可选 bid_1/bid_2/bid_3/double 的字典。

        Returns:
            叫分策略。

        Raises:
            TypeError: 数值类型错误。
            ValueError: 阈值不递增。
        """
        fields: dict[str, float] = {}
        for key in ("bid_1", "bid_2", "bid_3", "double"):
            if key in values:
                value = values[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{key} 必须是数字")
                fields[key] = float(value)
        return cls(**fields)


def hand_strength(cards: Sequence[str]) -> float:
    """按大牌和炸弹计算手牌强度。

    大王 4、小王 3、每张 2 记 2、每张 A 记 1、每个炸弹再加 4、王炸再加 2。

    Args:
        cards: Han 牌点。

    Returns:
        强度分数。

    Examples:
        >>> hand_strength(["大", "小", "2", "2", "A"])
        14.0
        >>> hand_strength(["5", "5", "5", "5"])
        4.0
    """
    counts = Counter(cards)
    score = 4.0*counts["大"]+3.0*counts["小"]+2.0*counts["2"]+1.0*counts["A"]
    score += 4.0*sum(1 for card, count in counts.items() if count == 4 and card not in ("小", "大"))
    if counts["大"] and counts["小"]:
        score += 2.0
    return score


def bid_level(strength: float, policy: BidPolicy) -> int:
    """把强度换算为想叫的分数。

    Args:
        strength: hand_strength 结果。
        policy: 阈值。

    Returns:
        0 表示不叫，1~3 为分数。

    Examples:
        >>> bid_level(9, BidPolicy())
        2
    """
    if strength >= policy.bid_3:
        return 3
    if strength >= policy.bid_2:
        return 2
    if strength >= policy.bid_1:
        return 1
    return 0


def choose_bid_button(level: int, available: Collection[str]) -> str | None:
    """在当前可见按钮中选择与目标分数相符的按钮。

    叫分模式取不超过目标的最高分，其次不叫；叫/抢地主模式下 1 分以上叫地主、3 分抢地主。

    Args:
        level: 目标分数 0~3。
        available: 当前可操作的按钮名。

    Returns:
        要点击的按钮；没有合适按钮时为 None。

    Raises:
        ValueError: level 越界。

    Examples:
        >>> choose_bid_button(2, {"bid_3", "bid_pass"})
        'bid_pass'
        >>> choose_bid_button(3, {"bid_1", "bid_2", "bid_3", "bid_pass"})
        'bid_3'
        >>> choose_bid_button(1, {"call_landlord", "no_call"})
        'call_landlord'
    """
    if not 0 <= level <= 3:
        raise ValueError("叫分只能是 0~3")
    for score in range(level, 0, -1):
        if f"bid_{score}" in available:
            return f"bid_{score}"
    if "bid_pass" in available:
        return "bid_pass"
    if "call_landlord" in available or "no_call" in available:
        return "call_landlord" if level >= 1 and "call_landlord" in available else (
            "no_call" if "no_call" in available else None)
    if "rob_landlord" in available or "no_rob" in available:
        return "rob_landlord" if level >= 3 and "rob_landlord" in available else (
            "no_rob" if "no_rob" in available else None)
    return None


def choose_double_button(strength: float, policy: BidPolicy, available: Collection[str]) -> str | None:
    """选择加倍或不加倍按钮。

    Args:
        strength: 手牌强度。
        policy: 阈值。
        available: 当前可操作的按钮名。

    Returns:
        要点击的按钮；不加倍但没有“不加倍”模板时返回 None（等待游戏默认）。

    Examples:
        >>> choose_double_button(12, BidPolicy(), {"double", "no_double"})
        'double'
        >>> choose_double_button(3, BidPolicy(), {"double"}) is None
        True
    """
    if strength >= policy.double and "double" in available:
        return "double"
    return "no_double" if "no_double" in available else None
