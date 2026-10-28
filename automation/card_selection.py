"""根据当前帧真实牌框判断已选牌并规划选牌点击；只生成计划，不发送输入。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import median

from automation.types import CardBox, ClickPlan, ClickPoint


class UnplannableAction(ValueError):
    """目标牌在当前手牌框中无法完整、无歧义地选出。"""


def raised_flags(hand: Sequence[CardBox], ratio: float = 0.3, baseline: float | None = None) -> tuple[bool, ...]:
    """判断每张手牌是否处于被选中（上移）状态。

    以未选中牌的中心纵坐标为基线；比基线高出 ``ratio`` 倍中位牌框高度视为已选。

    Args:
        hand: 当前手牌框。
        ratio: 上移阈值与牌框高度之比。
        baseline: 本局记录的未选中基线；None 时取最低一张牌。

    Returns:
        与 hand 顺序一致的布尔值。

    Raises:
        ValueError: ratio 不在 (0, 2) 内。

    Examples:
        >>> boxes = [CardBox("3", (0, 100, 10, 140)), CardBox("4", (20, 80, 30, 120))]
        >>> raised_flags(boxes)
        (False, True)
    """
    if not 0 < ratio < 2:
        raise ValueError("上移阈值必须在 0 与 2 之间")
    if not hand:
        return ()
    centers = [box.center[1] for box in hand]
    height = median(box.height for box in hand)
    base = max(centers) if baseline is None else baseline
    return tuple(base-center > ratio*height for center in centers)


def resting_baseline(hand: Sequence[CardBox], ratio: float = 0.3) -> float | None:
    """当所有牌都在同一高度时返回基线，否则 None。

    Args:
        hand: 当前手牌框。
        ratio: 与 raised_flags 一致的阈值。

    Returns:
        未选中牌中心纵坐标的中位数；存在上移牌或为空时为 None。
    """
    if not hand or any(raised_flags(hand, ratio)):
        return None
    return median(box.center[1] for box in hand)


def selected_cards(hand: Sequence[CardBox], flags: Sequence[bool]) -> Counter[str]:
    """统计已选中的牌点。

    Args:
        hand: 手牌框。
        flags: raised_flags 结果。

    Returns:
        已选牌点多重集。
    """
    return Counter(box.card for box, flag in zip(hand, flags) if flag)


def plan_card_clicks(action: Sequence[str], hand: Sequence[CardBox], flags: Sequence[bool] | None = None,
                     image_size: tuple[int, int] | None = None) -> ClickPlan:
    """规划把当前选中状态切换为目标出牌所需的点击。

    多余的已选牌会被点回，缺少的牌从左到右选取互不重复的牌框。

    Args:
        action: 目标出牌牌点。
        hand: 当前手牌框。
        flags: 每张牌是否已选；None 视为都未选。
        image_size: 截图宽高，用于拒绝越界点击。

    Returns:
        点击计划；已满足时为空计划。

    Raises:
        UnplannableAction: 手牌中缺少目标牌、标记数量不符或点击点越界。

    Examples:
        >>> hand = [CardBox("A", (0, 100, 20, 140)), CardBox("A", (30, 100, 50, 140))]
        >>> [c.x for c in plan_card_clicks(["A"], hand).clicks]
        [10]
    """
    states = tuple(flags) if flags is not None else tuple(False for _ in hand)
    if len(states) != len(hand):
        raise UnplannableAction("选中标记与手牌数量不一致")
    target = Counter(action)
    available = Counter(box.card for box in hand)
    missing = target-available
    if missing:
        raise UnplannableAction(f"手牌中缺少: {' '.join(sorted(missing.elements()))}")
    ordered = sorted(range(len(hand)), key=lambda index: hand[index].bbox[0])
    toggles: list[CardBox] = []
    for card in sorted(set(available) | set(target)):
        indices = [index for index in ordered if hand[index].card == card]
        raised = [index for index in indices if states[index]]
        resting = [index for index in indices if not states[index]]
        need = target.get(card, 0)
        if len(raised) > need:
            toggles.extend(hand[index] for index in raised[need:])
        elif len(raised) < need:
            toggles.extend(hand[index] for index in resting[:need-len(raised)])
    toggles.sort(key=lambda box: box.bbox[0])
    clicks: list[ClickPoint] = []
    for box in toggles:
        x, y = box.center
        if image_size is not None and not (0 <= x < image_size[0] and 0 <= y < image_size[1]):
            raise UnplannableAction("牌框中心超出截图范围")
        clicks.append(ClickPoint(int(round(x)), int(round(y)), box.card))
    return ClickPlan(tuple(clicks), "选牌 " + " ".join(sorted(target.elements())) if target else "取消选牌")
