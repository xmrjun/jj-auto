"""连续帧稳定门与暂停条件：画面未稳定或存在矛盾时不允许任何动作。"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from automation.card_selection import raised_flags
from automation.types import TableObservation

BLOCKING_BUTTONS: tuple[str, ...] = ("trusteeship", "confirm", "dialog")


def observation_signature(observation: TableObservation, raise_ratio: float = 0.3,
                          baseline: float | None = None) -> Hashable:
    """提取决定动作的观测要素，用于比较连续帧是否一致。

    Args:
        observation: 当前观测。
        raise_ratio: 选中判断阈值。
        baseline: 本局手牌基线。

    Returns:
        可哈希签名；未知区域与空区域不同。
    """
    hand: tuple[tuple[str, bool], ...] | None = None
    if observation.hand is not None:
        ordered = sorted(observation.hand, key=lambda box: box.bbox[0])
        hand = tuple(zip((box.card for box in ordered), raised_flags(ordered, raise_ratio, baseline)))
    played = tuple(sorted((seat, None if cards is None else tuple(sorted(cards)))
                          for seat, cards in observation.played.items()))
    history = tuple(sorted((seat, len(items)) for seat, items in observation.history.items()))
    buttons = tuple(sorted(name for name, match in observation.buttons.items() if match.actionable))
    bottom = None if observation.bottom is None else tuple(sorted(observation.bottom))
    return (hand, played, history, buttons, bottom, observation.window_rect, observation.image_size)


class StableObservationGate:
    """只有连续 required_frames 帧签名完全一致时才放行。

    Attributes:
        required_frames: 需要的连续一致帧数。
        raise_ratio: 选中判断阈值。
    """

    def __init__(self, required_frames: int = 3, raise_ratio: float = 0.3) -> None:
        """创建稳定门。

        Args:
            required_frames: 需要的连续一致帧数，至少 1。
            raise_ratio: 选中判断阈值。

        Raises:
            ValueError: required_frames 小于 1。
        """
        if required_frames < 1:
            raise ValueError("稳定帧数至少为 1")
        self.required_frames = required_frames
        self.raise_ratio = raise_ratio
        self._last: Hashable | None = None
        self._count = 0

    def reset(self) -> None:
        """清空连续计数，下一帧重新开始累计。"""
        self._last = None
        self._count = 0

    def observe(self, observation: TableObservation, baseline: float | None = None) -> bool:
        """加入一帧并返回是否已稳定。

        Args:
            observation: 当前观测。
            baseline: 本局手牌基线。

        Returns:
            连续一致帧数达到要求时为 True；无效帧总是 False 并清零。
        """
        if not observation.valid:
            self.reset()
            return False
        signature = observation_signature(observation, self.raise_ratio, baseline)
        if signature == self._last:
            self._count += 1
        else:
            self._last, self._count = signature, 1
        return self._count >= self.required_frames


def pause_reason(previous: TableObservation | None, current: TableObservation,
                 action_in_flight: bool, blocking: Sequence[str] = BLOCKING_BUTTONS) -> str | None:
    """返回必须立即暂停的原因。

    Args:
        previous: 上一帧观测。
        current: 当前观测。
        action_in_flight: 是否有尚未验证的点击。
        blocking: 出现即暂停的弹窗/托管按钮名。

    Returns:
        暂停原因；可以继续时为 None。
    """
    for name in blocking:
        if name in current.buttons:
            return f"检测到未处理的界面: {name}"
    if not action_in_flight:
        return None
    if not current.valid:
        return "操作后截图或识别失败"
    if previous is not None and previous.window_rect != current.window_rect:
        return "操作期间窗口移动或缩放"
    return None
