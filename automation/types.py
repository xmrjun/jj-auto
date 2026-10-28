"""自动化层不可变数据类型；只含纯数据，可安全跨线程复制。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

Box = tuple[float, float, float, float]
SEATS: tuple[str, str, str] = ("self", "right", "left")


class AutomationMode(str, Enum):
    """运行模式：关闭、只观察给建议、自动操作。"""

    DISABLED = "disabled"
    OBSERVE = "observe"
    AUTO = "auto"


class AutomationPhase(str, Enum):
    """自动化阶段。"""

    WAITING = "waiting"
    BIDDING = "bidding"
    PLAYING = "playing"
    VERIFYING = "verifying"
    PAUSED = "paused"
    FINISHED = "finished"


PHASE_TEXT: dict[AutomationPhase, str] = {
    AutomationPhase.WAITING: "等待开局",
    AutomationPhase.BIDDING: "叫分/加倍",
    AutomationPhase.PLAYING: "出牌阶段",
    AutomationPhase.VERIFYING: "等待操作生效",
    AutomationPhase.PAUSED: "已暂停",
    AutomationPhase.FINISHED: "本局结束",
}


@dataclass(frozen=True)
class CardBox:
    """当前帧中的一张手牌检测框（截图像素坐标）。

    Attributes:
        card: Han 牌点。
        bbox: (x1, y1, x2, y2)。
    """

    card: str
    bbox: Box

    @property
    def center(self) -> tuple[float, float]:
        """返回检测框中心。"""
        x1, y1, x2, y2 = self.bbox
        return (x1+x2)/2, (y1+y2)/2

    @property
    def height(self) -> float:
        """返回检测框高度。"""
        return self.bbox[3]-self.bbox[1]


@dataclass(frozen=True)
class ButtonMatch:
    """一个按钮的模板匹配结果（截图像素坐标）。

    Attributes:
        name: 按钮名。
        bbox: (x1, y1, x2, y2)。
        score: 归一化相关系数。
        actionable: 无歧义且置信度达标时为 True。
    """

    name: str
    bbox: Box
    score: float
    actionable: bool = True

    @property
    def center(self) -> tuple[float, float]:
        """返回按钮中心。"""
        x1, y1, x2, y2 = self.bbox
        return (x1+x2)/2, (y1+y2)/2


@dataclass(frozen=True)
class TableObservation:
    """一帧截图得到的牌桌观测。

    Attributes:
        timestamp: 单调时钟秒。
        valid: 截图与模型调用是否成功。
        image_size: 截图宽高。
        window_rect: 截图时目标窗口的屏幕矩形 (left, top, right, bottom)。
        hand: 本家手牌框；None 表示该区未知。
        played: 各座位当前出牌区牌点；值为 None 表示该区未知。
        bottom: 底牌区牌点；None 表示未知。
        buttons: 已匹配的按钮。
        history: 记牌器已确认的各座位出牌历史。
    """

    timestamp: float
    valid: bool
    image_size: tuple[int, int] = (0, 0)
    window_rect: tuple[int, int, int, int] | None = None
    hand: tuple[CardBox, ...] | None = None
    played: Mapping[str, tuple[str, ...] | None] = field(default_factory=dict)
    bottom: tuple[str, ...] | None = None
    buttons: Mapping[str, ButtonMatch] = field(default_factory=dict)
    history: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)

    def hand_cards(self) -> tuple[str, ...]:
        """返回按横坐标排序的手牌牌点，未知时为空元组。"""
        if self.hand is None:
            return ()
        return tuple(box.card for box in sorted(self.hand, key=lambda item: item.bbox[0]))

    def has_any(self, names: tuple[str, ...]) -> bool:
        """判断是否有任一按钮可操作。

        Args:
            names: 按钮名。

        Returns:
            至少一个按钮存在且无歧义时为 True。
        """
        return any(name in self.buttons and self.buttons[name].actionable for name in names)


@dataclass(frozen=True)
class ClickPoint:
    """一次计划点击（截图像素坐标）。

    Attributes:
        x: 横坐标。
        y: 纵坐标。
        label: 说明，例如牌点或按钮名。
    """

    x: int
    y: int
    label: str


@dataclass(frozen=True)
class ClickPlan:
    """一组按顺序执行的点击。

    Attributes:
        clicks: 点击点。
        description: 界面显示的说明。
    """

    clicks: tuple[ClickPoint, ...]
    description: str
