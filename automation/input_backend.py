"""受保护的点击执行：模式、紧急停止、窗口一致性、遮挡与人工接管检查。"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from automation.emergency_stop import AutomationStopped, EmergencyStop
from automation.types import AutomationMode, ClickPlan


class UnsafeClick(RuntimeError):
    """点击前检查失败，整个计划被放弃。"""


class NativeInput(Protocol):
    """平台输入接口；测试使用假实现，Windows 使用 windows_input.Win32Input。"""

    def window_rect(self, hwnd: int) -> tuple[int, int, int, int] | None:
        """返回窗口屏幕矩形，窗口不存在时为 None。"""
        ...

    def root_window_at(self, x: int, y: int) -> int:
        """返回屏幕点所在的顶层窗口句柄。"""
        ...

    def cursor_pos(self) -> tuple[int, int]:
        """返回鼠标屏幕坐标。"""
        ...

    def click(self, x: int, y: int) -> None:
        """在屏幕坐标执行一次左键单击。"""
        ...


class GuardedInputBackend:
    """只在自动模式下、所有检查通过后逐个发送点击。

    Attributes:
        native: 平台输入实现。
        stop: 紧急停止标志。
        settle: 每次点击后的等待秒数。
        takeover_distance: 鼠标被人为移动超过该像素即视为接管。
        executed: 已真实发送的点击总数。
        log: 最近计划的说明，观察模式也会记录。
    """

    def __init__(self, native: NativeInput | None, stop: EmergencyStop, settle: float = 0.08,
                 takeover_distance: int = 12, sleep: Callable[[float], None] = time.sleep) -> None:
        """创建执行器。

        Args:
            native: 平台输入；None 表示当前平台不能点击。
            stop: 紧急停止标志。
            settle: 点击间隔秒。
            takeover_distance: 人工接管判定距离。
            sleep: 可注入的等待函数。
        """
        self.native = native
        self.stop = stop
        self.settle = settle
        self.takeover_distance = takeover_distance
        self.executed = 0
        self.log: list[str] = []
        self._sleep = sleep

    def execute(self, plan: ClickPlan, mode: AutomationMode, hwnd: int | None,
                expected_rect: tuple[int, int, int, int] | None) -> int:
        """执行点击计划。

        Args:
            plan: 截图像素坐标的点击计划。
            mode: 当前模式；非 AUTO 只记录不点击。
            hwnd: 截图时的目标窗口句柄。
            expected_rect: 截图时的窗口矩形。

        Returns:
            本次真实发送的点击数。

        Raises:
            AutomationStopped: 紧急停止或检测到人工接管。
            UnsafeClick: 平台不可用、窗口变化、点击点越界或被其他窗口遮挡。
        """
        self.log = (self.log+[plan.description])[-20:]
        if mode != AutomationMode.AUTO or not plan.clicks:
            return 0
        if self.native is None:
            raise UnsafeClick("当前平台不支持自动点击")
        if hwnd is None or expected_rect is None:
            raise UnsafeClick("缺少目标窗口信息")
        sent = 0
        last: tuple[int, int] | None = None
        for point in plan.clicks:
            self.stop.check()
            rect = self.native.window_rect(hwnd)
            if rect != expected_rect:
                raise UnsafeClick("目标窗口已移动、缩放或关闭")
            left, top, right, bottom = rect
            x, y = left+point.x, top+point.y
            if not (left <= x < right and top <= y < bottom):
                raise UnsafeClick(f"点击点 {point.label} 超出窗口")
            if self.native.root_window_at(x, y) != hwnd:
                raise UnsafeClick(f"点击点 {point.label} 被其他窗口遮挡")
            if last is not None:
                cx, cy = self.native.cursor_pos()
                if abs(cx-last[0]) > self.takeover_distance or abs(cy-last[1]) > self.takeover_distance:
                    self.stop.trigger("检测到鼠标被人工移动，已停止自动操作")
                    raise AutomationStopped(self.stop.reason)
            self.native.click(x, y)
            sent += 1
            self.executed += 1
            last = (x, y)
            self._sleep(self.settle)
        return sent
