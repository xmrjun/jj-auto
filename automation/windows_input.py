"""Win32 鼠标输入实现；只在 Windows 上调用时才加载 ctypes 接口，导入无副作用。"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
GA_ROOT = 2


class _MouseInput(ctypes.Structure):
    """SendInput 的 MOUSEINPUT 结构。"""

    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _InputUnion(ctypes.Union):
    """INPUT 结构中的联合体，只使用鼠标分支。"""

    _fields_ = [("mi", _MouseInput), ("padding", ctypes.c_byte*32)]


class _Input(ctypes.Structure):
    """SendInput 的 INPUT 结构。"""

    _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]


class Win32Input:
    """使用 SetCursorPos + SendInput 执行真实点击。

    坐标与截图一致使用物理像素；截图模块已调用 SetProcessDPIAware。
    """

    def __init__(self) -> None:
        """绑定 user32 函数。

        Raises:
            RuntimeError: 非 Windows 平台。
        """
        if sys.platform != "win32":
            raise RuntimeError("Win32Input 只能在 Windows 上使用")
        self._user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        self._user32.WindowFromPoint.restype = wintypes.HWND
        self._user32.WindowFromPoint.argtypes = [wintypes.POINT]
        self._user32.GetAncestor.restype = wintypes.HWND
        self._user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]

    def window_rect(self, hwnd: int) -> tuple[int, int, int, int] | None:
        """返回窗口屏幕矩形。

        Args:
            hwnd: 窗口句柄。

        Returns:
            (left, top, right, bottom)；窗口无效时为 None。
        """
        if not self._user32.IsWindow(hwnd):
            return None
        rect = wintypes.RECT()
        if not self._user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return rect.left, rect.top, rect.right, rect.bottom

    def root_window_at(self, x: int, y: int) -> int:
        """返回屏幕点处的顶层窗口句柄。

        Args:
            x: 屏幕横坐标。
            y: 屏幕纵坐标。

        Returns:
            顶层窗口句柄；没有窗口时为 0。
        """
        child = self._user32.WindowFromPoint(wintypes.POINT(x, y))
        if not child:
            return 0
        return int(self._user32.GetAncestor(child, GA_ROOT) or 0)

    def cursor_pos(self) -> tuple[int, int]:
        """返回当前鼠标坐标。"""
        point = wintypes.POINT()
        self._user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y

    def click(self, x: int, y: int) -> None:
        """移动鼠标并发送一次左键按下/抬起。

        Args:
            x: 屏幕横坐标。
            y: 屏幕纵坐标。

        Raises:
            OSError: SendInput 未能注入全部事件。
        """
        self._user32.SetCursorPos(x, y)
        events = (_Input*2)()
        events[0].type = events[1].type = INPUT_MOUSE
        events[0].union.mi = _MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0)
        events[1].union.mi = _MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0)
        if self._user32.SendInput(2, events, ctypes.sizeof(_Input)) != 2:
            raise OSError("SendInput 被系统拒绝（目标可能以管理员权限运行）")
