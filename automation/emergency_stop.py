"""独立于 UI 线程和推理流程的紧急停止标志与全局热键监视。"""

from __future__ import annotations

import sys
import threading
import time
from typing import Protocol


class AutomationStopped(RuntimeError):
    """紧急停止已触发，必须放弃剩余点击。"""


class EventLike(Protocol):
    """threading.Event 与 multiprocessing.Event 的共同接口。"""

    def set(self) -> None:
        """置位。"""
        ...

    def clear(self) -> None:
        """清除。"""
        ...

    def is_set(self) -> bool:
        """返回是否已置位。"""
        ...


VIRTUAL_KEYS: dict[str, int] = {**{f"F{index}": 0x6F+index for index in range(1, 13)},
                                "PAUSE": 0x13, "SCROLLLOCK": 0x91}


class EmergencyStop:
    """跨进程共享的停止标志；UI 按钮和热键都可直接置位。

    Attributes:
        event: 底层事件，可为 multiprocessing.Event。
        reason: 最近一次触发原因（仅本进程可见）。
    """

    def __init__(self, event: EventLike | None = None) -> None:
        """创建停止标志。

        Args:
            event: 共享事件；None 时使用进程内事件。
        """
        self.event: EventLike = event if event is not None else threading.Event()
        self.reason = ""

    def trigger(self, reason: str = "紧急停止") -> None:
        """置位停止标志。

        Args:
            reason: 触发原因。
        """
        self.reason = reason
        self.event.set()

    def clear(self) -> None:
        """用户重新开启自动模式时清除标志。"""
        self.reason = ""
        self.event.clear()

    def is_set(self) -> bool:
        """返回是否已停止。"""
        return self.event.is_set()

    def check(self) -> None:
        """已停止时抛出异常。

        Raises:
            AutomationStopped: 停止标志已置位。
        """
        if self.event.is_set():
            raise AutomationStopped(self.reason or "紧急停止")


class HotkeyWatcher:
    """在后台线程轮询全局热键，按下即触发紧急停止；非 Windows 平台不启动。

    Attributes:
        stop: 要触发的停止标志。
        key: 热键名，如 ``F12``。
    """

    def __init__(self, stop: EmergencyStop, key: str = "F12", interval: float = 0.03) -> None:
        """创建监视器。

        Args:
            stop: 停止标志。
            key: 热键名，支持 F1~F12、PAUSE、SCROLLLOCK。
            interval: 轮询间隔秒。

        Raises:
            ValueError: 热键不支持。
        """
        if key.upper() not in VIRTUAL_KEYS:
            raise ValueError(f"不支持的紧急停止热键: {key}")
        self.stop = stop
        self.key = key.upper()
        self._interval = interval
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """启动守护线程。

        Returns:
            已在 Windows 上启动时为 True。
        """
        if sys.platform != "win32" or self._thread is not None:
            return self._thread is not None
        self._thread = threading.Thread(target=self._run, name="emergency-hotkey", daemon=True)
        self._thread.start()
        return True

    def close(self) -> None:
        """请求线程退出。"""
        self._quit.set()

    def _run(self) -> None:
        """轮询 GetAsyncKeyState，按下时置位停止标志。"""
        import ctypes

        get_state = ctypes.windll.user32.GetAsyncKeyState  # type: ignore[attr-defined]
        code = VIRTUAL_KEYS[self.key]
        while not self._quit.is_set():
            if get_state(code) & 0x8000:
                self.stop.trigger(f"按下紧急停止热键 {self.key}")
            time.sleep(self._interval)
