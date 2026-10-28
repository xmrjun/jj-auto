"""布局编辑器的窗口与配置服务。"""

from typing import List

try:
    import win32gui
except ImportError:  # pragma: no cover - Windows CI 覆盖真实 Win32 路径
    win32gui = None

from config import settings
from config.settings import save_window_layout
from core.screen_capture import ScreenCapture


def list_visible_window_titles() -> List[str]:
    """列出 Windows 桌面当前可见且有标题的窗口。"""
    if win32gui is None:
        raise RuntimeError("窗口枚举仅支持 Windows，且需要安装 pywin32")

    titles = []

    def _enum_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)

    win32gui.EnumWindows(_enum_callback, None)
    return sorted(set(titles))


def capture_window_by_title(window_title: str):
    """按窗口标题截图。"""
    cap = ScreenCapture(window_title)
    return cap.capture_window()


def get_layout_names() -> List[str]:
    """返回已保存的布局名称。"""
    return list(settings.WINDOW_LAYOUTS.keys())


def get_layout_config(layout_name: str):
    """返回布局对应的窗口标题与归一化配置。"""
    layout = settings.WINDOW_LAYOUTS.get(layout_name)
    if not layout:
        return None
    return layout.get("window_title"), layout.get("layout")


def save_layout(
    layout_name: str,
    window_title: str,
    normalized_layout: dict,
    set_current: bool,
) -> None:
    """保存一个窗口布局。"""
    save_window_layout(
        layout_name,
        window_title,
        normalized_layout,
        set_current=set_current,
    )
