try:
    import win32gui
except ImportError:  # pragma: no cover - Windows CI 覆盖真实 Win32 路径
    win32gui = None

def enum_windows_callback(hwnd, windows):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd)
        if title:
            windows.append(title)

def get_all_window_titles():
    if win32gui is None:
        raise RuntimeError("窗口枚举仅支持 Windows，且需要安装 pywin32")

    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    return windows

if __name__ == "__main__":
    print("所有窗口标题:")
    print("=" * 50)
    
    titles = get_all_window_titles()
    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")
    
    print("=" * 50)
    print(f"共找到 {len(titles)} 个窗口")
