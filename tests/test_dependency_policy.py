"""跨平台依赖策略测试。"""

from pathlib import Path
import importlib
import platform
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyPolicyTests(unittest.TestCase):
    """确保 Mac 开发依赖与 Windows 运行依赖保持隔离。"""

    def test_win32_dependency_is_windows_only(self) -> None:
        """pywin32 只能出现在 Windows 专用依赖文件。"""
        windows = (ROOT / "requirements" / "windows.txt").read_text(encoding="utf-8")
        base = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")

        self.assertIn("pywin32", windows)
        self.assertNotIn("pywin32", base)

    def test_runtime_does_not_force_cuda_build_on_mac(self) -> None:
        """基础依赖不能强制安装 CUDA 专用 PyTorch wheel。"""
        base = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")

        self.assertNotIn("+cu121", base)
        self.assertNotIn("--index-url https://download.pytorch.org/whl/cu121", base)

    def test_legacy_entrypoint_delegates_to_platform_files(self) -> None:
        """旧安装入口只组合基础依赖和当前平台依赖。"""
        legacy = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("-r requirements/base.txt", legacy)
        self.assertIn("-r requirements/windows.txt", legacy)

    @unittest.skipIf(platform.system() == "Windows", "仅验证非 Windows 导入边界")
    def test_windows_helpers_can_be_imported_on_mac(self) -> None:
        """Mac 审核环境导入模块时不应强制要求 pywin32。"""
        for module_name in (
            "core.screen_capture",
            "utils.layout_editor.service",
            "utils.add_layout.list_windows",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    @unittest.skipIf(platform.system() == "Windows", "仅验证非 Windows 错误信息")
    def test_windows_capture_fails_with_clear_error_on_mac(self) -> None:
        """真正调用 Windows 功能时应给出明确的平台错误。"""
        screen_capture = importlib.import_module("core.screen_capture")

        with self.assertRaisesRegex(RuntimeError, "Windows"):
            screen_capture.ScreenCapture("JJ斗地主").capture_window()


if __name__ == "__main__":
    unittest.main()

