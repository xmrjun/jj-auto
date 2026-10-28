"""自动化接入：运行时配置、Controller 整图五区路线、异常隔离、Worker 命令与界面控件。"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image

from automation.runtime import AutomationRuntime, parse_config
from config import settings
from core.game_controller import GameController
from core.inference_protocol import Response
from tests.test_automation_units import FakeNative


def raw_result(items, size):
    """构造 ultralytics 风格的结果：items 为 (类别名, 归一化中心x, 中心y)。"""
    names = sorted({name for name, _, _ in items})
    raw = Mock()
    raw.names = dict(enumerate(names))
    width, height = size
    raw.boxes.xyxy.cpu.return_value.tolist.return_value = [
        [x*width-10, y*height-15, x*width+10, y*height+15] for _, x, y in items]
    raw.boxes.cls.cpu.return_value.tolist.return_value = [float(names.index(name)) for name, _, _ in items]
    return raw


class RuntimeConfigTests(unittest.TestCase):
    """automation.yaml 校验与运行时加载。"""

    def test_defaults_and_validation(self) -> None:
        """缺省值可用；类型或范围错误明确报错。"""
        config, options = parse_config({})
        self.assertEqual((config.required_frames, options.stop_hotkey, options.douzero_model), (3, "F12", "resnet"))
        with self.assertRaises(TypeError):
            parse_config({"auto_continue": "yes"})
        with self.assertRaises(ValueError):
            parse_config({"required_frames": 0})
        with self.assertRaises(ValueError):
            parse_config({"douzero_model": "../evil"})

    def test_runtime_without_weights_refuses_to_enable(self) -> None:
        """没有权重时开启失败，模式保持关闭；缺失模板被列出。"""
        with tempfile.TemporaryDirectory() as folder:
            runtime = AutomationRuntime(folder, folder, native=FakeNative())
            self.addCleanup(runtime.close)
            self.assertIn("play", AutomationRuntime(folder, Path("."), native=FakeNative()).missing_templates)
            with self.assertRaises(FileNotFoundError):
                runtime.set_mode("observe")
            self.assertFalse(runtime.active)
            self.assertEqual(runtime.status()["mode"], "disabled")

    def test_repository_automation_config_parses(self) -> None:
        """仓库自带的 automation.yaml 合法。"""
        import yaml

        parse_config(yaml.safe_load(Path("config/automation.yaml").read_text(encoding="utf-8")))


class ControllerIntegrationTests(unittest.TestCase):
    """真实 GameController + 手动布局 + 假模型。"""

    def setUp(self) -> None:
        """创建手动布局控制器。"""
        snapshot = settings.runtime_snapshot()
        self.addCleanup(settings.apply_runtime_settings, snapshot)
        config = settings.validate_layout({"window_title": "Game", "mode": "manual", "layout": {
            "player_hand": [.04, .7, .96, .86], "player_played": [.04, .46, .96, .66],
            "opponent_left": [.08, .25, .48, .45], "opponent_right": [.5, .25, .94, .45],
            "landlord_cards": [.35, 0., .6, .2]}})
        self.image = Image.new("RGB", (1280, 800))
        with patch.object(settings, "WINDOW_LAYOUTS", {"M": config}), patch.object(
                GameController, "_create_inferencer", return_value=Mock()), patch(
                "core.game_controller.ScreenCapture") as capture:
            capture.return_value.capture_window.return_value = self.image
            capture.return_value.last_hwnd = 7
            capture.return_value.last_rect = (0, 0, 1280, 800)
            self.controller = GameController("doudizhu", "M")
        items = [("A", .2+.03*i, .78) for i in range(3)]+[("K", .45, .1)]
        self.controller.yolo_inferencer.detect.return_value = (raw_result(items, self.image.size), 1.)
        self.controller.yolo_inferencer.detect_packed.return_value = (None, 1.)

    def test_disabled_command_without_runtime_is_noop(self) -> None:
        """未开启过自动化时关闭命令不创建运行时。"""
        self.controller.execute_command("set_automation_mode", "disabled")
        self.assertIsNone(self.controller.automation)
        with self.assertRaises(TypeError):
            self.controller.execute_command("set_automation_mode", 1)

    def test_automation_gets_full_frame_with_hand_and_errors_are_isolated(self) -> None:
        """自动化开启时整图识别五区并拿到手牌；自动化异常不影响记牌结果。"""
        runtime = Mock(active=True)
        runtime.process.return_value = {"mode": "observe"}
        self.controller.automation = runtime
        remain, _, _ = self.controller.detect()
        args = runtime.process.call_args.args
        self.assertEqual(len(args[2]["player_hand"]), 3)
        self.assertEqual(set(args[3]), set(self.controller.layout_config["layout"]))
        self.assertEqual(args[7:], (7, (0, 0, 1280, 800)))
        self.assertEqual(self.controller.automation_status, {"mode": "observe"})
        self.assertIn("automation_ms", self.controller.last_performance)
        runtime.process.side_effect = RuntimeError("boom")
        runtime.status.return_value = {"mode": "observe", "reason": "x"}
        again, _, _ = self.controller.detect()
        self.assertEqual(again, remain)
        runtime.coordinator.fail.assert_called_once()

    def test_response_carries_optional_automation_status(self) -> None:
        """回执字段缺省为 None，兼容旧调用。"""
        self.assertIsNone(Response("ok").automation_status)


class WorkerAndPanelTests(unittest.TestCase):
    """主进程 Worker 与界面控件，不启动子进程。"""

    @classmethod
    def setUpClass(cls) -> None:
        """创建 QApplication。"""
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_worker_validates_mode_and_stop_sets_shared_event(self) -> None:
        """未知模式拒绝；紧急停止直接置位共享事件。"""
        import multiprocessing as mp

        from core.inference_process import InferenceWorker

        worker = InferenceWorker("x")
        with self.assertRaises(ValueError):
            worker.set_automation_mode("turbo")
        worker._stop_event = mp.Event()
        worker.emergency_stop()
        self.assertTrue(worker._stop_event.is_set())
        self.assertIsNone(worker.set_automation_mode("auto"))
        self.assertFalse(worker._stop_event.is_set(), "开启自动前清除停止标志")
        worker.deleteLater()

    def test_panel_follows_backend_and_hides_optional_templates(self) -> None:
        """后端报告关闭时下拉框同步；只提示必需模板。"""
        from ui.automation_panel import AutomationPanel

        panel = AutomationPanel()
        emitted = []
        panel.mode_requested.connect(emitted.append)
        panel.mode_combo.setCurrentIndex(1)
        self.assertEqual(emitted, ["observe"])
        panel.show_status({"mode": "observe", "phase_text": "出牌阶段", "role": "地主", "suggestion": "建议：3",
                           "candidates": ["3 (0.1)"], "missing_templates": ["no_rob", "play"], "hotkey": "F12"})
        text = panel.status_label.text()
        self.assertIn("建议：3", text)
        self.assertIn("出牌", text.splitlines()[-1])
        self.assertNotIn("no_rob", text)
        self.assertIn("F12", panel.btn_stop.text())
        panel.show_status({"mode": "disabled", "reason": "紧急停止"})
        self.assertEqual(panel.current_mode(), "disabled")
        self.assertEqual(emitted, ["observe"], "同步不回发信号")
        panel.deleteLater()


if __name__ == "__main__":
    unittest.main()
