"""模型制品完整性和运行格式策略测试。"""

import hashlib
import importlib.util
from pathlib import Path
import shutil
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from config import settings
from core.model_assets import (
    ModelIntegrityError,
    UnsafeModelFormatError,
    resolve_runtime_model,
    verify_model,
)


class ModelAssetTests(unittest.TestCase):
    """运行进程只能加载清单批准且哈希一致的安全格式。"""

    def setUp(self) -> None:
        """为每个测试创建一个独立的假模型文件。"""
        self.directory = Path(tempfile.mkdtemp(prefix="jj-auto-model-test-"))
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = self.directory / "cards.onnx"
        self.path.write_bytes(b"known-safe-onnx-fixture")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.manifest = {
            "schema_version": 1,
            "artifacts": [
                {
                    "path": "cards.onnx",
                    "format": "onnx",
                    "sha256": self.sha256,
                    "approved_for_runtime": True,
                }
            ],
        }

    def test_hash_mismatch_rejects_model(self) -> None:
        """文件内容与清单不一致时必须拒绝加载。"""
        with self.assertRaises(ModelIntegrityError):
            verify_model(self.path, "0" * 64)

    def test_runtime_policy_rejects_unapproved_pt(self) -> None:
        """PyTorch pickle 容器不能进入正式运行进程。"""
        with self.assertRaises(UnsafeModelFormatError):
            resolve_runtime_model("unknown.pt", manifest={})

    def test_approved_onnx_is_selected_without_pt_fallback(self) -> None:
        """清单批准且哈希一致的 ONNX 文件可以直接选择。"""
        selected = resolve_runtime_model(self.path, manifest=self.manifest)

        self.assertEqual(selected.format, "onnx")
        self.assertEqual(selected.path, self.path.resolve())
        self.assertEqual(selected.sha256, self.sha256)

    def test_settings_discovers_runtime_artifacts_but_ignores_pt_only_dirs(self) -> None:
        """模型选择列表不能把只有 pickle 权重的目录暴露为运行模型。"""
        safe_dir = self.directory / "safe"
        unsafe_dir = self.directory / "unsafe"
        safe_dir.mkdir()
        unsafe_dir.mkdir()
        (safe_dir / "best.onnx").write_bytes(b"onnx")
        (unsafe_dir / "best.pt").write_bytes(b"pt")

        with patch.object(settings, "YOLO_WEIGHTS_DIR", str(self.directory)):
            self.assertEqual(settings._scan_model_dirs(), ["safe"])
            self.assertTrue(settings._resolve_model_path("safe").endswith("best.onnx"))

    def test_inferencer_rejects_pt_before_ultralytics_loads_it(self) -> None:
        """即使文件存在，推理器也不能把 .pt 交给 Ultralytics。"""
        unsafe_path = self.directory / "unsafe.pt"
        unsafe_path.write_bytes(b"not-a-real-model")
        torch_stub = ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(is_available=Mock(return_value=False))
        ultralytics_stub = ModuleType("ultralytics")
        ultralytics_stub.YOLO = Mock()
        source = Path(__file__).parents[1] / "core" / "yolo_inferencer.py"
        spec = importlib.util.spec_from_file_location("_safe_yolo_loader", source)
        module = importlib.util.module_from_spec(spec)

        with (
            patch.object(settings, "YOLO_MODEL_PATH", str(unsafe_path)),
            patch.dict(
                "sys.modules",
                {"torch": torch_stub, "ultralytics": ultralytics_stub},
            ),
        ):
            spec.loader.exec_module(module)
            with self.assertRaises(UnsafeModelFormatError):
                module.YoloInferencer()
        ultralytics_stub.YOLO.assert_not_called()


if __name__ == "__main__":
    unittest.main()
