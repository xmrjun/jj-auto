"""YOLO 推理引擎模块。

仅负责加载模型与执行纯视觉推理，返回原始 YOLO 结果对象和耗时。
"""

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Tuple, cast

# 必须在导入 ultralytics 之前设置：禁止自动 pip 安装依赖、禁止联网检查与匿名统计。
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

import torch
from PIL import Image
from ultralytics import YOLO

try:
    from ultralytics import settings as _ultralytics_settings

    if _ultralytics_settings.get("sync", False):
        _ultralytics_settings.update({"sync": False})
except Exception:  # 统计开关失败不影响离线推理，YOLO_OFFLINE 仍阻止联网。
    pass

import config.settings as settings
from core.frame_packing import Detection
from core.model_assets import resolve_runtime_model
from config.packing_profile import PACKED_INPUT_SIZE

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


class YoloInferencer:
    """基于 YOLO 的纯视觉推理器。

    不包含业务坐标的计算和状态机的处理，只接受单帧图片返回原始边框和执行时间。
    
    Attributes:
        yolo_iou: YOLO IOU 阈值。
        yolo_conf: YOLO 置信度阈值。
        weight_path: 模型权重路径。
        model: 加载的 YOLO 模型。
        device: 推理设备（cuda/cpu）。
        _full_input_size: 首次完整推理的输入配置，避免拼图参数污染整图回退。
    """

    def __init__(self) -> None:
        """根据模型和设备配置初始化纯视觉推理器。"""
        self.yolo_iou: float = settings.YOLO_IOU_THRESHOLD
        self.yolo_conf: float = settings.YOLO_CONFIDENCE_THRESHOLD
        self.weight_path: str = settings.YOLO_MODEL_PATH

        self.model, self.device = self.__load_model()
        self._full_input_size: int | list[int] | tuple[int, int] | None = None

    def __load_model(self) -> Tuple[YOLO, str]:
        """校验清单和哈希后加载安全格式的本地模型。

        Returns:
            模型与实际设备名称。
        Raises:
            ModelAssetError: 模型未批准、格式不安全或哈希不一致。
            RuntimeError: TensorRT 模型与设备选择不兼容或模型加载失败。
        """
        selected = resolve_runtime_model(self.weight_path)
        self.weight_path = str(selected.path)

        device_choice = settings.DEVICE_CHOICE
        print(f"[YoloInferencer] 当前设备选择: {device_choice}")

        use_gpu = device_choice == "cuda" and torch.cuda.is_available()
        if device_choice == "cuda" and not use_gpu:
            print("[YoloInferencer] 警告: 用户选择了GPU，但CUDA不可用，回退到CPU")

        if selected.format == "engine" and not use_gpu:
            raise RuntimeError("TensorRT .engine 模型需要可用的 CUDA 设备")

        try:
            model = YOLO(self.weight_path)
        except Exception as exc:
            raise RuntimeError(f"安全模型加载失败: {selected.path.name}: {exc}") from exc
        return model, "cuda" if use_gpu else "cpu"

    def detect(self, img: Image.Image | None) -> Tuple[object, float]:
        """执行纯粹的 YOLO 推理。

        Returns:
            Tuple[object, float]: (YOLO 单帧 raw_result 或 None, 推理耗时毫秒)
        """
        if img is None:
            return None, 0.0

        backend = getattr(getattr(self.model, "predictor", None), "model", None)
        flexible = (bool(getattr(backend, "pt", False) or getattr(backend, "dynamic", False)) if backend is not None
                    else str(self.model.overrides.get("model", "")).lower().endswith(".pt"))
        width, height = PACKED_INPUT_SIZE
        return self._run(img, (height, width) if flexible else None)

    def detect_packed(self, img: Image.Image) -> tuple[object | None, float]:
        """按画布宽高执行矩形推理，静态引擎保留其固定输入。

        Args:
            img: 当前线程同步消费的 RGB 拼图画布。
        Returns:
            原始画布结果与模型调用耗时；不为静态模型自动重新导出。
        """
        backend = getattr(getattr(self.model, "predictor", None), "model", None)
        flexible = (bool(getattr(backend, "pt", False) or getattr(backend, "dynamic", False)) if backend is not None
                    else str(self.model.overrides.get("model", "")).lower().endswith(".pt"))
        size: int | tuple[int, int] = img.width if img.width == img.height else (img.height, img.width)
        return self._run(img, size if flexible else None, packed=True)

    def _run(self, img: Image.Image, input_size: int | tuple[int, int] | None = None, *, packed: bool = False) -> tuple[object | None, float]:
        """运行单模型并同步 GPU 计时。

        Args:
            img: 当前输入图。
            input_size: 可变模型的候选尺寸；None 恢复整图原始尺寸。
            packed: 是否拼图调用；严格使用画布形状，整图恢复原矩形策略。

        Returns:
            单帧结果及整个 Ultralytics 调用毫秒，包含其内部预处理和后处理。
        Raises:
            ValueError: 模型默认输入尺寸元数据不是整数或尺寸序列。
            Exception: 推理或设备同步失败时向控制器传播。
        """
        if self._full_input_size is None:
            configured = self.model.overrides.get("imgsz", 640)
            if not isinstance(configured, (int, list, tuple)):
                raise ValueError("模型输入尺寸元数据无效")
            self._full_input_size = configured.copy() if isinstance(configured, list) else configured
        selected_size = input_size if input_size is not None else self._full_input_size
        fixed_size = self._static_input_size()
        if fixed_size is not None:
            selected_size = fixed_size
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = self.model(
            img,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            device=self.device,
            verbose=False,
            half=(self.device == "cuda"),
            rect=not packed,
            imgsz=selected_size,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        yolo_ms = (time.perf_counter() - t0) * 1000
        # 首次 packed 调用也可能刚刚建立静态后端，立即记住其元数据形状。
        fixed_size = self._static_input_size()
        if fixed_size is not None:
            self._full_input_size = fixed_size
            self._full_size_confirmed = True
        if not packed and not getattr(self, "_full_size_confirmed", False):
            predictor = self.model.predictor
            if predictor is None:
                raise RuntimeError("模型未建立推理器")
            size = predictor.args.imgsz
            self._full_input_size = size.copy() if isinstance(size, list) else size
            self._full_size_confirmed = True

        return (results[0] if results else None), yolo_ms

    def _static_input_size(self) -> tuple[int, int] | None:
        """读取已加载静态后端的导出高宽，防止首帧拼图后被原默认尺寸覆盖。

        Returns:
            可信的固定 (height,width)；动态后端或没有二元尺寸元数据时为 None。
        """
        backend = getattr(getattr(self.model, "predictor", None), "model", None)
        if backend is None or getattr(backend, "pt", False) or getattr(backend, "dynamic", False):
            return None
        size = getattr(backend, "imgsz", None)
        if (isinstance(size, (list, tuple)) and len(size) == 2
                and all(type(v) is int and v > 0 for v in size)):
            return size[0], size[1]
        return None

    @staticmethod
    def restore_result(raw: object, img: Image.Image, cards: Sequence[Detection]) -> object:
        """构造原图结果供布局解析和标注。

        Args:
            raw: 当前模型画布结果，保留类别元数据。
            img: 当前原图。
            cards: 已恢复并去重的原图像素检测框。

        Returns:
            原始截图尺寸的 Ultralytics Results；不同位置的同点数牌均保留。
        """
        import numpy as np
        from ultralytics.engine.results import Results

        metadata = cast("Results", raw)
        names = metadata.names
        indices = {name: index for index, name in names.items()}
        boxes = torch.tensor([(*c.bbox, c.confidence, indices[c.name]) for c in cards], dtype=torch.float32).reshape(-1, 6)
        return Results(np.asarray(img.convert("RGB"))[:, :, ::-1].copy(), path=metadata.path, names=names, boxes=boxes)
