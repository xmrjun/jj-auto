"""推理子进程内的自动化运行时：读取配置、加载按钮模板与 DouZero、连接 Win32 输入。

只在子进程中使用，不创建 Qt 对象；重型依赖在首次开启观察/自动模式时才加载。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from automation.bidding import BidPolicy
from automation.buttons import ButtonDetector, load_button_detector
from automation.coordinator import AutomationConfig, AutomationCoordinator, observation_from_frame
from automation.emergency_stop import EmergencyStop, EventLike, HotkeyWatcher
from automation.input_backend import GuardedInputBackend, NativeInput
from automation.types import AutomationMode

if TYPE_CHECKING:
    from PIL import Image


def _read_yaml(path: Path) -> dict[str, object]:
    """读取 YAML 映射；文件损坏必须报错，不能静默使用空配置。

    Args:
        path: 文件路径。

    Returns:
        配置字典；文件不存在时为空字典。

    Raises:
        ValueError: 内容不是映射或 YAML 语法错误。
    """
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.name} 格式错误: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} 顶层必须是映射")
    return payload


def _locate(name: str, base_dir: Path, resource_dir: Path) -> Path:
    """优先使用可写目录中的文件，其次使用资源目录。

    Args:
        name: 相对路径。
        base_dir: 可写运行目录。
        resource_dir: 冻结资源目录。

    Returns:
        存在的路径；都不存在时返回可写目录路径。
    """
    for root in (base_dir, resource_dir):
        candidate = root/name
        if candidate.exists():
            return candidate
    return base_dir/name


def _flag(values: Mapping[str, object], key: str, default: bool) -> bool:
    """读取布尔配置。

    Args:
        values: 配置。
        key: 字段名。
        default: 缺省值。

    Returns:
        布尔值。

    Raises:
        TypeError: 不是布尔值。
    """
    value = values.get(key, default)
    if type(value) is not bool:
        raise TypeError(f"{key} 必须是 true/false")
    return value


def _number(values: Mapping[str, object], key: str, default: float, low: float, high: float) -> float:
    """读取有范围的数字配置。

    Args:
        values: 配置。
        key: 字段名。
        default: 缺省值。
        low: 下限。
        high: 上限。

    Returns:
        浮点数。

    Raises:
        TypeError: 不是数字。
        ValueError: 越界。
    """
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} 必须是数字")
    if not low <= float(value) <= high:
        raise ValueError(f"{key} 必须在 {low}~{high} 之间")
    return float(value)


@dataclass(frozen=True)
class RuntimeOptions:
    """协调器之外的运行参数。

    Attributes:
        douzero_model: douzero/weights 下的权重目录名。
        click_settle: 点击间隔秒。
        stop_hotkey: 紧急停止热键。
        buttons_config: 按钮配置文件相对路径。
    """

    douzero_model: str = "resnet"
    click_settle: float = 0.08
    stop_hotkey: str = "F12"
    buttons_config: str = "config/jj_buttons.yaml"


def parse_config(values: Mapping[str, object]) -> tuple[AutomationConfig, RuntimeOptions]:
    """把 automation.yaml 转换为协调器参数和运行时参数。

    Args:
        values: automation.yaml 内容。

    Returns:
        协调器参数与其余运行参数。

    Raises:
        TypeError: 字段类型错误。
        ValueError: 字段越界。
    """
    bidding = values.get("bidding", {})
    if not isinstance(bidding, Mapping):
        raise TypeError("bidding 必须是映射")
    config = AutomationConfig(
        required_frames=int(_number(values, "required_frames", 3, 1, 20)),
        raise_ratio=_number(values, "raise_ratio", 0.3, 0.05, 1.5),
        verify_timeout=_number(values, "verify_timeout", 3.0, 0.5, 30),
        max_corrections=int(_number(values, "max_corrections", 1, 0, 5)),
        resume_next_round=_flag(values, "resume_next_round", True),
        auto_continue=_flag(values, "auto_continue", False),
        round_end_seconds=_number(values, "round_end_seconds", 3.0, 0.5, 60),
        bid_policy=BidPolicy.from_mapping(bidding),
    )
    model = str(values.get("douzero_model", "resnet"))
    if not model or "/" in model or "\\" in model or model.startswith("."):
        raise ValueError("douzero_model 必须是 douzero/weights 下的目录名")
    options = RuntimeOptions(model, _number(values, "click_settle", 0.08, 0.0, 1.0),
                             str(values.get("stop_hotkey", "F12")),
                             str(values.get("buttons_config", "config/jj_buttons.yaml")))
    return config, options


class AutomationRuntime:
    """子进程中的自动化入口。

    Attributes:
        stop: 紧急停止标志（与 UI 共享）。
        coordinator: 闭环协调器。
        detector: 按钮检测器。
        missing_templates: 配置了但缺少图片的按钮。
    """

    def __init__(self, base_dir: str | Path, resource_dir: str | Path, stop_event: EventLike | None = None,
                 device: str = "cpu", native: NativeInput | None = None) -> None:
        """读取配置、加载按钮模板并启动热键监视；不加载 DouZero。

        Args:
            base_dir: 可写运行目录。
            resource_dir: 资源目录。
            stop_event: UI 共享的停止事件。
            device: DouZero 推理设备。
            native: 测试注入的输入实现；None 时 Windows 使用 Win32Input。

        Raises:
            TypeError: 配置字段类型错误。
            ValueError: 配置损坏或越界。
        """
        self._base = Path(base_dir)
        self._resource = Path(resource_dir)
        self._device = device
        config, options = parse_config(_read_yaml(_locate("config/automation.yaml", self._base, self._resource)))
        self._options = options
        self.stop = EmergencyStop(stop_event)
        if native is None and sys.platform == "win32":
            from automation.windows_input import Win32Input

            native = Win32Input()
        backend = GuardedInputBackend(native, self.stop, settle=options.click_settle)
        self.coordinator = AutomationCoordinator(None, backend, config)
        buttons = _read_yaml(_locate(options.buttons_config, self._base, self._resource))
        self.detector: ButtonDetector
        self.detector, self.missing_templates = load_button_detector(buttons, self._base)
        self._watcher = HotkeyWatcher(self.stop, options.stop_hotkey)
        self._watcher.start()

    @property
    def active(self) -> bool:
        """观察或自动模式时为 True。"""
        return self.coordinator.active

    def set_mode(self, mode: str) -> None:
        """切换模式，首次开启时加载并校验 DouZero 权重。

        Args:
            mode: disabled、observe 或 auto。

        Raises:
            ValueError: 模式未知。
            FileNotFoundError: 权重缺失。
            ModelAssetError: 权重未批准或哈希不一致。
        """
        selected = AutomationMode(mode)
        if selected != AutomationMode.DISABLED and self.coordinator.decider is None:
            from decision.douzero_adapter import DouZeroAdapter, TorchDouZeroScorer

            folder = _locate(f"douzero/weights/{self._options.douzero_model}", self._base, self._resource)
            scorer = TorchDouZeroScorer(folder, self._device)
            self.coordinator.decider = DouZeroAdapter(scorer, scorer.model_type)
        self.coordinator.set_mode(selected)

    def process(self, image: "Image.Image | None", valid: bool, region_boxes: Mapping[str, list[dict[str, object]]],
                known: set[str], mapping: Mapping[str, str], played_regions: Mapping[str, str],
                history: Mapping[str, list[list[str]]], hwnd: int | None,
                window_rect: tuple[int, int, int, int] | None) -> dict[str, object]:
        """处理一帧并返回界面状态。

        Args:
            image: 原始截图。
            valid: 截图与模型是否成功。
            region_boxes: 分区检测框。
            known: 已知区域。
            mapping: YOLO 类别到牌点。
            played_regions: 座位到出牌区域。
            history: 记牌器已确认历史。
            hwnd: 截图窗口句柄。
            window_rect: 截图窗口矩形。

        Returns:
            可跨进程传递的状态字典。
        """
        import numpy as np

        buttons = {}
        if image is not None and valid:
            buttons = self.detector.detect(np.asarray(image.convert("RGB")))
        observation = observation_from_frame(
            time.monotonic(), valid and image is not None, image.size if image is not None else (0, 0),
            window_rect, region_boxes, known, mapping, played_regions, buttons, history)
        status = self.coordinator.handle(observation, hwnd).to_dict()
        status["buttons"] = sorted(name for name, match in buttons.items() if match.actionable)
        status["missing_templates"] = list(self.missing_templates)
        status["hotkey"] = self._watcher.key
        return status

    def status(self) -> dict[str, object]:
        """返回不处理新帧的当前状态。

        Returns:
            状态字典。
        """
        status = self.coordinator.status.to_dict()
        status["buttons"] = []
        status["missing_templates"] = list(self.missing_templates)
        status["hotkey"] = self._watcher.key
        return status

    def close(self) -> None:
        """停止热键线程。"""
        self._watcher.close()
