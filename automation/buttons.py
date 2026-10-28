"""JJ 按钮模板匹配：多尺度、限定搜索区域，并拒绝存在歧义的匹配。

模板图片必须由使用者从自己的 JJ 窗口截图中裁剪，不随仓库分发。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from automation.types import Box, ButtonMatch

DEFAULT_SCALE_STEPS: tuple[float, ...] = (0.92, 1.0, 1.08)


@dataclass(frozen=True)
class ButtonTemplate:
    """一个按钮模板。

    Attributes:
        name: 按钮名。
        image: 灰度模板，uint8。
        reference_width: 截取模板时窗口截图宽度。
        threshold: 最低相关系数。
        region: 归一化搜索区域 (x1, y1, x2, y2)。
    """

    name: str
    image: np.ndarray
    reference_width: float
    threshold: float = 0.8
    region: Box = (0.0, 0.0, 1.0, 1.0)


def _gray(image: np.ndarray) -> np.ndarray:
    """把 RGB/RGBA/灰度数组转换为 uint8 灰度。

    Args:
        image: 输入数组。

    Returns:
        二维 uint8 数组。

    Raises:
        ValueError: 数组维度不支持。
    """
    import cv2

    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        code = cv2.COLOR_RGBA2GRAY if array.shape[2] == 4 else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array.astype(np.uint8, copy=False), code)
    raise ValueError("图像必须是灰度、RGB 或 RGBA 数组")


class ButtonDetector:
    """在截图中查找按钮。

    Attributes:
        templates: 按名称索引的模板。
        ambiguity_margin: 次优匹配与最优匹配之差小于该值时视为歧义。
        match_scale: 匹配前对截图的缩放比例，降低耗时。
        scale_steps: 围绕窗口宽度换算比例的微调倍数。
    """

    def __init__(self, templates: Sequence[ButtonTemplate], ambiguity_margin: float = 0.03,
                 match_scale: float = 0.5, scale_steps: Sequence[float] = DEFAULT_SCALE_STEPS) -> None:
        """创建检测器。

        Args:
            templates: 模板列表。
            ambiguity_margin: 歧义判定差值。
            match_scale: 截图缩放比例，(0, 1]。
            scale_steps: 尺度微调倍数。

        Raises:
            ValueError: 参数越界或模板重名。
        """
        if not 0 < match_scale <= 1:
            raise ValueError("match_scale 必须在 (0, 1] 内")
        if ambiguity_margin < 0:
            raise ValueError("ambiguity_margin 不能为负")
        names = [template.name for template in templates]
        if len(names) != len(set(names)):
            raise ValueError("按钮模板名称重复")
        self.templates = {template.name: template for template in templates}
        self.ambiguity_margin = ambiguity_margin
        self.match_scale = match_scale
        self.scale_steps = tuple(scale_steps)

    @property
    def names(self) -> tuple[str, ...]:
        """返回已加载的按钮名。"""
        return tuple(self.templates)

    def detect(self, image: np.ndarray, names: Sequence[str] | None = None) -> dict[str, ButtonMatch]:
        """匹配全部或指定按钮。

        Args:
            image: 截图数组（RGB 或灰度）。
            names: 只匹配这些按钮；None 匹配全部。

        Returns:
            达到阈值的按钮匹配，坐标为原截图像素。
        """
        import cv2

        gray = _gray(image)
        height, width = gray.shape
        small = gray if self.match_scale == 1 else cv2.resize(
            gray, (max(1, int(width*self.match_scale)), max(1, int(height*self.match_scale))),
            interpolation=cv2.INTER_AREA)
        results: dict[str, ButtonMatch] = {}
        for name in (self.templates if names is None else names):
            template = self.templates.get(name)
            if template is None:
                continue
            match = self._match(small, width, template)
            if match is not None:
                results[name] = match
        return results

    def _match(self, small: np.ndarray, full_width: int, template: ButtonTemplate) -> ButtonMatch | None:
        """在缩小截图的搜索区域中进行多尺度匹配。

        Args:
            small: 缩小后的灰度截图。
            full_width: 原截图宽度。
            template: 模板。

        Returns:
            匹配结果；低于阈值时为 None。
        """
        import cv2

        sh, sw = small.shape
        x1, y1, x2, y2 = template.region
        left, top = int(x1*sw), int(y1*sh)
        right, bottom = max(left+1, int(x2*sw)), max(top+1, int(y2*sh))
        area = small[top:bottom, left:right]
        base = full_width/template.reference_width*self.match_scale
        best: tuple[float, float, tuple[int, int], tuple[int, int]] | None = None
        for step in self.scale_steps:
            factor = base*step
            th, tw = template.image.shape
            size = (max(4, int(round(tw*factor))), max(4, int(round(th*factor))))
            if size[0] > area.shape[1] or size[1] > area.shape[0]:
                continue
            resized = cv2.resize(template.image, size, interpolation=cv2.INTER_AREA)
            scores = cv2.matchTemplate(area, resized, cv2.TM_CCOEFF_NORMED)
            _, value, _, location = cv2.minMaxLoc(scores)
            if best is not None and value <= best[0]:
                continue
            masked = scores.copy()
            mx, my = location
            masked[max(0, my-size[1]//2):my+size[1]//2+1, max(0, mx-size[0]//2):mx+size[0]//2+1] = -1
            second = float(masked.max()) if masked.size else -1.0
            best = (float(value), second, location, size)
        if best is None or best[0] < template.threshold:
            return None
        value, second, (mx, my), (tw, th) = best
        actionable = not (second >= template.threshold and value-second < self.ambiguity_margin)
        scale = 1/self.match_scale
        bbox = ((left+mx)*scale, (top+my)*scale, (left+mx+tw)*scale, (top+my+th)*scale)
        return ButtonMatch(template.name, bbox, value, actionable)


def read_image(path: Path) -> np.ndarray:
    """读取图片为灰度数组，兼容 Windows 中文路径。

    Args:
        path: 图片路径。

    Returns:
        uint8 灰度数组。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 图片无法解码。
    """
    import cv2

    if not path.is_file():
        raise FileNotFoundError(f"找不到按钮模板: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法解码按钮模板: {path}")
    return image


def _number(value: object, name: str) -> float:
    """把配置数值转换为 float。

    Args:
        value: 配置值。
        name: 字段名，用于报错。

    Returns:
        浮点数。

    Raises:
        TypeError: 不是数字。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数字")
    return float(value)


WIDTHS_FILE = "widths.yaml"


def read_widths(folder: Path) -> dict[str, float]:
    """读取模板目录中记录的各模板截取时窗口宽度。

    Args:
        folder: 模板目录。

    Returns:
        文件名到宽度；文件不存在时为空。

    Raises:
        ValueError: 文件不是“文件名: 数字”映射。
    """
    import yaml

    path = folder/WIDTHS_FILE
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须是 文件名: 宽度 映射")
    return {str(key): _number(value, f"{WIDTHS_FILE}:{key}") for key, value in payload.items()}


def write_width(folder: Path, filename: str, width: float) -> None:
    """记录一个模板的截取宽度（同目录临时文件 + 原子替换）。

    Args:
        folder: 模板目录。
        filename: 模板文件名。
        width: 截图宽度。
    """
    import os
    import tempfile

    import yaml

    widths = read_widths(folder)
    widths[filename] = float(width)
    folder.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=folder, suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(sorted(widths.items())), stream, allow_unicode=True)
    os.replace(temporary, folder/WIDTHS_FILE)


def load_button_detector(config: Mapping[str, object], base_dir: Path) -> tuple[ButtonDetector, list[str]]:
    """按配置加载模板；缺失的图片跳过并在第二个返回值中列出。

    Args:
        config: jj_buttons.yaml 的内容。
        base_dir: 解析 template_dir 的根目录。

    Returns:
        检测器与缺失模板名列表。

    Raises:
        TypeError: 配置字段类型错误。
        ValueError: 配置数值越界。
    """
    buttons = config.get("buttons", {})
    if not isinstance(buttons, Mapping):
        raise TypeError("buttons 必须是映射")
    folder = base_dir/str(config.get("template_dir", "config/jj_buttons"))
    default_width = _number(config.get("reference_width", 1280), "reference_width")
    widths = read_widths(folder)
    default_threshold = _number(config.get("threshold", 0.8), "threshold")
    templates: list[ButtonTemplate] = []
    missing: list[str] = []
    for name, spec in buttons.items():
        if not isinstance(name, str) or not isinstance(spec, Mapping):
            raise TypeError("每个按钮配置必须是 名称: {file, region}")
        raw_region = spec.get("region", (0, 0, 1, 1))
        if not isinstance(raw_region, (list, tuple)) or len(raw_region) != 4:
            raise TypeError(f"按钮 {name} 的 region 必须是四个数字")
        x1, y1, x2, y2 = (_number(v, f"{name}.region") for v in raw_region)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"按钮 {name} 的 region 必须是 0~1 的 x1,y1,x2,y2")
        threshold = _number(spec.get("threshold", default_threshold), f"{name}.threshold")
        filename = str(spec.get("file", f"{name}.png"))
        width = _number(spec.get("reference_width", widths.get(filename, default_width)),
                        f"{name}.reference_width")
        if not 0 < threshold <= 1 or width <= 0:
            raise ValueError(f"按钮 {name} 的阈值或参考宽度无效")
        try:
            image = read_image(folder/filename)
        except FileNotFoundError:
            missing.append(name)
            continue
        templates.append(ButtonTemplate(name, image, width, threshold, (x1, y1, x2, y2)))
    detector = ButtonDetector(templates, _number(config.get("ambiguity_margin", 0.03), "ambiguity_margin"),
                              _number(config.get("match_scale", 0.5), "match_scale"))
    return detector, missing
