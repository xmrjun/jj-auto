"""模型制品清单、格式白名单与完整性校验。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "security" / "model-manifest.json"
)
RUNTIME_FORMATS = frozenset({"onnx", "engine"})
DECISION_FORMAT = "torch_state_dict"


class ModelAssetError(RuntimeError):
    """模型制品策略错误基类。"""


class ModelManifestError(ModelAssetError):
    """模型清单缺失或结构无效。"""


class ModelIntegrityError(ModelAssetError):
    """模型文件哈希与清单不一致。"""


class UnsafeModelFormatError(ModelAssetError):
    """模型格式不允许进入正式运行进程。"""


class UnapprovedModelError(ModelAssetError):
    """模型未在清单中明确批准。"""


@dataclass(frozen=True)
class RuntimeModel:
    """已通过格式、清单与哈希检查的运行模型。"""

    path: Path
    format: str
    sha256: str


def _valid_sha256(value: object) -> bool:
    """判断值是否为规范的小写 SHA-256。"""
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def verify_model(path: str | Path, expected_sha256: str) -> str:
    """流式计算并核对模型文件 SHA-256。"""
    if not _valid_sha256(expected_sha256):
        raise ModelManifestError("模型清单中的 SHA-256 无效")

    model_path = Path(path)
    digest = hashlib.sha256()
    try:
        with model_path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到模型文件: {model_path}") from None

    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ModelIntegrityError(
            f"模型完整性校验失败: {model_path.name}，"
            f"期望 {expected_sha256}，实际 {actual}"
        )
    return actual


def load_model_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """读取并执行模型清单的最小结构校验。"""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ModelManifestError(f"模型清单不存在: {manifest_path}") from None
    except json.JSONDecodeError as exc:
        raise ModelManifestError(f"模型清单不是有效 JSON: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ModelManifestError("模型清单 schema_version 必须为 1")
    if not isinstance(payload.get("artifacts"), list):
        raise ModelManifestError("模型清单 artifacts 必须是列表")
    return payload


def _matches_manifest_path(requested: Path, manifest_value: object) -> bool:
    """按文件名或仓库相对路径匹配清单条目。"""
    if not isinstance(manifest_value, str) or not manifest_value:
        return False
    declared = Path(manifest_value)
    if declared.is_absolute() or ".." in declared.parts:
        raise ModelManifestError("模型清单 path 必须是安全的相对路径")
    if len(declared.parts) == 1:
        return requested.name == declared.name
    requested_parts = requested.parts
    return tuple(requested_parts[-len(declared.parts) :]) == declared.parts


def resolve_runtime_model(
    requested_path: str | Path,
    *,
    manifest: Mapping[str, object] | None = None,
) -> RuntimeModel:
    """只解析清单明确批准的 ONNX/TensorRT 运行模型。"""
    requested = Path(requested_path)
    model_format = requested.suffix.lower().lstrip(".")
    if model_format not in RUNTIME_FORMATS:
        raise UnsafeModelFormatError(
            f"正式运行不允许 .{model_format or '<无扩展名>'} 模型；"
            "仅允许清单批准的 .onnx 或 .engine"
        )

    payload: Mapping[str, object] = (
        load_model_manifest() if manifest is None else manifest
    )
    if payload.get("schema_version") != 1:
        raise ModelManifestError("模型清单 schema_version 必须为 1")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ModelManifestError("模型清单 artifacts 必须是列表")

    matching: Mapping[str, object] | None = None
    for candidate in artifacts:
        if not isinstance(candidate, Mapping):
            raise ModelManifestError("模型清单 artifact 必须是对象")
        if _matches_manifest_path(requested, candidate.get("path")):
            matching = candidate
            break

    if matching is None or matching.get("approved_for_runtime") is not True:
        raise UnapprovedModelError(f"模型未获准用于正式运行: {requested.name}")
    if matching.get("format") != model_format:
        raise ModelManifestError(f"模型格式与文件扩展名不一致: {requested.name}")

    expected_hash = matching.get("sha256")
    if not isinstance(expected_hash, str):
        raise ModelManifestError("批准的模型缺少 SHA-256")
    actual_hash = verify_model(requested, expected_hash)
    return RuntimeModel(requested.resolve(), model_format, actual_hash)


def resolve_decision_model(
    requested_path: str | Path,
    *,
    manifest: Mapping[str, object] | None = None,
) -> RuntimeModel:
    """只解析清单批准的 DouZero 张量状态字典（.ckpt）。

    决策权重格式记为 ``torch_state_dict``，调用方必须以
    ``torch.load(weights_only=True)`` 读取，禁止执行任意 pickle 对象。

    Args:
        requested_path: 权重文件路径。
        manifest: 测试注入的清单；省略时读取默认清单。

    Returns:
        已通过格式、批准与哈希校验的权重。

    Raises:
        UnsafeModelFormatError: 扩展名不是 .ckpt。
        ModelManifestError: 清单结构或格式字段无效。
        UnapprovedModelError: 未在清单中批准。
        ModelIntegrityError: 哈希不一致。
        FileNotFoundError: 文件不存在。
    """
    requested = Path(requested_path)
    if requested.suffix.lower() != ".ckpt":
        raise UnsafeModelFormatError(f"决策模型只允许清单批准的 .ckpt 状态字典: {requested.name}")
    payload: Mapping[str, object] = load_model_manifest() if manifest is None else manifest
    artifacts = payload.get("artifacts")
    if payload.get("schema_version") != 1 or not isinstance(artifacts, list):
        raise ModelManifestError("模型清单结构无效")
    matching: Mapping[str, object] | None = None
    for candidate in artifacts:
        if not isinstance(candidate, Mapping):
            raise ModelManifestError("模型清单 artifact 必须是对象")
        if (candidate.get("format") == DECISION_FORMAT
                and _matches_manifest_path(requested, candidate.get("path"))):
            matching = candidate
            break
    if matching is None or matching.get("approved_for_runtime") is not True:
        raise UnapprovedModelError(f"决策模型未获准用于运行: {requested.name}")
    expected_hash = matching.get("sha256")
    if not isinstance(expected_hash, str):
        raise ModelManifestError("批准的决策模型缺少 SHA-256")
    actual_hash = verify_model(requested, expected_hash)
    return RuntimeModel(requested.resolve(), DECISION_FORMAT, actual_hash)
