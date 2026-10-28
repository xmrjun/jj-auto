"""下载或复制运行所需模型，并按 security/model-manifest.json 校验 SHA-256。

只下载清单中已批准且固定到具体提交的文件；哈希不符的文件不会落盘。
这是唯一需要联网的步骤，正式运行程序本身不联网下载任何东西。

用法::

    python scripts/fetch_models.py                 # 下载 yolov11s 识牌模型 + DouZero ResNet 出牌模型
    python scripts/fetch_models.py --yolo n m      # 额外选择 n/m 识牌模型
    python scripts/fetch_models.py --from-dir D:/jjdouzeroAuto D:/han-card-tracker   # 从本地克隆复制
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"security"/"model-manifest.json"
YOLO_NAMES = {"n": "yolov11n_imgsz_960", "s": "yolov11s_imgsz_960", "m": "yolov11m_imgsz_960"}


def raw_url(source: str) -> str:
    """把 GitHub blob 链接转换为固定提交的 raw 下载链接。

    Args:
        source: https://github.com/<owner>/<repo>/blob/<commit>/<path>。

    Returns:
        raw.githubusercontent.com 链接。

    Raises:
        ValueError: 链接不是固定提交的 GitHub blob 链接。

    Examples:
        >>> raw_url("https://github.com/a/b/blob/" + "0"*40 + "/x/y.onnx")
        'https://raw.githubusercontent.com/a/b/0000000000000000000000000000000000000000/x/y.onnx'
    """
    prefix = "https://github.com/"
    parts = source[len(prefix):].split("/") if source.startswith(prefix) else []
    if len(parts) < 5 or parts[2] != "blob" or len(parts[3]) != 40:
        raise ValueError(f"来源必须是固定提交的 GitHub 链接: {source}")
    owner, repo, _, commit = parts[:4]
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{'/'.join(parts[4:])}"


def destination(entry: dict[str, object]) -> Path:
    """返回清单条目在程序目录中的存放位置。

    Args:
        entry: 清单条目。

    Returns:
        目标路径。

    Raises:
        ValueError: 格式不支持。
    """
    path = str(entry["path"])
    if entry["format"] == "onnx":
        return ROOT/"yolo"/"weights"/path
    if entry["format"] == "torch_state_dict":
        return ROOT/"douzero"/"weights"/path
    raise ValueError(f"不支持的格式: {entry['format']}")


def sha256(path: Path) -> str:
    """流式计算文件 SHA-256。

    Args:
        path: 文件。

    Returns:
        小写十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(entry: dict[str, object], local_roots: Sequence[Path]) -> str:
    """下载或复制一个文件，校验通过后原子替换到目标位置。

    Args:
        entry: 清单条目。
        local_roots: 本地克隆目录；在其中按文件名查找。

    Returns:
        结果说明。

    Raises:
        RuntimeError: 哈希不符或找不到来源。
    """
    target = destination(entry)
    expected = str(entry["sha256"])
    if target.is_file() and sha256(target) == expected:
        return f"已存在且校验通过: {target.relative_to(ROOT)}"
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(handle)
    try:
        source = str(entry["source"])
        relative = source.split("/blob/", 1)[1].split("/", 1)[1]
        local = next((root/relative for root in local_roots if (root/relative).is_file()), None)
        if local is not None:
            shutil.copyfile(local, temporary)
            origin = str(local)
        elif local_roots:
            raise RuntimeError(f"本地目录中找不到 {relative}")
        else:
            origin = raw_url(source)
            with urllib.request.urlopen(origin, timeout=60) as response, open(temporary, "wb") as output:
                shutil.copyfileobj(response, output, 1 << 20)
        actual = sha256(Path(temporary))
        if actual != expected:
            raise RuntimeError(f"{target.name} 哈希不符，已丢弃。期望 {expected}，实际 {actual}（来源 {origin}）")
        os.replace(temporary, target)
        return f"已安装并校验: {target.relative_to(ROOT)}"
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数；None 使用 sys.argv。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description="下载并校验识牌/出牌模型")
    parser.add_argument("--yolo", nargs="*", default=["s"], choices=sorted(YOLO_NAMES), help="识牌模型规格")
    parser.add_argument("--skip-douzero", action="store_true", help="不安装 DouZero 出牌模型")
    parser.add_argument("--from-dir", nargs="*", default=[], type=Path,
                        help="从本地 han-card-tracker / jjdouzeroAuto 克隆复制，不联网")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted_yolo = {YOLO_NAMES[key] for key in args.yolo}
    failures = 0
    for entry in manifest["artifacts"]:
        if entry.get("approved_for_runtime") is not True or "source" not in entry:
            continue
        if entry["format"] == "onnx" and str(entry["path"]).split("/")[0] not in wanted_yolo:
            continue
        if entry["format"] == "torch_state_dict" and args.skip_douzero:
            continue
        try:
            print(install(entry, [Path(p) for p in args.from_dir]))
        except Exception as exc:
            failures += 1
            print(f"失败: {entry['path']}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
