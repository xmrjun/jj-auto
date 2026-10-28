"""按钮模板工具：截取 JJ 窗口、裁剪模板、从本地 jjdouzeroAuto 克隆导入、在截图上核对匹配。

用法::

    python -m utils.button_templates capture --window JJ斗地主
    python -m utils.button_templates crop --image config/jj_buttons/_shot.png --name play --box 600 520 700 560
    python -m utils.button_templates import-jjdouzero D:/code/jjdouzeroAuto
    python -m utils.button_templates check --image config/jj_buttons/_shot.png
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import yaml

from automation.buttons import load_button_detector, write_width

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT/"config"/"jj_buttons"
# jjdouzeroAuto/Auto_JJ2/pics 中的文件名 → 本项目按钮名。这些图片只在你本机复制，不入库。
JJDOUZERO_PICS: dict[str, str] = {
    "play_card.png": "play.png", "pass_btn.png": "pass.png", "yaobuqi.png": "cannot_beat.png",
    "bujiao_btn.png": "bid_pass.png", "fen1.png": "bid_1.png", "fen2.png": "bid_2.png", "fen3.png": "bid_3.png",
    "jiabei_btn.png": "double.png", "once_more.png": "continue.png", "tuoguan.png": "trusteeship.png",
    "confirm.png": "confirm.png", "dati.png": "dialog.png",
}
JJDOUZERO_WIDTH = 1280


def capture(window: str) -> Path:
    """截取 JJ 窗口保存为 PNG，供裁剪模板。

    Args:
        window: 窗口精确标题。

    Returns:
        截图路径。

    Raises:
        RuntimeError: 找不到窗口或非 Windows。
    """
    from core.screen_capture import ScreenCapture

    image = ScreenCapture(window).capture_window()
    if image is None:
        raise RuntimeError(f"找不到窗口: {window}")
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPLATE_DIR/f"_shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    image.save(path)
    return path


def crop(image_path: Path, name: str, box: Sequence[int]) -> Path:
    """从截图裁剪一个按钮模板并记录截图宽度。

    Args:
        image_path: capture 生成的截图。
        name: 按钮名，如 play。
        box: 像素框 x1 y1 x2 y2。

    Returns:
        模板路径。

    Raises:
        ValueError: 框无效。
    """
    from PIL import Image

    x1, y1, x2, y2 = box
    with Image.open(image_path) as image:
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("裁剪框超出截图范围")
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        target = TEMPLATE_DIR/f"{name}.png"
        image.crop((x1, y1, x2, y2)).save(target)
        write_width(TEMPLATE_DIR, target.name, image.width)
    return target


def import_jjdouzero(clone: Path) -> list[str]:
    """从本地 jjdouzeroAuto 克隆复制按钮图片到模板目录。

    Args:
        clone: 仓库根目录或其 Auto_JJ2/pics 目录。

    Returns:
        已导入的目标文件名。

    Raises:
        FileNotFoundError: 找不到 pics 目录。
    """
    pics = next((p for p in (clone/"Auto_JJ2"/"pics", clone/"pics", clone) if (p/"play_card.png").is_file()), None)
    if pics is None:
        raise FileNotFoundError(f"{clone} 中找不到 Auto_JJ2/pics/play_card.png")
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    imported: list[str] = []
    for source, target in JJDOUZERO_PICS.items():
        if (pics/source).is_file():
            shutil.copyfile(pics/source, TEMPLATE_DIR/target)
            write_width(TEMPLATE_DIR, target, JJDOUZERO_WIDTH)
            imported.append(target)
    return imported


def check(image_path: Path) -> list[str]:
    """在截图上运行按钮检测并列出结果。

    Args:
        image_path: 截图。

    Returns:
        每个匹配一行说明。
    """
    import numpy as np
    from PIL import Image

    config = yaml.safe_load((ROOT/"config"/"jj_buttons.yaml").read_text(encoding="utf-8"))
    detector, missing = load_button_detector(config, ROOT)
    with Image.open(image_path) as image:
        matches = detector.detect(np.asarray(image.convert("RGB")))
    lines = [f"{m.name}: 相似度 {m.score:.3f} 位置 {tuple(int(v) for v in m.bbox)}"
             + ("" if m.actionable else "（有歧义，不会点击）") for m in matches.values()]
    if missing:
        lines.append("缺少模板: " + ", ".join(missing))
    return lines or ["没有匹配到任何按钮"]


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数；None 使用 sys.argv。

    Returns:
        退出码。
    """
    parser = argparse.ArgumentParser(description="JJ 按钮模板工具")
    sub = parser.add_subparsers(dest="command", required=True)
    p_capture = sub.add_parser("capture", help="截取 JJ 窗口")
    p_capture.add_argument("--window", default="JJ斗地主")
    p_crop = sub.add_parser("crop", help="从截图裁剪模板")
    p_crop.add_argument("--image", type=Path, required=True)
    p_crop.add_argument("--name", required=True)
    p_crop.add_argument("--box", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    p_import = sub.add_parser("import-jjdouzero", help="从本地 jjdouzeroAuto 克隆导入")
    p_import.add_argument("path", type=Path)
    p_check = sub.add_parser("check", help="在截图上核对匹配")
    p_check.add_argument("--image", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            print(f"已保存: {capture(args.window)}（用画图找按钮像素框后执行 crop）")
        elif args.command == "crop":
            print(f"已保存模板: {crop(args.image, args.name, args.box)}")
        elif args.command == "import-jjdouzero":
            print("已导入: " + ", ".join(import_jjdouzero(args.path)))
            print("提示：这些图片来自他人截图的 JJ 版本，若匹配不到请用 capture/crop 重新截取。")
        else:
            print("\n".join(check(args.image)))
    except Exception as exc:
        print(f"失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
