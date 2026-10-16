# -*- coding: utf-8 -*-
"""B1 自采集照片预处理。

把手机拍回来的原始照片，规范化成可直接进入标注/审计流程的形式：

    1. 递归收集图片（含 iPhone 的 .heic/.HEIC）
    2. EXIF 方向校正（关键：手机竖拍带 orientation tag，OpenCV 不认，
       若不做这一步，后面画出来的框会整体旋转 90°，标注全废）
    3. HEIC -> JPG 转换
    4. 统一重命名为 ASCII：b1_0001.jpg（避免项目已知的"中文路径"坑）
    5. 按 EXIF 拍摄时间排序，输出重命名映射表（可追溯）
    6. 列出无法读取/过小/重复尺寸的可疑文件，供人工剔除

用法：
    # 干跑（只体检，不写文件）
    python tools/prep_b1_photos.py --src D:/B1_raw --out data/raw/b1

    # 实际执行
    python tools/prep_b1_photos.py --src D:/B1_raw --out data/raw/b1 --apply

输出：
    <out>/images/b1_0001.jpg ...
    <out>/prep_report.txt      体检报告（分辨率分布、转换/失败清单）
    <out>/rename_map.csv       原名 <-> 新名 映射（含拍摄时间、尺寸）
"""
import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}

# 最低可用分辨率（低于此值的照片在 640 训练尺度下细节不足）
MIN_W, MIN_H = 640, 480


def ensure_heif():
    """注册 HEIC 解码器；失败则返回 False 并给出提示。"""
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
        return True
    except Exception:  # noqa: BLE001
        return False


def exif_dt(img: Image.Image) -> datetime | None:
    """取 EXIF 拍摄时间，失败返回 None。"""
    try:
        ex = img.getexif()
        for tag in (36867, 36868, 306):  # DateTimeOriginal / Digitized / DateTime
            v = ex.get(tag)
            if v:
                return datetime.strptime(str(v)[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="B1 自采集照片预处理")
    ap.add_argument("--src", required=True, help="原始照片目录（手机导出文件夹）")
    ap.add_argument("--out", default="data/raw/b1", help="输出目录（默认 data/raw/b1）")
    ap.add_argument("--prefix", default="b1", help="重命名前缀（默认 b1）")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 质量（默认 95）")
    ap.add_argument("--apply", action="store_true", help="实际写入；缺省为干跑")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print(f"[错误] 源目录不存在：{src}")
        return 2

    has_heif = ensure_heif()

    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not files:
        print(f"[错误] 在 {src} 下没找到图片（支持的扩展名：{', '.join(sorted(IMG_EXT))}）")
        return 2

    print(f"[扫描] {src}  ->  {out}")
    print(f"  找到 {len(files)} 个候选文件")
    heic_n = sum(1 for p in files if p.suffix.lower() in {".heic", ".heif"})
    if heic_n:
        print(f"  其中 HEIC/HEIF {heic_n} 个 —— HEIC 解码器：{'已加载' if has_heif else '缺失'}")

    rows: list[dict] = []
    failed: list[tuple[str, str]] = []
    small: list[str] = []
    sizes: Counter = Counter()

    for p in files:
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)  # 方向校正（同时清掉 orientation tag）
                im = im.convert("RGB")
                w, h = im.size
                dt = exif_dt(im)
                rows.append({"src": p, "img": im.copy(), "w": w, "h": h, "dt": dt,
                             "was_heic": p.suffix.lower() in {".heic", ".heif"}})
                sizes[(w, h)] += 1
                if w < MIN_W or h < MIN_H:
                    small.append(f"{p.name}  {w}x{h}")
        except Exception as exc:  # noqa: BLE001
            failed.append((p.name, f"{type(exc).__name__}: {exc}"))

    if not rows:
        print("[错误] 没有一张能成功读取。若全是 HEIC，请先安装解码器：")
        print("  .venv\\Scripts\\pip install pillow-heif")
        return 3

    # 按拍摄时间排序（无 EXIF 的排最后，保持原相对顺序）
    far = datetime(1970, 1, 1)
    idx_order = sorted(range(len(rows)), key=lambda i: (rows[i]["dt"] or far, i))
    rows = [rows[i] for i in idx_order]

    # 写文件
    img_dir = out / "images"
    if args.apply:
        img_dir.mkdir(parents=True, exist_ok=True)

    mapping: list[tuple[str, str, str, int, int]] = []
    n_heic_conv = 0
    for i, r in enumerate(rows, 1):
        new_name = f"{args.prefix}_{i:04d}.jpg"
        mapping.append((str(r["src"].relative_to(src)), new_name,
                        r["dt"].strftime("%Y-%m-%d %H:%M:%S") if r["dt"] else "",
                        r["w"], r["h"]))
        if r["was_heic"]:
            n_heic_conv += 1
        if args.apply:
            r["img"].save(img_dir / new_name, "JPEG", quality=args.quality, subsampling=0)

    # 报告
    lines = [
        f"B1 照片预处理报告    {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"源目录: {src}",
        f"输出目录: {out}" + ("" if args.apply else "   [干跑，未写文件]"),
        "",
        f"可用照片总数: {len(rows)}",
        f"  HEIC 转换: {n_heic_conv}",
        f"  读取失败: {len(failed)}",
        f"  低于 {MIN_W}x{MIN_H}: {len(small)}",
        "",
        "分辨率分布 (TOP10):",
        *[f"  {w}x{h:<6} {c:>5} 张" for (w, h), c in sizes.most_common(10)],
        "",
        "拍摄时间跨度: "
        + (
            f"{rows[0]['dt']:%Y-%m-%d %H:%M} ~ {rows[-1]['dt']:%Y-%m-%d %H:%M}"
            if rows[0]["dt"] and rows[-1]["dt"]
            else "（EXIF 时间缺失）"
        ),
    ]
    if small:
        lines += ["", f"⚠️ 分辨率偏低（建议剔除）：", *[f"  {s}" for s in small[:30]]]
    if failed:
        lines += ["", "❌ 读取失败：", *[f"  {n}  ->  {e}" for n, e in failed[:30]]]
    if n_heic_conv and not args.apply:
        lines += ["", "（干跑模式下未实际转换 HEIC；加 --apply 执行）"]

    report = "\n".join(lines) + "\n"
    print()
    print(report)

    if args.apply:
        out.mkdir(parents=True, exist_ok=True)
        (out / "prep_report.txt").write_text(report, encoding="utf-8")
        with open(out / "rename_map.csv", "w", newline="", encoding="utf-8-sig") as f:
            wtr = csv.writer(f)
            wtr.writerow(["原路径(相对)", "新文件名", "拍摄时间", "宽", "高"])
            wtr.writerows(mapping)
        print(f"[完成] {len(rows)} 张 -> {img_dir}")
        print(f"  报告: {out / 'prep_report.txt'}")
        print(f"  映射: {out / 'rename_map.csv'}")
        print()
        print("下一步：")
        print(f"  .venv\\Scripts\\python.exe tools\\dedup_new_dataset.py --src {out} "
              f"--name b1 --refs dataset1,dataset2,labcoat_si")
    else:
        print("[干跑] 未写入任何文件。确认无误后加 --apply 执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
