"""uploads 预标注草稿的逐张核实渲染（带框序号，供目测判定/定坐标）。

- 图：data/test/uploads/<name>
- 草稿：data/_prelabel/uploads_20261008/labels/<stem>.txt（0=person 1=mask 2=gloves 3=lab_coat 4=goggles）
- 编号：outputs/_prelabel_uploads_20261008/index.txt

用法:
    python tools/make_uploads_verify.py --out outputs/_prelabel_uploads_20261008/verify \
        --nos 1,3,4,5 --width 1200
    python tools/make_uploads_verify.py --out ... --range 1-43 --cols 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "test" / "uploads"
LBL_DIR = ROOT / "data" / "_prelabel" / "uploads_20261008" / "labels"
INDEX = ROOT / "outputs" / "_prelabel_uploads_20261008" / "index.txt"

CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}
COLOR = {0: (200, 200, 200), 1: (0, 0, 230), 2: (230, 120, 0), 3: (0, 170, 0), 4: (200, 0, 170)}


def imread_u(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p: Path, img):
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(p))


def read_index() -> dict:
    m = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        parts = [p for p in parts if p != ""]
        if len(parts) >= 2:
            m[int(parts[0].strip())] = parts[1].strip()
    return m


def parse_nos(spec: str, rng: str) -> list[int]:
    out = []
    if spec:
        out += [int(x) for x in spec.split(",") if x.strip()]
    if rng:
        a, b = rng.split("-")
        out += list(range(int(a), int(b) + 1))
    seen, res = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            res.append(n)
    return res


def render(n: int, rel: str, width: int, out_dir: Path, tag: str = "") -> bool:
    rel = rel.replace("\\", "/")
    name = rel.split("/")[-1]
    img_p = IMG_DIR / name
    if not img_p.exists():
        print(f"[跳过] #{n} 原图不存在 {name}")
        return False
    img = imread_u(img_p)
    if img is None:
        print(f"[跳过] #{n} 读图失败")
        return False
    h, w = img.shape[:2]
    lp = LBL_DIR / (Path(name).stem + ".txt")
    boxes = []
    if lp.exists():
        for i, line in enumerate(lp.read_text(encoding="utf-8").splitlines()):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            boxes.append((c, i, x1, y1, x2, y2))
    scale = width / w
    img = cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    for c, i, x1, y1, x2, y2 in boxes:
        X1, Y1, X2, Y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
        col = COLOR.get(c, (128, 128, 128))
        cv2.rectangle(img, (X1, Y1), (X2, Y2), col, 2)
        cv2.putText(img, f"{i}:{CLS.get(c,'?')}", (X1, max(14, Y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    header = np.full((34, width, 3), 30, np.uint8)
    cv2.putText(header, f"#{n} {tag} orig {w}x{h} boxes={len(boxes)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    out_dir.mkdir(parents=True, exist_ok=True)
    imwrite_u(out_dir / f"u_{n:03d}.jpg", np.vstack([header, img]))
    print(f"u_{n:03d}.jpg  {name}  boxes={len(boxes)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nos", default="")
    ap.add_argument("--range", default="")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    idx = read_index()
    nos = parse_nos(args.nos, args.range)
    out_dir = Path(args.out)
    ok = 0
    for n in nos:
        rel = idx.get(n)
        if rel is None:
            print(f"[跳过] #{n} 不在 index")
            continue
        if render(n, rel, args.width, out_dir, args.tag):
            ok += 1
    print(f"完成 {ok}/{len(nos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
