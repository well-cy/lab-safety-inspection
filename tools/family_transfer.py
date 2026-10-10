# -*- coding: utf-8 -*-
"""同族套用（video-frame family transfer）

同一段视频的相邻帧画面几乎不动、标注几乎一致，人工逐帧重复同样的删/改动作很浪费。
本工具把「参照帧」上的人工操作（目前支持**删除**非 person 框）按坐标映射到同族其他帧，
person 则采纳各帧**自己的 COCO 候选**（逐帧位置更准，不跨帧复制坐标）。

参照帧的"操作前"状态取自落库备份包（tar），"操作后"取她的编辑器导出。

用法:
  $PY tools/family_transfer.py \
      --batch outputs/person_review_p0_e \
      --ref-backup data/lab_ppe/_label_backup/labels_train_person_20261010_230422.tar.gz \
      --ref-export outputs/person_review_p0_d/_her_export.txt \
      --family 8381407 \
      --out-export outputs/person_review_p0_e/_family_8381407_export.txt \
      --apply-ws --sheet outputs/person_review_p0_e/_family_8381407_check.jpg
落库:
  $PY tools/apply_person_review.py --export outputs/person_review_p0_e/_family_8381407_export.txt
"""
import argparse
import io
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "lab_ppe" / "labels" / "train"
NAMES = ["person", "mask", "gloves", "lab_coat", "goggles"]
COLORS = [(0, 190, 0), (200, 140, 0), (0, 120, 255), (255, 90, 0), (200, 0, 160)]


def rows_of_text(txt: str):
    out = []
    for ln in txt.splitlines():
        q = ln.split()
        if len(q) == 5:
            out.append((int(q[0]), *(round(float(v), 6) for v in q[1:5])))
    return out


def rows_of_file(p: Path):
    return rows_of_text(p.read_text(encoding="utf-8")) if p.exists() else []


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    u = a[3] * a[4] + b[3] * b[4] - ix * iy
    return (ix * iy) / u if u > 0 else 0.0


def parse_export(txt: str):
    out, cur = {}, None
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln.startswith("###"):
            cur = ln[3:].strip()
            out[cur] = []
        elif cur and len(ln.split()) == 5:
            out[cur].append(rows_of_text(ln)[0])
    return out


def index_stems(p: Path):
    return [l.split("\t")[1].rsplit(".jpg", 1)[0] for l in
            p.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]


def fmt(rows):
    return "\n".join(f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows) + \
        ("\n" if rows else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="当前批次目录（含 index.txt）")
    ap.add_argument("--ref-backup", required=True, help="参照批落库前的标签备份 tar.gz")
    ap.add_argument("--ref-export", required=True, help="参照批她的编辑器导出 txt")
    ap.add_argument("--ref-index", default="", help="参照批 index.txt（默认取 ref-export 同级）")
    ap.add_argument("--family", required=True, help="族前缀，如 8381407")
    ap.add_argument("--iou", type=float, default=0.5, help="跨帧框匹配阈值")
    ap.add_argument("--out-export", required=True)
    ap.add_argument("--apply-ws", action="store_true", help="把结果回写工作区标签（保证后续采纳幂等）")
    ap.add_argument("--sheet", default="", help="核对图版输出路径 jpg")
    args = ap.parse_args()

    batch = (ROOT / args.batch) if not Path(args.batch).is_absolute() else Path(args.batch)
    assert (batch / "index.txt").exists(), f"批次无 index.txt: {batch}"
    tag = batch.name.replace("person_review_", "")
    ws_dir = ROOT / "data" / "_prelabel" / f"person_{tag}" / "labels"
    assert ws_dir.exists(), f"工作区标签目录不存在: {ws_dir}"

    # ---- 参照帧操作（落库前 → 她导出）----
    with tarfile.open(args.ref_backup, "r:gz") as tf:
        ref_lib = {}
        for m in tf.getmembers():
            if not m.name.endswith(".txt"):
                continue
            stem = Path(m.name).stem
            if not stem.startswith(args.family):
                continue
            ref_lib[stem] = rows_of_text(tf.extractfile(m).read().decode("utf-8"))
    ref_exp = parse_export(Path(args.ref_export).read_text(encoding="utf-8"))
    ref_idx = Path(args.ref_index) if args.ref_index else Path(args.ref_export).parent / "index.txt"
    ref_family = [s for s in index_stems(ref_idx) if s.startswith(args.family)]
    ref_family = [s for s in ref_family if s in ref_exp or s in ref_lib]
    print(f"参照族 {args.family}：批内 {len(ref_family)} 帧，其中已被她编辑 "
          f"{sum(1 for s in ref_family if s in ref_exp)} 帧")

    ops_raw = []            # (class, cx, cy, w, h)
    for s in ref_family:
        lib = ref_lib.get(s)
        if lib is None or s not in ref_exp:
            continue
        her = ref_exp[s]
        for r in lib:
            if max((iou(r, h) for h in her if h[0] == r[0]), default=0.0) < args.iou:
                ops_raw.append(r)
    # 同一操作在参照族的多帧里都有副本（坐标几乎一样）→ 按类合并成一条
    ops = []
    for r in ops_raw:
        if not any(r[0] == o[0] and iou(r, o) >= args.iou for o in ops):
            ops.append(r)
    print(f"参照导出提取到删除动作 {len(ops_raw)} 条 → 合并同族副本后 {len(ops)} 类操作：")
    for r in ops:
        print(f"  - 删 {NAMES[r[0]]:9} cx{r[1]:.3f} cy{r[2]:.3f} w{r[3]:.3f} h{r[4]:.3f}")

    # ---- 套用到目标帧 ----
    ref_all_lib = [r for s in ref_family for r in ref_lib.get(s, [])]   # 参照族全部原框
    targets = [s for s in index_stems(batch / "index.txt") if s.startswith(args.family)]
    print(f"\n本批同族目标 {len(targets)} 帧")
    export_txt, applied, flagged = [], [], []
    for s in targets:
        lib = rows_of_file(LABEL_DIR / (s + ".txt"))
        ws = rows_of_file(ws_dir / (s + ".txt"))
        final = list(ws)
        removed = []
        for op in ops:
            for r in lib:
                if r[0] == op[0] and iou(r, op) >= args.iou and r not in removed:
                    final = [f for f in final if not (f[0] == r[0] and all(
                        abs(f[k] - r[k]) < 1e-6 for k in range(1, 5)))]
                    removed.append(r)
        applied.append((s, lib, ws, final, removed))
        # 目标帧里参照帧没有的"新框" → 提示她人工看（保守：不动）
        for r in lib:
            if r[0] == 0:
                continue
            if not any(r[0] == o[0] and iou(r, o) >= args.iou for o in ops) and \
               not any(r[0] == p[0] and iou(r, p) >= args.iou for p in ref_all_lib):
                flagged.append((s, r))
        n_cand = len(ws) - len(lib)
        print(f"  {s[:44]:46} 库{len(lib)} 候选{n_cand} → 删{len(removed)} 终{len(final)}"
              + (f"  删除: {[NAMES[r[0]] for r in removed]}" if removed else ""))
        export_txt.append("### " + s)
        export_txt.append(fmt(final).rstrip("\n"))

    outp = Path(args.out_export)
    if not outp.is_absolute():
        outp = ROOT / outp
    outp.write_text("\n".join(x for x in export_txt if x is not None) + "\n", encoding="utf-8", newline="\n")
    print(f"\n导出（可直接落库）→ {outp.relative_to(ROOT)}")

    if args.apply_ws:
        for s, lib, ws, final, removed in applied:
            (ws_dir / (s + ".txt")).write_text(fmt(final), encoding="utf-8", newline="\n")
        print(f"工作区标签已回写 {len(applied)} 图（后续「未改动即采纳」幂等）")

    if flagged:
        print(f"\n⚠ 目标帧里存在参照帧没有的框（未自动处理，请人工看）：{len(flagged)} 个")
        for s, r in flagged:
            print(f"  {NAMES[r[0]]:9} {s[:44]:46} cx{r[1]:.3f} cy{r[2]:.3f} w{r[3]:.3f} h{r[4]:.3f}")

    if args.sheet:
        import cv2
        import numpy as np
        IMG = ROOT / "data" / "lab_ppe" / "images" / "train"
        W, COLS = 640, 2
        panels = []
        for s, lib, ws, final, removed in applied:
            ip = next(IMG.glob(s + ".*"), None)
            if ip is None:
                continue
            im = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
            h, w = im.shape[:2]
            sc = W / w
            im = cv2.resize(im, (W, int(h * sc)), interpolation=cv2.INTER_AREA)
            for c, cx, cy, bw, bh in final:
                x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * h * sc)
                x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * h * sc)
                col = (0, 0, 255) if c == 0 else COLORS[c]
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
                cv2.putText(im, NAMES[c], (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 1, cv2.LINE_AA)
            for r in removed:                     # 被删的原框：黑框 + ✕
                x1, y1 = int((r[1] - r[3] / 2) * W), int((r[2] - r[4] / 2) * h * sc)
                x2, y2 = int((r[1] + r[3] / 2) * W), int((r[2] + r[4] / 2) * h * sc)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0), 3)
                cv2.putText(im, "DEL " + NAMES[r[0]], (x1 + 2, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            bar = np.full((24, W, 3), 255, np.uint8)
            cv2.putText(bar, f"{s.split('mp4-')[-1][:4]}  红=person 黑框DEL=已删", (4, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(np.vstack([bar, im]))
        rows_img = []
        for i in range(0, len(panels), COLS):
            grp = panels[i:i + COLS]
            hh = max(p.shape[0] for p in grp)
            canvas = np.full((hh, (W + 6) * COLS, 3), 240, np.uint8)
            for j, p in enumerate(grp):
                canvas[:p.shape[0], j * (W + 6):j * (W + 6) + p.shape[1]] = p
            rows_img.append(canvas)
        sheet = np.vstack(rows_img)
        sp = Path(args.sheet)
        if not sp.is_absolute():
            sp = ROOT / sp
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(sp))
        print(f"核对图版 → {sp.relative_to(ROOT)}  ({sheet.shape[1]}x{sheet.shape[0]})")

    tot = Counter(NAMES[r[0]] for _, _, _, _, rem in applied for r in rem)
    print(f"\n合计删除 {sum(tot.values())} 个框：{dict(tot)}；目标帧 {len(applied)} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
