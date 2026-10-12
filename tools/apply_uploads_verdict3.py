# -*- coding: utf-8 -*-
"""
uploads 第 0 批预标注草稿 · 第三批判定落地（她判 #9 #10 #11 #17 #23 #26 #28 #31 #52 #59 #68 #76 #92）
剔图 6 / 删框 4 / 修框 4 / 补框 5 → 执行后 52 图 / 174 框
用法:  python tools/apply_uploads_verdict3.py          # 干跑
       python tools/apply_uploads_verdict3.py --apply  # 执行
要点:  DEL 与 SET 同文件时全部基于原索引在内存一次性解析(先算后写), 杜绝"先删后改索引前移"。
"""
import os, sys, shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG  = os.path.join(BASE, "data", "test", "uploads")
LBL  = os.path.join(BASE, "data", "_prelabel", "uploads_20261008", "labels")
BK   = os.path.join(BASE, "data", "_label_backup", "uploads_verdict3_20261008")
REJ  = os.path.join(BASE, "data", "_rejected", "uploads_reject_20261008")
APPLY = "--apply" in sys.argv

CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}

# ---------- 剔图（图片→REJ/images, 草稿标签→REJ/labels） ----------
REJECTS = [
    ("img_144e2bff.jpg", "#11 与#43 md5同图(hijab实验室)且无person框, #43已补全保留"),
    ("img_392f916c.jpg", "#23 与#24同姿势连拍且滤镜重画质差, 保留干净的#24"),
    ("img_47e8a531.jpg", "#26 电视综艺截图(SONY台标印剧布景), 非实验室场景"),
    ("img_5794be96.jpg", "#31 与#17同图且lab_coat错位, 保留标注正确的#17"),
    ("img_8a65437d.jpg", "#52 与#30 md5同图(安全帽+防毒面具+白大褂), 完全重复"),
    ("img_fbc5be5d.jpg", "#92 与#59 md5同图(红2招牌滤镜图), 完全重复"),
]

# ---------- 删框（stem, 原索引, 类别, 说明） ----------
DELS = [
    ("img_2905d870", 2, 0, "#17 内层重复person框(只罩躯干), 保留外层[1]全身框"),
    ("img_4b608c01", 3, 0, "#28 第二个重复person框, 保留[2]并扩为全身框"),
    ("img_b33fbdc0", 0, 2, "#68 误检大gloves框(罩胸口, 图内无手套)"),
    ("img_b33fbdc0", 1, 0, "#68 重复person框, 保留[2]并扩为可见全身框"),
]

# ---------- 修框（stem, 原索引, 类别, cx, cy, w, h, 说明） ----------
SETS = [
    ("img_0a4aeefa", 2, 1, 0.613, 0.762, 0.087, 0.094, "#9 病人口罩框原偏下(框在下巴), 上移对准青色口罩"),
    ("img_104e87bc", 5, 0, 0.477, 0.486, 0.343, 0.655, "#10 person[5]原一框罩中间两人, 缩为蓝手术衣者单人框"),
    ("img_4b608c01", 2, 0, 0.490, 0.539, 0.639, 0.866, "#28 保留person框扩为全身(含前伸手臂)"),
    ("img_b33fbdc0", 2, 0, 0.670, 0.500, 0.660, 1.000, "#68 保留person框扩为可见躯干全高(右贴边)"),
]

# ---------- 补框（stem, 类别, cx, cy, w, h, 说明） ----------
ADDS = [
    ("img_0a4aeefa", 0, 0.699, 0.833, 0.278, 0.334, "#9 床上病人(裸上身戴口罩)补person框"),
    ("img_104e87bc", 0, 0.481, 0.641, 0.241, 0.303, "#10 中间俯身白头套医护补person框"),
    ("img_104e87bc", 1, 0.356, 0.507, 0.058, 0.090, "#10 最左医护(侧脸)青色口罩补框"),
    ("img_97d0fec3", 2, 0.616, 0.574, 0.075, 0.060, "#59 画面右侧垂手的青色手套补框"),
    ("img_bca17019", 1, 0.292, 0.520, 0.047, 0.131, "#76 左侧病人(蓝点病号服)白口罩补框"),
]

def read_lines(stem):
    p = os.path.join(LBL, stem + ".txt")
    if not os.path.exists(p):
        raise SystemExit(f"[中断] 标签不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    rej_stems = {fn.rsplit(".", 1)[0] for fn, _ in REJECTS}
    mod_stems = ({s for s, *_ in DELS} | {s for s, *_ in SETS} | {s for s, *_ in ADDS})
    overlap = rej_stems & mod_stems
    assert not overlap, f"剔图与改框冲突: {overlap}"

    print(f"模式: {'APPLY(执行)' if APPLY else 'DRY-RUN(干跑,不落盘)'}")
    print(f"剔图 {len(REJECTS)} | 删框 {len(DELS)} | 修框 {len(SETS)} | 补框 {len(ADDS)} | 涉及标签 {len(rej_stems | mod_stems)} 个\n")

    # ---- 1) 剔图 ----
    for fn, why in REJECTS:
        src = os.path.join(IMG, fn)
        stem = fn.rsplit(".", 1)[0]
        lab = os.path.join(LBL, stem + ".txt")
        ok_img = os.path.exists(src); ok_lab = os.path.exists(lab)
        print(f"[剔] {fn}: {why}")
        print(f"     img={'√' if ok_img else '×缺失!'} label={'√' if ok_lab else '×缺失!'}")
        if APPLY and ok_img:
            os.replace(src, os.path.join(REJ, "images", fn))
        if APPLY and ok_lab:
            os.replace(lab, os.path.join(REJ, "labels", stem + ".txt"))

    # ---- 2) 改框：每个 stem 读入内存, 基于原索引先解析后写回(单次重写) ----
    stems = sorted(mod_stems)
    for stem in stems:
        lines = read_lines(stem)
        parsed = []
        for i, l in enumerate(lines):
            p = l.split()
            assert len(p) == 5, f"{stem}[{i}] 字段数!=5: {l}"
            parsed.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))

        backup_path = os.path.join(BK, stem + ".txt")
        if APPLY:
            os.makedirs(BK, exist_ok=True)
            shutil.copy2(os.path.join(LBL, stem + ".txt"), backup_path)

        dels  = [(idx, cls, why) for (s, idx, cls, why) in DELS if s == stem]
        sets  = [(idx, cls, cx, cy, w, h, why) for (s, idx, cls, cx, cy, w, h, why) in SETS if s == stem]
        adds  = [(cls, cx, cy, w, h, why) for (s, cls, cx, cy, w, h, why) in ADDS if s == stem]

        out = list(parsed)
        for idx, cls, why in dels:                       # 删(原索引)
            assert 0 <= idx < len(out), f"{stem} DEL[{idx}] 越界(共{len(out)})"
            assert out[idx][0] == cls, f"{stem} DEL[{idx}] 类别{out[idx][0]}!={cls}({CLS[cls]})"
            print(f"[删] {stem}[{idx}] {CLS[cls]} ({' '.join(lines[idx].split()[1:])}) — {why}")
            out[idx] = None
        for idx, cls, cx, cy, w, h, why in sets:         # 改(原索引, 不受删影响)
            assert 0 <= idx < len(out), f"{stem} SET[{idx}] 越界"
            assert out[idx] is not None and out[idx][0] == cls, f"{stem} SET[{idx}] 类别不符或已被删"
            print(f"[修] {stem}[{idx}] {CLS[cls]} -> ({cx},{cy},{w},{h}) — {why}")
            out[idx] = (cls, cx, cy, w, h)
        for cls, cx, cy, w, h, why in adds:              # 补(尾部追加)
            print(f"[补] {stem} +{CLS[cls]} ({cx},{cy},{w},{h}) — {why}")
            out.append((cls, cx, cy, w, h))

        out = [o for o in out if o is not None]
        for o in out:
            assert 0 <= o[0] <= 4 and all(0.0 <= v <= 1.0 for v in o[1:]), f"{stem} 非法行: {o}"
        if APPLY:
            with open(os.path.join(LBL, stem + ".txt"), "w", encoding="utf-8", newline="\n") as f:
                for cls, cx, cy, w, h in out:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    if not APPLY:
        print("\n(干跑结束, 未落盘。确认无误后加 --apply 执行)")
        return

    # ---- 3) 全库校验 + 对账 ----
    n_img = len([f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    labs  = [f for f in os.listdir(LBL) if f.endswith(".txt")]
    tally = {k: 0 for k in CLS}
    nbox = 0
    for lf in labs:
        stem = lf[:-4]
        assert os.path.exists(os.path.join(IMG, stem + ".jpg")) or \
               os.path.exists(os.path.join(IMG, stem + ".jpeg")) or \
               os.path.exists(os.path.join(IMG, stem + ".png")), f"标签无图: {stem}"
        with open(os.path.join(LBL, lf), "r", encoding="utf-8") as f:
            rows = [l.split() for l in f if l.strip()]
        for p in rows:
            assert len(p) == 5 and 0 <= int(p[0]) <= 4, f"{lf} 坏行: {p}"
            assert all(0.0 <= float(v) <= 1.0 for v in p[1:]), f"{lf} 坐标越界: {p}"
            tally[int(p[0])] += 1; nbox += 1
    for f in os.listdir(IMG):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            stem = f.rsplit(".", 1)[0]
            assert os.path.exists(os.path.join(LBL, stem + ".txt")), f"图无标签: {f}"
    print(f"\n执行后: {n_img} 图 / {nbox} 框 / 标签 {len(labs)} 个, 0 坏行, 图签配对无缺失")
    print("按类别: " + " ".join(f"{CLS[k]}={tally[k]}" for k in sorted(CLS)))
    print(f"备份: {BK}\n剔除: {REJ}")

if __name__ == "__main__":
    main()
