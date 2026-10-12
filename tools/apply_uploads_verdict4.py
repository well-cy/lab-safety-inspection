# -*- coding: utf-8 -*-
"""
uploads 第 0 批预标注草稿 · 第四批判定落地（她判 #16 #22 #24 #29 #63 #68 #76）
补框 7 / 修框 5 → 执行后 52 图 / 181 框
用法:  python tools/apply_uploads_verdict4.py          # 干跑
       python tools/apply_uploads_verdict4.py --apply  # 执行
要点:  SET 全部基于原索引在内存一次性解析(先算后写), 与 verdict3 同构。
她判原文: "16 22 24的实验服没有框选出来29另一只手套没有框选
          63的人物没有框选出来68的实验服没有框选出来76的手套没有框选"
附加修正(同图目检发现的客观短框, 已在回报中说明): #16/#22/#24/#29 person 延伸至脚,
          #29 lab_coat 下摆延伸(0.59->0.72)。
"""
import os, sys, shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG  = os.path.join(BASE, "data", "test", "uploads")
LBL  = os.path.join(BASE, "data", "_prelabel", "uploads_20261008", "labels")
BK   = os.path.join(BASE, "data", "_label_backup", "uploads_verdict4_20261008")
APPLY = "--apply" in sys.argv

CLS = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}

# ---------- 修框（stem, 原索引, 类别, cx, cy, w, h, 说明） ----------
SETS = [
    ("img_27f757ba", 0, 0, 0.500, 0.5025, 0.940, 0.945,
     "#16 person原只到y0.646, 延伸至鞋底y0.975并微放宽(头顶y0.030)"),
    ("img_369557a4", 2, 0, 0.4775, 0.5345, 0.825, 0.931,
     "#22 person原只到y0.682, 下摆被图像底边裁切, 延伸至y1.0并放宽x"),
    ("img_3afdb65b", 1, 0, 0.344330, 0.5035, 0.494098, 0.909,
     "#24 person原只到y0.617, 延伸至鞋底y0.958(x不变)"),
    ("img_4e99f35c", 1, 3, 0.519997, 0.452245, 0.923526, 0.525511,
     "#29 lab_coat原只到y0.592, 延伸至下摆y0.715(顶/x不变)"),
    ("img_4e99f35c", 2, 0, 0.500132, 0.512150, 0.919736, 0.875701,
     "#29 person原只到y0.663, 延伸至鞋底y0.950并放宽右手侧"),
]

# ---------- 补框（stem, 类别, cx, cy, w, h, 说明） ----------
ADDS = [
    ("img_27f757ba", 3, 0.5025, 0.4625, 0.945, 0.615,
     "#16 白大褂(K.G.J)补lab_coat框 y0.155-0.770"),
    ("img_369557a4", 3, 0.4825, 0.6525, 0.825, 0.695,
     "#22 白大褂(持注射器医护)补lab_coat框 y0.305-底边"),
    ("img_3afdb65b", 3, 0.386, 0.4635, 0.372, 0.497,
     "#24 白大褂(蓝安全帽举手者)补lab_coat框 y0.215-0.712"),
    ("img_4e99f35c", 2, 0.850, 0.400, 0.210, 0.090,
     "#29 右手青色手套补框(颜色检测x[1205,1529]y[1381,1690]+目检)"),
    ("img_a441277b", 0, 0.625, 0.510, 0.750, 0.980,
     "#63 单人特写补person框(安全帽+护目镜者, 肩部延伸至底边)"),
    ("img_b33fbdc0", 3, 0.670, 0.500, 0.660, 1.000,
     "#68 白大褂躯干特写补lab_coat框(与person同区域)"),
    ("img_bca17019", 2, 0.483, 0.915, 0.220, 0.170,
     "#76 防护服者紫色手套补框(颜色检测x[153,243]y[165-底边], 含板后拇指)"),
]

def read_lines(stem):
    p = os.path.join(LBL, stem + ".txt")
    if not os.path.exists(p):
        raise SystemExit(f"[中断] 标签不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    mod_stems = {s for s, *_ in SETS} | {s for s, *_ in ADDS}
    print(f"模式: {'APPLY(执行)' if APPLY else 'DRY-RUN(干跑,不落盘)'}")
    print(f"修框 {len(SETS)} | 补框 {len(ADDS)} | 涉及标签 {len(mod_stems)} 个\n")

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

        sets  = [(idx, cls, cx, cy, w, h, why) for (s, idx, cls, cx, cy, w, h, why) in SETS if s == stem]
        adds  = [(cls, cx, cy, w, h, why) for (s, cls, cx, cy, w, h, why) in ADDS if s == stem]

        out = list(parsed)
        for idx, cls, cx, cy, w, h, why in sets:         # 改(原索引)
            assert 0 <= idx < len(out), f"{stem} SET[{idx}] 越界"
            assert out[idx][0] == cls, f"{stem} SET[{idx}] 类别{out[idx][0]}!={cls}({CLS[cls]})"
            print(f"[修] {stem}[{idx}] {CLS[cls]} ({' '.join(lines[idx].split()[1:])}) -> ({cx},{cy},{w},{h}) — {why}")
            out[idx] = (cls, cx, cy, w, h)
        for cls, cx, cy, w, h, why in adds:              # 补(尾部追加)
            print(f"[补] {stem} +{CLS[cls]} ({cx},{cy},{w},{h}) — {why}")
            out.append((cls, cx, cy, w, h))

        for o in out:
            assert 0 <= o[0] <= 4 and all(0.0 <= v <= 1.0 for v in o[1:]), f"{stem} 非法行: {o}"
        if APPLY:
            with open(os.path.join(LBL, stem + ".txt"), "w", encoding="utf-8", newline="\n") as f:
                for cls, cx, cy, w, h in out:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    if not APPLY:
        print("\n(干跑结束, 未落盘。确认无误后加 --apply 执行)")
        return

    # ---- 全库校验 + 对账 ----
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
    print(f"备份: {BK}")

if __name__ == "__main__":
    main()
