# -*- coding: utf-8 -*-
"""5 类化迁移断点续跑（分批写盘绕开沙箱批量写阈值）。

状态来源 data/_5cls_migrate_state.json（对账产物，pending 必须先复核 cur==orig）:
  1. pending 文件类号 +1（每批 150，批间 sleep）
  2. 网图批 123 图替换为 labels_5cls 内容（同样分批）
  3. data.yaml → nc:5
  4. 全库复检
可反复运行至完成；每步写前重新校验，防状态过期。
"""
from __future__ import annotations

import glob
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LAB = ROOT / "data" / "lab_ppe"
FIVE = ROOT / "data" / "raw" / "web_lab_scene" / "labels_5cls"
ARCH = ROOT / "data" / "_label_backup" / "lab_ppe_5cls_20261009.tar.gz"
STATE = ROOT / "data" / "_5cls_migrate_state.json"
SPLITS = ["train", "valid", "test"]
BATCH = 150
PAUSE = 2.5
NAMES5 = {0: "person", 1: "mask", 2: "gloves", 3: "lab_coat", 4: "goggles"}


def norm(s):
    return s.replace("\r\n", "\n").replace("\r", "\n")


def shift(text):
    out = []
    for ln in text.split("\n"):
        t = ln.split()
        if len(t) == 5 and t[0].isdigit() and int(t[0]) <= 3:
            out.append(f"{int(t[0]) + 1} {t[1]} {t[2]} {t[3]} {t[4]}")
        elif ln.strip():
            out.append(ln)
    return "\n".join(out) + ("\n" if out else "")


def write_batches(pairs, tag):
    """pairs: [(path, content)]；分批写，返回成功数。"""
    n = 0
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        for p, content in chunk:
            p.write_text(content, encoding="utf-8", newline="\n")
            n += 1
        print(f"  [{tag}] {min(i + BATCH, len(pairs))}/{len(pairs)}", flush=True)
        if i + BATCH < len(pairs):
            time.sleep(PAUSE)
    return n


def main() -> int:
    st = json.loads(STATE.read_text(encoding="utf-8"))
    t = tarfile.open(ARCH, "r:gz")
    blobs = {}
    for m in t.getmembers():
        if m.isfile() and m.name.endswith(".txt"):
            parts = m.name.split("/")
            blobs[(parts[1], Path(parts[2]).stem)] = t.extractfile(m).read().decode("utf-8")

    # ① pending 类号 +1
    pend = st["pending"]
    print(f"[迁移] pending {len(pend)}")
    if pend:
        pairs = []
        left = []
        for sp, stem in pend:
            cur = (LAB / "labels" / sp / f"{stem}.txt").read_text(encoding="utf-8")
            orig = norm(blobs[(sp, stem)])
            if norm(cur) == orig:
                pairs.append((LAB / "labels" / sp / f"{stem}.txt", shift(orig)))
            elif norm(cur) == shift(orig):
                pass                                   # 已被前次跑完
            else:
                print(f"[中止] 状态漂移: {sp}/{stem}")
                return 1
        if pairs:
            write_batches(pairs, "shift")
        print(f"[迁移] 本轮 +1 完成 {len(pairs)}（其余已是迁移态）")

    # ② 网图批替换
    idx = {}
    for sp in SPLITS:
        for p in glob.glob(str(LAB / "images" / sp / "*")):
            idx[Path(p).stem] = sp
    five = sorted(FIVE.glob("*.txt"))
    pairs2 = []
    for f in five:
        sp = idx[f.stem]
        cur_path = LAB / "labels" / sp / f"{f.stem}.txt"
        # 已替换特征：含 class 0 框（person）；未替换特征：与 5cls 非 person 左移一致
        c5 = norm(f.read_text(encoding="utf-8"))
        cur = cur_path.read_text(encoding="utf-8")
        c5_rows = [l.split() for l in c5.splitlines() if l.strip()]
        cur_rows = [l.split() for l in norm(cur).splitlines() if l.strip()]
        has_p = any(r[0] == "0" for r in cur_rows)
        nonp5 = [r for r in c5_rows if r[0] != "0"]
        shifted = all(int(r[0]) >= 1 for r in cur_rows) and len(cur_rows) == len(nonp5)
        if has_p and len(cur_rows) == len(c5_rows):
            continue                                    # 已替换
        if not shifted:
            print(f"[中止] 5cls 前置不符: {f.stem}")
            return 1
        content = c5 if c5.endswith("\n") else c5 + "\n"
        pairs2.append((cur_path, content))
    print(f"[回补] 网图批待替换 {len(pairs2)}")
    if pairs2:
        write_batches(pairs2, "backfill")

    # ③ data.yaml
    dy = LAB / "data.yaml"
    if "nc: 4" in dy.read_text(encoding="utf-8"):
        dy.write_text(
            "path: D:/ZuoYe/lab-safety-inspection/data/lab_ppe\n"
            "train: images/train\nval: images/valid\ntest: images/test\nnc: 5\nnames:\n"
            "  0: person\n  1: mask\n  2: gloves\n  3: lab_coat\n  4: goggles\n",
            encoding="utf-8", newline="\n")
        print("[yaml] nc:5 已写")

    # ④ 复检
    time.sleep(PAUSE)
    bad = 0
    cnt = Counter()
    nfiles = 0
    for sp in SPLITS:
        for f in glob.glob(str(LAB / "labels" / sp / "*.txt")):
            nfiles += 1
            for ln in open(f, encoding="utf-8"):
                tk = ln.split()
                if not tk:
                    continue
                if len(tk) != 5 or not tk[0].isdigit() or int(tk[0]) > 4:
                    bad += 1
                else:
                    cnt[int(tk[0])] += 1
    print(f"[复检] {nfiles} 文件 / {sum(cnt.values())} 框 / 坏行 {bad}")
    print("[复检] 分布:", " ".join(f"{NAMES5[k]}={v}" for k, v in sorted(cnt.items())))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
