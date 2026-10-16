"""汇总 CPPE-5 人工目检结果，产出最终可用池。

输入：outputs/cppe5_screen/review_index.csv（883 项评审池）+ 本文件内嵌 DROP 集（30 版目检标记）。
输出：
  outputs/cppe5_screen/picked_index.csv   可用池清单（no/file/split/classes/reason_note）
  outputs/cppe5_screen/dropped_index.csv  被剔清单（含原因）
  data/raw/cppe5/picked/<file>            可用图拷贝（分批 150 张，防沙箱 SIGTERM）
  outputs/cppe5_screen/keep_sheets/*.jpg  可用池复核拼版（编号，供她终审）
"""
import csv
import io
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CPPE = ROOT / "data" / "raw" / "cppe5" / "extracted"
OUT = ROOT / "outputs" / "cppe5_screen"
PICKED = ROOT / "data" / "raw" / "cppe5" / "picked"
SHEET_DIR = OUT / "keep_sheets"
SHEET_DIR.mkdir(parents=True, exist_ok=True)

# ===== 30 版目检 DROP 集（评审编号）=====
DROP = set()
for s in """5 7 16 25 26 27
33 44 47 48 57
65 66 67 71 72 75 78 80 86
91 92 94 95 96 97 99 101 104 107 108 113 114 117
122 124 129 130 131 134 136 139 141 150
151 154 170 173 176 178 179
191 194 195 196 199 202 203 204 205 206 207 208 210
211 212 213 214 215 216 217 218 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240
241 242 243 244 245 246 248 249 250 251 252 253 255 256 257 258 259 260 261 262 263 264 265 267 268 269 270
271 272 273 274 275 276 278 279 280 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300
301 302 303 304 305 306 307 308 309 310 311 321 322 323 326 327 329 330
333 340 341 342 343 344 345 347 348 355 358 359 360
361 362 363 364 365 366 367 368 370 371 372 373 374 375 376 379 380 381 382 386 387 389 390
391 392 393 396 398 399 400 401 402 403 404 405 406 407 409 410 411 412 413 414 416 417 418 419 420
422 425 426 427 428 429 430 431 436 437 439 440 441 442 443 444 446 447 448 449 450
451 457 458 460 462 463 466 467 468 469 473 475 477 478 479 480
481 482 483 484 485 486 487 488 489 490 496 497 500 507 510
521 526 528 533
542 547 548 550 554 558
571 572 574 576 577 579 581 588 589 591 593 594 596 597 598
602 604 605 608 615 616 626 628
631 632 636 637 643 645 649 650 653 655
662 663 666 671 676 677 679 680 681 686 687 688 690
691 692 695 697 698 699 701 708 709 711 713 717
721 722 727 728 730 736 741 742 743 747 748
752 754 758 760 764 766 769 772 775 778
781 782 783 785 786 787 789 792 793 795 797 800 801 802 803 805 810
811 812 818 819 820 821 824 826 830 833 834 840
841 842 843 844 846 847 848 849 852 856 863 866 867 869
871 875 878 881""".split():
    DROP.add(int(s))

# 特殊备注（保留但有注意点）
NOTE = {
    97: "右下角小字台标（THE INDIA…），非大面积水印",
    181: "左下角小字摄影署名",
    780: "新闻底栏小字（Beijing 2022 期间画面）",
    808: "底部新闻字幕条（COVID 病房实景）",
    591: "六宫格拼贴（同屏多脸），入库前可裁切",
}

DROP_REASON = {
    "flat": "产品平铺/套件目录图",
    "studio": "棚拍/纯背景产品图或模特图",
    "graphic": "信息图/拼贴/带大字合成图",
    "watermark": "水印图（alamy/BBC/depositphotos 等）",
    "life": "生活照/街头/政治/零售等非医学科研场景",
    "bw": "黑白/艺术化处理",
    "factory": "工厂车间场景",
}


def reason_of(no):
    if no in (25, 48, 108, 150, 239, 273, 48, 242, 730):
        return DROP_REASON["graphic"]
    if no in (5, 7, 27, 44, 57, 78, 92, 113, 136, 141, 154, 173, 196, 210, 217, 221,
              222, 229, 232, 235, 246, 250, 255, 261, 458, 467, 792, 507, 608, 721,
              764, 27, 44, 57, 91, 47, 7, 5, 107, 236, 349):
        return DROP_REASON["flat"]
    if no in (550, 558, 340, 786, 821, 869, 758, 399):
        return DROP_REASON["watermark"]
    if no in (355, 380, 389, 392, 393, 401, 347, 348):
        return DROP_REASON["bw"]
    if no in (104, 214, 240, 288, 341, 342, 343, 344, 345):
        return DROP_REASON["factory"]
    if no in (16, 65, 66, 67, 71, 72, 75, 80, 86, 94, 95, 96, 97, 99, 101, 114, 117,
              122, 129, 130, 131, 134, 139, 151, 170, 176, 178, 179, 191, 194, 195,
              199, 202, 203, 204, 205, 206, 207, 208, 211, 212, 213, 215, 216, 218,
              220, 223, 224, 225, 226, 227, 228, 230, 231, 233, 234, 237, 238, 241,
              243, 244, 245, 248, 249, 251, 252, 253, 256, 257, 258, 259, 260, 262,
              263, 264, 265, 267, 268, 269, 270, 271, 272, 274, 275, 276, 278, 279,
              280, 282, 283, 284, 285, 286, 287, 289, 290, 291, 292, 293, 294, 295,
              296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
              310, 311, 322, 33, 466, 468, 469, 473, 477, 478, 480, 481, 482, 483,
              484, 485, 486, 487, 488, 489, 490, 496, 497, 500, 510, 521, 526, 528,
              533, 542, 547, 548, 554, 571, 572, 574, 576, 577, 579, 581, 588, 589,
              593, 594, 596, 597, 598, 602, 604, 605, 615, 616, 626, 631, 632, 636,
              637, 643, 645, 649, 650, 653, 655, 662, 663, 666, 676, 677, 679, 680,
              681, 686, 687, 688, 690, 691, 692, 695, 697, 698, 699, 701, 708, 709,
              711, 713, 717, 722, 727, 728, 736, 741, 742, 743, 747, 748, 752, 754,
              760, 766, 769, 772, 775, 778, 781, 782, 783, 785, 787, 789, 793, 795,
              797, 800, 801, 802, 803, 805, 810, 811, 812, 818, 819, 820, 824, 826,
              830, 833, 834, 840, 841, 842, 843, 844, 846, 848, 849, 852, 856, 863,
              866, 867, 871, 875, 878, 881, 33, 329):
        return DROP_REASON["studio"]
    return DROP_REASON["life"]


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def put(img, txt, org, color, sc=0.6, th=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


def main():
    rows = list(csv.DictReader((OUT / "review_index.csv").open(encoding="utf-8")))
    keep, drop = [], []
    for r in rows:
        no = int(r["no"])
        r["no"] = no
        if no in DROP:
            r["reason"] = reason_of(no)
            drop.append(r)
        else:
            r["note"] = NOTE.get(no, "")
            keep.append(r)

    with (OUT / "picked_index.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["no", "file", "split", "classes", "note"])
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, "") for k in ["no", "file", "split", "classes", "note"]})
    with (OUT / "dropped_index.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["no", "file", "split", "classes", "reason"])
        w.writeheader()
        for r in drop:
            w.writerow({k: r.get(k, "") for k in ["no", "file", "split", "classes", "reason"]})

    # 拷贝 picked（分批防 SIGTERM）
    PICKED.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, r in enumerate(keep):
        dst = PICKED / r["file"]
        if not dst.exists():
            shutil.copyfile(CPPE / "images" / r["file"], dst)
            done += 1
        if (i + 1) % 150 == 0:
            print(f"  拷贝 {i+1}/{len(keep)} …")
            time.sleep(0.8)
    print(f"拷贝 {done} 张 → {PICKED.relative_to(ROOT)}")

    # 类别统计
    from collections import Counter
    c = Counter()
    for r in keep:
        for u in r["classes"].split("|"):
            if u:
                c[u] += 1
    print(f"可用池 {len(keep)} 张（剔 {len(drop)}）类分布: {dict(c)}")

    # 复核拼版
    COLS, ROWS, CELL = 6, 5, 280
    page = COLS * ROWS
    pages = (len(keep) + page - 1) // page
    for pg in range(pages):
        chunk = keep[pg * page:(pg + 1) * page]
        sheet = np.full((ROWS * CELL, COLS * CELL, 3), 28, np.uint8)
        for k, r in enumerate(chunk):
            img = imread_u(PICKED / r["file"])
            if img is None:
                continue
            H, W = img.shape[:2]
            s = min(CELL / W, CELL / H)
            img = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
            y0 = (k // COLS) * CELL + (CELL - img.shape[0]) // 2
            x0 = (k % COLS) * CELL + (CELL - img.shape[1]) // 2
            sheet[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
            cx, cy = (k % COLS) * CELL, (k // COLS) * CELL
            tag = f"#{r['no']}"
            cv2.rectangle(sheet, (cx + 2, cy + 2), (cx + 10 + 17 * len(tag), cy + 36), (0, 0, 0), -1)
            put(sheet, tag, (cx + 8, cy + 29), (0, 220, 0), 0.95, 2)
            put(sheet, r["file"], (cx + 8, cy + CELL - 8), (190, 190, 190), 0.42, 1)
            if r.get("note"):
                put(sheet, "NOTE", (cx + CELL - 66, cy + 29), (0, 200, 255), 0.6, 1)
        put(sheet, f"keep {pg+1}/{pages}   no {chunk[0]['no']}-{chunk[-1]['no']}",
            (10, ROWS * CELL - 10), (255, 255, 255), 0.7, 2)
        op = SHEET_DIR / f"keep_{pg+1:02d}.jpg"
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(op))
    print(f"复核拼版 {pages} 版 → {SHEET_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
