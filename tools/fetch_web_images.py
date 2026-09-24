# -*- coding: utf-8 -*-
"""
从 Wikimedia Commons 批量抓取「真实实验室场景」候选图，用于补 lab_coat 的域覆盖（E4 §A3）。

定位：**只做采集，不做筛选**。
    - 图片级去重/泄漏  -> tools/dedup_new_dataset.py
    - 棚拍/白底/排版图 -> tools/screen_dirty_graphics.py --dir
    - 场景与类别终筛   -> tools/make_contact_sheet.py 人工拼版
    - 标注一致性       -> 人工标注（tools/labeler.html）

为什么是 Commons 而不是图库：dataset1 已有的 178 张 istockphoto 是 612x612 缩略图、
单人近景、打光充足，和它同类的再抓多少都治不了 A3 的域差。Commons 上是科研机构/
高校的实拍原图（常见 2000-6000px），真实实验室、多人、中远距离、自然顶光——正是
dataset2/t01-t07 那一侧的特征。

许可：仅收录 CC / PD 等自由许可条目，license 与作者写入 manifest.csv 备查。
      （课程项目内部使用，口径同 docs/E4_lab_coat_plan.md 风险3；调用方自行决定是否使用。）

用法：
    python tools/fetch_web_images.py --dry-run                 # 只看命中量，不下载
    python tools/fetch_web_images.py --limit 1200              # 抓取
    python tools/fetch_web_images.py --only cat               # 只跑 category 那批
    python tools/fetch_web_images.py --only search --page 3     # 只跑 search 第 3 页
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "web_lab" / "images"
OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT_DIR.parent / "manifest.csv"

API = "https://commons.wikimedia.org/w/api.php"
UA = "lab-ppe-dataset-research/1.0 (course project; contact: local)"
_tls = threading.local()

MIN_W = 800
MIN_H = 600
PAGE_SIZE = 50

# ---------- 检索式 ----------
# 原则：必须出现「人」（scientist/researcher/technician/student/worker），
#       且必须出现「实验室语义」（laboratory/lab/cleanroom/fume hood...）。
SEARCH_QUERIES = [
    "laboratory scientist working",
    "researcher laboratory lab coat",
    "laboratory technician working",
    "chemist laboratory experiment person",
    "students laboratory practical class",
    "laboratory microscope researcher",
    "cell culture laboratory worker",
    "pipetting laboratory person",
    "biological laboratory work person",
    "medical laboratory technician",
    "laboratory safety goggles person",
    "cleanroom laboratory worker",
    "fume hood laboratory worker",
    "research institute laboratory team",
    "analytical chemistry laboratory person",
    "laboratory worker protective equipment",
    "scientist pipette laboratory bench",
    "laboratory staff working bench",
    # ---- 2026-09-24 扩容：更多机构/场景语义，提高 lab_coat 域覆盖 ----
    "university laboratory students experiment",
    "teaching laboratory students coats",
    "graduate student laboratory bench",
    "postdoc laboratory research",
    "hospital laboratory technician samples",
    "clinical laboratory worker analysis",
    "pathology laboratory technician",
    "microbiology laboratory worker",
    "chemistry student laboratory experiment",
    "biology laboratory students microscope",
    "laboratory internship student",
    "lab coat scientist portrait laboratory",
    "laboratory team group photo coats",
    "vaccine laboratory researcher",
    "quality control laboratory technician",
    "food science laboratory worker",
    "environmental laboratory technician sampling",
    "forensic laboratory technician",
    "laboratory safety training workers",
    "laboratory assistant preparing samples",
    "laboratory staff centrifuge",
    "laboratory worker gloves samples bench",
    "research laboratory two people working",
    "laboratory technician notebook bench",
]

# ---------- 分类目录（比全文检索精度更高，但覆盖面窄）----------
CATEGORIES = [
    "Category:Laboratory technicians",
    "Category:Laboratory work",
    "Category:People in laboratories",
    "Category:Biological laboratories",
    "Category:Chemical laboratories",
    "Category:Medical laboratories",
    "Category:Cleanrooms",
    "Category:Microscopy laboratories",
    # ---- 2026-09-24 扩容 ----
    "Category:Laboratories",
    "Category:Laboratory equipment in use",
    "Category:Scientists in laboratories",
    "Category:Researchers",
    "Category:Physics laboratories",
    "Category:Biochemistry laboratories",
    "Category:Analytical laboratories",
    "Category:Teaching laboratories",
    "Category:University laboratories",
    "Category:Hospital laboratories",
    "Category:People working in laboratories",
    "Category:Laboratory glassware in use",
    "Category:Cell culture",
    "Category:Microbiology laboratories",
    "Category:Veterinary laboratories",
]

# ---------- 标题黑名单：明显不是「真实实验室里有人」的照片 ----------
BAD_TITLE = re.compile(
    r"(svg|logo|icon|diagram|schema|schematic|drawing|sketch|chart|graph|plot|map|"
    r"poster|logo|seal|coat of arms|stamp|banknote|screenshot|"
    r"glassware|beaker|flask|test tube|microscope\b(?!.*(person|worker|scientist))|"
    r"building|exterior|floor plan|apparatus|equipment only|museum|portrait|"
    r"painting|engraving|lithograph|stamp|coin)",
    re.IGNORECASE,
)

IMG_MIME = {"image/jpeg", "image/png"}

# 下载用 Commons 服务端缩略图的宽度。训练 imgsz=640，标注 1920 足够看清手套/口罩；
# 直接拉原图（8256px 一张 20MB+）纯属浪费磁盘和时间。
THUMB_PX = 1920

SESS = requests.Session()
SESS.headers.update({"User-Agent": UA})


def _session() -> requests.Session:
    """每个线程一个 Session（requests.Session 非线程安全）。"""
    global SESS
    if not hasattr(_tls, "sess"):
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _tls.sess = s
    return _tls.sess

# Commons 对匿名调用限流较紧（连续请求会 429），统一节流 + 指数退避
MIN_API_GAP = 2.5          # 相邻 API 调用最小间隔（秒）
_last_api = [0.0]


def _throttle() -> None:
    gap = time.time() - _last_api[0]
    if gap < MIN_API_GAP:
        time.sleep(MIN_API_GAP - gap)
    _last_api[0] = time.time()


def api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    waits = [4, 12, 30, 45]
    for attempt in range(len(waits) + 1):
        _throttle()
        try:
            r = SESS.get(API, params=params, timeout=45)
            if r.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            is429 = "429" in str(e)
            if attempt == len(waits):
                print(f"  [API 失败] {e}", file=sys.stderr)
                return {}
            w = waits[attempt]
            if is429:
                print(f"  [限流 429] 等待 {w}s 后重试 ...", file=sys.stderr)
            time.sleep(w)
    return {}


def norm_page(p: dict, origin: str) -> dict | None:
    """把一条 Commons 结果规整成统一结构；不合格返回 None。"""
    title = p.get("title", "")
    ii = (p.get("imageinfo") or [{}])[0]
    mime = ii.get("mime", "")
    w, h = ii.get("width", 0), ii.get("height", 0)
    # 优先服务端缩略图；/thumb 失败时退回原图
    url = ii.get("thumburl") or ii.get("url", "")

    if mime not in IMG_MIME or not url or w < MIN_W or h < MIN_H:
        return None
    if BAD_TITLE.search(title):
        return None
    ratio = w / h if h else 0
    if not (0.4 <= ratio <= 2.8):  # 极端长条多为拼接/文档
        return None

    em = ii.get("extmetadata") or {}

    def e(key: str) -> str:
        v = em.get(key, {}).get("value", "")
        v = re.sub(r"<[^>]+>", "", str(v))  # 去 HTML
        return re.sub(r"\s+", " ", v).strip()

    return dict(
        title=title,
        pageid=p.get("pageid", ""),
        url=url,
        mime=mime,
        width=w,
        height=h,
        license=e("LicenseShortName") or "UNKNOWN",
        author=e("Artist") or "",
        credit=e("Credit") or "",
        desc_url=ii.get("descriptionurl", ""),
        orig_url=ii.get("url", ""),
        origin=origin,
    )


def harvest_search(query: str, page: int, cap: int) -> list[dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": PAGE_SIZE,
        "gsroffset": (page - 1) * PAGE_SIZE,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": THUMB_PX,
    }
    d = api(params)
    pages = (d.get("query") or {}).get("pages") or []
    out = []
    for p in pages:
        n = norm_page(p, f"search:{query}")
        if n:
            out.append(n)
        if len(out) >= cap:
            break
    return out


def harvest_category(cat: str, cap: int) -> list[dict]:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": cat,
        "gcmtype": "file",
        "gcmlimit": PAGE_SIZE,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": THUMB_PX,
    }
    d = api(params)
    pages = (d.get("query") or {}).get("pages") or []
    out = []
    for p in pages:
        n = norm_page(p, f"cat:{cat}")
        if n:
            out.append(n)
        if len(out) >= cap:
            break
    return out


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return {row["title"]: row for row in csv.DictReader(f)}


def append_manifest(rows: list[dict]) -> None:
    new = not MANIFEST.exists()
    fields = ["name", "title", "pageid", "width", "height", "mime",
              "license", "author", "credit", "url", "orig_url", "desc_url", "origin"]
    with MANIFEST.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def download(item: dict, idx: int, out_dir: Path) -> dict | None:
    ext = ".jpg" if item["mime"] == "image/jpeg" else ".png"
    # 文件名用 Commons pageid（稳定唯一），不用清单位置——
    # 位置编号会随候选清单顺序变化而串号（2026-09-24 踩过：换清单后
    # dst.exists() 把新候选误判为已下载，整批跳过并写入错误元数据）
    name = f"web_p{item['pageid']}{ext}"
    dst = out_dir / name
    if dst.exists() and dst.stat().st_size > 4096:
        return {**item, "name": name}
    last_err = None
    for attempt in range(3):
        try:
            with _session().get(item["url"], stream=True, timeout=90) as r:
                if r.status_code == 429:
                    raise requests.HTTPError("429", response=r)
                r.raise_for_status()
                tmp = dst.with_suffix(dst.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                tmp.replace(dst)  # 原子落盘，不回退到 unlink 路径
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    if last_err is not None:
        print(f"  [下载失败] {item['title'][:60]}: {last_err}", file=sys.stderr)
        return None
    if dst.stat().st_size < 4096:
        print(f"  [文件过小] {name}", file=sys.stderr)
        return None
    size_kb = dst.stat().st_size // 1024
    print(f"  [{idx:04d}] {item['width']}x{item['height']:<5} {size_kb:>5}KB "
          f"{item['license']:<16} {item['title'][:58]}")
    return {**item, "name": name}


def main() -> int:
    ap = argparse.ArgumentParser(description="Commons 真实实验室场景候选图采集")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=2000, help="本次最多下载张数")
    ap.add_argument("--pages", type=int, default=1, help="每组检索式翻几页")
    ap.add_argument("--only", choices=["all", "search", "cat"], default="all")
    ap.add_argument("--page", type=int, default=0, help="只跑检索式第 N 页（配合 --only search）")
    ap.add_argument("--sleep", type=float, default=0.5, help="下载间隔秒（workers=1 时生效）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发下载线程数（>1 走线程池；Commons 缩略图 CDN 对并发较宽容，"
                         "单线程实测 ~9s/张过慢，建议 6）")
    ap.add_argument("--collect-only", action="store_true",
                    help="只枚举并写出 candidates.json，不下载（枚举受 Commons API 限流，慢）")
    ap.add_argument("--from-json", default="",
                    help="跳过枚举，直接读取已有 candidates.json 进入下载阶段（推荐）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cands_json = out_dir.parent / "candidates.json"

    have = load_manifest()
    print(f"已有 manifest 条目: {len(have)}")

    # ---------- 1. 枚举候选 ----------
    if args.from_json:
        cands = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        print(f"[载入] {args.from_json} -> {len(cands)} 条候选")
    else:
        cands: list[dict] = []
        seen: set[str] = set(have)

        if args.only in ("all", "search"):
            pages = [args.page] if args.page else list(range(1, args.pages + 1))
            for q in SEARCH_QUERIES:
                got = 0
                for pg in pages:
                    for n in harvest_search(q, pg, PAGE_SIZE):
                        if n["title"] in seen:
                            continue
                        seen.add(n["title"])
                        cands.append(n)
                        got += 1
                print(f"[search] {q:<45} +{got}", flush=True)
        if args.only in ("all", "cat"):
            for cat in CATEGORIES:
                got = 0
                for n in harvest_category(cat, PAGE_SIZE * 2):
                    if n["title"] in seen:
                        continue
                    seen.add(n["title"])
                    cands.append(n)
                    got += 1
                print(f"[cat]    {cat:<45} +{got}", flush=True)

        print(f"\n去重后候选: {len(cands)}")
        cands.sort(key=lambda x: -(x["width"] * x["height"]))  # 分辨率降序
        cands_json.write_text(json.dumps(cands, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"[落盘] {cands_json}（枚举结果已保存，后续可 --from-json 续跑）")

    if args.collect_only:
        for n in cands[:30]:
            print(f"  {n['width']}x{n['height']:<6} {n['license']:<18} {n['title'][:64]}")
        print(f"  ... 共 {len(cands)} 张（--collect-only 未下载）")
        return 0

    # 跳过已抓取项：按 Commons pageid 判定（与文件名/清单顺序解耦）
    have_ids = {str(r.get("pageid", "")).strip()
                for r in have.values() if str(r.get("pageid", "")).strip()}
    n_before = len(cands)
    cands = [c for c in cands if str(c.get("pageid", "")).strip() not in have_ids]
    print(f"[去重] 跳过已抓取 {n_before - len(cands)} 张，待抓 {len(cands)} 张")

    cands = cands[: args.limit]
    print(f"本次下载配额: {len(cands)}")

    # ---------- 2. 下载 ----------
    # 文件名由 pageid 决定（唯一稳定），与清单位置无关 -> 并发/断点续跑都安全
    done, failed = [], 0
    batch: list[dict] = []

    def flush(force: bool = False) -> None:
        nonlocal batch
        if len(batch) >= 25 or (force and batch):
            append_manifest(batch)
            batch = []

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"并发下载 workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download, item, i, out_dir): item
                    for i, item in enumerate(cands, start=1)}
            for n, fut in enumerate(as_completed(futs), start=1):
                rec = fut.result()
                if rec is None:
                    failed += 1
                else:
                    done.append(rec)
                    batch.append(rec)
                    flush()
                if n % 50 == 0:
                    print(f"  ... 进度 {n}/{len(cands)}  成功 {len(done)} 失败 {failed}", flush=True)
    else:
        for i, item in enumerate(cands, start=1):
            rec = download(item, i, out_dir)
            if rec is None:
                failed += 1
                continue
            done.append(rec)
            batch.append(rec)
            flush()
            time.sleep(args.sleep)
    flush(force=True)

    print("\n" + "=" * 60)
    print(f"下载成功 {len(done)} / 失败 {failed}")
    print(f"图片目录: {out_dir}")
    print(f"清单文件: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
