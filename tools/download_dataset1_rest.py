#!/usr/bin/env python3
"""download_dataset1_rest.py — REST 直下 dataset1（断点续传 + 自动重试版）

背景：roboflow SDK 会在目标目录已存在时静默跳过下载（version.py L394-396），
且直连下载在部分网络下会频繁卡死。本脚本：
  1. 走 REST API 拿导出链接（format 放路径：/{ws}/{proj}/{ver}/yolov8）
  2. 流式下载，每 20MB 打印一次进度和速度
  3. 读超时 45 秒即判定卡死，自动断点续传重试（最多 50 次），无需人工值守
  4. 下载完校验解压后三 split 总数 ≥7000 张才算成功

用法:
    python tools/download_dataset1_rest.py <ROBOFLOW_API_KEY>

中断后重新运行同一命令即可：已下载的部分会自动续传，不会从头开始。
"""
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DEST = RAW / "dataset1"
WORKSPACE, PROJECT, VERSION = "vitor-ferraz-marini", "lab_ppe-g8hja", "1"

READ_TIMEOUT = 45      # 读超时（秒）：超过即判卡死，自动续传
MAX_RETRY = 50         # 最大自动重试次数
PROGRESS_STEP = 20 << 20  # 每 20MB 打印一次


def get_export_link(api_key: str, ws: str, proj: str, ver: str) -> str:
    """拿导出链接；HTTP 202 表示 Roboflow 正在打包，轮询等待。"""
    meta_url = f"https://api.roboflow.com/{ws}/{proj}/{ver}/yolov8"
    for i in range(60):  # 最多等 5 分钟
        r = requests.get(meta_url, params={"api_key": api_key, "nocache": "true"},
                         timeout=60)
        r.raise_for_status()
        d = r.json()
        link = d.get("export", {}).get("link")
        if link:
            return link
        print(f"  [等待] Roboflow 正在生成导出包（{d.get('progress', 0)}），"
              f"15s 后再查（首次请求该格式需打包 1-2 分钟）...", flush=True)
        time.sleep(15)
    sys.exit("[错误] 等了 5 分钟导出包仍未生成，稍后重跑同一命令再试")


def human(n: float) -> str:
    return f"{n / 1e6:.0f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def download_with_resume(link: str, zip_path: Path) -> None:
    """带断点续传的循环下载：卡死/断网自动重试，已下载字节不丢。"""
    attempt = 0
    while True:
        have = zip_path.stat().st_size if zip_path.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        attempt += 1
        t0 = time.time()
        try:
            with requests.get(link, headers=headers, stream=True,
                              timeout=(15, READ_TIMEOUT)) as resp:
                if resp.status_code == 416:  # Range 超界 = 已经下完了
                    return
                resp.raise_for_status()
                if have and resp.status_code != 206:
                    # 服务器不支持续传，从头来
                    have = 0
                total = int(resp.headers.get("content-length", 0)) + have
                print(f"[重试 {attempt}] 从 {human(have)} 续传，目标 {human(total)}", flush=True)
                mode = "ab" if have else "wb"
                last_mark = have
                with open(zip_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if have - last_mark >= PROGRESS_STEP:
                            speed = (have - last_mark) / max(time.time() - t0, 1) / 1e6
                            pct = f" ({have * 100 // total}%)" if total else ""
                            print(f"  {human(have)} / {human(total)}{pct}"
                                  f"  {speed:.1f} MB/s", flush=True)
                            last_mark = have
                return  # 正常读完
        except (requests.RequestException, OSError) as e:
            wait = min(5 * attempt, 30)
            print(f"  [中断] {type(e).__name__}：{human(have)} 已保留，{wait}s 后自动续传",
                  flush=True)
            time.sleep(wait)
            if attempt >= MAX_RETRY:
                sys.exit(f"[错误] 重试 {MAX_RETRY} 次仍失败，已下载 {human(have)} 在 {zip_path}，"
                         f"稍后重跑同一命令即可续传")


def main(api_key: str, ws: str = WORKSPACE, proj: str = PROJECT,
         ver: str = VERSION, dest: Path = DEST) -> None:
    global DEST
    DEST = dest  # 供下载/解压路径使用
    print(f"[1/4] 拿导出链接: {ws}/{proj}/v{ver} ...", flush=True)
    link = get_export_link(api_key, ws, proj, ver)
    zip_path = dest / "roboflow.zip"
    dest.mkdir(parents=True, exist_ok=True)

    print("[2/4] 下载（断点续传+自动重试，卡死会自己恢复，不用管）：", flush=True)
    download_with_resume(link, zip_path)
    size = zip_path.stat().st_size
    print(f"[3/4] 下载完成 {human(size)}，解压中...", flush=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    zip_path.unlink()

    n_total = 0
    for sp in ("train", "valid", "test"):
        n = len(list((dest / sp / "images").glob("*")))
        n_total += n
        print(f"      {sp}: {n} 张", flush=True)
    print(f"[4/4] 完成：共 {n_total} 张 → {dest}", flush=True)


if __name__ == "__main__":
    # 用法 1（默认 dataset1）: python tools/download_dataset1_rest.py <KEY>
    # 用法 2（任意数据集）:
    #   python tools/download_dataset1_rest.py <KEY> <workspace> <project> <version> <dest_dir>
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 6:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5]))
    else:
        print(__doc__)
        sys.exit(1)
