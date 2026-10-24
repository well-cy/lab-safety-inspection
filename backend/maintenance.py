# -*- coding: utf-8 -*-
"""
存储维护：统计并清理运行期产生的文件。

对应分工任务「清理临时视频、结果视频和截图目录」。

问题
----
运行期有 4 个目录持续产生文件，且**原先没有任何清理机制**：

======================  ================================================
目录                     说明
======================  ================================================
``data/test/uploads``    每次检测都把上传文件存一份（含各类临时上传）
``outputs/videos``       标注后的视频，每个几十 MB
``outputs/annotated``    标注后的图片
``outputs/screenshots``  违规截图（被 violation_events.screenshot_path 引用）
======================  ================================================

安全设计
--------
1. **默认 dry_run**：只统计不删除，必须显式传 ``dry_run=false`` 才真正删除
2. **按修改时间保留**：只删早于 ``今天 - keep_days`` 的文件，不会误删新数据
3. **只在配置目录内操作**：遍历的是 ``backend/config.py`` 里那几个目录的子文件，
   不会触碰其他路径
4. **截图目录默认不清理**：``violation_events.screenshot_path`` 引用了截图文件，
   删掉会让"违规记录"页出现裂图。因此默认清理目标为
   ``uploads / annotated / videos``，需要清截图时必须显式指定 ``screenshots``
5. **不碰数据库**：本模块只删文件，不删记录 —— 历史违规记录始终保留
"""
import time
from datetime import datetime
from pathlib import Path

from backend.config import config

# 可清理的目标（名称 → 对应目录由 config 提供）
TARGET_NAMES = ("uploads", "annotated", "videos", "screenshots")

# 默认清理目标：不含 screenshots（原因见模块 docstring 第 4 条）
DEFAULT_TARGETS = ("uploads", "annotated", "videos")


def _target_dirs() -> dict:
    """名称 → 目录（全部来自集中配置）"""
    return {
        "uploads": config.UPLOAD_DIR,
        "annotated": config.ANNOTATED_DIR,
        "videos": config.VIDEO_DIR,
        "screenshots": config.SCREENSHOT_DIR,
    }


def _iter_files(directory: Path) -> list:
    """列出目录下所有文件（递归；目录不存在时返回空列表）"""
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.is_file()]


def storage_report() -> dict:
    """
    统计各运行期目录的占用情况。

    返回每个目录的文件数、字节数、最旧文件时间，以及总计。
    """
    targets = {}
    total_bytes = 0
    for name, directory in _target_dirs().items():
        files = _iter_files(directory)
        size = 0
        oldest = None
        for path in files:
            try:
                stat = path.stat()
            except OSError:
                continue          # 文件在处理途中被删/权限问题，跳过即可
            size += stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime)
            if oldest is None or mtime < oldest:
                oldest = mtime
        targets[name] = {
            "path": str(directory),
            "exists": directory.exists(),
            "files": len(files),
            "bytes": size,
            "mb": round(size / 1024 / 1024, 2),
            "oldest": oldest.strftime("%Y-%m-%d %H:%M:%S") if oldest else "",
        }
        total_bytes += size
    return {
        "targets": targets,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
    }


def cleanup(keep_days: int = 7, dry_run: bool = True,
            targets: list | None = None) -> dict:
    """
    清理运行期文件。

    keep_days: 保留最近 N 天的文件（按文件修改时间判断）
    dry_run  : True 时只统计不删除（默认）
    targets  : 要清理的目标名列表；None 表示使用 DEFAULT_TARGETS

    返回每个目标的 deleted / kept / freed_bytes，以及总计。
    目标名非法时抛 ValueError（由 API 层转成 400）。
    """
    if keep_days < 0:
        raise ValueError(f"keep_days 不能为负数，当前为 {keep_days}")

    dirs = _target_dirs()
    selected = list(targets) if targets else list(DEFAULT_TARGETS)
    unknown = [t for t in selected if t not in dirs]
    if unknown:
        raise ValueError(
            f"未知的清理目标：{','.join(unknown)}；"
            f"可选值：{','.join(TARGET_NAMES)}")

    cutoff = time.time() - keep_days * 86400
    detail = {}
    freed_bytes = 0
    for name in selected:
        directory = dirs[name]
        deleted = kept = 0
        deleted_bytes = 0
        errors = []
        for path in _iter_files(directory):
            try:
                stat = path.stat()
                if stat.st_mtime >= cutoff:
                    kept += 1
                    continue
                if not dry_run:
                    path.unlink()
                deleted += 1
                deleted_bytes += stat.st_size
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")
        detail[name] = {
            "path": str(directory),
            "deleted": deleted,
            "kept": kept,
            "freed_bytes": deleted_bytes,
            "freed_mb": round(deleted_bytes / 1024 / 1024, 2),
            "errors": errors[:5],
        }
        freed_bytes += deleted_bytes

    return {
        "dry_run": dry_run,
        "keep_days": keep_days,
        "targets": selected,
        "detail": detail,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / 1024 / 1024, 2),
        "note": ("仅统计，未删除任何文件（dry_run=true）" if dry_run
                 else "已按修改时间删除超期文件；数据库记录未改动"),
    }
