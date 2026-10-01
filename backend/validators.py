# -*- coding: utf-8 -*-
"""
参数校验工具（被 backend/main.py 与 backend/export.py 共用）。

为什么单独成文件：
    /api/violations 与 /api/export/violations 使用同一套筛选参数
    （date / severity / vtype），若各写一份校验逻辑，很容易出现
    "导出接口会拦非法日期、查询接口却静默放过" 这类不一致。
    统一放在这里，保证两个接口的校验口径完全相同。
"""
import re
from datetime import datetime

# 日期格式：YYYY-MM-DD
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


def is_valid_date(value: str | None) -> bool:
    """
    校验是否为真实存在的日历日期。

    只用正则不够：'2026-13-99'（13 月 99 日）、'2026-02-30'（2 月 30 日）
    位数都合法，但不是有效日期，必须用 strptime 兜底。

    空值（None / ""）视为"未提供该筛选条件"，返回 True。
    """
    if not value:
        return True
    if not re.match(DATE_PATTERN, value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def find_invalid_date(*pairs) -> tuple[str, str] | None:
    """
    批量校验日期参数。

    参数：若干 (参数名, 值) 二元组，例如 ("date_from", date_from)。
    返回：第一个非法参数的 (参数名, 值)；全部合法时返回 None。
    """
    for label, value in pairs:
        if not is_valid_date(value):
            return label, value
    return None
