# -*- coding: utf-8 -*-
"""
数据库层：SQLite（第一阶段）。

表结构：
  laboratories      实验室
  areas             检测区域（矩形 ROI，归一化坐标）
  safety_rules      区域的 PPE 要求（可配置）
  detection_records 每次巡检（图片/视频处理）的记录
  violation_events  违规事件
"""
import sqlite3
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "backend" / "labsafety.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS laboratories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS areas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lab_id INTEGER NOT NULL REFERENCES laboratories(id),
    name TEXT NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS safety_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    area_id INTEGER NOT NULL REFERENCES areas(id),
    ppe_type TEXT NOT NULL,          -- mask / gloves / lab_coat / goggles / helmet
    required INTEGER NOT NULL DEFAULT 1,
    UNIQUE(area_id, ppe_type)
);

CREATE TABLE IF NOT EXISTS detection_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lab_id INTEGER REFERENCES laboratories(id),
    source_type TEXT NOT NULL,       -- image / video
    source_name TEXT NOT NULL,
    person_count INTEGER DEFAULT 0,
    violation_count INTEGER DEFAULT 0,
    normal_count INTEGER DEFAULT 0,
    process_time_ms REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS violation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER REFERENCES detection_records(id),
    lab_id INTEGER REFERENCES laboratories(id),
    area_name TEXT NOT NULL,
    person_id INTEGER NOT NULL,
    violation_types TEXT NOT NULL,   -- 逗号分隔
    severity TEXT NOT NULL,
    screenshot_path TEXT DEFAULT '',
    frame_time REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(seed: bool = True):
    """初始化数据库；seed=True 时写入演示用默认配置"""
    conn = get_conn()
    conn.executescript(SCHEMA)
    if seed:
        cur = conn.execute("SELECT COUNT(*) c FROM laboratories")
        if cur.fetchone()["c"] == 0:
            conn.execute(
                "INSERT INTO laboratories(name, description) VALUES(?, ?)",
                ("实验室A", "化学实验操作实验室（演示默认配置）"))
            lab_id = conn.execute(
                "SELECT id FROM laboratories WHERE name='实验室A'").fetchone()["id"]
            # 默认操作区：画面中部偏下的大矩形
            conn.execute(
                "INSERT INTO areas(lab_id, name, x1, y1, x2, y2) VALUES(?,?,?,?,?,?)",
                (lab_id, "实验操作区", 0.22, 0.18, 0.88, 0.92))
            area_id = conn.execute(
                "SELECT id FROM areas WHERE lab_id=? AND name='实验操作区'",
                (lab_id,)).fetchone()["id"]
            for ppe in ("mask", "gloves", "lab_coat"):
                conn.execute(
                    "INSERT INTO safety_rules(area_id, ppe_type, required) VALUES(?,?,1)",
                    (area_id, ppe))
    conn.commit()
    conn.close()


# ---------------- 查询/写入辅助 ----------------

def load_areas(lab_id: int):
    """读取某实验室的全部区域 + PPE 规则，返回 AreaROI 列表"""
    from ai.roi import AreaROI
    conn = get_conn()
    rows = conn.execute("SELECT * FROM areas WHERE lab_id=?", (lab_id,)).fetchall()
    areas = []
    for r in rows:
        rules = conn.execute(
            "SELECT ppe_type FROM safety_rules WHERE area_id=? AND required=1",
            (r["id"],)).fetchall()
        areas.append(AreaROI(
            area_id=r["id"], name=r["name"],
            x1=r["x1"], y1=r["y1"], x2=r["x2"], y2=r["y2"],
            required_ppe=[x["ppe_type"] for x in rules]))
    conn.close()
    return areas


def lab_exists(lab_id: int) -> bool:
    """判断实验室是否存在（用于 API 参数校验，把外键崩溃转成 4xx）"""
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM laboratories WHERE id=?",
                       (lab_id,)).fetchone()
    conn.close()
    return row is not None


def save_detection_record(lab_id, source_type, source_name, person_count,
                          violation_count, normal_count, process_time_ms) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO detection_records(lab_id, source_type, source_name,"
        " person_count, violation_count, normal_count, process_time_ms)"
        " VALUES(?,?,?,?,?,?,?)",
        (lab_id, source_type, source_name, person_count,
         violation_count, normal_count, process_time_ms))
    record_id = cur.lastrowid
    conn.commit()
    conn.close()
    return record_id


def save_violation_events(record_id, lab_id, events: list,
                          include_normal: bool = False):
    """把 ViolationEvent 列表写入数据库（默认只写违规；include_normal=True 时也写正常评估）"""
    conn = get_conn()
    for ev in events:
        if not getattr(ev, "is_violation", False) and not include_normal:
            continue
        conn.execute(
            "INSERT INTO violation_events(record_id, lab_id, area_name, person_id,"
            " violation_types, severity, screenshot_path, frame_time)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (record_id, lab_id, ev.area_name, ev.person_id,
             ",".join(ev.violation_types), ev.severity,
             getattr(ev, "screenshot", ""), getattr(ev, "frame_time", 0)))
    conn.commit()
    conn.close()


def save_event_dicts(record_id, lab_id, event_dicts: list):
    """把 process_video 返回的 dict 事件写入数据库"""
    conn = get_conn()
    for ev in event_dicts:
        conn.execute(
            "INSERT INTO violation_events(record_id, lab_id, area_name, person_id,"
            " violation_types, severity, screenshot_path, frame_time)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (record_id, lab_id, ev.get("area", ""), ev.get("person_id", 0),
             ",".join(ev.get("violation_types", [])), ev.get("severity", ""),
             ev.get("screenshot", ""), ev.get("frame_time", 0)))
    conn.commit()
    conn.close()


# 违规查询允许的排序字段（白名单，拼接进 SQL 前校验，避免注入）
VIOLATION_SORT_FIELDS = {
    "id": "v.id",
    "created_at": "v.created_at",
    "severity": "v.severity",
    "area_name": "v.area_name",
}


def _violation_where(vtype=None, severity=None, date_str=None, lab_id=None,
                     date_from=None, date_to=None):
    """
    构造违规查询的 WHERE 片段与参数。
    query_violations / count_violations / 导出功能共用，保证筛选口径一致。
    """
    where = " WHERE 1=1"
    args = []
    if vtype:
        where += " AND v.violation_types LIKE ?"
        args.append(f"%{vtype}%")
    if severity:
        where += " AND v.severity = ?"
        args.append(severity)
    else:
        where += " AND v.severity != '正常'"  # 默认隐藏正常评估记录
    if date_str:
        where += " AND date(v.created_at) = ?"
        args.append(date_str)
    if date_from:
        where += " AND date(v.created_at) >= ?"
        args.append(date_from)
    if date_to:
        where += " AND date(v.created_at) <= ?"
        args.append(date_to)
    if lab_id:
        where += " AND v.lab_id = ?"
        args.append(lab_id)
    return where, args


def query_violations(vtype=None, severity=None, date_str=None, lab_id=None,
                     limit=200, offset=0, date_from=None, date_to=None,
                     sort="id", order="desc"):
    """
    查询违规记录。

    新增参数均为可选，原有调用方式完全不变（向后兼容）：
      date_from / date_to : 日期区间过滤（含端点，YYYY-MM-DD）
      sort                : id | created_at | severity | area_name（白名单）
      order               : asc | desc
    """
    where, args = _violation_where(vtype, severity, date_str, lab_id,
                                   date_from, date_to)
    col = VIOLATION_SORT_FIELDS.get(str(sort).lower(), "v.id")
    direction = "ASC" if str(order).lower() == "asc" else "DESC"
    sql = ("SELECT v.*, l.name AS lab_name FROM violation_events v"
           " LEFT JOIN laboratories l ON v.lab_id = l.id"
           f"{where} ORDER BY {col} {direction}, v.id DESC LIMIT ? OFFSET ?")
    args = args + [limit, offset]
    conn = get_conn()
    rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
    conn.close()
    return rows


def count_violations(vtype=None, severity=None, date_str=None, lab_id=None,
                     date_from=None, date_to=None) -> int:
    """按与 query_violations 相同的筛选条件统计总条数（用于分页 total_count）"""
    where, args = _violation_where(vtype, severity, date_str, lab_id,
                                   date_from, date_to)
    conn = get_conn()
    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM violation_events v{where}", args).fetchone()
    conn.close()
    return int(row["c"])


def dashboard_stats():
    """首页统计：今日检测次数、违规次数、正常次数、违规率、违规类型分布"""
    today = date.today().isoformat()
    conn = get_conn()
    det = conn.execute(
        "SELECT COUNT(*) c FROM detection_records WHERE date(created_at)=?",
        (today,)).fetchone()["c"]
    vio = conn.execute(
        "SELECT COUNT(*) c FROM violation_events WHERE date(created_at)=?"
        " AND severity != '正常'", (today,)).fetchone()["c"]
    # 正常次数：按"帧级评估结果"统计（今日违规事件之外的正常评估记录）
    normal = conn.execute(
        "SELECT COUNT(*) c FROM violation_events WHERE date(created_at)=?"
        " AND severity='正常'", (today,)).fetchone()["c"]
    by_type = [dict(r) for r in conn.execute(
        "SELECT violation_types t, COUNT(*) c FROM violation_events"
        " WHERE date(created_at)=? AND severity != '正常'"
        " GROUP BY violation_types ORDER BY c DESC LIMIT 10", (today,)).fetchall()]
    by_day = [dict(r) for r in conn.execute(
        "SELECT date(created_at) d, COUNT(*) c FROM violation_events"
        " WHERE severity != '正常' AND created_at >= date('now','-6 days','localtime')"
        " GROUP BY d ORDER BY d",).fetchall()]
    total_eval = vio + normal
    rate = (vio / total_eval * 100) if total_eval else 0.0
    conn.close()
    return {
        "date": today,
        "detection_count": det,
        "violation_count": vio,
        "normal_count": normal,
        "violation_rate": round(rate, 1),
        "by_type": by_type,
        "by_day": by_day,
    }


def statistics():
    """统计页数据：每日违规、各类违规数量、正常/违规比例"""
    conn = get_conn()
    by_day = [dict(r) for r in conn.execute(
        "SELECT date(created_at) d, COUNT(*) c FROM violation_events"
        " WHERE severity != '正常' GROUP BY d ORDER BY d LIMIT 30").fetchall()]
    by_type = [dict(r) for r in conn.execute(
        "SELECT violation_types t, COUNT(*) c FROM violation_events"
        " WHERE severity != '正常' GROUP BY violation_types ORDER BY c DESC").fetchall()]
    by_severity = [dict(r) for r in conn.execute(
        "SELECT severity s, COUNT(*) c FROM violation_events GROUP BY severity").fetchall()]
    conn.close()
    return {"by_day": by_day, "by_type": by_type, "by_severity": by_severity}


# ---------------- 配置管理（实验室 / 区域 / PPE 规则） ----------------
# 说明：原先这些 SQL 直接写在 backend/main.py 里（技术债清单 #5：
#       "设置端点裸 SQL 绕过 database 层"），本次统一收敛到这里。

def list_labs_with_areas() -> list:
    """
    读取全部实验室，及其下属区域与各区域的 PPE 要求（供 GET /api/settings 使用）。

    统一在此加 ORDER BY，保证返回顺序稳定：
      - laboratories / areas 按 id 升序
      - required_ppe 按 safety_rules.id（即配置时的顺序）升序

    修复的问题：原实现无 ORDER BY，SQLite 返回顺序不保证，实测
    required_ppe 会返回 ["gloves","lab_coat","mask"]，与
    docs/INTERFACE_CONTRACT.md 6.8 示例的 ["mask","gloves","lab_coat"] 不一致；
    若前端按顺序渲染会显示错乱。加 ORDER BY 后与契约一致。
    """
    conn = get_conn()
    labs = [dict(r) for r in conn.execute(
        "SELECT * FROM laboratories ORDER BY id").fetchall()]
    for lab in labs:
        lab["areas"] = [dict(r) for r in conn.execute(
            "SELECT * FROM areas WHERE lab_id=? ORDER BY id",
            (lab["id"],)).fetchall()]
        for area in lab["areas"]:
            area["required_ppe"] = [r["ppe_type"] for r in conn.execute(
                "SELECT ppe_type FROM safety_rules WHERE area_id=? AND required=1"
                " ORDER BY id", (area["id"],)).fetchall()]
    conn.close()
    return labs


def create_lab(name: str, description: str = "") -> int:
    """
    新建实验室，返回新记录 id。
    名称重复时抛出 sqlite3.IntegrityError，由调用方（API 层）转为 400。
    """
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO laboratories(name, description) VALUES(?,?)",
            (name, description))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def create_area(lab_id: int, name: str, x1: float, y1: float, x2: float,
                y2: float, required_ppe: list) -> int:
    """
    新建区域并写入其 PPE 要求，返回新区域 id。
    required_ppe 为该区域要求的 PPE 业务类别列表（如 ["mask","gloves"]）。
    lab_id 不存在时抛出 sqlite3.IntegrityError（外键约束）。
    """
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO areas(lab_id, name, x1, y1, x2, y2) VALUES(?,?,?,?,?,?)",
            (lab_id, name, x1, y1, x2, y2))
        area_id = int(cur.lastrowid)
        for ppe in required_ppe:
            conn.execute("INSERT INTO safety_rules(area_id, ppe_type, required)"
                         " VALUES(?,?,1)", (area_id, ppe))
        conn.commit()
        return area_id
    finally:
        conn.close()


def delete_area(area_id: int) -> None:
    """
    删除区域及其 PPE 规则（级联清理 safety_rules）。
    幂等：area_id 不存在时不报错。
    """
    conn = get_conn()
    try:
        conn.execute("DELETE FROM safety_rules WHERE area_id=?", (area_id,))
        conn.execute("DELETE FROM areas WHERE id=?", (area_id,))
        conn.commit()
    finally:
        conn.close()
