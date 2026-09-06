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


def query_violations(vtype=None, severity=None, date_str=None, lab_id=None,
                     limit=200, offset=0):
    sql = ("SELECT v.*, l.name AS lab_name FROM violation_events v"
           " LEFT JOIN laboratories l ON v.lab_id = l.id WHERE 1=1")
    args = []
    if vtype:
        sql += " AND v.violation_types LIKE ?"
        args.append(f"%{vtype}%")
    if severity:
        sql += " AND v.severity = ?"
        args.append(severity)
    else:
        sql += " AND v.severity != '正常'"  # 默认隐藏正常评估记录
    if date_str:
        sql += " AND date(v.created_at) = ?"
        args.append(date_str)
    if lab_id:
        sql += " AND v.lab_id = ?"
        args.append(lab_id)
    sql += " ORDER BY v.id DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    conn = get_conn()
    rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
    conn.close()
    return rows


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
