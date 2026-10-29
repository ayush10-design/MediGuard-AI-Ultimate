"""SQLite storage for analysis history. Images are not stored."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    filename TEXT,
    analysis_type TEXT NOT NULL,
    predicted_class TEXT,
    fake_score REAL,
    real_score REAL,
    medicine_name TEXT,
    report_json TEXT
)
"""


@contextmanager
def _connect(db_path: Path | None = None):
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def add_record(report: dict[str, Any], db_path: Path | None = None) -> int:
    cls = report.get("classification") or {}
    scores = cls.get("scores") or {}

    def pick(name):
        for k, v in scores.items():
            if k.lower() == name:
                return v
        return None

    name = ((report.get("fields") or {}).get("medicine_name") or {}).get("value")
    with _connect(db_path) as con:
        cur = con.execute(
            "INSERT INTO analyses (timestamp, filename, analysis_type, predicted_class, fake_score, real_score, medicine_name, report_json)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (report.get("created_at") or datetime.now().isoformat(timespec="seconds"),
             report.get("filename"), report.get("analysis_type", "image"),
             cls.get("predicted_class") if cls.get("ok") else None,
             pick("fake") if cls.get("ok") else None, pick("real") if cls.get("ok") else None,
             name if name and name != "Not detected" else None,
             json.dumps(report, default=str)),
        )
        return int(cur.lastrowid)


def list_records(search: str = "", predicted: str | None = None, db_path: Path | None = None,
                 limit: int = 500) -> pd.DataFrame:
    q = ("SELECT id, timestamp, filename, analysis_type, predicted_class, fake_score, real_score, medicine_name "
         "FROM analyses WHERE 1=1")
    args: list[Any] = []
    if search:
        q += " AND (LOWER(IFNULL(filename,'')) LIKE ? OR LOWER(IFNULL(medicine_name,'')) LIKE ?)"
        args += [f"%{search.lower()}%"] * 2
    if predicted:
        q += " AND predicted_class = ?"
        args.append(predicted)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _connect(db_path) as con:
        return pd.read_sql_query(q, con, params=args)


def get_report(record_id: int, db_path: Path | None = None) -> dict | None:
    with _connect(db_path) as con:
        row = con.execute("SELECT report_json FROM analyses WHERE id = ?", (record_id,)).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def delete_record(record_id: int, db_path: Path | None = None) -> None:
    with _connect(db_path) as con:
        con.execute("DELETE FROM analyses WHERE id = ?", (record_id,))


def clear_all(db_path: Path | None = None) -> None:
    with _connect(db_path) as con:
        con.execute("DELETE FROM analyses")


def stats(db_path: Path | None = None) -> dict[str, int]:
    with _connect(db_path) as con:
        total = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        rows = con.execute("SELECT predicted_class, COUNT(*) FROM analyses WHERE predicted_class IS NOT NULL "
                           "GROUP BY predicted_class").fetchall()
    return {"total": total, **{str(k): v for k, v in rows}}
