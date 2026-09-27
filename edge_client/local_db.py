"""
Local SQLite queue used as a fallback when the cloud backend is unreachable.
Records are stored here and synced later by scheduler.py's retry job.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "local_attendance.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            period TEXT NOT NULL,
            classroom_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            synced INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def queue_attendance(student_name, period, classroom_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO pending_attendance (student_name, period, classroom_id, timestamp) VALUES (?, ?, ?, ?)",
        (student_name, period, classroom_id, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def already_marked_today(student_name, period):
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    cur = conn.execute(
        "SELECT COUNT(*) FROM pending_attendance WHERE student_name=? AND period=? AND timestamp LIKE ?",
        (student_name, period, f"{today}%"),
    )
    count = cur.fetchone()[0]
    conn.close()
    return count > 0


def get_unsynced():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT id, student_name, period, classroom_id, timestamp FROM pending_attendance WHERE synced=0")
    rows = cur.fetchall()
    conn.close()
    return rows


def mark_synced(record_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE pending_attendance SET synced=1 WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
