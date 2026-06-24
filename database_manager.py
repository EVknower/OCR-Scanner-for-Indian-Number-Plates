import sqlite3
import os
from datetime import datetime, timedelta
from config import DB_PATH, COOLDOWN_SECONDS

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                plate      TEXT NOT NULL,
                timestamp  DATETIME NOT NULL,
                confidence FLOAT,
                image_path TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate ON vehicles(plate)")
        conn.commit()

def is_duplicate(plate):
    cutoff = (datetime.now() - timedelta(seconds=COOLDOWN_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM vehicles WHERE plate=? AND timestamp>? LIMIT 1",
            (plate, cutoff)
        ).fetchone()
    return row is not None

def insert(plate, confidence, image_path=None):
    if is_duplicate(plate):
        return False
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO vehicles (plate, timestamp, confidence, image_path) VALUES (?,?,?,?)",
            (plate, ts, confidence, image_path)
        )
        conn.commit()
    return True

def fetch_all():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT plate, timestamp, confidence, image_path FROM vehicles ORDER BY timestamp DESC"
        ).fetchall()
    return rows

def fetch_today():
    today = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT plate, timestamp, confidence, image_path FROM vehicles WHERE timestamp LIKE ? ORDER BY timestamp DESC",
            (f"{today}%",)
        ).fetchall()
    return rows

def search(plate):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT plate, timestamp, confidence, image_path FROM vehicles WHERE plate LIKE ? ORDER BY timestamp DESC",
            (f"%{plate}%",)
        ).fetchall()
    return rows

init_db()
