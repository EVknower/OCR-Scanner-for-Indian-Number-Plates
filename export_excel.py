import pandas as pd
import os
from datetime import datetime
from database_manager import fetch_all, fetch_today
from config import EXPORT_DIR

os.makedirs(EXPORT_DIR, exist_ok=True)

def _rows_to_df(rows):
    return pd.DataFrame(rows, columns=["Plate", "Timestamp", "Confidence", "Image Path"])

def export_all(filename=None):
    rows = fetch_all()
    df = _rows_to_df(rows)
    if not filename:
        filename = f"vehicle_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(EXPORT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[Export] Saved → {path}")
    return path

def export_today(filename=None):
    rows = fetch_today()
    df = _rows_to_df(rows)
    if not filename:
        filename = f"vehicle_log_today_{datetime.now().strftime('%Y%m%d')}.csv"
    path = os.path.join(EXPORT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[Export] Today's log → {path}")
    return path
