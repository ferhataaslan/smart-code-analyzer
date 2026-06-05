import sqlite3
import os
from datetime import datetime

# .env dosyasından ortam değişkenlerini yükle (harici paket gerektirmez)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

DB_FILE = "data/dbs/review_state.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            cwe_id TEXT DEFAULT 'Unknown',
            raw_code TEXT NOT NULL,
            processed_code TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            device_id TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_record(source: str, raw_code: str, processed_code: str, cwe_id: str = "Unknown", conn=None):
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True
        
    cursor = conn.cursor()
    
    device_id = os.environ.get("DEVICE_ID", "unknown_device")
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO records (source, cwe_id, raw_code, processed_code, status, device_id, timestamp)
        VALUES (?, ?, ?, ?, 'pending', ?, ?)
    ''', (source, cwe_id, raw_code, processed_code, device_id, timestamp))
    
    conn.commit()
    if close_after:
        conn.close()

def get_pending_record(source_filter: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if source_filter:
        cursor.execute("SELECT * FROM records WHERE status = 'pending' AND source = ? ORDER BY id DESC LIMIT 1", (source_filter,))
    else:
        cursor.execute("SELECT * FROM records WHERE status = 'pending' ORDER BY id DESC LIMIT 1")
        
    row = cursor.fetchone()
    conn.close()
    return row

def update_status(record_id: int, status: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    device_id = os.environ.get("DEVICE_ID", "unknown_device")
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        UPDATE records 
        SET status = ?, device_id = ?, timestamp = ? 
        WHERE id = ?
    ''', (status, device_id, timestamp, record_id))
    
    conn.commit()
    conn.close()

def get_approved_records():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM records WHERE status = 'approved'")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Initialize db when imported
init_db()
