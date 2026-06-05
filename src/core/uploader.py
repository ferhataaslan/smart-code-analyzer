#!/usr/bin/env python3
"""
uploader.py — Parquet Export & Hugging Face Hub Push

Veritabanından onaylı kayıtları (approved) Parquet formatında export eder
ve Hugging Face Hub'a pushlar.

Şema Koruması:
  Tüm karmaşık alanlar (security_context, ast_metadata, nl_alignment,
  data_flow_graph) datasets.Features ile KESİN Value("string") olarak
  tanımlanır. PyArrow'un otomatik nested struct tahmini engellenir.
"""

import os
import json
import sqlite3

import pandas as pd
from datetime import datetime
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi

from src.database import database

# .env dosyasından ortam değişkenlerini yükle (harici paket gerektirmez)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

# Hibrit DB dosyaları
DOCUMENT_DB_FILE = "data/dbs/document_store.db"
VECTOR_DB_FILE = "data/dbs/vector_store.db"

# ============================================================================
# Katı HF Features Şeması — Sütun sırası ve tipleri sabitlenir (Flat String)
# ============================================================================
HF_FEATURES = Features({
    # ── Kimlik ──
    "entity_id":              Value("string"),
    # ── Kaynak ──
    "source_system":          Value("string"),
    # ── Ham Kod ──
    "raw_snippet":            Value("string"),
    # ── Normalize Edilmiş Yapı ──
    "normalized_structure":   Value("string"),
    # ── Uygulanan Algoritma ──
    "applied_algorithm":      Value("string"),
    # ── Karmaşıklık (Big O) ──
    "complexity":             Value("string"),
    # ── Platform Meta Verileri (JSON string) ──
    "ast_metadata":           Value("string"),
    "nl_alignment":           Value("string"),
    # ── Güvenlik Bağlamı (JSON string — risk_scoring dahil) ──
    "security_context":       Value("string"),
    # ── Veri Akış Grafiği (JSON string — DFG kenar listesi) ──
    "data_flow_graph":        Value("string"),
    # ── Dedup Hash ──
    "code_hash":              Value("string"),
    # ── Zaman Damgası ──
    "created_at":             Value("string"),
})

# Features'daki sütun sırası (DataFrame sütun düzeni için referans)
HF_COLUMN_ORDER = list(HF_FEATURES.keys())


def count_approved_records() -> int:
    """Eski Review Station'daki onaylı kayıt sayısını döndürür."""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM records WHERE status = 'approved'")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def _get_vector_store_records() -> list:
    """
    Vector Store ve Document Store'dan kayıtları birleştirerek okur.
    Böylece hem normalized_structure hem de raw_snippet alınmış olur.
    """
    try:
        conn = sqlite3.connect(VECTOR_DB_FILE)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        
        # Document store'u bağla (raw_snippet için)
        conn.execute("ATTACH DATABASE 'data/dbs/document_store.db' AS doc_db")

        # Sütun varlığını kontrol et
        cursor = conn.execute("PRAGMA table_info(vectors)")
        columns = [row[1] for row in cursor.fetchall()]

        select_cols = [
            "v.entity_id", "v.source_system", "d.raw_snippet", "v.normalized_structure",
            "v.applied_algorithm", "v.ast_metadata", "v.nl_alignment",
            "v.security_context", "v.code_hash", "v.created_at"
        ]
        if "data_flow_graph" in columns:
            select_cols.append("v.data_flow_graph")
        if "complexity" in columns:
            select_cols.append("v.complexity")

        cols_str = ", ".join(select_cols)
        rows = conn.execute(
            f"SELECT {cols_str} "
            f"FROM vectors v "
            f"LEFT JOIN doc_db.documents d ON v.entity_id = d.entity_id"
        ).fetchall()

        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[-] Vector Store okunamadı: {e}")
        return []


def _ensure_json_string(value, fallback="{}") -> str:
    """
    Değerin saf JSON string olduğunu garanti eder.
    Python dict/list ise json.dumps ile serileştirir.
    None veya boş ise fallback değerini döner.
    """
    if value is None:
        return fallback
    if isinstance(value, str):
        return value if value.strip() else fallback
    # dict, list veya diğer Python objeleri → JSON string
    return json.dumps(value, ensure_ascii=False)


def export_and_push():
    """
    Hibrit veritabanından tüm kayıtları Parquet olarak export eder
    ve Hugging Face Hub'a pushlar.

    Şema Koruması:
      - Tüm sütunlar HF_FEATURES ile Value("string") olarak sabitlenir
      - PyArrow'un otomatik nested struct tahmini ENGELLENİR
      - Dataset Viewer'da her alan düz metin/JSON olarak görünür
    """
    # ── 1. Hibrit DB'den kayıtları al ──
    records = _get_vector_store_records()

    if not records:
        # Fallback: Eski Review Station DB'den al
        print("[*] Vector Store boş, Review Station'dan okunuyor...")
        records = database.get_approved_records()

    if not records:
        print("[*] No records to export.")
        return

    device_id = os.environ.get("DEVICE_ID", "unknown_device")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_{device_id}_{timestamp}.parquet"

    # ── 2. DataFrame oluştur ──
    df = pd.DataFrame(records)

    # complexity sütunu yoksa varsayılan ekle
    if "complexity" not in df.columns:
        df["complexity"] = "1"

    # data_flow_graph sütunu yoksa boş JSON listesi olarak ekle
    if "data_flow_graph" not in df.columns:
        df["data_flow_graph"] = "[]"

    # ── 3. Tüm karmaşık alanları saf JSON string'e dönüştür ──
    # (PyArrow'un dict/list tahminini kesin olarak engelle)
    json_columns_with_fallback = {
        "ast_metadata":     "{}",
        "nl_alignment":     "{}",
        "security_context": "{}",
        "data_flow_graph":  "[]",
    }
    for col, fallback in json_columns_with_fallback.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x, fb=fallback: _ensure_json_string(x, fb))

    # ── 4. Eksik sütunları varsayılan değerlerle ekle ──
    column_defaults = {
        "entity_id": "",
        "source_system": "",
        "raw_snippet": "",
        "normalized_structure": "",
        "applied_algorithm": "",
        "complexity": "1",
        "ast_metadata": "{}",
        "nl_alignment": "{}",
        "security_context": "{}",
        "data_flow_graph": "[]",
        "code_hash": "",
        "created_at": "",
    }
    for col, default_val in column_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # ── 5. Sütun sırasını HF_FEATURES şemasına göre sabitle ──
    df = df[HF_COLUMN_ORDER]

    # ── 6. NaN/None değerleri temizle (tüm sütunlar string) ──
    df = df.astype(str)
    df = df.fillna("")

    record_count = len(df)

    # ── 7. HF Dataset objesi oluştur (KATİ ŞEMA ile) ──
    hf_dataset = Dataset.from_pandas(df, features=HF_FEATURES, preserve_index=False)

    # Parquet dosyasını HF Dataset üzerinden yaz (şema korumalı)
    hf_dataset.to_parquet(filename)
    print(f"[+] Exported {record_count} records to {filename}")

    # ── 8. Parquet alanlarını doğrula ──
    print(f"[+] Parquet sütunları: {HF_COLUMN_ORDER}")
    print(f"[OK] Tüm karmaşık alanlar Value('string') olarak sabitlendi.")

    # ── 9. HF Hub'a push ──
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[-] HF_TOKEN environment variable not set. Skipping push to Hub.")
        return

    repo_id = "smart-code-analyzer-team/cpp-vulnerability-dataset"
    api = HfApi(token=token)

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=f"data/{filename}",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"[+] Successfully pushed {filename} ({record_count} records) to {repo_id}")
    except Exception as e:
        print(f"[-] Failed to push to Hugging Face: {e}")


if __name__ == "__main__":
    export_and_push()
