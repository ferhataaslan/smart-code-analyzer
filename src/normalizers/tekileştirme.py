#!/usr/bin/env python3
"""
tekileştirme.py — HF Dataset Tekilleştirme & Birleştirme

HF'deki TÜM parquet dosyalarını stream ile okur, entity_id'ye göre
tekrar edenleri kaldırır ve TEK bir birleşik parquet dosyası olarak
aynı repo'ya yükler. Mevcut parquet dosyaları SİLİNMEZ.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

load_dotenv()

REPO_ID = "smart-code-analyzer-team/cpp-vulnerability-dataset"
TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    print("[-] HF_TOKEN bulunamadi! .env dosyasini kontrol edin.")
    exit(1)

# ── 1. HF'den stream olarak tüm parquet'leri oku ──
print(f"[*] HF'den veri yukleniyor (streaming): {REPO_ID}")
dataset = load_dataset(REPO_ID, split="train", token=TOKEN, streaming=True)

seen_ids = set()
cleaned_data = []
duplicates = 0
total = 0

# ── 2. entity_id bazlı tekilleştirme ──
print("[*] Tekilleştirme basliyor...")
for example in dataset:
    total += 1
    record_id = example.get("entity_id")
    if record_id and record_id not in seen_ids:
        seen_ids.add(record_id)
        cleaned_data.append(example)
    else:
        duplicates += 1

    if total % 5000 == 0:
        print(f"  ... {total} kayit islendi, {duplicates} tekrar bulundu")

print(f"\n[*] Toplam taranan: {total}")
print(f"[*] Tekil kayit: {len(cleaned_data)}")
print(f"[*] Silinen tekrar: {duplicates}")

# ── 3. Temizlenmiş dataset oluştur ──
print("[*] Birlestirilmis dataset olusturuluyor...")
new_dataset = Dataset.from_list(cleaned_data)
print(f"[*] Dataset: {len(new_dataset)} kayit, {new_dataset.column_names}")

# ── 4. Tek bir parquet dosyasına kaydet ──
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"merged_deduplicated_{timestamp}.parquet"
new_dataset.to_parquet(filename)
print(f"[*] Lokal parquet olusturuldu: {filename}")

# ── 5. HF'ye tek dosya olarak yükle (mevcut dosyalara DOKUNMAZ) ──
print(f"[*] HF'ye yukleniyor: data/{filename}")
api = HfApi(token=TOKEN)
api.upload_file(
    path_or_fileobj=filename,
    path_in_repo=f"data/{filename}",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print(f"\n[OK] Basariyla yuklendi!")
print(f"     Dosya: data/{filename}")
print(f"     Kayit: {len(new_dataset)} tekil kayit")
print(f"     Tekrar: {duplicates} duplikat silindi")