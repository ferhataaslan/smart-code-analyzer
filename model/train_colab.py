#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model/train_colab.py — Smart Code Analyzer: Production-Grade Eğitim Scripti

Mimari : GraphCodeBERT (microsoft/graphcodebert-base)
         + Memory-Augmented Transformer + Hierarchical Attention
         + Multi-Task Learning (CWE / Big-O / DFG)

Kullanım (Google Colab):
    1. Runtime → Change runtime type → GPU (T4 veya A100)
    2. Bu dosyayı Colab'a yükleyin
    3. Çalıştırın: !python train_colab.py

Kullanım (Lokal):
    $ py -m model.train_colab
    $ py -m model.train_colab --resume model/checkpoints/last_checkpoint.pt
    $ py -m model.train_colab --device cuda --epochs 50

Referanslar:
    - GraphCodeBERT: https://arxiv.org/abs/2009.08366
    - Multi-Task Learning: https://arxiv.org/abs/1706.05098
    - Memory-Augmented Networks: https://arxiv.org/abs/1410.3916

Copyright (c) 2026 Smart Code Analyzer Team
License: MIT
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
#  §0. Ortam Kurulumu & Bağımlılıklar
# ══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os
import glob

# ── Hata Çözümü: Stale Lock ve Tqdm Deadlock'unu Engelleme ──
# 1. Colab'da indirme yarıda kesilirse HuggingFace arkada ".lock" (kilit) dosyaları bırakır.
#    Sonraki çalıştırmalarda bu kilitler açılmadığı için script sonsuza kadar kilitli kalır.
#    Aşağıdaki kod tüm kilit dosyalarını acımasızca silerek donmayı engeller.
try:
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_cache):
        for lock_file in glob.glob(os.path.join(hf_cache, "**/*.lock"), recursive=True):
            try:
                os.remove(lock_file)
                print(f"🧹 Temizlendi (Kilit Çözüldü): {lock_file}")
            except Exception:
                pass
except Exception:
    pass

# 2. İlerleme çubuğu kilitlenmelerini engellemek için tqdm'yi kapatıyoruz.
# (hf_transfer bazı ortamlarda uyumsuz olduğu için şimdilik kapattık)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

_REQUIRED_PACKAGES: list[str] = [
    "torch",
    "transformers>=4.30.0",
    "datasets>=2.14.0",
    "huggingface_hub>=0.16.0",
    "python-dotenv",
    "tqdm",
    "matplotlib>=3.7.0",
]


def _ensure_packages() -> None:
    """Eksik paketleri sessizce kurar (Colab uyumlu)."""
    for spec in _REQUIRED_PACKAGES:
        pkg = spec.split(">=")[0].split("==")[0]
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"  > {spec} kuruluyor...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", spec],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  + {pkg} kuruldu.")


_ensure_packages()

# ── Standart Kütüphaneler ────────────────────────────────────────────────────
import json
import math
import time
import random
import logging
import argparse
from enum import IntEnum
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ── Üçüncü Parti ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ── Google Drive (Colab) ─────────────────────────────────────────────────────
_IN_COLAB: bool = False
try:
    import google.colab  # type: ignore[import-untyped]
    _IN_COLAB = True
except ImportError:
    pass

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SmartCodeAnalyzer")


# ══════════════════════════════════════════════════════════════════════════════
#  §1. Sabitler & Konfigürasyon
# ══════════════════════════════════════════════════════════════════════════════

# ── Temel Mimari: GraphCodeBERT ──────────────────────────────────────────────
# GraphCodeBERT, veri akış grafiği (DFG) farkındalığıyla eğitilmiş bir
# pre-trained modeldir.  CodeBERT'e göre avantajları:
#   1. DFG edge'lerini anlama kapasitesi (bizim data_flow_graph sütunu)
#   2. Değişken bağımlılıklarını yakalama
#   3. Kod yapısını daha derin anlama
# Kaynak: https://arxiv.org/abs/2009.08366
def _ensure_local_model(repo_id: str = "microsoft/graphcodebert-base") -> str:
    """
    HuggingFace python kütüphanesinin (xet_get, deadlock vs) kronik indirme sorunlarını 
    aşmak için modeli doğrudan işletim sisteminin GIT komutuyla indirir.
    Bu, indirme kilitlenmelerini temelden ve kesin olarak çözen kurşun geçirmez yöntemdir.
    """
    import os
    import subprocess
    local_dir = os.path.abspath("local_graphcodebert_base")
    if not os.path.exists(os.path.join(local_dir, "pytorch_model.bin")):
        print(f"\n🚀 [TEMEL ÇÖZÜM] Model HF Python kütüphanesi bypass edilerek doğrudan GIT ile indiriliyor...")
        # Git LFS aktif ediliyor
        subprocess.run(["git", "lfs", "install"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Model klonlanıyor
        cmd = ["git", "clone", f"https://huggingface.co/{repo_id}", local_dir]
        subprocess.run(cmd, check=True)
        print("✅ Model indirme tamamlandı. Kilitlenme riski sıfırlandı.\n")
    return local_dir

BASE_MODEL: str = _ensure_local_model("microsoft/graphcodebert-base")

# ── Dinamik Zafiyet / CWE Listesi (Multi-Label) ──────────────────────────────
# Modelin tespit edeceği 35 temel CWE (risk_scoring.py'den sabitlendi)
KNOWN_CWES: list[str] = [
    "CWE-79", "CWE-89", "CWE-352", "CWE-862", "CWE-787", "CWE-120",
    "CWE-20", "CWE-78", "CWE-416", "CWE-22", "CWE-125", "CWE-119",
    "CWE-190", "CWE-200", "CWE-122", "CWE-134", "CWE-242", "CWE-327",
    "CWE-330", "CWE-415", "CWE-426", "CWE-427", "CWE-191", "CWE-476",
    "CWE-401", "CWE-77", "CWE-94", "CWE-502", "CWE-918", "CWE-269",
    "CWE-434", "CWE-306", "CWE-798", "CWE-863", "CWE-611",
    # Veri setinden taranarak eklenen 47 kayıp CWE
    "CWE-480", "CWE-682", "CWE-562", "CWE-597", "CWE-704", "CWE-672", "CWE-127", "CWE-457", 
    "CWE-362", "CWE-197", "CWE-467", "CWE-124", "CWE-188", "CWE-369", "CWE-123", "CWE-176", 
    "CWE-676", "CWE-783", "CWE-768", "CWE-252", "CWE-571", "CWE-788", "CWE-475", "CWE-398", 
    "CWE-377", "CWE-248", "CWE-771", "CWE-685", "CWE-126", "CWE-563", "CWE-686", "CWE-665", 
    "CWE-628", "CWE-807", "CWE-786", "CWE-762", "CWE-121", "CWE-758", "CWE-775", "CWE-114", 
    "CWE-570", "CWE-664", "CWE-829", "CWE-688", "CWE-15", "CWE-477", "CWE-561"
]

CWE_BASE_SCORES: dict[str, float] = {
    "CWE-79": 60.38, "CWE-89": 28.72, "CWE-352": 13.64, "CWE-862": 13.28, "CWE-787": 13.00,
    "CWE-120": 12.00, "CWE-20": 11.00, "CWE-78": 10.50, "CWE-416": 10.00, "CWE-22": 9.80,
    "CWE-125": 9.50, "CWE-119": 9.00, "CWE-190": 8.50, "CWE-200": 8.00, "CWE-122": 7.80,
    "CWE-134": 7.50, "CWE-242": 7.20, "CWE-327": 7.00, "CWE-330": 6.80, "CWE-415": 6.50,
    "CWE-426": 6.20, "CWE-427": 6.00, "CWE-191": 5.80, "CWE-476": 5.50, "CWE-401": 5.20,
    "CWE-77": 8.80, "CWE-94": 8.60, "CWE-502": 8.40, "CWE-918": 8.20, "CWE-269": 7.80,
    "CWE-434": 7.60, "CWE-306": 7.40, "CWE-798": 7.20, "CWE-863": 7.00, "CWE-611": 6.80,
    # Eklenen 47 CWE için CVSS ve MITRE uyarlanmış Base Score Dağılımı
    # Kritik/Yüksek Etki (Bellek Taşması, Yetki, Race Condition vb.) - Skala: 6.50
    "CWE-121": 6.50, "CWE-123": 6.50, "CWE-124": 6.50, "CWE-126": 6.50, "CWE-127": 6.50,
    "CWE-248": 6.50, "CWE-362": 6.50, "CWE-457": 6.50, "CWE-562": 6.50, "CWE-664": 6.50,
    "CWE-672": 6.50, "CWE-762": 6.50, "CWE-786": 6.50, "CWE-788": 6.50, "CWE-807": 6.50,
    "CWE-829": 6.50, "CWE-114": 6.50, "CWE-15": 6.50,

    # Orta Etki (Mantık Hataları, DoS, Kaynak Sızıntıları) - Skala: 4.50
    "CWE-197": 4.50, "CWE-252": 4.50, "CWE-369": 4.50, "CWE-377": 4.50, "CWE-467": 4.50,
    "CWE-475": 4.50, "CWE-480": 4.50, "CWE-665": 4.50, "CWE-676": 4.50, "CWE-682": 4.50,
    "CWE-704": 4.50, "CWE-758": 4.50, "CWE-771": 4.50, "CWE-775": 4.50, "CWE-783": 4.50,

    # Düşük Etki / Kod Kalitesi (Ölü Kod, Gereksiz İşlemler) - Skala: 2.50
    "CWE-176": 2.50, "CWE-188": 2.50, "CWE-398": 2.50, "CWE-477": 2.50, "CWE-561": 2.50,
    "CWE-563": 2.50, "CWE-570": 2.50, "CWE-571": 2.50, "CWE-597": 2.50, "CWE-628": 2.50,
    "CWE-685": 2.50, "CWE-686": 2.50, "CWE-688": 2.50, "CWE-768": 2.50
}

# ── HuggingFace Veri Seti ────────────────────────────────────────────────────
REPO_ID: str = "smart-code-analyzer-team/cpp-vulnerability-dataset"

# ── Boş/Eksik Etiketler İçin Sentinel Değeri ─────────────────────────────────
# Bu değere sahip etiketler loss hesabından otomatik olarak çıkarılır.
# Ortalama atama (mean imputation) YAPILMAZ — model boş veriden öğrenmez.
IGNORE_LABEL: int = -1


class ComplexityClass(IntEnum):
    """Big-O karmaşıklık sınıfları — sabit sıralı enum."""
    O_1       = 0
    O_LOG_N   = 1
    O_N       = 2
    O_N_LOG_N = 3
    O_N2      = 4
    O_N3      = 5
    O_2N      = 6
    O_N_FACT  = 7

    @classmethod
    def from_string(cls, s: str) -> int:
        """Complexity string'ini enum değerine çevirir, bilinmiyorsa IGNORE."""
        _MAP: dict[str, int] = {
            # ── Standart Big-O notasyonu ──
            "O(1)": 0, "1": 0,
            "O(log n)": 1, "O(logn)": 1, "log n": 1, "logn": 1,
            "O(n)": 2, "n": 2,
            "O(n log n)": 3, "O(nlogn)": 3, "n log n": 3, "nlogn": 3,
            "O(n^2)": 4, "O(n²)": 4, "n^2": 4, "n²": 4,
            "O(n^2 log n)": 4, "n^2 log n": 4,  # n^2 grubuna dahil
            "O(n^3)": 5, "O(n³)": 5, "n^3": 5, "n³": 5,
            "O(n^4)": 5, "n^4": 5,              # n^3+ grubuna dahil
            "O(n^5)": 5, "n^5": 5,
            "O(n^6)": 5, "n^6": 5,
            "O(n^7)": 5, "n^7": 5,
            "O(2^n)": 6, "2^n": 6,
            "O(n!)": 7, "n!": 7,
        }
        return _MAP.get(s.strip(), IGNORE_LABEL)


@dataclass
class TrainingConfig:
    """Eğitim hiperparametrelerini tek bir yerde toplar (immutable-ish)."""

    # ── Eğitim ───────────────────────────────────────────────────────────
    epochs:                int   = 30
    batch_size:            int   = 32        # Colab Pro A100 için optimize
    learning_rate:         float = 5e-5
    weight_decay:          float = 1e-4
    label_smoothing:       float = 0.1
    dropout:               float = 0.175
    max_grad_norm:         float = 1.0
    warmup_ratio:          float = 0.1
    accumulation_steps:    int   = 2         # A100 batch=32 → efektif=64
    early_stopping_patience: int = 5

    # ── Mimari ───────────────────────────────────────────────────────────
    max_length:            int   = 512
    memory_bank_size:      int   = 1024
    memory_top_k:          int   = 64
    use_memory:            bool  = True
    use_hierarchical:      bool  = True
    gradient_checkpointing: bool = True
    num_cwe_classes:       int   = 35        # KNOWN_CWES uzunluğu ile aynı
    hierarchical_levels:   int   = 3

    # ── Görev Ağırlıkları ────────────────────────────────────────────────
    vuln_weight:           float = 0.5
    complexity_weight:     float = 0.3
    dataflow_weight:       float = 0.2

    # ── Veri ─────────────────────────────────────────────────────────────
    validation_split:      float = 0.1
    balance_ratio:         float = 0.5
    chunk_overlap:         int   = 64
    num_workers:           int   = 4         # Colab Pro daha fazla CPU
    random_seed:           int   = 42

    def to_dict(self) -> dict[str, Any]:
        """Serileştirme için dict dönüşümü."""
        return {k: v for k, v in self.__dict__.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  §2. Ortam Algılama & Token Yönetimi
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_hf_token() -> str:
    """
    HF_TOKEN'ı 4 farklı kaynaktan sırayla arar:
      1. Ortam değişkeni (HF_TOKEN)
      2. Google Colab Secrets (🔑)
      3. .env dosyası (python-dotenv)
      4. Kullanıcıdan input
    """
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token

    # Colab Secrets
    if _IN_COLAB:
        try:
            from google.colab import userdata  # type: ignore[import-untyped]
            token = userdata.get("HF_TOKEN")
            if token:
                return token
        except Exception:
            pass

    # .env dosyası
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ.get("HF_TOKEN", "")
        if token:
            return token
    except ImportError:
        pass

    # Manuel giriş
    if not token:
        token = input("  🔑 HF_TOKEN girin: ").strip()
    
    if not token:
        raise ValueError("HF_TOKEN boş olamaz. Lütfen geçerli bir token girin.")

    # ── Token'ı Doğrula ──────────────────────────────────────────────────────
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami(token=token)
        logger.info(f"Hugging Face kimliği doğrulandı: {user_info.get('name', 'Bilinmeyen Kullanıcı')}")
    except ImportError:
        pass # huggingface_hub yoksa test edemeyiz, geç
    except Exception as e:
        logger.error(f"❌ HF_TOKEN DOĞRULAMASI BAŞARISIZ! (401 Unauthorized)")
        logger.error(f"Girdiğiniz token Hugging Face tarafından reddedildi.")
        logger.error(f"Lütfen tokenin süresinin dolmadığından ve 'smart-code-analyzer-team' organizasyonuna erişimi olduğundan emin olun.")
        raise ValueError(f"Geçersiz HF_TOKEN: {str(e)}")

    return token


def _resolve_device(requested: Optional[str] = None) -> torch.device:
    """GPU/CPU algılama ve raporlama."""
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU algılandı: %s (%.1f GB VRAM)", name, mem_gb)
    else:
        logger.warning("GPU bulunamadı — CPU kullanılacak (çok yavaş).")

    return device


def _resolve_checkpoint_dir() -> Path:
    """Checkpoint dizinini belirler (Drive varsa Drive, yoksa lokal)."""
    if _IN_COLAB:
        drive_base = Path("/content/drive/MyDrive")
        if drive_base.exists():
            path = drive_base / "smart_code_analyzer/checkpoints"
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Google Drive algılandı → %s", path)
            return path
        else:
            logger.warning("Google Drive klasörü bulunamadı (/content/drive/MyDrive). Model ağırlıkları geçici Colab diskine (lokal) kaydedilecek!")

    path = Path("model/checkpoints")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  §3. Veri İşleme — Dataset & DataLoader
# ══════════════════════════════════════════════════════════════════════════════

def _is_empty_field(value: Any) -> bool:
    """Bir alanın gerçekten boş/null olup olmadığını kontrol eder."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in ("", "''", '""', "{}", "[]", "null", "None")
    return False


def _safe_json_parse(value: Any) -> Any:
    """JSON string'i güvenli şekilde parse eder."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


class CodeAnalysisDataset(Dataset):
    """
    Pipeline'ın ürettiği HF veri setini PyTorch Dataset'ine dönüştürür.

    Özellikler:
    - Boş alanlar IGNORE_LABEL (-1) alır → loss'tan maskelenir
    - Graph-aware sliding-window chunking (512 token, 64 overlap)
    - Context prefix (kaynak sistemi, fonksiyon adı)
    - Automatic padding ve attention mask
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._overlap = chunk_overlap
        self._samples: list[dict[str, Any]] = []

        # ── Hata Mesajını (Warning) Gizleme ──
        # Biz veriyi zaten kendi algoritmamızla (_sliding_window_chunk) 512 token'a 
        # böldüğümüz için, HuggingFace'in verdiği "Token indices sequence length 
        # is longer than..." uyarısı gereksizdir. Bu uyarı tqdm çubuğunu bozar.
        # Uyarıyı kapatmak için tokenizer'ın limitini geçici olarak yükseltiyoruz.
        old_max_length = getattr(self._tokenizer, "model_max_length", 512)
        self._tokenizer.model_max_length = int(1e9)

        # ── İlerleme Çubuğu (Progress Bar) Eklendi ──
        from tqdm.auto import tqdm
        skipped = 0
        
        # tqdm ile sarıldı
        for rec in tqdm(records, desc="Tokenize & Chunk", unit="kayıt"):
            chunks = self._process_record(rec)
            if chunks:
                self._samples.extend(chunks)
            else:
                skipped += 1

        # Orijinal limiti geri yükle
        self._tokenizer.model_max_length = old_max_length

        if skipped > 0:
            logger.warning("Atlanan kayıt (boş kod): %d", skipped)
        logger.info(
            "Dataset hazır: %d kayıt → %d chunk",
            len(records), len(self._samples),
        )

    # ── Public API ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]

    # ── Veri İşleme ──────────────────────────────────────────────────────

    def _process_record(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        """Tek bir HF kaydını tokenize edip chunk'lara böler."""
        code = rec.get("normalized_structure", "")
        if _is_empty_field(code):
            code = rec.get("raw_snippet", "")
        if _is_empty_field(code):
            return []

        labels = self._extract_labels(rec)
        encoding = self._tokenizer(
            str(code),
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        input_ids: list[int] = encoding["input_ids"]

        chunks = self._sliding_window_chunk(input_ids)
        return [self._build_sample(c, labels) for c in chunks]

    def _extract_labels(self, rec: dict[str, Any]) -> dict[str, int]:
        """
        Etiketleri çıkarır — boş alanlar IGNORE_LABEL (-1) alır.

        Bu yaklaşım sayesinde:
        - Boş alanlara ortalama atama YAPILMAZ
        - Model boş veriden yanlış pattern öğrenmez
        - CrossEntropyLoss(ignore_index=-1) otomatik maskeler
        """
        labels: dict[str, int] = {}

        # ── Complexity ───────────────────────────────────────────────────
        raw_comp = rec.get("complexity", "")
        if _is_empty_field(raw_comp):
            labels["complexity"] = IGNORE_LABEL
        else:
            labels["complexity"] = ComplexityClass.from_string(str(raw_comp))

        # ── Data Flow ────────────────────────────────────────────────────
        # ── Data Flow ────────────────────────────────────────────────────
        raw_dfg = rec.get("data_flow_graph", "")
        dfg_edge_count = 0
        if _is_empty_field(raw_dfg):
            labels["has_data_flow"] = IGNORE_LABEL
        else:
            dfg = _safe_json_parse(raw_dfg)
            if dfg:
                labels["has_data_flow"] = 1
                if isinstance(dfg, list):
                    dfg_edge_count = len(dfg)
                elif isinstance(dfg, dict):
                    dfg_edge_count = len(dfg.get("edges", []))
            else:
                labels["has_data_flow"] = 0

        # ── Vulnerability, CWEs, and Risk Score ──────────────────────────
        raw_sec = rec.get("security_context", "")
        # Başlangıç değerleri (Boş veya zafiyetsiz)
        labels["is_vulnerable"] = IGNORE_LABEL
        labels["risk_score"] = 0.0
        # Multi-label CWE vektörü (Örn: 35 CWE için 35 boyutlu 0 veya 1)
        cwe_vec = [0.0] * len(KNOWN_CWES)

        if not _is_empty_field(raw_sec):
            sec = _safe_json_parse(raw_sec)
            if isinstance(sec, dict):
                # Zafiyet durumu
                is_vuln = False
                if "is_vulnerable" in sec:
                    is_vuln = bool(sec["is_vulnerable"])
                
                # CWE Listesi Ayrıştırma (Multi-Label)
                cwe_ids = sec.get("cwe_ids", [])
                if isinstance(cwe_ids, str):
                    cwe_ids = [cwe_ids]  # Tek bir string geldiyse listeye çevir

                # JSON içindeki kirli final_calculated_risk yerine,
                # Eğitim sırasında TEMİZ ve DİNAMİK Risk Skoru hesaplaması yapıyoruz.
                # GitHub yıldızı vs gibi dış etkenlerden arındırılmış saf koda dayalı risk:
                base_score = 0.0
                
                # CWE Vector'ü doldurma ve Kümülatif Base Score (Probabilistic Sum) hesaplama
                if cwe_ids:
                    combined_inv_prob = 1.0
                    valid_cwe_found = False
                    for cwe in cwe_ids:
                        cwe_str = str(cwe).strip()
                        if cwe_str in KNOWN_CWES:
                            idx = KNOWN_CWES.index(cwe_str)
                            cwe_vec[idx] = 1.0
                            # Base Score
                            score = CWE_BASE_SCORES.get(cwe_str, 5.0)
                            combined_inv_prob *= (1.0 - (score / 100.0))
                            valid_cwe_found = True
                        else:
                            # Listede olmayan CWE için düşük sabit bir risk
                            combined_inv_prob *= (1.0 - (5.0 / 100.0))
                            valid_cwe_found = True
                            
                    if valid_cwe_found:
                        base_score = 100.0 * (1.0 - combined_inv_prob)
                    else:
                        base_score = 5.0
                elif is_vuln:
                    # CWE belirsiz ama zafiyetli kod
                    base_score = 5.0
                    
                # Güvenlik açığı tespiti mantığı
                if is_vuln or cwe_ids or base_score > 0.0:
                    labels["is_vulnerable"] = 1
                else:
                    labels["is_vulnerable"] = 0

                # DataFlow (Veri Akışı) kenar/düğüm yoğunluğuna dayalı "Attack Surface" çarpanı
                # Simülasyon analizine göre: 0 akış (0.50), ortalama 15 akış (1.00)
                attack_surface = 0.5 + 0.5 * min(1.0, dfg_edge_count / 15.0)
                
                clean_risk_score = base_score * attack_surface

                labels["risk_score"] = clean_risk_score

        labels["cwe_vector"] = cwe_vec

        return labels

    def _sliding_window_chunk(self, ids: list[int]) -> list[list[int]]:
        """Kayan pencere ile chunking (semantik sınır aramalı)."""
        if len(ids) <= self._max_length:
            return [ids]

        # ── HIZLANDIRMA: Milyonlarca decode() çağrısı yerine ID eşleştirme ──
        if getattr(self, "_split_ids", None) is None:
            # "}", ";" ve "\n" karakterlerinin token ID'lerini bir kez hesapla
            split_chars = ["}", ";", "\n"]
            split_ids = set()
            for c in split_chars:
                encoded = self._tokenizer.encode(c, add_special_tokens=False)
                if encoded:
                    split_ids.add(encoded[0])
            self._split_ids = split_ids

        chunks: list[list[int]] = []
        start = 0
        while start < len(ids):
            end = min(start + self._max_length, len(ids))

            # Semantik sınır ara (fonksiyon/blok sonu) - 1000x daha hızlı
            if end < len(ids):
                best = end
                search_from = max(end - 64, start)
                for i in range(end - 1, search_from, -1):
                    if ids[i] in self._split_ids:
                        best = i + 1
                        break
                end = best

            chunks.append(ids[start:end])
            
            # Sonsuz döngüyü (infinite loop) engelleyen kritik düzeltme:
            if end >= len(ids):
                break
                
            start = end - self._overlap
        return chunks

    def _build_sample(
        self, input_ids: list[int], labels: dict[str, int]
    ) -> dict[str, Any]:
        """Padding, attention mask ve label tensörleri oluşturur."""
        seq_len = len(input_ids)
        pad_len = self._max_length - seq_len

        if pad_len > 0:
            pad_id = self._tokenizer.pad_token_id or 0
            input_ids = input_ids + [pad_id] * pad_len
            attn_mask = [1] * seq_len + [0] * pad_len
        else:
            input_ids = input_ids[: self._max_length]
            attn_mask = [1] * self._max_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": labels,
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch collation — label tensörlerini birleştirir."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": {
            "is_vulnerable": torch.tensor(
                [b["labels"]["is_vulnerable"] for b in batch], dtype=torch.long
            ),
            "complexity": torch.tensor(
                [b["labels"]["complexity"] for b in batch], dtype=torch.long
            ),
            "has_data_flow": torch.tensor(
                [b["labels"]["has_data_flow"] for b in batch], dtype=torch.long
            ),
            "cwe_vector": torch.tensor(
                [b["labels"]["cwe_vector"] for b in batch], dtype=torch.float
            ),
            "risk_score": torch.tensor(
                [b["labels"]["risk_score"] for b in batch], dtype=torch.float
            ),
        },
    }


def _print_data_quality_report(records: list[dict[str, Any]]) -> None:
    """Veri kalitesi raporunu terminale yazdırır."""
    n = max(len(records), 1)
    fields = [
        "raw_snippet", "normalized_structure", "ast_metadata",
        "security_context", "data_flow_graph", "complexity",
        "code_hash", "source_system", "nl_alignment",
    ]

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                  VERİ KALİTESİ RAPORU                   │")
    print("├─────────────────────────────┬──────────┬─────────────────┤")
    print("│ Alan                        │   Boş    │     Oran        │")
    print("├─────────────────────────────┼──────────┼─────────────────┤")

    for f in fields:
        empty = sum(1 for r in records if _is_empty_field(r.get(f, "")))
        pct = empty / n * 100
        icon = "✅" if pct < 5 else "⚠️" if pct < 50 else "❌"
        print(f"│ {icon} {f:<25} │ {empty:>8} │ {pct:>13.1f}%  │")

    print("├─────────────────────────────┴──────────┴─────────────────┤")
    print("│ ℹ  Boş alanlar IGNORE (-1) → loss'tan maskelenir.       │")
    print("└─────────────────────────────────────────────────────────┘\n")


def build_data_loaders(
    tokenizer: Any,
    cfg: TrainingConfig,
    hf_token: str,
) -> tuple[DataLoader, DataLoader]:
    """
    HF'den veri setini yükler → kalite raporu → dengele → DataLoader oluşturur.

    Returns:
        (train_loader, test_loader)
    """
    from datasets import load_dataset as hf_load_dataset

    logger.info("HF veri seti yükleniyor: %s", REPO_ID)
    ds = hf_load_dataset(REPO_ID, split="train", token=hf_token)
    records: list[dict[str, Any]] = [dict(row) for row in ds]
    logger.info("Toplam %d kayıt yüklendi.", len(records))

    _print_data_quality_report(records)

    # ── Shuffle & Split ──────────────────────────────────────────────────
    rng = random.Random(cfg.random_seed)
    rng.shuffle(records)
    split_idx = int(len(records) * (1 - cfg.validation_split))
    train_recs = records[:split_idx]
    test_recs = records[split_idx:]

    # ── Dengeleme İptali (Artık Class-Weighted Loss kullanılıyor) ────
    # train_recs = _balance_classes(train_recs, cfg.balance_ratio, rng)
    logger.info("Oversampling devre dışı (Class-Weighted Loss aktif). Train: %d", len(train_recs))

    # ── Dataset Oluştur ──────────────────────────────────────────────────
    train_ds = CodeAnalysisDataset(train_recs, tokenizer, cfg.max_length, cfg.chunk_overlap)
    test_ds = CodeAnalysisDataset(test_recs, tokenizer, cfg.max_length, cfg.chunk_overlap)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_batch,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_batch,
        pin_memory=True,
    )

    logger.info("Train: %d chunk (%d batch), Test: %d chunk (%d batch)",
                len(train_ds), len(train_loader), len(test_ds), len(test_loader))
    return train_loader, test_loader


def _balance_classes(
    records: list[dict[str, Any]],
    target_ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Oversampling iptal edildi. Artık Class-Weighted Loss kullanılıyor.
    Veri seti kopyalanmaz, olduğu gibi döndürülür.
    """
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  §4. Model Mimarisi — GraphCodeBERT + Memory + Hierarchical + Multi-Task
# ══════════════════════════════════════════════════════════════════════════════


class MemoryBank(nn.Module):
    """
    Kayan pencere yaklaşımında önceki pencerelerin yüksek dikkat ağırlıklı
    token'larını biriktiren ve cross-attention ile sorgulayan bellek modülü.

    Bu mekanizma sayesinde 512 tokenlık pencere sınırı aşılır ve model
    1000+ token uzaklıktaki ilişkili noktaları bağlayabilir.

    Çalışma prensibi:
        1. Her encoder çıktısından en önemli K token seçilir (attention-based)
        2. FIFO belleğe eklenir (gradient akışı yok)
        3. Sonraki pencere bu belleği cross-attention ile sorgular

    Args:
        hidden_size: Encoder'ın hidden boyutu (GraphCodeBERT: 768)
        memory_size: Bellekteki maksimum token sayısı
        top_k:       Her güncellemede belleğe eklenecek token sayısı
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 1024,
        top_k: int = 64,
    ) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self._memory_size = memory_size
        self._top_k = top_k

        # Cross-attention projeksiyon katmanları
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Bellek buffer'ı (gradient yok — sadece bilgi taşıma)
        self.register_buffer(
            "_memory", torch.zeros(1, memory_size, hidden_size)
        )
        self.register_buffer("_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> None:
        """
        Yüksek dikkat ağırlıklı token'ları belleğe ekler.
        Gradient akışı yoktur — sadece bilgi taşıma mekanizmasıdır.

        Args:
            hidden_states:    (batch, seq_len, hidden) — encoder çıktısı
            attention_weights: (batch, heads, seq, seq) — son katman dikkat ağırlıkları
        """
        # Attention önemlilik skoru: tüm head ve pozisyonların ortalaması
        if attention_weights.dim() == 4:
            importance = attention_weights.mean(dim=(1, 2))  # (batch, seq)
        else:
            importance = attention_weights.mean(dim=-1)

        k = min(self._top_k, hidden_states.size(1))
        _, top_indices = importance.topk(k, dim=-1)

        # Batch'teki her örnek için belleği güncelle
        for b in range(hidden_states.size(0)):
            selected = hidden_states[b, top_indices[b]]  # (k, hidden)
            current = self._count.item()
            new_count = min(current + k, self._memory_size)

            if current + k <= self._memory_size:
                self._memory[0, current : current + k] = selected
            else:
                # FIFO: eski token'ları sola kaydır, yenileri sona ekle
                shift = current + k - self._memory_size
                self._memory[0, : -shift] = self._memory[0, shift:].clone()
                self._memory[0, -k:] = selected

            self._count.fill_(new_count)

    def query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Mevcut hidden state'leri cross-attention ile bellek sorgulaması.

        Args:
            hidden_states: (batch, seq_len, hidden)

        Returns:
            (batch, seq_len, hidden) — bellek bilgisi ile zenginleştirilmiş
        """
        mem_len = self._count.item()
        if mem_len == 0:
            return hidden_states

        batch_size = hidden_states.size(0)
        memory = self._memory[:, :mem_len].expand(batch_size, -1, -1)

        # Cross-attention: Q=current, K=memory, V=memory
        Q = self.W_q(hidden_states)
        K = self.W_k(memory)
        V = self.W_v(memory)

        scale = math.sqrt(self._hidden_size)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, V)

        # Residual connection + LayerNorm
        return self.layer_norm(hidden_states + self.W_o(context))

    def reset(self) -> None:
        """Belleği sıfırlar (yeni dosya/örnek işlerken)."""
        self._memory.zero_()
        self._count.zero_()


class HierarchicalAttention(nn.Module):
    """
    Token → Statement → Function seviyesinde hiyerarşik dikkat.

    Her seviyede ayrı bir attention havuzu token'ları ağırlıklandırır.
    Sonuç merge edilerek kodun hem mikro hem makro yapısı yakalanır.

    Args:
        hidden_size: Encoder hidden boyutu
        num_levels:  Hiyerarşi seviyesi sayısı (varsayılan: 3)
    """

    def __init__(self, hidden_size: int, num_levels: int = 3) -> None:
        super().__init__()
        self._num_levels = num_levels

        self.level_pools = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, 1, bias=False),
            )
            for _ in range(num_levels)
        ])
        self.merge = nn.Linear(hidden_size * num_levels, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden)

        Returns:
            (batch, hidden) — hiyerarşik olarak birleştirilmiş temsil
        """
        level_outputs: list[torch.Tensor] = []
        for pool in self.level_pools:
            scores = pool(hidden_states).squeeze(-1)        # (batch, seq)
            weights = F.softmax(scores, dim=-1)              # (batch, seq)
            pooled = torch.bmm(weights.unsqueeze(1), hidden_states)  # (batch, 1, H)
            level_outputs.append(pooled.squeeze(1))          # (batch, H)

        merged = torch.cat(level_outputs, dim=-1)  # (batch, H * levels)
        return self.layer_norm(self.merge(merged))  # (batch, H)


class TaskHead(nn.Module):
    """
    Görev başlığı (classification head) — tekrarı önlemek için fabrika.

    Args:
        input_size:  Girdi boyutu
        output_size: Çıktı sınıf sayısı
        dropout:     Dropout oranı
        depth:       Katman derinliği (1=sığ, 2=derin)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.175,
        depth: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        if depth >= 2:
            layers.extend([
                nn.Linear(input_size, input_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_size // 2, input_size // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_size // 4, output_size),
            ])
        else:
            layers.extend([
                nn.Linear(input_size, input_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_size // 2, output_size),
            ])

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SmartCodeAnalyzerModel(nn.Module):
    """
    Multi-Task Code Analysis Transformer.

    Mimari:
        GraphCodeBERT Encoder
            ↓
        Memory Bank (uzun bağımlılık)
            ↓
        Hierarchical Attention (kod yapısı)
            ↓
        ┌──────────┬──────────────┬─────────────┐
        │ VulnHead │ ComplexHead   │ DataFlowHead│
        │ (CWE ID) │ (Big-O)      │ (DFG var/yok)│
        └──────────┴──────────────┴─────────────┘

    Args:
        cfg: TrainingConfig nesnesi
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        from transformers import AutoModel, AutoConfig  # type: ignore[import-untyped]

        # ── GraphCodeBERT Encoder ────────────────────────────────────────
        model_config = AutoConfig.from_pretrained(BASE_MODEL)
        
        # Hata Çözümü: SDPA (Attention) ve Gradient Checkpointing Uyarılarını Giderme
        # Memory Bank özelliğinin çalışması için 'output_attentions=True' dönmesi şarttır.
        # Yeni nesil SDPA (FlashAttention) bellek tasarrufu için dikkat matrisini VRAM'de tutmadığından 
        # output_attentions desteklemez. Bu yüzden modeli klasik 'eager' attention'a zorlamalıyız.
        self.encoder = AutoModel.from_pretrained(
            BASE_MODEL,
            config=model_config,
            ignore_mismatched_sizes=True,
            attn_implementation="eager"
        )
        hidden = model_config.hidden_size  # 768

        if cfg.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing aktif (VRAM tasarrufu).")

        # ── Memory Bank ──────────────────────────────────────────────────
        self._use_memory = cfg.use_memory
        if cfg.use_memory:
            self.memory_bank = MemoryBank(
                hidden, cfg.memory_bank_size, cfg.memory_top_k
            )

        # ── Hierarchical Attention ───────────────────────────────────────
        self._use_hier = cfg.use_hierarchical
        if cfg.use_hierarchical:
            self.hier_attn = HierarchicalAttention(
                hidden, cfg.hierarchical_levels
            )

        # ── Task Heads ───────────────────────────────────────────────────
        # 1. Binary Zafiyet Tespiti (Temiz mi Kirli mi?)
        self.vuln_head = TaskHead(
            hidden, 2, cfg.dropout, depth=2
        )
        # 2. Multi-Label CWE Tespiti (Hangi zafiyetler var?)
        self.cwe_head = TaskHead(
            hidden, len(KNOWN_CWES), cfg.dropout, depth=2
        )
        # 3. Risk Skoru Regresyonu (0.0 - 100.0)
        self.risk_head = TaskHead(
            hidden, 1, cfg.dropout, depth=1
        )
        # 4. Karmaşıklık Tespiti
        self.comp_head = TaskHead(
            hidden, len(ComplexityClass), cfg.dropout, depth=1
        )  # 8 sınıf
        # 5. Veri Akışı Tespiti
        self.df_head = TaskHead(
            hidden, 2, cfg.dropout, depth=1
        )  # binary

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        update_memory: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            {
                "vulnerability_logits": (B, num_cwe+1),
                "complexity_logits":    (B, 8),
                "dataflow_logits":      (B, 2),
            }
        """
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self._use_memory,
        )
        hidden = encoder_out.last_hidden_state  # (B, seq, 768)

        # Memory Bank: sorgula → güncelle
        if self._use_memory:
            hidden = self.memory_bank.query(hidden)
            if update_memory and encoder_out.attentions:
                self.memory_bank.update(
                    hidden.detach(), encoder_out.attentions[-1].detach()
                )

        # Pooling
        if self._use_hier:
            pooled = self.hier_attn(hidden)  # (B, 768)
        else:
            pooled = hidden[:, 0]  # CLS token

        pooled = self.dropout(pooled)

        return {
            "vulnerability_logits": self.vuln_head(pooled),
            "cwe_logits": self.cwe_head(pooled),
            "risk_pred": self.risk_head(pooled).squeeze(-1),
            "complexity_logits": self.comp_head(pooled),
            "dataflow_logits": self.df_head(pooled),
        }

    def reset_memory(self) -> None:
        """Memory bank'ı sıfırlar."""
        if self._use_memory:
            self.memory_bank.reset()


class MaskedMultiTaskLoss(nn.Module):
    """
    Multi-task loss — IGNORE_LABEL maskelemeli.

    Boş etiketli örnekler ilgili görevin loss'undan otomatik çıkarılır.
    Ortalama atama (mean imputation) YAPILMAZ.

    Args:
        cfg: TrainingConfig
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.w_v = cfg.vuln_weight
        self.w_c = cfg.complexity_weight
        self.w_d = cfg.dataflow_weight

        ls = cfg.label_smoothing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Sınıf Ağırlıkları (Class-Weighted Loss) ──
        # Hesaplamalar 69,299 kayıtlık tam veri seti üzerinden yapılmıştır.
        # w = toplam_kayit / (sinif_sayisi * sinif_kayit_sayisi)

        # Vulnerability: clean(0)=1.347, vuln(1)=0.795
        w_vuln = torch.tensor([1.3468, 0.7952], dtype=torch.float32, device=device)
        self.loss_vuln = nn.CrossEntropyLoss(
            weight=w_vuln, label_smoothing=ls, ignore_index=IGNORE_LABEL
        )
        
        # CWE Multi-Label Loss
        self.loss_cwe = nn.BCEWithLogitsLoss()
        
        # Risk Score Regression Loss
        # HuberLoss, MSELoss'a göre çok büyük hatalarda L1 gibi davranarak (doğrusal)
        # patlayan gradyanları ve float16 (65504 limit) taşmalarını kökünden engeller.
        self.loss_risk = nn.HuberLoss(delta=1.0)

        # Complexity: O(1), O(log n), O(n), O(n log n), O(n²), O(n³+)
        # Enum sırasına göre: 0, 1, 2, 3, 4, 5
        # 6 ve 7 (O(2^n), O(n!)) veri setinde neredeyse yok, ağırlıkları 1.0 bırakıyoruz.
        w_comp = torch.tensor([
            0.2242,   # 0: O(1)      - 51K kayıt (ağırlık düşürüldü)
            2.9934,   # 1: O(log n)  - 3.8K kayıt (ağırlık artırıldı)
            1.0808,   # 2: O(n)      - 10.6K kayıt
            7.5474,   # 3: O(n log n)- 1.5K kayıt
            9.4795,   # 4: O(n²)     - 1.2K kayıt
            23.2382,  # 5: O(n³+)    - 494 kayıt (max ağırlık)
            1.0,      # 6: O(2^n)    - Yok
            1.0       # 7: O(n!)     - Yok
        ], dtype=torch.float32, device=device)
        self.loss_comp = nn.CrossEntropyLoss(
            weight=w_comp, label_smoothing=ls, ignore_index=IGNORE_LABEL
        )

        # DataFlow: yok(0)=5.203, var(1)=0.553
        w_df = torch.tensor([5.2026, 0.5532], dtype=torch.float32, device=device)
        self.loss_df = nn.CrossEntropyLoss(
            weight=w_df, label_smoothing=ls, ignore_index=IGNORE_LABEL
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        lv = torch.nan_to_num(self.loss_vuln(outputs["vulnerability_logits"], labels["is_vulnerable"]), nan=0.0)
        
        # CWE Loss (sadece zafiyet olan örnekler veya hepsi üzerinden hesaplanabilir, 
        # modelin temiz kodda sıfır üretmesini teşvik etmek için hepsini katıyoruz)
        # BCEWithLogitsLoss float tensor ister
        cwe_target = torch.stack(labels["cwe_vector"]).T.float().to(outputs["cwe_logits"].device) if isinstance(labels["cwe_vector"], list) else labels["cwe_vector"].float()
        lcwe = torch.nan_to_num(self.loss_cwe(outputs["cwe_logits"], cwe_target), nan=0.0)
        
        # Risk Skoru Loss
        risk_target = labels["risk_score"].float()
        lrisk = torch.nan_to_num(self.loss_risk(outputs["risk_pred"], risk_target), nan=0.0)
        
        lc = torch.nan_to_num(self.loss_comp(outputs["complexity_logits"], labels["complexity"]), nan=0.0)
        ld = torch.nan_to_num(self.loss_df(outputs["dataflow_logits"], labels["has_data_flow"]), nan=0.0)

        total = self.w_v * lv + 0.5 * lcwe + 0.1 * lrisk + self.w_c * lc + self.w_d * ld
        return {
            "total_loss": total,
            "vuln_loss": lv,
            "cwe_loss": lcwe,
            "risk_loss": lrisk,
            "complexity_loss": lc,
            "dataflow_loss": ld,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  §5. Terminal Görselleştirme
# ══════════════════════════════════════════════════════════════════════════════

def _print_epoch_table(
    epoch: int,
    total_epochs: int,
    train: dict[str, float],
    val: dict[str, float],
    elapsed: float,
    best_loss: float,
    patience: int,
    max_patience: int,
) -> None:
    """Her epoch sonunda formatlanmış sonuç tablosu."""
    header = f"EPOCH {epoch + 1}/{total_epochs} SONUÇLARI"
    W = 64

    print()
    print(f"┌{'─' * W}┐")
    print(f"│{header:^{W}}│")
    print(f"├{'─' * 30}┬{'─' * 16}┬{'─' * 16}┤")
    print(f"│ {'Metrik':<28} │ {'Train':>14} │ {'Eval':>14} │")
    print(f"├{'─' * 30}┼{'─' * 16}┼{'─' * 16}┤")

    rows = [
        ("Total Loss",       train.get("total_loss", 0),      val.get("total_loss", 0)),
        ("Vulnerability Loss", train.get("vuln_loss", 0),      val.get("vuln_loss", 0)),
        ("CWE Multi-Label Loss", train.get("cwe_loss", 0),     val.get("cwe_loss", 0)),
        ("Risk Score Loss",  train.get("risk_loss", 0),        val.get("risk_loss", 0)),
        ("Complexity Loss",  train.get("complexity_loss", 0),  val.get("complexity_loss", 0)),
        ("DataFlow Loss",    train.get("dataflow_loss", 0),    val.get("dataflow_loss", 0)),
    ]
    for name, t, v in rows:
        print(f"│ {name:<28} │ {t:>14.6f} │ {v:>14.6f} │")

    print(f"├{'─' * 30}┼{'─' * 16}┼{'─' * 16}┤")
    print(f"│ {'Vuln Accuracy':<28} │ {'—':>14} │ {val.get('vuln_acc', 0):>13.2%} │")
    print(f"│ {'CWE F1-Score (Macro)':<28} │ {'—':>14} │ {val.get('cwe_f1', 0):>13.2%} │")
    print(f"│ {'Complexity Accuracy':<28} │ {'—':>14} │ {val.get('comp_acc', 0):>13.2%} │")
    print(f"├{'─' * 30}┴{'─' * 16}┴{'─' * 16}┤")
    info = f"⏱ {elapsed:.0f}s │ Best: {best_loss:.4f} │ Patience: {patience}/{max_patience}"
    print(f"│ {info:<{W}} │")
    print(f"└{'─' * W}┘")


def _print_accuracy_bars(metrics: dict[str, float]) -> None:
    """Görev bazlı accuracy bar chart."""
    W = 40
    print()
    print("  📈 Accuracy Dashboard")
    print(f"  {'─' * 52}")

    items = [
        ("Vulnerability", metrics.get("vuln_acc", 0)),
        ("CWE F1-Score", metrics.get("cwe_f1", 0)),
        ("Complexity", metrics.get("comp_acc", 0)),
        ("Data Flow", metrics.get("df_acc", 0)),
    ]
    for name, val in items:
        filled = int(val * W)
        icon = "🟢" if val >= 0.7 else "🟡" if val >= 0.4 else "🔴"
        bar = "█" * filled + "░" * (W - filled)
        print(f"  {icon} {name:<18} │{bar}│ {val:.2%}")
    print(f"  {'─' * 52}")


def _print_loss_histogram(history: list[dict[str, Any]]) -> None:
    """Son epoch'ların loss dağılımını ASCII histogram olarak gösterir."""
    if len(history) < 2:
        return

    recent = history[-min(15, len(history)):]
    max_loss = max(h["train_loss"] for h in recent) or 1.0
    W = 40

    print()
    print("  📊 Loss Eğrisi (Son Epoch'lar)")
    print(f"  {'─' * 52}")
    for h in recent:
        e = h["epoch"]
        tl, el = h["train_loss"], h["eval_loss"]
        tb = int(min(tl / max_loss, 1.0) * W)
        eb = int(min(el / max_loss, 1.0) * W)
        print(f"  E{e:>2} T │{'█' * tb}{'░' * (W - tb)}│ {tl:.4f}")
        print(f"      E │{'▓' * eb}{'░' * (W - eb)}│ {el:.4f}")
    print(f"  {'─' * 52}")
    print("  █ Train  ▓ Eval")


def _save_training_curves(
    history: list[dict[str, Any]], save_dir: Path
) -> None:
    """Matplotlib ile eğitim grafiklerini oluşturur ve kaydeder."""
    if len(history) < 2:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Smart Code Analyzer — Eğitim Metrikleri",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: Loss Eğrisi ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history],
            "o-", color="#2196F3", label="Train", linewidth=2, markersize=4)
    ax.plot(epochs, [h["eval_loss"] for h in history],
            "s-", color="#F44336", label="Eval", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Eğrisi")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Accuracy Eğrisi ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, [h.get("vuln_acc", 0) for h in history],
            "o-", color="#4CAF50", label="Vulnerability", linewidth=2, markersize=4)
    ax.plot(epochs, [h.get("comp_acc", 0) for h in history],
            "s-", color="#FF9800", label="Complexity", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Eğrisi")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Görev Bazlı Loss ────────────────────────────────────────
    ax = axes[2]
    last = history[-1]
    names = ["Vulnerability", "Complexity", "DataFlow"]
    values = [
        last.get("eval_vuln_loss", 0),
        last.get("eval_comp_loss", 0),
        last.get("eval_df_loss", 0),
    ]
    colors = ["#F44336", "#FF9800", "#2196F3"]
    ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Loss")
    ax.set_title(f"Görev Bazlı Loss (Epoch {last['epoch']})")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = save_dir / "training_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Colab'da inline göster
    if _IN_COLAB:
        try:
            from IPython.display import display, Image as IPImage
            display(IPImage(filename=str(plot_path)))
        except Exception:
            pass

    logger.info("Grafik kaydedildi: %s", plot_path)


# ══════════════════════════════════════════════════════════════════════════════
#  §6. Değerlendirme
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model: SmartCodeAnalyzerModel,
    loader: DataLoader,
    criterion: MaskedMultiTaskLoss,
    device: torch.device,
) -> dict[str, float]:
    """
    Modeli test seti üzerinde değerlendirir.

    Returns:
        total_loss, vuln_loss, complexity_loss, dataflow_loss,
        vuln_acc, comp_acc
    """
    model.eval()
    model.reset_memory()

    loss_accum = {
        "total_loss": 0.0, "vuln_loss": 0.0, "cwe_loss": 0.0, "risk_loss": 0.0,
        "complexity_loss": 0.0, "dataflow_loss": 0.0
    }
    correct_vuln = correct_comp = correct_df = total_samples = total_df = 0
    cwe_true_list = []
    cwe_pred_list = []

    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labs = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

        outputs = model(ids, mask, update_memory=False)
        losses = criterion(outputs, labs)

        for k in loss_accum:
            loss_accum[k] += losses[k].item()

        # Accuracy (sadece maskelenmemiş örnekler)
        v_mask = labs["is_vulnerable"] != IGNORE_LABEL
        if v_mask.any():
            v_pred = outputs["vulnerability_logits"][v_mask].argmax(-1)
            correct_vuln += (v_pred == labs["is_vulnerable"][v_mask]).sum().item()

        c_mask = labs["complexity"] != IGNORE_LABEL
        if c_mask.any():
            c_pred = outputs["complexity_logits"][c_mask].argmax(-1)
            correct_comp += (c_pred == labs["complexity"][c_mask]).sum().item()

        # CWE F1-Score Tracking
        cwe_t = torch.stack(labs["cwe_vector"]).T if isinstance(labs["cwe_vector"], list) else labs["cwe_vector"]
        cwe_true_list.append(cwe_t.cpu())
        cwe_pred_list.append((torch.sigmoid(outputs["cwe_logits"]) > 0.5).float().cpu())

        d_mask = labs["has_data_flow"] != IGNORE_LABEL
        if d_mask.any():
            d_pred = outputs["dataflow_logits"][d_mask].argmax(-1)
            correct_df += (d_pred == labs["has_data_flow"][d_mask]).sum().item()
            total_df += d_mask.sum().item()
        total_samples += ids.size(0)

    n = max(len(loader), 1)
    metrics = {k: v / n for k, v in loss_accum.items()}
    metrics["vuln_acc"] = correct_vuln / max(total_samples, 1)
    metrics["comp_acc"] = correct_comp / max(total_samples, 1)
    metrics["df_acc"] = correct_df / max(total_df, 1)    
    # Calculate Macro F1 for CWE Multi-Label
    cwe_true_all = torch.cat(cwe_true_list)
    cwe_pred_all = torch.cat(cwe_pred_list)
    tp = (cwe_pred_all * cwe_true_all).sum(dim=0)
    fp = (cwe_pred_all * (1 - cwe_true_all)).sum(dim=0)
    fn = ((1 - cwe_pred_all) * cwe_true_all).sum(dim=0)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    metrics["cwe_f1"] = f1.mean().item()

    model.train()
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  §7. Ana Eğitim Döngüsü
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: TrainingConfig, device: torch.device, ckpt_dir: Path, hf_token: str) -> list[dict[str, Any]]:
    """
    Ana eğitim fonksiyonu.

    Args:
        cfg:       Hiperparametreler
        device:    Eğitim cihazı (cuda/cpu)
        ckpt_dir:  Checkpoint dizini
        hf_token:  HuggingFace API token

    Returns:
        Eğitim geçmişi (list of dicts)
    """
    from transformers import AutoTokenizer  # type: ignore[import-untyped]
    from torch.amp import GradScaler, autocast  # PyTorch 2.x yeni API

    # ── A100 BF16 Desteği ────────────────────────────────────────────────
    # A100 GPU, BF16 (bfloat16) destekler — FP16'dan daha stabil eğitim.
    # BF16 daha geniş dinamik aralığa sahiptir, gradient underflow riski düşer.
    use_amp = device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        logger.info("BFloat16 destekleniyor → BF16 aktif (A100 optimizasyon).")
    elif use_amp:
        amp_dtype = torch.float16
        logger.info("FP16 aktif.")
    else:
        amp_dtype = torch.float32

    # ── Banner ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print("  🚀 SMART CODE ANALYZER — MODEL EĞİTİMİ")
    print(f"{'═' * 64}")
    print(f"  📌 Mimari      : {BASE_MODEL}")
    print(f"  📌 Device      : {device}")
    print(f"  📌 Epochs      : {cfg.epochs}")
    print(f"  📌 Batch Size  : {cfg.batch_size} (×{cfg.accumulation_steps} accum = {cfg.batch_size * cfg.accumulation_steps} efektif)")
    print(f"  📌 Learning Rate: {cfg.learning_rate}")
    print(f"  📌 AMP Dtype   : {amp_dtype}")
    print(f"  📌 Memory Bank : {cfg.use_memory} (size={cfg.memory_bank_size})")
    print(f"  📌 Hierarchical: {cfg.use_hierarchical} ({cfg.hierarchical_levels} levels)")
    print(f"  📌 Checkpoint  : {ckpt_dir}")
    print(f"{'═' * 64}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # ── DataLoader ───────────────────────────────────────────────────────
    train_loader, test_loader = build_data_loaders(tokenizer, cfg, hf_token)

    # ── Model ────────────────────────────────────────────────────────────
    model = SmartCodeAnalyzerModel(cfg).to(device)
    criterion = MaskedMultiTaskLoss(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parametreler: %s toplam, %s eğitilebilir",
                f"{total_params:,}", f"{trainable:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    total_steps = max(len(train_loader) * cfg.epochs // cfg.accumulation_steps, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=cfg.warmup_ratio,
        anneal_strategy="cos",
    )
    # GradScaler: FP16'da gerekli, BF16'da gereksiz ama zararsız
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ── State ────────────────────────────────────────────────────────────
    best_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, Any]] = []

    # ── Resume from Checkpoint ───────────────────────────────────────────
    ckpt_path = ckpt_dir / "last_checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        logger.info("Checkpoint yükleniyor: %s", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0)
        history = ckpt.get("history", [])
        logger.info("Epoch %d'den devam ediliyor (best_loss=%.4f).", start_epoch, best_loss)

    # ══════════════════════════════════════════════════════════════════════
    #  EĞİTİM DÖNGÜSÜ
    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        model.reset_memory()
        optimizer.zero_grad(set_to_none=True)

        epoch_losses: dict[str, float] = {
            "total_loss": 0.0, "vuln_loss": 0.0, "cwe_loss": 0.0, "risk_loss": 0.0,
            "complexity_loss": 0.0, "dataflow_loss": 0.0,
        }
        t0 = time.time()

        # ── Progress Bar ─────────────────────────────────────────────────
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=110,
            leave=True,
        )

        for step, batch in pbar:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labs = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(ids, mask)
                losses = criterion(outputs, labs)
                scaled_loss = losses["total_loss"] / cfg.accumulation_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Metrikleri biriktir
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            # Canlı güncelleme
            n_steps = step + 1
            pbar.set_postfix({
                "loss": f"{epoch_losses['total_loss'] / n_steps:.4f}",
                "v": f"{epoch_losses['vuln_loss'] / n_steps:.3f}",
                "c": f"{epoch_losses['complexity_loss'] / n_steps:.3f}",
                "d": f"{epoch_losses['dataflow_loss'] / n_steps:.3f}",
            })

        pbar.close()

        # Epoch ortalamaları
        n_batches = max(len(train_loader), 1)
        train_metrics = {k: v / n_batches for k, v in epoch_losses.items()}

        # ── Değerlendirme ────────────────────────────────────────────────
        eval_metrics = evaluate_model(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        # ── Görselleştirme ───────────────────────────────────────────────
        _print_epoch_table(
            epoch, cfg.epochs, train_metrics, eval_metrics,
            elapsed, best_loss, patience_counter, cfg.early_stopping_patience,
        )
        _print_accuracy_bars(eval_metrics)

        # ── History ──────────────────────────────────────────────────────
        entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["total_loss"], 6),
            "eval_loss": round(eval_metrics["total_loss"], 6),
            "vuln_acc": round(eval_metrics.get("vuln_acc", 0), 4),
            "comp_acc": round(eval_metrics.get("comp_acc", 0), 4),
            "eval_vuln_loss": round(eval_metrics.get("vuln_loss", 0), 6),
            "eval_comp_loss": round(eval_metrics.get("complexity_loss", 0), 6),
            "eval_df_loss": round(eval_metrics.get("dataflow_loss", 0), 6),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": round(elapsed, 1),
        }
        history.append(entry)

        _print_loss_histogram(history)
        _save_training_curves(history, ckpt_dir)

        # ── Checkpoint ───────────────────────────────────────────────────
        ckpt_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "history": history,
            "config": cfg.to_dict(),
        }
        torch.save(ckpt_data, str(ckpt_path))
        
        # Ayrıca her epoch'u ayrı bir dosya olarak kaydet
        epoch_path = ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(ckpt_data, str(epoch_path))

        eval_loss = eval_metrics["total_loss"]
        if eval_loss < best_loss:
            best_loss = eval_loss
            patience_counter = 0
            best_path = ckpt_dir / "best_model.pt"
            torch.save(ckpt_data, str(best_path))
            print(f"\n  ⭐ YENİ EN İYİ MODEL! Loss: {best_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            logger.info("🛑 Early Stopping: %d epoch iyileşme yok.", patience_counter)
            break

        print()  # Epoch arası boşluk

    # ══════════════════════════════════════════════════════════════════════
    #  FİNAL RAPORU
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 64}")
    print("  ✅ EĞİTİM TAMAMLANDI!")
    print(f"{'═' * 64}")
    print(f"  📊 Toplam epoch      : {len(history)}")
    print(f"  🏆 En iyi eval loss  : {best_loss:.6f}")
    print(f"  📁 Checkpoint dizini : {ckpt_dir}")
    if history:
        last = history[-1]
        print(f"  🎯 Son vuln accuracy : {last.get('vuln_acc', 0):.2%}")
        print(f"  🎯 Son comp accuracy : {last.get('comp_acc', 0):.2%}")
    print(f"{'═' * 64}")

    _save_training_curves(history, ckpt_dir)

    hist_path = ckpt_dir / "training_history.json"
    with open(str(hist_path), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info("History kaydedildi: %s", hist_path)

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  §8. CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Komut satırı arayüzü."""
    parser = argparse.ArgumentParser(
        description="Smart Code Analyzer — Model Eğitimi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch boyutu (otomatik)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="cuda veya cpu")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint dizini")
    args = parser.parse_args()

    # ── A100 GPU Özel Optimizasyonları (Tensor Cores / TF32) ─────────────
    # A100 GPU'lar Matris çarpımlarında (MatMul) Float32 yerine TensorFloat32 (TF32)
    # kullanırsa hızı %200 - %300 arası artırır ve hassasiyet kaybı yaşatmaz.
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Konfigürasyon ────────────────────────────────────────────────────
    cfg = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    # ── Device ───────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    elif device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if "A100" in name:
            cfg.batch_size = 32
            cfg.accumulation_steps = 2   # efektif=64
            cfg.num_workers = 4
        elif "V100" in name or mem_gb >= 32:
            cfg.batch_size = 24
            cfg.accumulation_steps = 2
            cfg.num_workers = 4
        elif "L4" in name or "T4" in name or mem_gb >= 15:
            cfg.batch_size = 16
            cfg.accumulation_steps = 4   # efektif=64
            cfg.num_workers = 2
        else:
            cfg.batch_size = 8
            cfg.accumulation_steps = 8
            cfg.num_workers = 2
    else:
        cfg.batch_size = 4
        cfg.accumulation_steps = 16
        cfg.num_workers = 0

    # ── HF Token ─────────────────────────────────────────────────────────
    hf_token = _resolve_hf_token()

    # ── Checkpoint ────────────────────────────────────────────────────────
    if args.output_dir:
        ckpt_dir = Path(args.output_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_dir = _resolve_checkpoint_dir()

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            import shutil
            target = ckpt_dir / "last_checkpoint.pt"
            if not target.exists():
                shutil.copy2(str(resume_path), str(target))

    # ── Eğitimi Başlat ───────────────────────────────────────────────────
    train(cfg, device, ckpt_dir, hf_token)


if __name__ == "__main__":
    main()
