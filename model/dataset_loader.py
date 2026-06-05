#!/usr/bin/env python3
"""
model/dataset_loader.py — Veri Seti Yükleme & Graph-Aware Chunking

Pipeline'ın HF'ye pushladığı veri setini yükler, tokenize eder,
graph-aware chunking uygular ve train/test split oluşturur.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("MODEL.DatasetLoader")


class CodeAnalysisDataset(Dataset):
    """
    HF veri setini PyTorch Dataset'ine dönüştürür.
    Her örnek: (input_ids, attention_mask, labels_dict)
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int = 512,
        chunk_overlap: int = 64,
        graph_aware: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap
        self.graph_aware = graph_aware
        self.samples = []

        for record in records:
            chunks = self._prepare_record(record)
            self.samples.extend(chunks)

        logger.info(
            f"Dataset olusturuldu: {len(records)} kayit -> "
            f"{len(self.samples)} chunk"
        )

    def _prepare_record(self, record: Dict) -> List[Dict]:
        """Tek bir kaydı tokenize edip chunk'lara böler."""
        # ── Girdileri hazırla ──
        code = record.get("normalized_structure", "")
        if not code or code == "''":
            code = record.get("raw_snippet", "")
        if not code:
            return []

        # AST ve DFG bağlamını ekle
        ast_meta = self._safe_json_parse(record.get("ast_metadata", "{}"))
        dfg_data = self._safe_json_parse(record.get("data_flow_graph", "[]"))

        # Bağlam bilgisi oluştur (modele ek sinyal)
        context_prefix = self._build_context_prefix(record, ast_meta)

        # ── Etiketleri hazırla ──
        labels = self._extract_labels(record)

        # ── Tokenize et ──
        full_text = context_prefix + code
        encoding = self.tokenizer(
            full_text,
            truncation=False,  # Biz kendimiz chunk'layacağız
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"]

        # ── Graph-Aware Chunking ──
        if self.graph_aware and dfg_data:
            chunks = self._graph_aware_chunk(input_ids, dfg_data)
        else:
            chunks = self._sliding_window_chunk(input_ids)

        # Her chunk için tam örnek oluştur
        result = []
        for chunk_ids in chunks:
            # Padding ve attention mask
            padded = self._pad_and_mask(chunk_ids)
            padded["labels"] = labels
            result.append(padded)

        return result if result else []

    def _build_context_prefix(self, record: Dict, ast_meta: dict) -> str:
        """Modele verilecek bağlam prefix'i oluşturur."""
        parts = []

        # Kaynak sistemi
        source = record.get("source_system", "")
        if source:
            parts.append(f"[SOURCE:{source}]")

        # Karmaşıklık
        complexity = record.get("complexity", "")
        if complexity and complexity != "1":
            parts.append(f"[COMPLEXITY:{complexity}]")

        # AST'den fonksiyon adı
        if isinstance(ast_meta, dict):
            func_name = ast_meta.get("function_name", "")
            if func_name:
                parts.append(f"[FUNC:{func_name}]")

        return " ".join(parts) + " " if parts else ""

    def _extract_labels(self, record: Dict) -> Dict:
        """Etiketleri çıkarır: güvenlik, karmaşıklık, akış."""
        labels = {}

        # ── Güvenlik etiketi (CWE ID) ──
        sec_ctx = self._safe_json_parse(
            record.get("security_context", "{}")
        )
        if isinstance(sec_ctx, dict):
            is_vulnerable = sec_ctx.get("is_vulnerable", False)
            cwe_ids = sec_ctx.get("cwe_ids", [])
            labels["is_vulnerable"] = 1 if is_vulnerable else 0
            labels["cwe_ids"] = cwe_ids if cwe_ids else []
        else:
            labels["is_vulnerable"] = 0
            labels["cwe_ids"] = []

        # ── Karmaşıklık etiketi ──
        complexity = record.get("complexity", "1")
        labels["complexity"] = self._complexity_to_label(complexity)

        # ── DFG etiketi ──
        dfg = self._safe_json_parse(record.get("data_flow_graph", "[]"))
        labels["has_data_flow"] = 1 if dfg else 0

        return labels

    def _complexity_to_label(self, complexity_str: str) -> int:
        """Big-O karmaşıklığını sayısal etikete dönüştürür."""
        mapping = {
            "O(1)": 0, "1": 0,
            "O(log n)": 1, "O(logn)": 1,
            "O(n)": 2,
            "O(n log n)": 3, "O(nlogn)": 3,
            "O(n^2)": 4, "O(n²)": 4,
            "O(n^3)": 5, "O(n³)": 5,
            "O(2^n)": 6,
            "O(n!)": 7,
        }
        return mapping.get(str(complexity_str).strip(), 0)

    def _graph_aware_chunk(
        self, input_ids: List[int], dfg_data: list
    ) -> List[List[int]]:
        """
        Graph-Aware Chunking: DFG bağlantılı düğümleri aynı chunk'ta tutar.
        Fallback olarak sliding window kullanır.
        """
        total_len = len(input_ids)
        if total_len <= self.max_length:
            return [input_ids]

        # DFG edge'lerinden bağlantılı segment grupları oluştur
        # (basitleştirilmiş: fonksiyon sınırlarında böl)
        chunks = []
        start = 0
        while start < total_len:
            end = min(start + self.max_length, total_len)

            # Semantik sınır ara (fonksiyon/blok sonu)
            if end < total_len:
                # Son 64 tokenda bir kırılma noktası ara
                search_start = max(end - 64, start)
                best_break = end
                for i in range(end - 1, search_start, -1):
                    # Fonksiyon/blok sonu tokenları (';', '}', vb.)
                    token_text = self.tokenizer.decode([input_ids[i]])
                    if token_text.strip() in ("}", ";", "\n\n"):
                        best_break = i + 1
                        break
                end = best_break

            chunks.append(input_ids[start:end])
            start = end - self.chunk_overlap  # Overlap ile kaydır

        return chunks

    def _sliding_window_chunk(
        self, input_ids: List[int]
    ) -> List[List[int]]:
        """Basit kayan pencere chunking."""
        total_len = len(input_ids)
        if total_len <= self.max_length:
            return [input_ids]

        chunks = []
        start = 0
        while start < total_len:
            end = min(start + self.max_length, total_len)
            chunks.append(input_ids[start:end])
            start = end - self.chunk_overlap

        return chunks

    def _pad_and_mask(self, input_ids: List[int]) -> Dict:
        """Padding ve attention mask oluşturur."""
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id or 0] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    @staticmethod
    def _safe_json_parse(value) -> any:
        """JSON string'i güvenli şekilde parse eder."""
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════
#  Veri Seti Yükleme Fonksiyonları
# ══════════════════════════════════════════════════════════════════════════


def load_hf_dataset(
    repo_id: str = "smart-code-analyzer-team/cpp-vulnerability-dataset",
    validation_split: float = 0.1,
) -> Tuple[list, list]:
    """
    HF'den veri setini yükler ve train/test split yapar.

    Returns:
        (train_records, test_records) tuple'ı
    """
    from datasets import load_dataset

    logger.info(f"HF'den veri seti yukleniyor: {repo_id}")
    ds = load_dataset(repo_id, split="train")

    records = [dict(row) for row in ds]
    logger.info(f"Toplam {len(records)} kayit yuklendi.")

    # Shuffle ve split
    import random
    random.seed(42)
    random.shuffle(records)

    split_idx = int(len(records) * (1 - validation_split))
    train_records = records[:split_idx]
    test_records = records[split_idx:]

    logger.info(
        f"Split: {len(train_records)} train, {len(test_records)} test "
        f"(%{validation_split * 100:.0f} validation)"
    )

    return train_records, test_records


def balance_dataset(
    records: List[Dict], ratio: float = 0.5
) -> List[Dict]:
    """
    Temiz ve hatalı kod örneklerini dengeleyerek
    istenen orana getirir (varsayılan %50-50).
    """
    vulnerable = []
    clean = []

    for r in records:
        sec_ctx = CodeAnalysisDataset._safe_json_parse(
            r.get("security_context", "{}")
        )
        if isinstance(sec_ctx, dict) and sec_ctx.get("is_vulnerable"):
            vulnerable.append(r)
        else:
            clean.append(r)

    logger.info(
        f"Dengeleme oncesi: {len(vulnerable)} vulnerable, "
        f"{len(clean)} clean"
    )

    # Oversample minority class
    import random
    if len(vulnerable) < len(clean):
        target = int(len(clean) * ratio / (1 - ratio))
        while len(vulnerable) < target:
            vulnerable.append(random.choice(vulnerable))
    elif len(clean) < len(vulnerable):
        target = int(len(vulnerable) * (1 - ratio) / ratio)
        while len(clean) < target:
            clean.append(random.choice(clean))

    balanced = vulnerable + clean
    random.shuffle(balanced)

    logger.info(
        f"Dengeleme sonrasi: {len(vulnerable)} vulnerable, "
        f"{len(clean)} clean = {len(balanced)} toplam"
    )

    return balanced


def create_data_loaders(
    tokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    repo_id: str = "smart-code-analyzer-team/cpp-vulnerability-dataset",
    validation_split: float = 0.1,
    balance_ratio: float = 0.5,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Tam pipeline: HF'den yükle → dengele → tokenize → chunk → DataLoader

    Returns:
        (train_loader, test_loader)
    """
    # 1. HF'den yükle
    train_records, test_records = load_hf_dataset(repo_id, validation_split)

    # 2. Eğitim setini dengele
    train_records = balance_dataset(train_records, balance_ratio)

    # 3. Dataset oluştur
    train_dataset = CodeAnalysisDataset(
        train_records, tokenizer, max_length=max_length
    )
    test_dataset = CodeAnalysisDataset(
        test_records, tokenizer, max_length=max_length
    )

    # 4. DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )

    return train_loader, test_loader


def _collate_fn(batch: List[Dict]) -> Dict:
    """Batch içindeki örnekleri birleştirir."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    # Labels
    labels = {
        "is_vulnerable": torch.tensor(
            [b["labels"]["is_vulnerable"] for b in batch], dtype=torch.long
        ),
        "complexity": torch.tensor(
            [b["labels"]["complexity"] for b in batch], dtype=torch.long
        ),
        "has_data_flow": torch.tensor(
            [b["labels"]["has_data_flow"] for b in batch], dtype=torch.long
        ),
    }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
