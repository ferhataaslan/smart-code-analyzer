#!/usr/bin/env python3
"""
model/evaluator.py — Değerlendirme Metrikleri & Raporlama

Her görev için ayrı metrikler:
- Vulnerability: Precision, Recall, F1, Accuracy
- Complexity: Accuracy, Confusion Matrix
- Data Flow: Accuracy, F1
- Confidence Calibration (halüsinasyon önleme)
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("MODEL.Evaluator")


def evaluate_model(
    model,
    test_loader,
    criterion,
    device: torch.device,
    confidence_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Modeli test seti üzerinde değerlendirir.

    Returns:
        Dict: total_loss, vuln_acc, vuln_f1, complexity_acc,
              dataflow_acc, abstention_rate
    """
    model.eval()
    model.reset_memory()

    total_losses = {
        "total_loss": 0.0,
        "vuln_loss": 0.0,
        "complexity_loss": 0.0,
        "dataflow_loss": 0.0,
    }

    # Tahmin ve etiketleri biriktir
    all_vuln_preds = []
    all_vuln_labels = []
    all_complexity_preds = []
    all_complexity_labels = []
    all_dataflow_preds = []
    all_dataflow_labels = []
    all_max_confidences = []

    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {
                k: v.to(device) for k, v in batch["labels"].items()
            }

            outputs = model(input_ids, attention_mask, update_memory=False)
            losses = criterion(outputs, labels)

            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

            # ── Vulnerability tahminleri ──
            vuln_probs = F.softmax(outputs["vulnerability_logits"], dim=-1)
            vuln_preds = vuln_probs.argmax(dim=-1)
            vuln_confidence = vuln_probs.max(dim=-1).values

            # İlk sınıf (0) = zafiyet yok → binary: 0 ise clean, >0 ise vuln
            vuln_binary_pred = (vuln_preds > 0).long()
            all_vuln_preds.extend(vuln_binary_pred.cpu().tolist())
            all_vuln_labels.extend(labels["is_vulnerable"].cpu().tolist())

            # ── Complexity tahminleri ──
            comp_probs = F.softmax(outputs["complexity_logits"], dim=-1)
            comp_preds = comp_probs.argmax(dim=-1)
            all_complexity_preds.extend(comp_preds.cpu().tolist())
            all_complexity_labels.extend(labels["complexity"].cpu().tolist())

            # ── Data Flow tahminleri ──
            df_probs = F.softmax(outputs["dataflow_logits"], dim=-1)
            df_preds = df_probs.argmax(dim=-1)
            all_dataflow_preds.extend(df_preds.cpu().tolist())
            all_dataflow_labels.extend(labels["has_data_flow"].cpu().tolist())

            # ── Confidence (halüsinasyon metrikleri) ──
            all_max_confidences.extend(vuln_confidence.cpu().tolist())

    # Ortalamaları al
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    # ── Metrik hesaplama ──
    metrics = total_losses.copy()

    # Vulnerability metrikleri
    vuln_metrics = _binary_metrics(all_vuln_preds, all_vuln_labels)
    metrics["vuln_accuracy"] = vuln_metrics["accuracy"]
    metrics["vuln_precision"] = vuln_metrics["precision"]
    metrics["vuln_recall"] = vuln_metrics["recall"]
    metrics["vuln_f1"] = vuln_metrics["f1"]

    # Complexity metrikleri
    metrics["complexity_accuracy"] = _accuracy(
        all_complexity_preds, all_complexity_labels
    )

    # Data Flow metrikleri
    df_metrics = _binary_metrics(all_dataflow_preds, all_dataflow_labels)
    metrics["dataflow_accuracy"] = df_metrics["accuracy"]
    metrics["dataflow_f1"] = df_metrics["f1"]

    # Confidence / Abstention (halüsinasyon kontrolü)
    if all_max_confidences:
        low_conf = sum(
            1 for c in all_max_confidences if c < confidence_threshold
        )
        metrics["abstention_rate"] = low_conf / len(all_max_confidences)
        metrics["avg_confidence"] = sum(all_max_confidences) / len(
            all_max_confidences
        )

    model.train()
    return metrics


def print_metrics(
    epoch: int, train_metrics: Dict, eval_metrics: Dict
) -> None:
    """Eğitim ve değerlendirme metriklerini formatlanmış şekilde yazdırır."""
    print()
    print("=" * 65)
    print(f"  EPOCH {epoch + 1} SONUCLARI")
    print("=" * 65)

    print(f"  {'Metrik':<30} {'Train':>12} {'Eval':>12}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12}")

    # Loss metrikleri
    for key in ["total_loss", "vuln_loss", "complexity_loss", "dataflow_loss"]:
        train_val = train_metrics.get(key, 0)
        eval_val = eval_metrics.get(key, 0)
        print(f"  {key:<30} {train_val:>12.4f} {eval_val:>12.4f}")

    print(f"  {'-' * 30} {'-' * 12} {'-' * 12}")

    # Eval-only metrikleri
    eval_only_keys = [
        "vuln_accuracy", "vuln_precision", "vuln_recall", "vuln_f1",
        "complexity_accuracy", "dataflow_accuracy", "dataflow_f1",
        "abstention_rate", "avg_confidence",
    ]
    for key in eval_only_keys:
        if key in eval_metrics:
            val = eval_metrics[key]
            print(f"  {key:<30} {'—':>12} {val:>12.4f}")

    print("=" * 65)
    print()


# ══════════════════════════════════════════════════════════════════════════
#  Yardımcı Fonksiyonlar
# ══════════════════════════════════════════════════════════════════════════


def _accuracy(preds: List[int], labels: List[int]) -> float:
    """Basit doğruluk hesaplama."""
    if not preds:
        return 0.0
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(preds)


def _binary_metrics(
    preds: List[int], labels: List[int]
) -> Dict[str, float]:
    """Binary sınıflandırma metrikleri: accuracy, precision, recall, f1."""
    if not preds:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
