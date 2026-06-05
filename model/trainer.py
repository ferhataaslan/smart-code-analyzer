#!/usr/bin/env python3
"""
model/trainer.py — Multi-Task Eğitim Döngüsü

Özellikler:
- Multi-task learning (güvenlik, karmaşıklık, veri akışı)
- Mixed precision (fp16) eğitim
- Gradient accumulation & checkpointing
- Early stopping
- Checkpoint kaydetme/yükleme
- Halüsinasyon önleme (confidence calibration)
"""

import os
import json
import time
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from model.config import TRAINING_CONFIG, DATASET_CONFIG, ARCHITECTURE_CONFIG
from model.architecture import CodeAnalyzerModel, MultiTaskLoss
from model.dataset_loader import create_data_loaders
from model.evaluator import evaluate_model, print_metrics

logger = logging.getLogger("MODEL.Trainer")


class Trainer:
    """Multi-Task Code Analyzer eğitmeni."""

    def __init__(
        self,
        model: CodeAnalyzerModel = None,
        output_dir: str = "model/checkpoints",
        device: str = None,
    ):
        # ── Cihaz ──
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"GPU kullaniliyor: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("GPU bulunamadi, CPU kullaniliyor.")

        # ── Hiperparametreler ──
        hp = TRAINING_CONFIG["hyperparameters"]
        self.learning_rate = hp["learning_rate"]
        self.weight_decay = hp["weight_decay"]
        self.epochs = hp["epochs"]
        self.batch_size = hp["batch_size"]
        self.max_grad_norm = hp["max_grad_norm"]
        self.warmup_ratio = hp["warmup_ratio"]
        self.accumulation_steps = TRAINING_CONFIG["memory_efficiency"][
            "accumulation_steps"
        ]

        # ── Regularization ──
        reg = TRAINING_CONFIG["regularization"]
        self.early_stopping_patience = reg["early_stopping_patience"]
        self.label_smoothing = reg["label_smoothing"]

        # ── Mixed precision ──
        self.use_fp16 = (
            TRAINING_CONFIG["memory_efficiency"]["mixed_precision"] == "fp16"
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler(enabled=self.use_fp16)

        # ── Model ──
        if model is None:
            model = self._build_model()
        self.model = model.to(self.device)

        # ── Loss ──
        task_cfg = TRAINING_CONFIG["tasks"]
        self.criterion = MultiTaskLoss(
            vuln_weight=task_cfg["vulnerability_detection"]["weight"],
            complexity_weight=task_cfg["complexity_estimation"]["weight"],
            dataflow_weight=task_cfg["data_flow_prediction"]["weight"],
            label_smoothing=self.label_smoothing,
        )

        # ── Optimizer ──
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # ── Output ──
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # ── Eğitim durumu ──
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.training_history = []

    def _build_model(self) -> CodeAnalyzerModel:
        """Config'den model oluşturur."""
        arch = ARCHITECTURE_CONFIG
        hp = TRAINING_CONFIG["hyperparameters"]

        return CodeAnalyzerModel(
            pretrained_model_name="microsoft/graphcodebert-base",
            num_cwe_classes=50,
            dropout=hp["dropout"],
            memory_bank_size=arch["memory_bank"]["size"],
            use_memory=arch["memory_augmented"],
            use_hierarchical_attention=arch["hierarchical_attention"]["enabled"],
            gradient_checkpointing=TRAINING_CONFIG["memory_efficiency"][
                "gradient_checkpointing"
            ],
        )

    def _get_scheduler(self, num_training_steps: int):
        """Warmup + cosine decay scheduler."""
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=num_training_steps,
            pct_start=self.warmup_ratio,
            anneal_strategy="cos",
        )

    def train(
        self,
        train_loader=None,
        test_loader=None,
        tokenizer=None,
        resume_from: str = None,
    ) -> Dict:
        """
        Ana eğitim döngüsü.

        Args:
            train_loader: Eğitim DataLoader (None ise otomatik oluşturur)
            test_loader: Test DataLoader
            tokenizer: Tokenizer (DataLoader oluşturmak için)
            resume_from: Checkpoint dosya yolu (devam etmek için)

        Returns:
            Eğitim geçmişi (dict)
        """
        # ── DataLoader oluştur ──
        if train_loader is None:
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/graphcodebert-base"
                )
            train_loader, test_loader = create_data_loaders(
                tokenizer=tokenizer,
                batch_size=self.batch_size,
                max_length=ARCHITECTURE_CONFIG["context_window"],
                repo_id=DATASET_CONFIG["repo_id"],
                validation_split=DATASET_CONFIG["validation_split"],
                balance_ratio=DATASET_CONFIG["balance"]["clean_to_vulnerable_ratio"],
            )

        # ── Checkpoint'tan devam ──
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self._load_checkpoint(resume_from)
            logger.info(f"Checkpoint yuklendi: epoch {start_epoch}")

        # ── Scheduler ──
        num_training_steps = len(train_loader) * self.epochs // self.accumulation_steps
        scheduler = self._get_scheduler(num_training_steps)

        # ── Eğitim döngüsü ──
        logger.info("=" * 60)
        logger.info("  EGITIM BASLIYOR")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Accumulation: {self.accumulation_steps}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  FP16: {self.use_fp16}")
        logger.info(f"  Train samples: {len(train_loader.dataset)}")
        if test_loader:
            logger.info(f"  Test samples: {len(test_loader.dataset)}")
        logger.info("=" * 60)

        for epoch in range(start_epoch, self.epochs):
            # ── Train ──
            train_metrics = self._train_epoch(
                train_loader, scheduler, epoch
            )

            # ── Evaluate ──
            eval_metrics = {}
            if test_loader:
                eval_metrics = evaluate_model(
                    self.model, test_loader, self.criterion, self.device
                )
                print_metrics(epoch, train_metrics, eval_metrics)

            # ── Checkpoint ──
            epoch_loss = eval_metrics.get(
                "total_loss", train_metrics["total_loss"]
            )
            self._save_history(epoch, train_metrics, eval_metrics)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(
                    f"[Epoch {epoch + 1}] Yeni en iyi model! "
                    f"Loss: {epoch_loss:.4f}"
                )
            else:
                self.patience_counter += 1
                self._save_checkpoint(epoch, is_best=False)

            # ── Early Stopping ──
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"[Early Stopping] {self.early_stopping_patience} epoch "
                    f"boyunca iyilesme yok. Egitim durduruluyor."
                )
                break

        logger.info("Egitim tamamlandi!")
        return {"history": self.training_history, "best_loss": self.best_loss}

    def _train_epoch(
        self, train_loader, scheduler, epoch: int
    ) -> Dict[str, float]:
        """Tek bir epoch eğitimi."""
        self.model.train()
        self.model.reset_memory()

        total_losses = {
            "total_loss": 0.0,
            "vuln_loss": 0.0,
            "complexity_loss": 0.0,
            "dataflow_loss": 0.0,
        }
        num_batches = 0
        start_time = time.time()

        self.optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = {
                k: v.to(self.device) for k, v in batch["labels"].items()
            }

            # ── Forward (mixed precision) ──
            with autocast(enabled=self.use_fp16):
                outputs = self.model(input_ids, attention_mask)
                losses = self.criterion(outputs, labels)
                loss = losses["total_loss"] / self.accumulation_steps

            # ── Backward ──
            self.scaler.scale(loss).backward()

            # ── Gradient accumulation ──
            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # ── Metrikleri biriktir ──
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

            # ── İlerleme logu ──
            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                avg_loss = total_losses["total_loss"] / num_batches
                logger.info(
                    f"  [Epoch {epoch + 1}] Step {step + 1}/{len(train_loader)} "
                    f"| Loss: {avg_loss:.4f} "
                    f"| LR: {scheduler.get_last_lr()[0]:.2e} "
                    f"| {elapsed:.0f}s"
                )

        # Ortalama al
        for key in total_losses:
            total_losses[key] /= max(num_batches, 1)

        elapsed = time.time() - start_time
        logger.info(
            f"  [Epoch {epoch + 1}] Train tamamlandi "
            f"({elapsed:.0f}s) | Loss: {total_losses['total_loss']:.4f}"
        )

        return total_losses

    # ── Checkpoint Yönetimi ──

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Model checkpoint'ı kaydeder."""
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "global_step": self.global_step,
            "training_history": self.training_history,
        }

        # Her zaman son checkpoint'ı kaydet
        path = os.path.join(self.output_dir, "last_checkpoint.pt")
        torch.save(checkpoint, path)

        # En iyi modeli ayrıca kaydet
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"  En iyi model kaydedildi: {best_path}")

    def _load_checkpoint(self, path: str) -> int:
        """Checkpoint'tan yükler. Başlanacak epoch'u döner."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.global_step = checkpoint.get("global_step", 0)
        self.training_history = checkpoint.get("training_history", [])
        return checkpoint.get("epoch", 0)

    def _save_history(
        self, epoch: int, train_metrics: Dict, eval_metrics: Dict
    ) -> None:
        """Eğitim geçmişini kaydeder."""
        entry = {
            "epoch": epoch + 1,
            "train": {k: round(v, 6) for k, v in train_metrics.items()},
            "eval": {k: round(v, 6) for k, v in eval_metrics.items()},
        }
        self.training_history.append(entry)

        # JSON olarak da kaydet
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Smart Code Analyzer — Model Egitimi"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint dosyasindan devam et"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Cihaz: cuda veya cpu"
    )
    parser.add_argument(
        "--output-dir", type=str, default="model/checkpoints",
        help="Checkpoint kayit dizini"
    )
    args = parser.parse_args()

    trainer = Trainer(output_dir=args.output_dir, device=args.device)
    result = trainer.train(resume_from=args.resume)

    print(f"\nEgitim tamamlandi! En iyi loss: {result['best_loss']:.4f}")
