#!/usr/bin/env python3
"""
model/architecture.py — Multi-Task Code Analysis Transformer

Mimari:
- Pre-trained CodeBERT/CodeT5 tabanlı encoder
- Memory-Augmented Transformer (yüksek dikkat ağırlıklarını taşır)
- Hiyerarşik dikkat mekanizması (token → statement → function → file)
- 3 görev başlığı: Güvenlik (CWE), Karmaşıklık (Big-O), Veri Akışı
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ══════════════════════════════════════════════════════════════════════════
#  Memory Bank — Önceki pencerelerin high-attention bilgisini taşır
# ══════════════════════════════════════════════════════════════════════════


class MemoryBank(nn.Module):
    """
    Kayan pencere yaklaşımında önceki pencerelerin önemli
    bağlam bilgisini biriktiren bellek modülü.
    """

    def __init__(self, hidden_size: int, memory_size: int = 1024):
        super().__init__()
        self.memory_size = memory_size
        self.hidden_size = hidden_size

        # Cross-attention ile belleği sorgulama
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

        # Bellek buffer'ı (eğitim sırasında güncellenir, gradient yok)
        self.register_buffer(
            "memory",
            torch.zeros(1, memory_size, hidden_size),
        )
        self.register_buffer("memory_count", torch.tensor(0))

    def update_memory(self, hidden_states: torch.Tensor,
                      attention_weights: torch.Tensor) -> None:
        """
        Yüksek dikkat ağırlıklı token'ları belleğe ekler.
        Gradient akışı yok — sadece bilgi taşıma.
        """
        with torch.no_grad():
            # Attention ortalamasını al → en önemli token'ları bul
            if attention_weights.dim() == 4:
                # (batch, heads, seq, seq) → (batch, seq)
                importance = attention_weights.mean(dim=(1, 2))
            else:
                importance = attention_weights.mean(dim=-1)

            batch_size = hidden_states.size(0)

            # En önemli K token'ı seç
            k = min(64, hidden_states.size(1))
            _, top_indices = importance.topk(k, dim=-1)

            # Belleği genişlet
            for b in range(batch_size):
                selected = hidden_states[b, top_indices[b]]  # (k, hidden)
                current_count = self.memory_count.item()
                new_count = min(current_count + k, self.memory_size)

                if current_count + k <= self.memory_size:
                    self.memory[0, current_count:current_count + k] = selected
                else:
                    # FIFO: eski belleği kaydır, yenileri ekle
                    shift = current_count + k - self.memory_size
                    self.memory[0, :-shift] = self.memory[0, shift:].clone()
                    self.memory[0, -k:] = selected

                self.memory_count.fill_(new_count)

    def query_memory(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Mevcut hidden state'lerle belleği cross-attention ile sorgular.
        """
        if self.memory_count.item() == 0:
            return hidden_states

        batch_size = hidden_states.size(0)
        mem_len = int(self.memory_count.item())

        # Memory'yi batch boyutuna genişlet
        memory = self.memory[:, :mem_len].expand(batch_size, -1, -1)

        # Cross-attention: Q=current, K,V=memory
        Q = self.query_proj(hidden_states)
        K = self.key_proj(memory)
        V = self.value_proj(memory)

        scale = math.sqrt(self.hidden_size)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        context = torch.matmul(attn_weights, V)
        output = self.out_proj(context)

        # Residual connection + layer norm
        return self.layer_norm(hidden_states + output)

    def reset(self) -> None:
        """Belleği sıfırlar (yeni dosya/örnek için)."""
        self.memory.zero_()
        self.memory_count.zero_()


# ══════════════════════════════════════════════════════════════════════════
#  Hiyerarşik Dikkat Mekanizması
# ══════════════════════════════════════════════════════════════════════════


class HierarchicalAttention(nn.Module):
    """
    Token → Statement → Function seviyesinde hiyerarşik dikkat.
    Kod bloklarının yapısal ilişkilerini yakalar.
    """

    def __init__(self, hidden_size: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Her seviye için dikkat havuzu
        self.level_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, 1),
            )
            for _ in range(num_levels)
        ])

        # Seviyeleri birleştirme
        self.merge = nn.Linear(hidden_size * num_levels, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, hidden_states: torch.Tensor,
        level_boundaries: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            level_boundaries: İsteğe bağlı seviye sınırları

        Returns:
            (batch, hidden_size) — Hiyerarşik olarak birleştirilmiş temsil
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        level_outputs = []

        for level_idx in range(self.num_levels):
            # Seviye bazında dikkat ağırlığı hesapla
            attn_scores = self.level_attention[level_idx](hidden_states)
            attn_scores = attn_scores.squeeze(-1)  # (batch, seq_len)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Ağırlıklı toplam
            weighted = torch.bmm(
                attn_weights.unsqueeze(1), hidden_states
            )  # (batch, 1, hidden)
            level_outputs.append(weighted.squeeze(1))

        # Tüm seviyeleri birleştir
        merged = torch.cat(level_outputs, dim=-1)  # (batch, hidden * levels)
        output = self.merge(merged)  # (batch, hidden)
        output = self.layer_norm(output)

        return output


# ══════════════════════════════════════════════════════════════════════════
#  Görev Başlıkları (Task Heads)
# ══════════════════════════════════════════════════════════════════════════


class VulnerabilityHead(nn.Module):
    """CWE ID bazlı güvenlik zafiyeti tespiti."""

    def __init__(self, hidden_size: int, num_cwe_classes: int = 50,
                 dropout: float = 0.175):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_cwe_classes + 1),
            # +1: ilk sınıf "zafiyet yok" demek
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)


class ComplexityHead(nn.Module):
    """Big-O karmaşıklık tahmini (sınıflandırma)."""

    # O(1), O(logn), O(n), O(nlogn), O(n²), O(n³), O(2^n), O(n!)
    NUM_CLASSES = 8

    def __init__(self, hidden_size: int, dropout: float = 0.175):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.NUM_CLASSES),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)


class DataFlowHead(nn.Module):
    """Veri akışı tahmini (binary: akış var/yok)."""

    def __init__(self, hidden_size: int, dropout: float = 0.175):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 2),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════
#  Ana Model: Multi-Task Code Analyzer
# ══════════════════════════════════════════════════════════════════════════


class CodeAnalyzerModel(nn.Module):
    """
    Multi-Task Code Analysis Transformer.

    Girdi: Tokenize edilmiş C/C++ kodu
    Çıktılar:
        1. vulnerability_logits — CWE sınıflandırma
        2. complexity_logits   — Big-O sınıflandırma
        3. dataflow_logits     — Veri akışı tespiti
    """

    def __init__(
        self,
        pretrained_model_name: str = "microsoft/graphcodebert-base",
        num_cwe_classes: int = 50,
        dropout: float = 0.175,
        memory_bank_size: int = 1024,
        use_memory: bool = True,
        use_hierarchical_attention: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        # ── Encoder (Pre-trained CodeBERT) ──
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = config.hidden_size

        # Gradient checkpointing (VRAM tasarrufu)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # ── Memory Bank ──
        self.use_memory = use_memory
        if use_memory:
            self.memory_bank = MemoryBank(
                self.hidden_size, memory_bank_size
            )

        # ── Hiyerarşik Dikkat ──
        self.use_hierarchical = use_hierarchical_attention
        if use_hierarchical_attention:
            self.hierarchical_attn = HierarchicalAttention(
                self.hidden_size, num_levels=3
            )

        # ── Görev Başlıkları ──
        self.vuln_head = VulnerabilityHead(
            self.hidden_size, num_cwe_classes, dropout
        )
        self.complexity_head = ComplexityHead(self.hidden_size, dropout)
        self.dataflow_head = DataFlowHead(self.hidden_size, dropout)

        # ── Dropout ──
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        update_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dict with keys: vulnerability_logits, complexity_logits,
            dataflow_logits
        """
        # ── Encoder ──
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.use_memory,
        )
        hidden_states = encoder_output.last_hidden_state  # (B, seq, H)

        # ── Memory Bank sorgulama ve güncelleme ──
        if self.use_memory:
            hidden_states = self.memory_bank.query_memory(hidden_states)

            if update_memory and encoder_output.attentions:
                last_attn = encoder_output.attentions[-1]
                self.memory_bank.update_memory(
                    hidden_states.detach(), last_attn.detach()
                )

        # ── Pooling ──
        if self.use_hierarchical:
            pooled = self.hierarchical_attn(hidden_states)
        else:
            # CLS token pooling
            pooled = hidden_states[:, 0, :]

        pooled = self.dropout(pooled)

        # ── Görev çıktıları ──
        return {
            "vulnerability_logits": self.vuln_head(pooled),
            "complexity_logits": self.complexity_head(pooled),
            "dataflow_logits": self.dataflow_head(pooled),
        }

    def reset_memory(self) -> None:
        """Memory bank'ı sıfırlar (yeni dosya işlerken)."""
        if self.use_memory:
            self.memory_bank.reset()


# ══════════════════════════════════════════════════════════════════════════
#  Multi-Task Loss
# ══════════════════════════════════════════════════════════════════════════


class MultiTaskLoss(nn.Module):
    """
    Çok görevli kayıp fonksiyonu.
    Her görevin ağırlığı config'den alınır.
    """

    def __init__(
        self,
        vuln_weight: float = 0.5,
        complexity_weight: float = 0.3,
        dataflow_weight: float = 0.2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.vuln_weight = vuln_weight
        self.complexity_weight = complexity_weight
        self.dataflow_weight = dataflow_weight

        self.vuln_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.complexity_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.dataflow_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(
        self, outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'total_loss', 'vuln_loss', 'complexity_loss',
            'dataflow_loss'
        """
        vuln_loss = self.vuln_loss_fn(
            outputs["vulnerability_logits"], labels["is_vulnerable"]
        )
        complexity_loss = self.complexity_loss_fn(
            outputs["complexity_logits"], labels["complexity"]
        )
        dataflow_loss = self.dataflow_loss_fn(
            outputs["dataflow_logits"], labels["has_data_flow"]
        )

        total = (
            self.vuln_weight * vuln_loss
            + self.complexity_weight * complexity_loss
            + self.dataflow_weight * dataflow_loss
        )

        return {
            "total_loss": total,
            "vuln_loss": vuln_loss,
            "complexity_loss": complexity_loss,
            "dataflow_loss": dataflow_loss,
        }
