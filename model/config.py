#!/usr/bin/env python3
"""
model/config.py — AI-Powered Smart Code & Dependency Analyzer
Eğitim Konfigürasyonu (Generation 2.9)

Bu dosya, modelin mimari, veri işleme ve eğitim stratejilerini tanımlar.
Pipeline (collector, automator, uploader vb.) dosyalarından BAĞIMSIZDIR.
"""

# ══════════════════════════════════════════════════════════════════════════
#  Veri Seti Konfigürasyonu
# ══════════════════════════════════════════════════════════════════════════

DATASET_CONFIG = {
    # HuggingFace repo bilgisi
    "repo_id": "smart-code-analyzer-team/cpp-vulnerability-dataset",

    # Veri setindeki sütunlar ve kullanım amaçları
    "columns": {
        "raw_snippet":            "Ham C/C++ kaynak kodu",
        "normalized_structure":   "Normalize edilmiş kod yapısı (AST korumalı)",
        "ast_metadata":           "Tree-sitter AST çıktısı (JSON string)",
        "security_context":       "Güvenlik analizi: CWE ID, cppcheck, flawfinder (JSON string)",
        "data_flow_graph":        "Veri Akış Grafiği / DFG (JSON string)",
        "complexity":             "Big-O karmaşıklık tahmini",
        "code_hash":              "SHA-256 dedup hash",
        "source_system":          "Kaynak: OWASP, HUGGINGFACE, GITHUB",
        "nl_alignment":           "Doğal dil açıklaması (JSON string)",
        "applied_algorithm":      "Uygulanan normalizasyon algoritması",
    },

    # Eğitim için kullanılacak birincil sütunlar
    "input_columns": [
        "normalized_structure",   # Model girdisi (normalize kod)
        "ast_metadata",           # Yapısal bağlam
        "data_flow_graph",        # Akış bağlamı
    ],

    # Etiket (label) sütunları
    "label_columns": {
        "vulnerability":  "security_context",   # CWE ID tahmini için
        "complexity":     "complexity",          # Big-O tahmini için
        "data_flow":      "data_flow_graph",     # Akış tahmini için
    },

    # Veri dengeleme
    "balance": {
        "clean_to_vulnerable_ratio": 0.5,  # %50 temiz, %50 hatalı
        "strategy": "oversample_minority",  # Azınlık sınıfı çoğalt
    },

    # Train/Test ayrımı
    "validation_split": 0.1,  # %10 test seti
}


# ══════════════════════════════════════════════════════════════════════════
#  Model Mimarisi
# ══════════════════════════════════════════════════════════════════════════

ARCHITECTURE_CONFIG = {
    # Temel mimari tipi
    "type": "multi_task_transformer",

    # Bağlam penceresi
    "context_window": 512,      # Token cinsinden maksimum girdi uzunluğu
    "sliding_window": True,     # Kayan pencere mekanizması

    # Uzun Bağımlılık Yönetimi (Long-Range Dependencies)
    "memory_augmented": True,   # Hafıza destekli mimari
    "memory_bank": {
        "enabled": True,
        "size": 1024,           # Memory bank token kapasitesi
        "strategy": "high_attention_carry",  # Yüksek dikkat ağırlıklarını taşı
        "merge_method": "cross_attention",   # Önceki pencereleri birleştir
    },

    # Hiyerarşik Dikkat Mekanizması
    "hierarchical_attention": {
        "enabled": True,
        "levels": [
            "token",      # Token seviyesi dikkat
            "statement",  # İfade seviyesi dikkat
            "function",   # Fonksiyon seviyesi dikkat
            "file",       # Dosya seviyesi dikkat
        ],
    },

    # Alternatif mimari seçenekleri (ileride test edilecek)
    "alternative_architectures": [
        "mamba_based",           # Mamba (State Space Model) tabanlı
        "longformer_variant",    # Longformer tarzı yerel+global dikkat
    ],
}


# ══════════════════════════════════════════════════════════════════════════
#  Eğitim Stratejisi & Hiperparametreler
# ══════════════════════════════════════════════════════════════════════════

TRAINING_CONFIG = {
    # Multi-Task Learning görevleri
    "tasks": {
        "vulnerability_detection": {
            "type": "multi_label_classification",
            "description": "CWE ID bazlı güvenlik zafiyeti tespiti",
            "loss": "binary_cross_entropy",
            "weight": 0.5,  # Görev ağırlığı
        },
        "complexity_estimation": {
            "type": "regression",
            "description": "Big-O karmaşıklık tahmini",
            "loss": "mse",
            "weight": 0.3,
        },
        "data_flow_prediction": {
            "type": "sequence_labeling",
            "description": "Mantıksal akış ve veri bağımlılığı tahmini",
            "loss": "cross_entropy",
            "weight": 0.2,
        },
    },

    # Hiperparametreler
    "hyperparameters": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "dropout": 0.175,           # 0.15 - 0.2 aralığının ortası
        "dropout_range": [0.15, 0.2],  # Deneme aralığı
        "batch_size": 16,
        "epochs": 30,
        "warmup_ratio": 0.1,        # İlk %10 epoch warmup
        "max_grad_norm": 1.0,       # Gradient clipping
    },

    # Bellek verimliliği
    "memory_efficiency": {
        "gradient_checkpointing": True,   # VRAM tasarrufu
        "mixed_precision": "fp16",        # Karma hassasiyet eğitimi
        "accumulation_steps": 4,          # Gradient biriktirme
    },

    # Overfitting önleme
    "regularization": {
        "early_stopping_patience": 5,     # 5 epoch iyileşme yoksa dur
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "data_augmentation": {
            "variable_renaming": True,     # Değişken ismi rastgele değiştir
            "dead_code_injection": False,  # Ölü kod ekleme (dikkatli)
            "comment_removal": True,       # Yorum kaldırma
        },
    },

    # Halüsinasyon minimizasyonu
    "hallucination_prevention": {
        "calibration": True,              # Olasılık kalibrasyonu
        "confidence_threshold": 0.7,      # Eşik altı tahminleri "bilinmiyor" say
        "abstention_enabled": True,       # Emin değilse tahmin yapmasın
    },
}


# ══════════════════════════════════════════════════════════════════════════
#  Kod Parçalama (Graph-Aware Chunking)
# ══════════════════════════════════════════════════════════════════════════

CHUNKING_CONFIG = {
    "strategy": "graph_aware",
    "max_tokens": 512,

    # Grafik farkındalıklı parçalama kuralları
    "rules": {
        "preserve_data_flow_edges": True,  # DFG bağlantılı düğümleri ayırma
        "keep_function_intact": True,      # Fonksiyonları bölme
        "overlap_tokens": 64,              # Pencereler arası örtüşme
        "semantic_boundary": True,         # Semantik sınırlarda böl
    },

    # Uzun kod için strateji
    "long_code_handling": {
        "method": "hierarchical_split",    # Önce fonksiyon, sonra blok bazlı
        "max_chunks_per_file": 20,
        "context_carry_tokens": 128,       # Önceki parçadan bağlam taşı
    },
}


# ══════════════════════════════════════════════════════════════════════════
#  Çıktı & CI/CD Entegrasyonu
# ══════════════════════════════════════════════════════════════════════════

OUTPUT_CONFIG = {
    # Model çıktı formatı
    "report_format": {
        "vulnerability_report": {
            "cwe_id": "str",           # Tespit edilen CWE ID
            "severity": "str",         # Ciddiyet seviyesi
            "confidence": "float",     # Güven skoru (0-1)
            "line_range": "tuple",     # Etkilenen satır aralığı
            "suggestion": "str",       # Düzeltme önerisi
        },
        "complexity_report": {
            "big_o": "str",            # O(n), O(n²), vb.
            "confidence": "float",
            "bottleneck_lines": "list", # Darboğaz satırları
        },
    },

    # CI/CD entegrasyonu
    "ci_cd": {
        "output_format": "sarif",      # SARIF (Static Analysis Results)
        "fail_on_severity": "high",    # HIGH ve üstü hatalarda build durdur
        "report_path": "analysis_report.json",
    },
}


# ══════════════════════════════════════════════════════════════════════════
#  Birleşik Konfigürasyon
# ══════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    "dataset": DATASET_CONFIG,
    "architecture": ARCHITECTURE_CONFIG,
    "training": TRAINING_CONFIG,
    "chunking": CHUNKING_CONFIG,
    "output": OUTPUT_CONFIG,
}


if __name__ == "__main__":
    import json
    print(json.dumps(MODEL_CONFIG, indent=2, ensure_ascii=False))
