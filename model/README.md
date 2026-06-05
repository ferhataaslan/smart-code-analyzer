# AI-Powered Smart Code & Dependency Analyzer — Model Modülü

## Genel Bakış

Bu klasör, C/C++ kodları için güvenlik zafiyeti tespiti, Big-O karmaşıklık tahmini ve
veri akış analizi yapacak olan AI modelinin eğitim, konfigürasyon ve çıkarım dosyalarını içerir.

> **Not:** Pipeline (veri toplama, normalizasyon, otomasyon) dosyaları kök dizinde kalır.
> Bu klasör sadece model eğitimi ile ilgili dosyaları içerir.

## Dosya Yapısı

```
model/
├── __init__.py      # Paket tanımlayıcı
├── config.py        # Eğitim konfigürasyonu (mimari, hiperparametreler, veri seti)
├── README.md        # Bu dosya
└── (ileride)
    ├── trainer.py           # Eğitim döngüsü
    ├── dataset_loader.py    # HF'den veri yükleme ve parçalama
    ├── architecture.py      # Model mimarisi tanımı
    ├── evaluator.py         # Değerlendirme metrikleri
    └── inference.py         # CI/CD entegrasyonu için çıkarım
```

## Mimari Özet

```
                    ┌─────────────────┐
                    │  Kod Girdisi    │
                    │  (C/C++ snippet)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Graph-Aware    │
                    │  Chunking       │
                    │  (512 token)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌────▼────┐ ┌──────▼───────┐
     │ Normalize Kod │ │  AST    │ │     DFG      │
     │   Encoder     │ │ Encoder │ │   Encoder    │
     └────────┬──────┘ └────┬────┘ └──────┬───────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Memory-Aug     │
                    │  Transformer    │
                    │  + Hierarchical │
                    │    Attention    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
  ┌────────▼──────┐ ┌───────▼───────┐ ┌──────▼───────┐
  │  Vulnerability│ │  Complexity   │ │  Data Flow   │
  │  Detection    │ │  Estimation   │ │  Prediction  │
  │  (CWE ID)     │ │  (Big-O)      │ │  (CFG/DFG)   │
  └───────────────┘ └───────────────┘ └──────────────┘
```

## Veri Seti Uyumluluğu

Model, pipeline'ın ürettiği HF veri setiyle %100 uyumludur:

| Sütun | Model Kullanımı |
|-------|----------------|
| `normalized_structure` | Ana girdi (kod) |
| `ast_metadata` | Yapısal bağlam |
| `data_flow_graph` | Akış bağlamı |
| `security_context` | Güvenlik etiketi (CWE) |
| `complexity` | Big-O etiketi |

## Eğitim Başlatma (İleride)

```bash
# Konfigürasyonu doğrula
python -m model.config

# Eğitim başlat (henüz uygulanmadı)
python -m model.trainer --config model/config.py
```
