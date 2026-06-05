# Smart Code Analyzer

Bu proje, C/C++ kod tabanlarındaki güvenlik açıklarını (CWE) ve karmaşıklık metriklerini (Big-O) tespit etmek için geliştirilmiş, **GraphCodeBERT** tabanlı bir yapay zeka analiz aracıdır.

## 👥 Katılımcılar
- Ferhat Aslan - ferhataaslan
- İsmail Bağkazan - ismailbgkzn

## 🚀 Proje Hakkında
Smart Code Analyzer, geleneksel statik analiz araçlarının (lizard) ötesine geçerek, derin öğrenme (Deep Learning) modelleri ile kodun anlamsal yapısını anlar. Geliştiricilerin kodlarını CI/CD süreçlerine entegre edilebilir bir şekilde analiz etmelerini sağlar.

## 🛠 Kullanılan Teknolojiler
- **Model:** Microsoft GraphCodeBERT (Fine-tuned)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Analiz:** Lizard (Statik Analiz)
- **Veri İşleme:** Apache Parquet, SQLite
- **Altyapı:** Docker, Docker Compose

## 📦 Proje Yapısı
```text
/
├── data/           # (Not: Veri setleri git üzerinde tutulmaz, .gitignore'a eklenmiştir)
├── model/          # Model mimarisi ve checkpoint dosyaları
├── src/            # Kaynak kodlar (API, Extractors, Normalizers)
├── tests/          # Test senaryoları
├── docker-compose.yml
├── .env.example    # Örnek çevre değişkenleri
└── requirements.txt
