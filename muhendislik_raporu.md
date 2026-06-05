# Akıllı Kod Analizatörü (Smart Code Analyzer)
## Detaylı Mühendislik Raporu

### 1. Genel Bakış ve Mimari
Proje, C/C++ kaynak kodlarındaki güvenlik açıklarını (CWE zafiyetleri), zaman/alan karmaşıklığını (Big-O) ve potansiyel veri akışı (Taint) problemlerini eşzamanlı olarak tespit etmek üzere tasarlanmış **Multi-Task Code Analysis Transformer** tabanlı bir sistemdir.

Sistem Mimarisi 3 ana katmandan oluşmaktadır:
1. **Veri Toplama ve İşleme Katmanı:** GitHub ve OWASP tabanlı zafiyetli kodların toplanması, AST (Abstract Syntax Tree) ve DFG (Data Flow Graph) çıkarımı.
2. **Derin Öğrenme ve Model Katmanı:** Microsoft GraphCodeBERT tabanlı çoklu görev (Multi-Task) Transformer mimarisi.
3. **API ve Arayüz Katmanı:** FastAPI tabanlı yüksek performanslı backend ve Streamlit tabanlı kullanıcı dostu frontend.

### 2. Veri Setleri ve Kaynaklar
- **Toplanan Veriler:** OWASP zafiyet veritabanları ve GitHub üzerindeki zafiyetli/güvenli C/C++ depolarından toplanan kod parçacıkları (`collector.py` ve `uploader.py` aracılığıyla).
- **Veri Saklama (Storage):**
  - **SQLite (`database.py`):** Ham kodlar, durum takibi (pending, approved) ve meta veriler `data/dbs/review_state.db` üzerinde tutulmaktadır.
  - **Parquet:** Yüksek performanslı okuma/yazma işlemleri ve Hugging Face dataset yüklemeleri için işlenmiş veriler `data/parquets/` dizininde sıkıştırılmış olarak saklanmaktadır.

### 3. Kullanılan Modeller ve Yapay Zeka Mimarisi
Ana model mimarisi `model/architecture.py` dosyası üzerinde tanımlanmıştır.

- **Temel Model:** `microsoft/graphcodebert-base` pre-trained (ön-eğitimli) modeli kullanılmıştır.
- **Mimari Bileşenler:**
  - **Memory Bank:** Kayan pencere (sliding window) yaklaşımıyla, önceki kod pencerelerinin en yüksek dikkat (attention) ağırlığına sahip token bilgilerini cross-attention mekanizması ile taşıyan bir bellek modülü.
  - **Hiyerarşik Dikkat (Hierarchical Attention):** Token seviyesinden başlayarak ifade (statement) ve fonksiyon seviyesine doğru çıkan 3 seviyeli bir dikkat havuzu (attention pooling) sistemi.
  - **Multi-Task Heads (Görev Başlıkları):**
    - `VulnerabilityHead`: 50 farklı CWE (Common Weakness Enumeration) sınıfına göre zafiyet tahmini.
    - `ComplexityHead`: 8 farklı Big-O sınıfı tahmini (O(1), O(n), O(n^2), vs.).
    - `DataFlowHead`: İki sınıflı (Binary) veri akışı/taint tespiti (Var/Yok).

### 4. Statik Analiz Entegrasyonu
Yapay zeka çıkarımlarının yanı sıra statik analiz araçları da projeye entegre edilmiştir.
- **Lizard (`src/core/api.py`):** Kodun satır sayısı (NLOC), Cyclomatic Complexity değerleri ve fonksiyon metriklerini tespit eder.
- AST ve DFG çıkarımları `src/extractors` modülleri (örn: `build_parser.py`, `dfg_extractor.py`) ile sağlanır. Ağaç yapıları üzerinden yapısal normalizasyon ve sekürite özellikleri `src/normalizers` klasöründeki betikler ile işlenir.

### 5. API ve Kullanıcı Arayüzü
- **Backend (`src/core/api.py`):** `FastAPI` ile yazılmış olup `uvicorn` üzerinden asenkron olarak 8000 portunda servis verir. İstekler hem statik analiz hem de GraphCodeBERT modeli ile değerlendirilip JSON olarak döner. Model, bellek yönetimini optimize etmek amacıyla `Singleton` deseniyle yüklenmektedir.
- **Frontend (`src/core/frontend/app.py`):** `Streamlit` kullanılarak geliştirilmiş, son kullanıcıların kod bloklarını analiz edip, hataları görsel olarak görebilecekleri ve sonuçları PDF olarak alabilecekleri (`pdf_generator.py`) dinamik bir arayüzdür.

### 6. Model Doğruluk Metrikleri (Accuracy Metrics)
Modelin çoklu görev (multi-task) yapısından dolayı her bir başlık için ayrı değerlendirme metrikleri (Accuracy, F1-Score, Precision, Recall) kullanılmaktadır:
- **Güvenlik (Vulnerability/CWE):** Zafiyetli ve zafiyetsiz kod ayrımında ağırlıklı F1-Score kullanılır. Dengesiz veri setleri için Focal Loss ve ağırlıklandırma (Class Weight) stratejileri uygulanmaktadır.
- **Zaman/Alan Karmaşıklığı (Complexity):** Big-O sınıflarının doğru tahmin oranı (Accuracy) ile ölçülür. O(n) ve O(1) gibi baskın sınıflar için özel örneklem stratejileri uygulanmıştır.
- **Veri Akışı (Data Flow/Taint):** Hassas verinin dışarı akıp akmadığını belirleyen ikili sınıflandırmada (Binary Classification) Recall (Duyarlılık) metriği kritik olarak ele alınır ve yüksek tutulması hedeflenir.

*(Not: Nihai doğruluk yüzdeleri, eğitim döngüsünün tam olarak tamamlanmasının ardından sistem loglarına kaydedilmek üzere tasarlanmıştır.)*

### 7. Yapay Zeka Halüsinasyonu (Hallucination) ve Alınan Önlemler
Genel amaçlı Büyük Dil Modellerinde (LLM) sıkça karşılaşılan "halüsinasyon" (olmayan fonksiyon uydurma, alakasız kod üretme) problemi, bu projenin temel mimari tasarımı sayesinde yapısal olarak ortadan kaldırılmıştır:
- **Encoder-Only Tasarım:** Kullanılan GraphCodeBERT, yeni bir metin veya kod üretmeyen (non-autoregressive) sadece var olan kodu analiz edip sınıflandıran bir modeldir. Girdi olarak verilen kod dışında hayali bir çıktı üretemez.
- **Deterministik Sınıflandırma:** Çıktılar serbest metin formatında değil, matematiksel olasılıklara dayanan kategorik sınıflar (örn: CWE-78, O(n)) şeklindedir.
- **Grafik ve Ağaç Tabanlı Doğrulama:** Model, düz metin (plain text) yerine kodun yapısal sözdizimi ağacı (AST) ve veri akış grafiği (DFG) üzerinden eğitildiği için, kodun çalışma mantığı dışına çıkması engellenmiştir.

### 8. Dağıtım (Deployment)
Sistem konteynerize edilmiş durumdadır:
- `docker-compose.yml` dosyası backend ve frontend servislerini ayağa kaldırır. Ortam değişkenleri `.env` üzerinden, donanım kaynakları ise volume mount işlemleri ile host makineyle (data, checkpoints, local_graphcodebert_base vb.) paylaşılır.

---
**Tarih:** 05.06.2026
**Raporu Hazırlayan:** AI Asistanı
