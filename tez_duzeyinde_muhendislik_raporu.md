# Akıllı Kod Analizatörü (Smart Code Analyzer)
## Tez Düzeyinde Mimari ve Algoritmik Analiz Raporu

**Rapor Sürümü:** 1.0 (Akademik & İleri Mühendislik İncelemesi)  
**Tarih:** 05.06.2026

---

## 1. Özet (Abstract)

C/C++ kaynak kodlarının güvenliğini, zaman/alan karmaşıklığını ve veri akışı tehlikelerini eşzamanlı olarak tespit etmeyi amaçlayan bu proje, Geleneksel Statik Kod Analiz (SCA) yöntemleri ile Derin Öğrenme (Deep Learning) tabanlı Doğal Dil İşleme (NLP) teknolojilerini birleştiren hibrit bir yapay zeka sistemidir.

Projenin merkezinde, kaynak kodları sadece metin (text) olarak değil, yapısal ağaçlar (AST - Abstract Syntax Tree) ve veri akış grafikleri (DFG - Data Flow Graph) olarak anlayabilen, Microsoft tarafından geliştirilmiş **GraphCodeBERT** mimarisi yer almaktadır. Ancak standart GraphCodeBERT'in 512 tokenlık bellek sınırını aşmak amacıyla projeye özgün **Memory Bank (Hafıza Bankası)** ve **Hierarchical Attention (Hiyerarşik Dikkat)** katmanları eklenmiş, sistem sıfırdan **Multi-Task Learning (Çoklu Görev Öğrenimi)** yapısına büründürülmüştür. 

Bu rapor, projenin en alt katmanından (veritabanı yönetimi) en üst katmanına (FastAPI ve Risk Algoritmaları) kadar olan mimari ve kod seviyesindeki tercihleri en ince ayrıntısıyla analiz etmektedir.

---

## 2. Veri İşleme ve Veritabanı Katmanı (Data & Storage Layer)

Sistemin veri toplama ve işleme boru hattı (pipeline) iki temel mekanizma üzerine inşa edilmiştir: SQLite ve Parquet.

### 2.1. İlişkisel Veritabanı Mimarisi (`src/database/database.py`)
Model için toplanacak veriler ve kullanıcıların arayüzden yapacağı sorgular, anlık durum takibi gerektirdiği için `review_state.db` isimli bir SQLite veritabanında tutulmaktadır. 
- **`records` Tablosu**: `id`, `source`, `cwe_id`, `raw_code`, `processed_code`, `status`, `device_id` ve `timestamp` sütunlarından oluşur. 
- **Veri Tutarlılığı:** `insert_record`, `update_status` ve `get_pending_record` fonksiyonlarında veritabanı bağlantıları (cursor) fonksiyon bazlı açılıp kapatılarak (Context Isolation) olası eşzamanlı okuma/yazma çakışmalarının önüne geçilmiştir.

### 2.2. O(1) Parquet Serileştirmesi (`src/database/collector.py`)
Milyonlarca satır C/C++ kodunun Hugging Face Dataset objelerine dönüştürülmesi için geleneksel JSON formatı yerine **Apache Parquet** tercih edilmiştir. 
Parquet yapısının kolon-bazlı (columnar storage) formatı sayesinde, modelin eğitim esnasında sadece "etiketler" veya sadece "ast_metadata" sütunlarını çekmesi gerektiğinde tüm dosyayı RAM'e yüklemesine gerek kalmaz, O(1) karmaşıklığında doğrudan belleğe transfer sağlanır.

---

## 3. Kaynak Kod Çıkarım ve Normalizasyon Katmanı (Extraction & Normalization)

Projenin en yenilikçi ve agresif mühendislik yaklaşımlarından biri bu katmanda yer almaktadır. Ham C/C++ kodları (makrolar, minified kodlar, anlamsız değişken isimleri vb.) model için aşırı yüksek bir varyans (gürültü) oluşturur. Bu gürültüyü gidermek için `tree-sitter` kütüphanesi üzerine özgün bir **Capture-Avoiding Alpha-Conversion (Kapsam Kaçınan Alfa-Dönüşümü)** algoritması yazılmıştır.

### 3.1. Full-Fidelity AST Ayrıştırma (`src/extractors/build_parser.py`)
Sistem, statik C dili kurallarına bağımlı kalmamak adına `tree_sitter_c` ve `tree_sitter_cpp` kullanarak kaynak kodların soyut sözdizim ağaçlarını (AST) milisaniyeler içerisinde çıkartır. Bu sayede kod derlenemez (syntax error) durumda olsa dahi hata düğümleri (ERROR nodes) izole edilerek kodun geri kalanından semantik anlam çıkartılabilir.

### 3.2. Yapısal Normalizasyon: Alpha-Conversion Algoritması (`src/normalizers/github_ast_structnorm.py`)
Algoritma şu şekilde işler:
1. **DFS (Depth-First Search) ile Tanımlayıcı Toplama:** Kodun en üst düğümünden başlayarak tüm değişken (`identifier`), tip (`type_identifier`) ve fonksiyon (`function_declarator`) isimleri toplanır.
2. **Allow-List (İzin Listesi) Filtrasyonu:** C/C++ dilinin standart anahtar kelimeleri (`int`, `malloc`, `FILE`, `AF_INET`, `sizeof`) 120+ kelimelik bir "Beyaz Liste" (Allow-List) ile korunur. Bu sayede `malloc` fonksiyonu `FUNC_1` olarak değiştirilip semantik anlamı yok edilmez.
3. **Scope (Kapsam) Takibi:** Değişkenler global olarak `VAR_1`, `VAR_2` diye değiştirilmez. Algoritma kendi içerisinde bir `Scope` sınıfı tanımlar (`CaptureAvoidingSymbolTable`). Eğer `if` bloğu içerisinde yerel bir değişken tanımlanmışsa (Shadowing), üst kapsamdaki (parent scope) değişkenin ismi ezilmez.
4. **AST -> SBT Serileştirme:** Alfa-Dönüşümü uygulanmış kod, salt metin olarak modele verilmeden önce **SBT (Structure-Based Traversal)** adı verilen bir parantezli dizi diline çevrilir. Örn: `( if_statement ( condition ) ... ) if_statement`. Bu sayede transformer modeli if bloğunun nerede başlayıp nerede bittiğini matematiksel olarak öğrenir.

### 3.3. Veri Akış Grafiği (DFG) Çıkarımı (`src/extractors/dfg_extractor.py`)
`dfg_extractor.py` değişken atamalarını ve kullanım yerlerini analiz ederek kod üzerindeki "Taint (kirlilik/hassas veri) Analizini" simüle eden bir yönlü grafik (Directed Graph) çizer. Bir değişkene değer atanması "yazma" (write), fonksiyon parametresi olarak iletilmesi "okuma" (read) düğümleri (edges) olarak çıkartılır ve doğrudan GraphCodeBERT modelinin özel dikkat maskesine (attention mask) entegre edilir.

---

## 4. Model Katmanı: Özgün Çoklu Görev Mimarisi (Deep Learning Architecture)

Bu projenin omurgası, `model/architecture.py` içerisinde tanımlanan **GraphCodeBERT tabanlı Multi-Task Transformer** modelidir.

### 4.1. 512 Token Sınırının Aşılması: Memory Bank (Hafıza Bankası)
Standart Transformer modellerinin `O(N^2)` olan Attention karmaşıklığından dolayı girdi sınırı 512 token'dır. Ancak C/C++ dosyaları genellikle binlerce satır sürer. Bu sınır şu özgün algoritma ile aşılmıştır:
- Kodlar 512 tokenlık parçalara (chunk) 64 tokenlık örtüşmeler (overlap) ile bölünür.
- İlk parça kodlayıcıdan (encoder) geçirildiğinde, **en yüksek Attention skoru alan K adet token (örn: 64 token)** seçilerek modele özel bir `MemoryBank` sınıfına (FIFO bellek) kaydedilir.
- İkinci parça işlenirken, Cross-Attention Layer devreye girer: İkinci parçanın `Q (Query)` matrisi ile `MemoryBank` içerisindeki eski tokenların `K (Key)` ve `V (Value)` matrisleri çarpılır.
- **Sonuç:** Model 2000. satırdaki bir fonksiyonu incelerken, 50. satırda (bellekteki) yapılmış `malloc` tanımını hatırlayabilir.

### 4.2. Hiyerarşik Dikkat (Hierarchical Attention Pooling)
Geleneksel modeller sınıflandırma yapmak için kodun en başındaki `[CLS]` token'ını baz alır. Ancak bu projede 3 aşamalı bir havuzlama (Pooling) kullanılmıştır:
1. **Token Pooling**: Her cümlenin/satırın tokenları birleşip o ifadeyi (statement) temsil eden bir vektöre dönüşür.
2. **Statement Pooling**: İfadeler (if, for, return) fonksiyon vektörlerini oluşturur.
3. **Function Pooling**: Fonksiyon vektörleri, programın global `[CLS]` token'ını güncelleyerek global bağlamı son derece isabetli bir vektör uzayına çeker.

### 4.3. Multi-Task Learning (Çoklu Görev Öğrenimi) Heads
Modelin üç adet ayrı sınıflandırma "başı" (Classification Head) vardır:
1. **Vulnerability Head (CWE Tespiti):** 35 temel CWE (CWE-79, CWE-120 vb.) üzerinde çalışan Lineer bir sinir ağı katmanı. Çıktı olarak `CrossEntropyLoss` kullanarak olasılık (probability) dağılımı döner.
2. **Complexity Head (Big-O Tahmini):** $O(1), O(\log n), O(n), O(n \log n), O(n^2), O(2^n), O(n!)$ olmak üzere 8 sınıfa ayrıştırılmış karmaşıklık başlığı.
3. **Data Flow Head (Taint Tespiti):** Veri akışı tehlikelerinin olup olmadığını % (yüzdelik) cinsinden ölçen bir Binary (İkili) Classifier.

*Eğitim stratejisinde (`model/train_colab.py`), dengesiz veri sınıflarına (örn: O(1) çokken O(n!) azdır) karşı modelin önyargılı (bias) olmasını engellemek için Focal Loss türevi bir `Class-Weighted Loss` (Sınıf Ağırlıklı Kayıp) formülü uygulanmaktadır.*

---

## 5. API Entegrasyonu ve Risk Skorlama Algoritmaları

### 5.1. FastAPI Asenkron Mimari (`src/core/api.py`)
Yapay zeka modellerinin GPU/CPU RAM'lerini tüketmemesi (OOM - Out of Memory Error) için, model sunucu ilk açıldığında değil, "ilk istek geldiğinde" (`ModelSingleton` deseni ile) yüklenmektedir. Bellek temizliği Python Garbage Collector ve PyTorch `.eval()` ve `torch.no_grad()` metodları ile kesin olarak sağlanmıştır. Gelen istekler statik ve derin öğrenme motorlarına paralel (asenkron) dağıtılır.

### 5.2. Statik Doğrulama (Lizard) ve Hibrit Algoritma
Arayüzden kod girildiğinde sadece yapay zeka kullanılmaz. Aynı anda `lizard` kütüphanesi devreye girerek kodun NLOC (satır sayısı) ve Cyclomatic Complexity (Koşul karmaşıklığı) skorlarını fiziksel olarak hesaplar. Bu ikili çapraz kontrol (cross-validation) sistem güvenilirliğini artırır.

### 5.3. Dinamik Risk Skorlaması (`src/extractors/risk_scoring.py`)
Tespit edilen Zafiyetin ciddiyetini %100 üzerinden hesaplayan mühendislik formülü şudur:
`Risk Score = Kümülatif Base Score (CVSS) * Attack Surface Çarpanı`

- **Kümülatif Base Score:** Eğer kodda hem CWE-120 (Buffer Overflow - CVSS 12.0) hem de CWE-89 (SQLi - CVSS 28.7) bulunursa, bu iki skor düz toplanmaz. İstatistiksel olasılık formülü olan probablistik toplam: `1.0 - ((1.0 - s1) * (1.0 - s2))` formülü kullanılarak üst sınır (100) aşılmadan normalize edilir.
- **Attack Surface Çarpanı:** Çıkarılan Data Flow Graph (DFG) üzerindeki düğüm (edge) sayısı arttıkça, bu kod bloğuna erişim yollarının fazla olduğu (Saldırı yüzeyinin geniş olduğu) anlaşılır. 15 akış (flow) ve üzeri, çarpanı 1.0 (maksimum tehlike) seviyesine çeker. 0 akış ise 0.5 (izole kod) çarpanı üretir.

---

## 6. Model Doğruluk Metrikleri ve Halüsinasyon Riskinin Mimari Engeli

### 6.1. Metrik Tasarımları
Model değerlendirmesinde:
- **CWE Sınıflandırması:** Dengesiz zafiyet veritabanlarında salt Doğruluk (Accuracy) yanıltıcı olduğundan **Ağırlıklı F1-Score (Macro F1)** kullanılır.
- **Data Flow:** Tehlikeli durumların kaçırılmaması çok kritik olduğu için False Negative'i cezalandıran **Recall (Duyarlılık)** metriği hedeflenir.

### 6.2. Neden Halüsinasyon (Hallucination) Görülmez?
GPT-4 veya Llama gibi modeller "Oto-Regresif (Autoregressive - Generative)" modellerdir; boşlukları doldurarak yeni kod/yazı üretirler. Bu yüzden "olmayan kütüphaneleri" icat etme (halüsinasyon) eğilimindedirler.
**GraphCodeBERT ise bir Encoder-Only (Sadece Kodlayıcı) modeldir.** Hiçbir zaman metin çıktısı üretmez. Var olan kodu alır, devasa matrislere (vektör uzayına) dönüştürür ve sondaki sınıflandırma başlıklarında sadece 0-1 arası kesin sayılar üretir. Bu nedenle yapay zekanın "yanlış bir kod uydurması" matematiksel ve mimari olarak imkânsızdır.

---

## 7. Sonuç

Bu Akıllı Kod Analizörü;
1. **tree-sitter ve AST** ile derin kod manipülasyonu yapan,
2. Kapsam ve İzin listesi (Allow-List) takibi ile **Alpha-Conversion** yapabilen,
3. 512 tokenlık sınırı donanımsal mantıkla aşan kendi yazdığımız **MemoryBank Transformer** mimarisini kullanan,
4. Çıktılarını statik analiz ve **Kümülatif CVSS algoritmasıyla** kesin kurallara bağlayan **tam teşekküllü, son teknoloji bir DevSecOps aracıdır.** Mimarisi tamamen tez, bilimsel makale ve akademik sunumlara hazır bir zemin üzerine oturtulmuştur.
