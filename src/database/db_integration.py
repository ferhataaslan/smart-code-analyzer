#!/usr/bin/env python3
"""
db_integration.py — Hibrit Veritabanı Entegrasyon Modülü

3 normalizasyon algoritmasının çıktısını hibrit veritabanına yazar.

Mimari:
  - Document DB tarafı (SQLite JSON): raw_snippet (orijinal ham kod)
  - SQL / Vektör tarafı (SQLite indeksli): normalized_structure (eğitim verisi)

Entity Şeması:
  - entity_id          (UUID)
  - source_system      (VARCHAR: GITHUB, HUGGINGFACE, OWASP)
  - raw_snippet        (Orijinal ham kod — Document DB yapısında)
  - normalized_structure (Normalize dizi/JSON — SQL/Vektör indeksli)
  - applied_algorithm  (Uygulanan script adı)
  - ast_metadata       (GitHub verileri için)
  - nl_alignment       (Hugging Face verileri için)
  - security_context   (OWASP verileri için)
"""

import json
import uuid
import hashlib
import sqlite3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field

# Normalizasyon motorları
from src.normalizers.github_ast_structnorm import normalize_github_code
from src.normalizers.hf_crossnorm import normalize_hf_code
from src.normalizers.owasp_secnorm import normalize_owasp_code

# Risk Puanlama ve DFG modülleri
from src.extractors.risk_scoring import calculate_final_risk, get_base_score, calculate_attack_surface, calculate_criticality
from src.extractors.dfg_extractor import extract_dfg
from src.extractors.complexity_analyzer import estimate_complexity

logger = logging.getLogger(__name__)

# ============================================================================
# Veritabanı Dosyaları
# ============================================================================
DOCUMENT_DB_FILE: str = "data/dbs/document_store.db"    # Ham veri (NoSQL-benzeri)
VECTOR_DB_FILE: str = "data/dbs/vector_store.db"        # Normalize eğitim verisi (SQL/Vektör)


# ============================================================================
# Entity Veri Yapısı
# ============================================================================
@dataclass
class NormalizationEntity:
    """Hibrit veritabanı entity şeması."""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_system: str = ""               # GITHUB | HUGGINGFACE | OWASP
    raw_snippet: str = ""                  # Orijinal ham kod
    normalized_structure: str = ""         # Normalize edilmiş dizi/JSON
    applied_algorithm: str = ""            # Uygulanan script adı
    ast_metadata: Optional[Dict] = None    # GitHub verileri için
    nl_alignment: Optional[Dict] = None    # Hugging Face verileri için
    security_context: Optional[Dict] = None  # OWASP verileri için
    data_flow_graph: Optional[List] = None  # DFG kenar listesi (JSON)
    complexity: str = "1"                  # Big O Zaman Karmaşıklığı
    code_hash: str = ""                    # SHA-256 hash (dedup için)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ============================================================================
# Document Store (NoSQL-Benzeri) — Ham Veri
# ============================================================================
class DocumentStore:
    """
    Ham kod parçalarını (raw_snippet) document-oriented yapıda saklar.
    SQLite JSON1 extension kullanarak NoSQL benzeri sorgulama sağlar.
    """

    def __init__(self, db_path: str = DOCUMENT_DB_FILE) -> None:
        self._db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        """Document store şemasını oluşturur."""
        try:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    entity_id       TEXT PRIMARY KEY,
                    source_system   TEXT NOT NULL,
                    raw_snippet     TEXT NOT NULL,
                    doc_metadata    TEXT DEFAULT '{}',
                    code_hash       TEXT,
                    created_at      TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_source
                ON documents(source_system)
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_hash
                ON documents(code_hash)
            """)
            conn.commit()
            conn.close()
            logger.info("[DocumentStore] Şema başarıyla oluşturuldu.")
        except Exception as e:
            logger.error(f"[DocumentStore] Şema oluşturma hatası: {e}")
            raise

    def insert(self, entity: NormalizationEntity) -> None:
        """Ham veriyi document store'a yazar."""
        try:
            conn = self._get_connection()
            # Platform'a göre meta-veri seç
            meta = {}
            if entity.ast_metadata:
                meta["ast_metadata"] = entity.ast_metadata
            if entity.nl_alignment:
                meta["nl_alignment"] = entity.nl_alignment
            if entity.security_context:
                meta["security_context"] = entity.security_context
            if entity.data_flow_graph:
                meta["data_flow_graph"] = entity.data_flow_graph

            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (entity_id, source_system, raw_snippet, doc_metadata, code_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity.entity_id,
                    entity.source_system,
                    entity.raw_snippet,
                    json.dumps(meta, ensure_ascii=False),
                    entity.code_hash,
                    entity.created_at,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[DocumentStore] Ekleme hatası: {e}")
            raise

    def hash_exists(self, code_hash: str) -> bool:
        """Verilen SHA-256 hash değerinin document store'da olup olmadığını kontrol eder."""
        if not code_hash:
            return False
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT 1 FROM documents WHERE code_hash = ? LIMIT 1", (code_hash,)
            ).fetchone()
            conn.close()
            return row is not None
        except Exception as e:
            logger.error(f"[DocumentStore] Hash kontrol hatası: {e}")
            return False

    def get_by_id(self, entity_id: str) -> Optional[Dict]:
        """Entity ID ile belge sorgular."""
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM documents WHERE entity_id = ?", (entity_id,)
            ).fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"[DocumentStore] Sorgulama hatası: {e}")
            return None

    def get_by_source(self, source_system: str) -> List[Dict]:
        """Kaynak sisteme göre belgeleri listeler."""
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM documents WHERE source_system = ?",
                (source_system,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[DocumentStore] Sorgulama hatası: {e}")
            return []


# ============================================================================
# Vector Store (SQL/Vektör) — Normalize Eğitim Verisi
# ============================================================================
class VectorStore:
    """
    Normalize edilmiş eğitim verisini (normalized_structure) indeksli
    SQL yapısında saklar. Vektör aramaları için hazır.
    """

    def __init__(self, db_path: str = VECTOR_DB_FILE) -> None:
        self._db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        """Vector store şemasını oluşturur."""
        try:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    entity_id              TEXT PRIMARY KEY,
                    source_system          TEXT NOT NULL,
                    normalized_structure   TEXT NOT NULL,
                    applied_algorithm      TEXT NOT NULL,
                    ast_metadata           TEXT DEFAULT '{}',
                    nl_alignment           TEXT DEFAULT '{}',
                    security_context       TEXT DEFAULT '{}',
                    data_flow_graph        TEXT DEFAULT '[]',
                    complexity             TEXT DEFAULT '1',
                    code_hash              TEXT,
                    created_at             TEXT NOT NULL
                )
            """)
            # data_flow_graph sütunu mevcut tablolara güvenli ekleme (migration)
            try:
                conn.execute("ALTER TABLE vectors ADD COLUMN data_flow_graph TEXT DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass  # Sütun zaten var
            # complexity sütunu mevcut tablolara güvenli ekleme (migration)
            try:
                conn.execute("ALTER TABLE vectors ADD COLUMN complexity TEXT DEFAULT '1'")
            except sqlite3.OperationalError:
                pass  # Sütun zaten var
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vec_source
                ON vectors(source_system)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vec_algorithm
                ON vectors(applied_algorithm)
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_vec_hash
                ON vectors(code_hash)
            """)
            conn.commit()
            conn.close()
            logger.info("[VectorStore] Şema başarıyla oluşturuldu.")
        except Exception as e:
            logger.error(f"[VectorStore] Şema oluşturma hatası: {e}")
            raise

    def insert(self, entity: NormalizationEntity) -> None:
        """Normalize veriyi vector store'a yazar."""
        try:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO vectors
                    (entity_id, source_system, normalized_structure,
                     applied_algorithm, ast_metadata, nl_alignment,
                     security_context, data_flow_graph, complexity, code_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity.entity_id,
                    entity.source_system,
                    entity.normalized_structure,
                    entity.applied_algorithm,
                    json.dumps(entity.ast_metadata or {}, ensure_ascii=False),
                    json.dumps(entity.nl_alignment or {}, ensure_ascii=False),
                    json.dumps(entity.security_context or {}, ensure_ascii=False),
                    json.dumps(entity.data_flow_graph or [], ensure_ascii=False),
                    entity.complexity,
                    entity.code_hash,
                    entity.created_at,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[VectorStore] Ekleme hatası: {e}")
            raise

    def get_by_id(self, entity_id: str) -> Optional[Dict]:
        """Entity ID ile normalize veri sorgular."""
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM vectors WHERE entity_id = ?", (entity_id,)
            ).fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"[VectorStore] Sorgulama hatası: {e}")
            return None

    def get_by_source(self, source_system: str) -> List[Dict]:
        """Kaynak sisteme göre normalize verileri listeler."""
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM vectors WHERE source_system = ?",
                (source_system,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[VectorStore] Sorgulama hatası: {e}")
            return []

    def count_by_source(self) -> Dict[str, int]:
        """Kaynak sistemlere göre kayıt sayılarını döner."""
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT source_system, COUNT(*) as cnt FROM vectors GROUP BY source_system"
            ).fetchall()
            conn.close()
            return {r["source_system"]: r["cnt"] for r in rows}
        except Exception as e:
            logger.error(f"[VectorStore] Sayım hatası: {e}")
            return {}


# ============================================================================
# Ana Pipeline: Senkronize Hibrit Yazıcı
# ============================================================================
class HybridDBPipeline:
    """
    3 normalizasyon algoritmasının çıktısını hibrit veritabanına
    senkronize ve kayıpsız yazan pipeline.

    Akış:
      1. Kaynak sisteme göre uygun normalizasyon algoritmasını seç
      2. Ham veriyi (raw_snippet) Document Store'a yaz
      3. Normalize veriyi (normalized_structure) Vector Store'a yaz
      4. Her iki yazma atomik olarak gerçekleşir (senkronize)
    """

    def __init__(
        self,
        doc_db_path: str = DOCUMENT_DB_FILE,
        vec_db_path: str = VECTOR_DB_FILE,
    ) -> None:
        self._doc_store = DocumentStore(db_path=doc_db_path)
        self._vec_store = VectorStore(db_path=vec_db_path)
        logger.info("[HybridDBPipeline] Pipeline hazır.")

    # ── Hash & Dedup Yardımcıları ─────────────────────────────────────────

    @staticmethod
    def compute_hash(raw_code: str) -> str:
        """Ham kod için SHA-256 hash hesaplar."""
        return hashlib.sha256(raw_code.encode("utf-8")).hexdigest()

    def check_duplicate(self, code_hash: str) -> bool:
        """
        Verilen hash değerinin document_store'da zaten mevcut olup olmadığını kontrol eder.
        True dönerse veri TEKRAR (duplicate) demektir, atlanmalıdır.
        """
        return self._doc_store.hash_exists(code_hash)

    # ── Platform-Spesifik İşleme ─────────────────────────────────────────

    def _process_github(
        self, raw_code: str, language: str = "cpp",
        code_hash: str = "", security_context: Optional[Dict] = None,
        platform_metadata: Optional[Dict] = None
    ) -> NormalizationEntity:
        """GitHub verisini normalize eder ve entity oluşturur."""
        result = normalize_github_code(raw_code, language=language)

        # DFG Çıkarımı
        dfg_edges = extract_dfg(raw_code, language=language)

        # Risk Puanlaması — security_context içinden is_vulnerable ve cwe_ids çıkar
        is_vulnerable = False
        cwe_id = None
        if security_context:
            is_vulnerable = security_context.get("is_vulnerable", False)
            cwe_ids = security_context.get("cwe_ids", [])
            cwe_id = cwe_ids[0] if cwe_ids else None

        risk_info = calculate_final_risk(
            cwe_id=cwe_id,
            is_vulnerable=is_vulnerable,
            raw_code=raw_code,
            source="github",
            metadata=platform_metadata,
        )

        # Risk bilgisini security_context'e kaydet
        if security_context is None:
            security_context = {}
        security_context["risk_scoring"] = risk_info

        entity = NormalizationEntity(
            source_system="GITHUB",
            raw_snippet=raw_code,
            normalized_structure=result.get("normalized_code", ""),
            applied_algorithm=result.get("applied_algorithm", "github_ast_structnorm"),
            ast_metadata=result.get("ast_metadata"),
            security_context=security_context,
            data_flow_graph=dfg_edges,
            complexity=estimate_complexity(raw_code, language=language),
            code_hash=code_hash,
        )

        # SBT dizisini ast_metadata içine ekle
        if entity.ast_metadata is not None:
            entity.ast_metadata["sbt_sequence_preview"] = (
                result.get("ast_sbt_sequence", "")[:500]
            )

        return entity

    def _process_huggingface(
        self, raw_code: str, docstring: str = "",
        code_hash: str = "", security_context: Optional[Dict] = None,
        platform_metadata: Optional[Dict] = None
    ) -> NormalizationEntity:
        """Hugging Face verisini normalize eder ve entity oluşturur."""
        result = normalize_hf_code(raw_code, docstring=docstring)

        # DFG Çıkarımı
        dfg_edges = extract_dfg(raw_code, language="cpp")

        # Risk Puanlaması
        is_vulnerable = False
        cwe_id = None
        if security_context:
            is_vulnerable = security_context.get("is_vulnerable", False)
            cwe_ids = security_context.get("cwe_ids", [])
            cwe_id = cwe_ids[0] if cwe_ids else None

        risk_info = calculate_final_risk(
            cwe_id=cwe_id,
            is_vulnerable=is_vulnerable,
            raw_code=raw_code,
            source="hf",
            metadata=platform_metadata,
        )

        # Risk bilgisini security_context'e kaydet
        if security_context is None:
            security_context = {}
        security_context["risk_scoring"] = risk_info

        nl_align = result.get("nl_alignment") or {}
        nl_align["normalized_docstring"] = result.get("normalized_docstring", "")
        nl_align["chunks"] = result.get("chunks", [])

        entity = NormalizationEntity(
            source_system="HUGGINGFACE",
            raw_snippet=raw_code,
            normalized_structure=result.get("normalized_code", ""),
            applied_algorithm=result.get("applied_algorithm", "hf_crossnorm"),
            nl_alignment=nl_align,
            security_context=security_context,
            data_flow_graph=dfg_edges,
            complexity=estimate_complexity(raw_code, language="cpp"),
            code_hash=code_hash,
        )
        return entity

    def _process_owasp(
        self, raw_code: str, language: str = "cpp", cwe_hint: str = "Unknown",
        code_hash: str = ""
    ) -> NormalizationEntity:
        """OWASP verisini normalize eder ve entity oluşturur."""
        result = normalize_owasp_code(
            raw_code, language=language, cwe_hint=cwe_hint
        )

        # DFG Çıkarımı
        dfg_edges = extract_dfg(raw_code, language=language)

        # Risk Puanlaması — OWASP verisi için security_context'ten al
        sec_ctx = result.get("security_context", {}) or {}
        cwe_ids = sec_ctx.get("cwe_ids", [])
        is_vulnerable = bool(cwe_ids) or bool(sec_ctx.get("taint_paths", []))
        first_cwe = cwe_ids[0] if cwe_ids else (cwe_hint if cwe_hint != "Unknown" else None)

        risk_info = calculate_final_risk(
            cwe_id=first_cwe,
            is_vulnerable=is_vulnerable,
            raw_code=raw_code,
            source="owasp",
        )

        # Risk bilgisini security_context'e kaydet
        sec_ctx["risk_scoring"] = risk_info

        entity = NormalizationEntity(
            source_system="OWASP",
            raw_snippet=raw_code,
            normalized_structure=result.get("normalized_code", ""),
            applied_algorithm=result.get("applied_algorithm", "owasp_secnorm"),
            security_context=sec_ctx,
            data_flow_graph=dfg_edges,
            complexity=estimate_complexity(raw_code, language=language),
            code_hash=code_hash,
        )
        return entity

    # ── Senkronize Yazma ─────────────────────────────────────────────────

    def _write_synchronized(self, entity: NormalizationEntity) -> str:
        """
        Entity'yi her iki store'a atomik olarak yazar.
        Herhangi bir hata durumunda exception fırlatır.

        Returns:
            entity_id
        """
        try:
            # 1) Document Store: ham veri
            self._doc_store.insert(entity)
            logger.debug(
                f"[Pipeline] Document Store yazıldı: {entity.entity_id}"
            )

            # 2) Vector Store: normalize veri
            self._vec_store.insert(entity)
            logger.debug(
                f"[Pipeline] Vector Store yazıldı: {entity.entity_id}"
            )

            return entity.entity_id

        except Exception as e:
            logger.error(
                f"[Pipeline] Senkronize yazma hatası "
                f"(entity={entity.entity_id}): {e}"
            )
            raise

    # ── Public API ───────────────────────────────────────────────────────

    def ingest_github(
        self, raw_code: str, language: str = "cpp",
        security_context: Optional[Dict] = None,
        platform_metadata: Optional[Dict] = None
    ) -> str:
        """
        GitHub ham kodunu işler ve hibrit DB'ye yazar.
        Dedup kontrolü yapılır: hash mevcutsa atlanır.

        Args:
            raw_code: Ham C/C++ kaynak kodu
            language: "c" veya "cpp"
            security_context: Opsiyonel auto-label sonucu
            platform_metadata: GitHub API meta verileri (stars vb.)

        Returns:
            entity_id (UUID) veya "" (duplicate ise)
        """
        try:
            code_hash = self.compute_hash(raw_code)
            if self.check_duplicate(code_hash):
                logger.info(f"[Pipeline] GitHub DEDUP: Hash zaten mevcut, atlanıyor.")
                return ""
            entity = self._process_github(
                raw_code, language, code_hash, security_context, platform_metadata
            )
            eid = self._write_synchronized(entity)
            logger.info(f"[Pipeline] GitHub verisi eklendi: {eid}")
            return eid
        except Exception as e:
            logger.error(f"[Pipeline] GitHub ingest hatası: {e}")
            raise

    def ingest_huggingface(
        self, raw_code: str, docstring: str = "",
        security_context: Optional[Dict] = None,
        platform_metadata: Optional[Dict] = None
    ) -> str:
        """
        Hugging Face ham kodunu işler ve hibrit DB'ye yazar.
        Dedup kontrolü yapılır: hash mevcutsa atlanır.

        Args:
            raw_code: Ham C/C++ kaynak kodu
            docstring: NL açıklama metni
            security_context: Opsiyonel auto-label sonucu
            platform_metadata: HF dataset meta verileri (downloads vb.)

        Returns:
            entity_id (UUID) veya "" (duplicate ise)
        """
        try:
            code_hash = self.compute_hash(raw_code)
            if self.check_duplicate(code_hash):
                logger.info(f"[Pipeline] HuggingFace DEDUP: Hash zaten mevcut, atlanıyor.")
                return ""
            entity = self._process_huggingface(
                raw_code, docstring, code_hash, security_context, platform_metadata
            )
            eid = self._write_synchronized(entity)
            logger.info(f"[Pipeline] HuggingFace verisi eklendi: {eid}")
            return eid
        except Exception as e:
            logger.error(f"[Pipeline] HuggingFace ingest hatası: {e}")
            raise

    def ingest_owasp(
        self, raw_code: str, language: str = "cpp", cwe_hint: str = "Unknown"
    ) -> str:
        """
        OWASP ham kodunu işler ve hibrit DB'ye yazar.
        Dedup kontrolü yapılır: hash mevcutsa atlanır.
        Not: OWASP verisinin security_context'i owasp_secnorm tarafından üretilir,
        dışarıdan eklenmez.

        Args:
            raw_code: Ham C/C++ kaynak kodu
            language: "c" veya "cpp"
            cwe_hint: Bilinen CWE kimliği

        Returns:
            entity_id (UUID) veya "" (duplicate ise)
        """
        try:
            code_hash = self.compute_hash(raw_code)
            if self.check_duplicate(code_hash):
                logger.info(f"[Pipeline] OWASP DEDUP: Hash zaten mevcut, atlanıyor.")
                return ""
            entity = self._process_owasp(raw_code, language, cwe_hint, code_hash)
            eid = self._write_synchronized(entity)
            logger.info(f"[Pipeline] OWASP verisi eklendi: {eid}")
            return eid
        except Exception as e:
            logger.error(f"[Pipeline] OWASP ingest hatası: {e}")
            raise

    def ingest(
        self,
        raw_code: str,
        source_system: str,
        language: str = "cpp",
        docstring: str = "",
        cwe_hint: str = "Unknown",
        security_context: Optional[Dict] = None,
        platform_metadata: Optional[Dict] = None,
    ) -> str:
        """
        Genel giriş noktası — kaynak sisteme göre doğru algoritmayı seçer.
        SHA-256 dedup kontrolü her platform için otomatik yapılır.
        Risk puanlama ve DFG çıkarımı otomatik uygulanır.

        Args:
            raw_code: Ham kaynak kod
            source_system: "GITHUB" | "HUGGINGFACE" | "OWASP"
            language: "c" veya "cpp"
            docstring: NL metin (sadece HF için)
            cwe_hint: CWE kimliği (sadece OWASP için)
            security_context: Opsiyonel auto-label sonucu (GitHub/HF için)
            platform_metadata: Platform meta verileri (stars, downloads vb.)

        Returns:
            entity_id (UUID) veya "" (duplicate ise)
        """
        source_upper = source_system.upper()

        if source_upper == "GITHUB":
            return self.ingest_github(raw_code, language, security_context, platform_metadata)
        elif source_upper in ("HUGGINGFACE", "HF"):
            return self.ingest_huggingface(raw_code, docstring, security_context, platform_metadata)
        elif source_upper == "OWASP":
            return self.ingest_owasp(raw_code, language, cwe_hint)
        else:
            raise ValueError(
                f"Bilinmeyen source_system: {source_system}. "
                f"Geçerli değerler: GITHUB, HUGGINGFACE, OWASP"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Veritabanı istatistiklerini döner."""
        return {
            "vector_store_counts": self._vec_store.count_by_source(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================================
# CLI Demo
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pipeline = HybridDBPipeline()

    # ── GitHub demo ──
    github_sample = '''
    #include <stdio.h>
    #include <stdlib.h>
    int compute(int *arr, int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) sum += arr[i];
        return sum;
    }
    int main() {
        int *buf = (int*)malloc(10 * sizeof(int));
        printf("Result: %d\\n", compute(buf, 10));
        free(buf);
        return 0;
    }
    '''
    eid_gh = pipeline.ingest(github_sample, "GITHUB", language="c")
    print(f"[+] GitHub entity: {eid_gh}")

    # ── HuggingFace demo ──
    hf_sample = '''
    void process_auth_token(const char *auth_token, int retry_count) {
        char *session = (char*)malloc(256);
        strcpy(session, auth_token);
        for (int i = 0; i < retry_count; i++) {
            printf("Attempt %d\\n", i);
        }
        free(session);
    }
    '''
    hf_doc = "Processes auth_token with retry_count attempts using session buffer."
    eid_hf = pipeline.ingest(hf_sample, "HUGGINGFACE", docstring=hf_doc)
    print(f"[+] HuggingFace entity: {eid_hf}")

    # ── OWASP demo ──
    owasp_sample = '''
    #include <stdio.h>
    #include <string.h>
    void vuln(char *input) {
        char buf[64];
        strcpy(buf, input);
        printf("%s\\n", buf);
    }
    '''
    eid_ow = pipeline.ingest(owasp_sample, "OWASP", language="c", cwe_hint="CWE-120")
    print(f"[+] OWASP entity: {eid_ow}")

    # ── İstatistikler ──
    stats = pipeline.get_stats()
    print(f"\n=== Pipeline Stats ===")
    print(json.dumps(stats, indent=2))
