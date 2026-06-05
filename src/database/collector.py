#!/usr/bin/env python3
import os
import sys
import re
import requests
import time
import argparse
import logging
import hashlib
import subprocess
import tempfile

from datasets import load_dataset
from src.database.database import insert_record, get_db_connection
from src.normalizers.processor import process_code
from src.database.db_integration import HybridDBPipeline

# .env dosyasından ortam değişkenlerini yükle (harici paket gerektirmez)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Task 2: Koşullu Otomatik Zafiyet Etiketleme (Auto-Labeling)
# ============================================================================
def auto_label_vulnerability(raw_code: str, language: str = "cpp") -> dict:
    """
    cppcheck ve flawfinder ile otomatik zafiyet etiketleme.

    Geçici dosyaya yazılan ham kodu SAST araçlarından geçirir.
    Eğer zafiyet bulunursa is_vulnerable: True ve CWE numaraları döner.
    Temiz çıkarsa is_vulnerable: False döner.

    Args:
        raw_code: Ham C/C++ kaynak kodu
        language: "c" veya "cpp"

    Returns:
        security_context dict:
            {
                "is_vulnerable": bool,
                "cwe_ids": list[str],
                "tool_findings": list[dict],
                "auto_labeled": True
            }
    """
    ext = ".c" if language == "c" else ".cpp"
    security_context = {
        "is_vulnerable": False,
        "cwe_ids": [],
        "tool_findings": [],
        "auto_labeled": True,
    }

    tmp_path = None
    try:
        # Geçici dosyaya yaz
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(raw_code)
            tmp_path = f.name

        # ── cppcheck (XML çıktı — cwe attribute yakalama) ──
        try:
            result = subprocess.run(
                ["cppcheck", "--enable=warning,style,portability",
                 "--force", "--quiet", "--xml", tmp_path],
                capture_output=True, text=True, timeout=30
            )
            # cppcheck XML çıktısını stderr'e yazar
            cppcheck_output = result.stderr or ""
            if cppcheck_output.strip():
                for line in cppcheck_output.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # XML <error> satırlarından bilgi çıkar
                    if "<error" in line or "error" in line.lower():
                        security_context["tool_findings"].append({
                            "tool": "cppcheck",
                            "detail": line,
                        })
                    # XML attribute: cwe="788" veya cwe="120"
                    cwe_attr = re.search(r'cwe="?(\d+)"?', line)
                    if cwe_attr:
                        cwe = f"CWE-{cwe_attr.group(1)}"
                        if cwe not in security_context["cwe_ids"]:
                            security_context["cwe_ids"].append(cwe)
                        security_context["is_vulnerable"] = True
                    # Metin formatı: CWE-120 veya CWE120
                    cwe_text = re.search(r'CWE-?(\d+)', line)
                    if cwe_text:
                        cwe = f"CWE-{cwe_text.group(1)}"
                        if cwe not in security_context["cwe_ids"]:
                            security_context["cwe_ids"].append(cwe)
                        security_context["is_vulnerable"] = True
                    # severity="error" XML attribute
                    if 'severity="error"' in line:
                        security_context["is_vulnerable"] = True
        except FileNotFoundError:
            logger.debug("[AutoLabel] cppcheck bulunamadı, atlanıyor.")
        except subprocess.TimeoutExpired:
            logger.debug("[AutoLabel] cppcheck timeout, atlanıyor.")

        # ── flawfinder ──
        try:
            result = subprocess.run(
                ["flawfinder", "--columns", "--quiet", "--dataonly", tmp_path],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    security_context["tool_findings"].append({
                        "tool": "flawfinder",
                        "detail": line,
                    })
                    # CWE-120 veya (CWE-120) formatı
                    cwe_match = re.search(r'CWE-?(\d+)', line)
                    if cwe_match:
                        cwe = f"CWE-{cwe_match.group(1)}"
                        if cwe not in security_context["cwe_ids"]:
                            security_context["cwe_ids"].append(cwe)
                        security_context["is_vulnerable"] = True
                    # CWE bulunamasa bile flawfinder finding varsa zafiyet
                    elif line and not line.startswith("#"):
                        security_context["is_vulnerable"] = True
        except FileNotFoundError:
            logger.debug("[AutoLabel] flawfinder bulunamadı, atlanıyor.")
        except subprocess.TimeoutExpired:
            logger.debug("[AutoLabel] flawfinder timeout, atlanıyor.")

    except Exception as e:
        logger.warning(f"[AutoLabel] Beklenmeyen hata: {e}")
    finally:
        # Geçici dosyayı temizle
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return security_context


class Collector:
    def __init__(self):
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self.gh_headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            self.gh_headers["Authorization"] = f"token {self.github_token}"
        
        # Yeni Hibrit DB Entegrasyonu
        self.hybrid_pipeline = HybridDBPipeline()

    # ========================================================================
    # Task 3: Üstel Geri Çekilme (Exponential Backoff) ile API İsteği
    # ========================================================================
    def _api_request(self, url: str, max_retries: int = 5) -> requests.Response:
        """
        GitHub API isteği — HTTP 429/403 durumunda üstel geri çekilme uygular.

        Retry-After veya X-RateLimit-Reset header'larını okuyarak
        limit sıfırlanana kadar bekler, ardından tekrar dener.

        Args:
            url: API endpoint URL'i
            max_retries: Maksimum deneme sayısı

        Returns:
            requests.Response nesnesi

        Raises:
            requests.RequestException: Tüm denemeler başarısız olursa
        """
        last_response = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=self.gh_headers, timeout=30)
                last_response = resp

                # Başarılı veya rate-limit dışı hata → hemen dön
                if resp.status_code not in (429, 403):
                    return resp

                # ── Rate Limit Algılandı ──
                # 403 ama rate limit değilse (örn: repo yok) → hemen dön
                if resp.status_code == 403:
                    remaining = resp.headers.get("X-RateLimit-Remaining")
                    if remaining is not None and int(remaining) > 0:
                        return resp  # Rate limit değil, gerçek 403

                # Bekleme süresini hesapla
                retry_after = resp.headers.get("Retry-After")
                rate_reset = resp.headers.get("X-RateLimit-Reset")

                if retry_after:
                    wait_seconds = int(retry_after)
                elif rate_reset:
                    reset_time = int(rate_reset)
                    wait_seconds = max(0, reset_time - int(time.time())) + 1
                else:
                    # Header yoksa üstel geri çekilme: 5, 10, 20, 40, 80 saniye
                    wait_seconds = min((2 ** attempt) * 5, 300)

                logger.warning(
                    f"[RateLimit] HTTP {resp.status_code} — "
                    f"{wait_seconds}s bekleniyor... "
                    f"(Deneme {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_seconds)

            except requests.exceptions.Timeout:
                wait_seconds = (2 ** attempt) * 2
                logger.warning(
                    f"[RateLimit] Timeout — {wait_seconds}s bekleniyor... "
                    f"(Deneme {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_seconds)
            except requests.exceptions.RequestException as e:
                logger.error(f"[RateLimit] İstek hatası: {e}")
                if attempt == max_retries - 1:
                    raise

        # Tüm denemeler tükendi
        logger.error(f"[RateLimit] {max_retries} deneme tükendi: {url}")
        if last_response is not None:
            return last_response
        raise requests.exceptions.ConnectionError(
            f"API isteği {max_retries} denemede başarısız: {url}"
        )

    # ========================================================================
    # Task 4: SHA-256 Hash ile Tekrar Önleme (Dedup) — Yardımcı
    # ========================================================================
    def _is_duplicate(self, raw_code: str) -> bool:
        """
        Ham kodun SHA-256 hash'ini hesaplayıp hibrit DB'de sorgular.
        True dönerse bu kod zaten işlenmiştir, atlanmalıdır.
        """
        code_hash = self.hybrid_pipeline.compute_hash(raw_code)
        return self.hybrid_pipeline.check_duplicate(code_hash)

    # ========================================================================
    # OWASP/Juliet CWE Çıkarma — Dosya adı ve yorum satırlarından
    # ========================================================================
    def _extract_cwe_from_context(self, filename: str, raw_code: str) -> str:
        """
        Juliet Test Suite dosya adından veya kod yorumlarından CWE ID çıkarır.
        
        Örnekler:
            Dosya adı: "CWE114_Process_Control__w32_char_connect_socket_01.c"
            Yorum: /* CWE: 114 Process Control */ veya * CWE-120
        
        Args:
            filename: Dosya yolu/adı
            raw_code: Ham kaynak kodu
        
        Returns:
            "CWE-114" formatında string veya None
        """
        # 1) Dosya adından çıkar (büyük/küçük harf duyarsız)
        # CWE-120, CWE120, CWE_120, CWE: 120, CWE 120 formatları
        cwe_pattern = r'CWE[-_:\s]?\s*(\d+)'
        match = re.search(cwe_pattern, filename, re.IGNORECASE)
        if match:
            return f"CWE-{match.group(1)}"
        
        # 2) Kod yorumlarından çıkar (ilk 30 satır — genelde header'da olur)
        lines = raw_code.split("\n")[:30]
        for line in lines:
            match = re.search(cwe_pattern, line, re.IGNORECASE)
            if match:
                return f"CWE-{match.group(1)}"
        
        # 3) Bulunamazsa None (Unknown DEĞİL — zafiyet olmayabilir)
        return None

    def collect_from_github(self, limit: int = 20, target_repos: list = None):
        logger.info("[*] Starting GitHub collection with Recursive Tree Search...")
        conn = get_db_connection()
        try:
            if not target_repos:
                target_repos = [
                    "redis/redis",
                    "torvalds/linux",
                    "openssl/openssl",
                    "TheAlgorithms/C-Plus-Plus"
                ]
            
            count = 0
            for repo in target_repos:
                if count >= limit: break
                logger.info(f"[{repo}] Hedef repo inceleniyor...")
                
                try:
                    # Get default branch (Task 3: rate-limited request)
                    repo_url = f"https://api.github.com/repos/{repo}"
                    repo_resp = self._api_request(repo_url)
                    if repo_resp.status_code != 200:
                        logger.warning(f"[{repo}] Repo bilgisi alınamadı (API Limit veya Repo yok)")
                        continue
                    
                    repo_json = repo_resp.json()
                    default_branch = repo_json.get("default_branch", "master")
                    repo_stars = repo_json.get("stargazers_count", 0)
                    platform_metadata = {"stars": repo_stars}
                    
                    # Get recursive tree (Task 3: rate-limited request)
                    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{default_branch}?recursive=1"
                    tree_resp = self._api_request(tree_url)
                    if tree_resp.status_code != 200:
                        logger.warning(f"[{repo}] Ağaç (Tree) bilgisi alınamadı.")
                        continue
                        
                    tree_data = tree_resp.json().get("tree", [])
                    logger.info(f"[{repo}] {len(tree_data)} obje bulundu, tarama başlıyor...")
                    
                    for item in tree_data:
                        if count >= limit: break
                        
                        if item["type"] == "tree":
                            # This is a directory
                            logger.info(f"[{repo}] Klasöre giriliyor: {item['path']}")
                            continue
                            
                        if item["type"] == "blob":
                            filename = item["path"]
                            
                            # Strict Extension Filtering
                            if not filename.endswith(('.c', '.cpp', '.cc', '.h', '.hpp')):
                                logger.debug(f"[{repo}] Dosya atlandı (Filtreye takıldı): {filename}")
                                continue
                                
                            raw_url = f"https://raw.githubusercontent.com/{repo}/{default_branch}/{filename}"
                            try:
                                raw_code_resp = self._api_request(raw_url)
                                if raw_code_resp.status_code == 200:
                                    raw_code = raw_code_resp.text

                                    # Task 4: Dedup — normalizasyondan ÖNCE kontrol
                                    if self._is_duplicate(raw_code):
                                        logger.info(f"[{repo}] DEDUP: Zaten mevcut, atlanıyor: {filename}")
                                        continue

                                    processed = process_code(raw_code, source="github")
                                    if not processed:
                                        logger.info(f"[{repo}] Dosya parse edilemedi veya boş: {filename}")
                                        continue
                                    
                                    # Task 2: Auto-Label — GitHub verisinin etiketi YOK,
                                    # cppcheck/flawfinder ile zafiyet tara
                                    sec_ctx = auto_label_vulnerability(raw_code, language="cpp")

                                    # CWE belirleme: zafiyet varsa auto-label'dan al,
                                    # temiz çıkarsa None ("Unknown" DEĞİL)
                                    if sec_ctx.get("is_vulnerable") and sec_ctx.get("cwe_ids"):
                                        cwe_id = sec_ctx["cwe_ids"][0]
                                    elif sec_ctx.get("is_vulnerable"):
                                        cwe_id = "Unknown"  # Zafiyet var ama CWE ID bulunamadı
                                    else:
                                        cwe_id = None  # Temiz kod — NULL

                                    # 1) Eski Review Station Veritabanına Yaz
                                    insert_record("github", raw_code, processed, cwe_id, conn=conn)

                                    # 2) Yeni Hibrit Veritabanına (Document + Vector DB) Yaz
                                    self.hybrid_pipeline.ingest(
                                        raw_code, "GITHUB", language="cpp",
                                        security_context=sec_ctx,
                                        platform_metadata=platform_metadata
                                    )
                                    
                                    conn.commit()
                                    count += 1
                                    vuln_flag = "🔴 VULN" if sec_ctx.get("is_vulnerable") else "🟢 CLEAN"
                                    logger.info(f"[{repo}] {vuln_flag} Eklendi: {filename} ({count}/{limit})")
                                else:
                                    logger.warning(f"[{repo}] Dosya indirilemedi (HTTP {raw_code_resp.status_code}): {filename}")
                            except Exception as e:
                                logger.warning(f"[{repo}] Dosya işleme hatası ({filename}): {e}")
                                continue
                except Exception as e:
                    logger.warning(f"[{repo}] Repo işleme hatası: {e}")
                    continue
        except Exception as e:
            logger.warning(f"GitHub search error: {e}")
        finally:
            conn.close()

    def collect_from_owasp(self, limit: int = 20, target_repos: list = None):
        logger.info("[*] Starting OWASP collection with Recursive Tree Search...")
        conn = get_db_connection()
        try:
            if not target_repos:
                target_repos = [
                    "arichardson/juliet-test-suite-c"
                ]
            
            count = 0
            for repo in target_repos:
                if count >= limit: break
                logger.info(f"[OWASP - {repo}] Hedef repo inceleniyor...")
                
                try:
                    # Get default branch (Task 3: rate-limited request)
                    repo_url = f"https://api.github.com/repos/{repo}"
                    repo_resp = self._api_request(repo_url)
                    if repo_resp.status_code != 200:
                        logger.warning(f"[OWASP - {repo}] Repo bilgisi alınamadı (API Limit veya Repo yok)")
                        continue
                    
                    default_branch = repo_resp.json().get("default_branch", "master")
                    
                    # Get recursive tree (Task 3: rate-limited request)
                    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{default_branch}?recursive=1"
                    tree_resp = self._api_request(tree_url)
                    if tree_resp.status_code != 200:
                        logger.warning(f"[OWASP - {repo}] Ağaç (Tree) bilgisi alınamadı.")
                        continue
                        
                    tree_data = tree_resp.json().get("tree", [])
                    logger.info(f"[OWASP - {repo}] {len(tree_data)} obje bulundu, tarama başlıyor...")
                    
                    for item in tree_data:
                        if count >= limit: break
                        
                        if item["type"] == "tree":
                            logger.info(f"[OWASP - {repo}] Klasöre giriliyor: {item['path']}")
                            continue
                            
                        if item["type"] == "blob":
                            filename = item["path"]
                            
                            # Strict Extension Filtering
                            if not filename.endswith(('.c', '.cpp', '.cc', '.h', '.hpp')):
                                logger.debug(f"[OWASP - {repo}] Dosya atlandı (Filtreye takıldı): {filename}")
                                continue
                                
                            raw_url = f"https://raw.githubusercontent.com/{repo}/{default_branch}/{filename}"
                            try:
                                raw_code_resp = self._api_request(raw_url)
                                if raw_code_resp.status_code == 200:
                                    raw_code = raw_code_resp.text

                                    # Task 4: Dedup — normalizasyondan ÖNCE kontrol
                                    if self._is_duplicate(raw_code):
                                        logger.info(f"[OWASP - {repo}] DEDUP: Zaten mevcut, atlanıyor: {filename}")
                                        continue

                                    processed = process_code(raw_code, source="owasp")
                                    if not processed:
                                        logger.info(f"[OWASP - {repo}] Dosya parse edilemedi veya boş: {filename}")
                                        continue
                                    
                                    # CWE ID'yi dosya adından ve kod yorumlarından çıkar
                                    cwe_id = self._extract_cwe_from_context(
                                        filename, raw_code
                                    )
                                    
                                    # 1) Eski Review Station Veritabanına Yaz
                                    insert_record("owasp", raw_code, processed, cwe_id, conn=conn)

                                    # Task 2: OWASP verisi zaten owasp_secnorm tarafından
                                    # security_context üretir → OTOMATİK ETİKETLEME YAPMA
                                    # Orijinal güvenlik bağlamına KESİNLİKLE DOKUNULMAZ
                                    
                                    # 2) Yeni Hibrit Veritabanına Yaz
                                    # security_context=None → owasp_secnorm kendi üretir
                                    self.hybrid_pipeline.ingest(
                                        raw_code, "OWASP", language="cpp",
                                        cwe_hint=cwe_id
                                    )
                                    
                                    conn.commit()
                                    count += 1
                                    logger.info(f"[OWASP - {repo}] Başarıyla eklendi: {filename} CWE={cwe_id} ({count}/{limit})")
                                else:
                                    logger.warning(f"[OWASP - {repo}] Dosya indirilemedi (HTTP {raw_code_resp.status_code}): {filename}")
                            except Exception as e:
                                logger.warning(f"[OWASP - {repo}] Dosya işleme hatası ({filename}): {e}")
                                continue
                except Exception as e:
                    logger.warning(f"[OWASP - {repo}] Repo işleme hatası: {e}")
                    continue
        except Exception as e:
            logger.warning(f"OWASP search error: {e}")
        finally:
            conn.close()

    def collect_from_hf(self, limit: int = 20):
        print("[*] Starting HF CodeXGLUE collection...")
        conn = get_db_connection()

        # HF dataset meta verileri (downloads)
        hf_platform_metadata = {"downloads": 0}
        try:
            from datasets import get_dataset_config_info
            ds_info = get_dataset_config_info("google/code_x_glue_cc_defect_detection", "default")
            if hasattr(ds_info, "download_size") and ds_info.download_size:
                hf_platform_metadata["downloads"] = ds_info.download_size
        except Exception:
            pass  # Meta veri alınamazsa varsayılan kullan

        try:
            ds = load_dataset("google/code_x_glue_cc_defect_detection", split="train", streaming=True)
            count = 0
            for item in ds:
                try:
                    raw_code = item["func"]

                    # Task 4: Dedup — normalizasyondan ÖNCE kontrol
                    if self._is_duplicate(raw_code):
                        logger.info(f"[HF] DEDUP: Zaten mevcut, atlanıyor.")
                        continue

                    processed = process_code(raw_code, source="hf")
                    if not processed: continue
                    
                    # Task 2: Auto-Label — HF verisinin etiketi YOK,
                    # cppcheck/flawfinder ile zafiyet tara
                    sec_ctx = auto_label_vulnerability(raw_code, language="cpp")

                    # CWE belirleme: zafiyet varsa auto-label'dan al,
                    # temiz çıkarsa None ("Unknown" DEĞİL)
                    if sec_ctx.get("is_vulnerable") and sec_ctx.get("cwe_ids"):
                        cwe_id = sec_ctx["cwe_ids"][0]
                    elif sec_ctx.get("is_vulnerable"):
                        cwe_id = "Unknown"  # Zafiyet var ama CWE ID bulunamadı
                    else:
                        cwe_id = None  # Temiz kod — NULL
                    
                    # 1) Eski Review Station Veritabanına Yaz
                    insert_record("hf", raw_code, processed, cwe_id, conn=conn)

                    # 2) Yeni Hibrit Veritabanına Yaz
                    self.hybrid_pipeline.ingest(
                        raw_code, "HUGGINGFACE", docstring="",
                        security_context=sec_ctx,
                        platform_metadata=hf_platform_metadata
                    )
                    
                    conn.commit()
                    count += 1
                    vuln_flag = "🔴 VULN" if sec_ctx.get("is_vulnerable") else "🟢 CLEAN"
                    cwe_display = cwe_id if cwe_id else "N/A"
                    print(f"[+] {vuln_flag} [{cwe_display}] Added HF record {count}/{limit}")
                    if count >= limit: break
                except Exception as e:
                    logger.warning(f"HF item processing error: {e}")
                    continue
        except Exception as e:
            logger.error(f"HF Collection Load Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Analysis Collector")
    parser.add_argument("--source", choices=["hf", "github", "owasp"], required=True, help="Veri kaynağı seçimi (hf, github, owasp)")
    parser.add_argument("--verbose", action="store_true", help="Detaylı loglama (Filtrelenen dosyaları gösterir)")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    c = Collector()
    
    if args.source == "hf":
        print(f"[*] Fetching from HF...")
        c.collect_from_hf(limit=20)
    elif args.source == "github":
        print(f"[*] Fetching from GitHub...")
        c.collect_from_github(limit=20)
    elif args.source == "owasp":
        print(f"[*] Fetching from OWASP...")
        c.collect_from_owasp(limit=20)
        
    print("[*] Process completed. Exiting cleanly...")
    sys.exit(0)
