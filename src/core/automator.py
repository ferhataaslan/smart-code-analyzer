#!/usr/bin/env python3
"""
automator.py — Interactive Continuous Automation & Capacity Management (Generation 2.9)

Ozellikler:
- Interaktif terminal menusu ile platform secimi
- Ctrl+C ile guvenli duraklama ve komut modu
- Manuel yukleme (HF push) — 5000 esigine ulasilmadan da mumkun
- Sistem kapansa bile kaldigin yerden devam
- Platform degistirip geri donme (indeks korunur)
- Batch commit ile ~%30 hizlanma
"""

import os
import json
import sqlite3
import time
import signal
import logging

from src.database.collector import Collector, auto_label_vulnerability
from src.database.database import get_db_connection, insert_record
from src.normalizers.processor import process_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("AUTOMATOR")

# Her N kayitta bir toplu commit (performans optimizasyonu)
BATCH_COMMIT_SIZE = 10


class AutomatorAction(Exception):
    """Collection loop'tan cikmak icin kullanilan ozel exception."""
    def __init__(self, action):
        self.action = action  # "change_platform" veya "exit"


class AutomatedCollector(Collector):
    def __init__(self, state_file="data/states/automation_state.json"):
        super().__init__()
        self.state_file = state_file
        self.state = self.load_state()
        self._interrupted = False

        # Ctrl+C signal handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Override the hybrid pipeline's ingest method — SADECE kapasite kontrolu
        # (Pause kontrolu ana dongu icinde yapilir, ingest icinde degil)
        original_ingest = self.hybrid_pipeline.ingest
        def automated_ingest(*args, **kwargs):
            res = original_ingest(*args, **kwargs)
            self._check_capacity()
            return res
        self.hybrid_pipeline.ingest = automated_ingest

    # ══════════════════════════════════════════════════════════════════════
    #  State Management
    # ══════════════════════════════════════════════════════════════════════

    def load_state(self) -> dict:
        if not os.path.exists(self.state_file):
            default_state = {
                "status": "RUNNING",
                "github_page": 1,
                "github_last_repo": "",
                "hf_last_index": 0,
                "hf_completed": False,
                "owasp_last_index": 0,
                "owasp_completed": False,
                "active_tasks": [],
                "total_uploaded": 0
            }
            self.save_state(default_state)
            return default_state
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                # Migration: yeni alanlar yoksa ekle
                state.setdefault("active_tasks", [])
                state.setdefault("total_uploaded", 0)
                return state
        except Exception as e:
            logger.error(f"[Automator] State dosyasi yuklenemedi: {e}")
            time.sleep(2)
            return getattr(self, "state", {})

    def save_state(self, state: dict = None) -> None:
        if state is not None:
            self.state = state
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Automator] State dosyasi kaydedilemedi: {e}")

    # ══════════════════════════════════════════════════════════════════════
    #  Signal & Pause
    # ══════════════════════════════════════════════════════════════════════

    def _handle_interrupt(self, signum, frame):
        """Ctrl+C sinyalini yakalar."""
        self._interrupted = True
        print("\n[!] Ctrl+C algilandi. Mevcut islem tamamlaniyor...")

    def _check_pause(self) -> None:
        """
        Her dongu adiminda cagirilir.
        1. Ctrl+C basildi mi? → Komut menusu goster
        2. State dosyasinda PAUSED mi? → Bekle
        """
        # ── Ctrl+C kontrolu ──
        if self._interrupted:
            self._interrupted = False
            self.save_state()
            action = self._show_command_menu()

            if action == "continue":
                return
            elif action == "upload_continue":
                self._manual_upload()
                return
            elif action == "change_platform":
                raise AutomatorAction("change_platform")
            elif action == "upload_exit":
                self._manual_upload()
                raise AutomatorAction("exit")
            elif action == "exit":
                raise AutomatorAction("exit")

        # ── Dosya bazli duraklatma kontrolu ──
        while True:
            self.state = self.load_state()
            if self.state.get("status") == "PAUSED":
                logger.info("[Automator] Status PAUSED. Bekleniyor...")
                time.sleep(1)
            else:
                break

    # ══════════════════════════════════════════════════════════════════════
    #  HF Sync & Verification
    # ══════════════════════════════════════════════════════════════════════

    def _get_local_count(self) -> int:
        """Lokal vector_store'daki kayit sayisini doner."""
        try:
            conn = sqlite3.connect("data/dbs/vector_store.db")
            count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def _sync_with_hf(self) -> None:
        """
        Baslangicta HF'deki gercek kayit sayisini kontrol eder.
        total_uploaded degerini HF'deki gercek sayiyla gunceller.
        """
        try:
            print("[*] HF ile senkronizasyon kontrol ediliyor...")
            from huggingface_hub import HfApi
            import os as _os
            from dotenv import load_dotenv
            load_dotenv()

            token = _os.environ.get("HF_TOKEN")
            if not token:
                print("[-] HF_TOKEN bulunamadi, senkronizasyon atlandi.")
                return

            repo_id = "smart-code-analyzer-team/cpp-vulnerability-dataset"
            api = HfApi(token=token)

            # HF'deki parquet dosyalarini listele
            try:
                files = api.list_repo_files(repo_id, repo_type="dataset")
            except Exception:
                print("[-] HF repo'ya erisilemiyor, senkronizasyon atlandi.")
                return

            parquets = [f for f in files if f.endswith(".parquet")]
            print(f"[*] HF'de {len(parquets)} parquet dosyasi bulundu.")

            if not parquets:
                self.state["total_uploaded"] = 0
                self.save_state()
                return

            # Gercek kayit sayisini al (dataset yukle)
            from datasets import load_dataset
            ds = load_dataset(repo_id, split="train")
            hf_count = len(ds)

            local_pending = self._get_local_count()

            old_total = self.state.get("total_uploaded", 0)
            self.state["total_uploaded"] = hf_count
            self.save_state()

            if old_total != hf_count:
                print(
                    f"[*] HF senkronize edildi: "
                    f"{old_total} -> {hf_count} kayit"
                )
            print(
                f"[OK] HF: {hf_count} kayit | "
                f"Lokal bekleyen: {local_pending} kayit"
            )

        except Exception as e:
            logger.warning(f"[Automator] HF senkronizasyon hatasi: {e}")
            print(f"[-] HF senkronizasyon hatasi: {e} (devam ediliyor)")

    def _verify_upload(self, filename: str) -> bool:
        """
        Upload edilen dosyanin HF'de gercekten var olup olmadigini dogrular.
        Returns: True ise dosya HF'de var, False ise yok.
        """
        try:
            from huggingface_hub import HfApi
            import os as _os
            from dotenv import load_dotenv
            load_dotenv()

            token = _os.environ.get("HF_TOKEN")
            if not token:
                return False

            repo_id = "smart-code-analyzer-team/cpp-vulnerability-dataset"
            api = HfApi(token=token)
            files = api.list_repo_files(repo_id, repo_type="dataset")

            expected_path = f"data/{filename}"
            exists = expected_path in files

            if exists:
                logger.info(f"[Automator] Upload dogrulandi: {expected_path} HF'de mevcut.")
            else:
                logger.error(
                    f"[Automator] UYARI: {expected_path} HF'de BULUNAMADI! "
                    f"Mevcut dosyalar: {[f for f in files if f.endswith('.parquet')]}"
                )

            return exists
        except Exception as e:
            logger.error(f"[Automator] Upload dogrulama hatasi: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════
    #  Capacity Management & Manual Upload
    # ══════════════════════════════════════════════════════════════════════

    def _check_capacity(self) -> None:
        """Esik degerine ulasildiginda otomatik yukleme tetikler."""
        try:
            count = self._get_local_count()
            threshold = int(os.environ.get("BATCH_THRESHOLD", 5000))
            if count >= threshold:
                logger.info(
                    f"[Automator] Kayit sayisi ({count}) esige ulasti "
                    f"({threshold}). Otomatik yukleme basliyor..."
                )
                self._manual_upload()
        except Exception as e:
            logger.error(f"[Automator] Kapasite kontrol hatasi: {e}")

    def _manual_upload(self) -> None:
        """
        HF'ye manuel push. 5000 esigine ulasilmamis olsa bile calisir.
        GUVENLK: Upload dogrulandiktan SONRA lokal DB flush edilir.
        """
        try:
            count = self._get_local_count()
            if count == 0:
                print("[*] Yuklenecek kayit yok.")
                return

            print(f"[*] {count} kayit HF'ye yukleniyor...")

            # Export parquet dosyasini olustur ve HF'ye push et
            from src.core.uploader import export_and_push
            export_and_push()

            # Yuklenen dosya adini bul (en yeni parquet)
            import glob
            parquet_files = sorted(
                glob.glob("data_*.parquet"),
                key=os.path.getmtime, reverse=True
            )
            uploaded_filename = parquet_files[0] if parquet_files else None

            # ── DOGRULAMA: Dosya HF'de var mi? ──
            if uploaded_filename and self._verify_upload(uploaded_filename):
                # Upload dogrulandi — lokal DB'yi guvenlice flush et
                self.state["total_uploaded"] = (
                    self.state.get("total_uploaded", 0) + count
                )
                self.save_state()

                logger.info(
                    "[Automator] Upload dogrulandi. "
                    "Disk alani geri kazaniliyor..."
                )

                # 1. Document store: raw_snippet temizle (code_hash korunur)
                conn_doc = sqlite3.connect("data/dbs/document_store.db")
                conn_doc.execute("UPDATE documents SET raw_snippet = ''")
                conn_doc.commit()
                conn_doc.execute("VACUUM")
                conn_doc.close()

                # 2. Vector store: satirlari sil (tekrar yuklenmemesi icin)
                conn_vec = sqlite3.connect("data/dbs/vector_store.db")
                conn_vec.execute("DELETE FROM vectors")
                conn_vec.commit()
                conn_vec.execute("VACUUM")
                conn_vec.close()

                logger.info("[Automator] DB flush ve VACUUM tamamlandi.")
                print(
                    f"[OK] {count} kayit basariyla yuklendi ve dogrulandi. "
                    f"Toplam: {self.state['total_uploaded']}"
                )
            else:
                # Upload dogrulanamadi — lokal veriyi SILME!
                print(
                    f"[UYARI] Upload dogrulanamadi! "
                    f"Lokal {count} kayit korunuyor (silinmedi). "
                    f"Bir sonraki denemede tekrar yuklenecek."
                )
                logger.error(
                    "[Automator] Upload dogrulanamadi. "
                    "Lokal DB flush ENGELLENDI — veri kaybi onlendi."
                )

        except Exception as e:
            logger.error(f"[Automator] Yukleme hatasi: {e}")
            print(f"[-] Yukleme hatasi: {e}")

    # ══════════════════════════════════════════════════════════════════════
    #  Interactive Menus
    # ══════════════════════════════════════════════════════════════════════

    def _show_main_menu(self):
        """
        Interaktif baslangic menusu.
        Returns:
            "exit"     - Cikis
            "upload"   - Manuel yukleme
            "continue" - Kaldigin yerden devam
            list       - Secilen platform listesi (orn: ["hf", "owasp"])
        """
        local_count = self._get_local_count()
        total_uploaded = self.state.get("total_uploaded", 0)
        active_tasks = self.state.get("active_tasks", [])

        # Platform durumlari
        owasp_idx = self.state.get("owasp_last_index", 0)
        owasp_done = self.state.get("owasp_completed", False)
        owasp_status = ("tamamlandi" if owasp_done
                        else f"indeks {owasp_idx}" if owasp_idx > 0
                        else "baslamadi")

        hf_idx = self.state.get("hf_last_index", 0)
        hf_done = self.state.get("hf_completed", False)
        hf_status = ("tamamlandi" if hf_done
                     else f"indeks {hf_idx}" if hf_idx > 0
                     else "baslamadi")

        gh_page = self.state.get("github_page", 1)
        gh_repo = self.state.get("github_last_repo", "")
        gh_status = f"sayfa {gh_page}"
        if gh_repo:
            gh_status += f" ({gh_repo})"

        print()
        print("=" * 55)
        print("   OTOMASYON KONTROL PANELI v2.9")
        print("=" * 55)
        print(f"   Mevcut Durum:")
        print(f"     OWASP  : {owasp_status}")
        print(f"     HF     : {hf_status}")
        print(f"     GitHub : {gh_status}")
        print(f"     Lokal  : {local_count} kayit (yuklenmeyi bekliyor)")
        print(f"     Toplam : {total_uploaded} kayit yuklendi (HF)")
        if active_tasks:
            print(f"     Son oturum: {', '.join(active_tasks)}")
        print("-" * 55)
        print("   [1] Platform sec ve baslat")
        if active_tasks:
            print(f"   [2] Kaldigi yerden devam et ({', '.join(active_tasks)})")
        else:
            print("   [2] Kaldigi yerden devam et (oturum yok)")
        print("   [3] Yukle (HF'ye push)")
        print("   [4] Cikis")
        print("=" * 55)

        while True:
            try:
                choice = input("   Seciminiz [1-4]: ").strip()
            except (EOFError, KeyboardInterrupt):
                return "exit"

            if choice == "1":
                return self._select_platforms()
            elif choice == "2":
                if active_tasks:
                    return "continue"
                else:
                    print("   [!] Onceki oturum bulunamadi. Lutfen platform secin (1).")
            elif choice == "3":
                return "upload"
            elif choice == "4":
                return "exit"
            else:
                print("   [!] Gecersiz secim. 1-4 arasinda bir sayi girin.")

    def _select_platforms(self):
        """
        Platform secim menusu.
        Returns: platform listesi (list) veya "exit"
        """
        print()
        print("-" * 55)
        print("   Platform secin (virgul ile ayirin):")
        print("     owasp  - OWASP Juliet Test Suite")
        print("     hf     - HuggingFace CodeXGLUE")
        print("     github - GitHub C/C++ projeleri (sonsuz)")
        print()
        print("   Ornekler: hf  |  owasp,hf  |  github")
        print("-" * 55)

        while True:
            try:
                raw = input("   Platformlar: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "exit"

            if not raw:
                print("   [!] En az bir platform secmelisiniz.")
                continue

            # Virgul, bosluk veya ikisiyle ayirma destegi
            platforms = [p.strip() for p in raw.replace(" ", ",").split(",")
                         if p.strip()]
            valid = {"owasp", "hf", "github"}
            invalid = [p for p in platforms if p not in valid]

            if invalid:
                print(
                    f"   [!] Gecersiz: {', '.join(invalid)}. "
                    f"Gecerli: owasp, hf, github"
                )
                continue

            # Tekrarlari kaldir, sirayi koru
            seen = set()
            unique = []
            for p in platforms:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)

            return unique

    def _show_command_menu(self):
        """
        Duraklama komut menusu (Ctrl+C sonrasi).
        Returns: "continue", "upload_continue", "change_platform",
                 "upload_exit", "exit"
        """
        local_count = self._get_local_count()
        total_uploaded = self.state.get("total_uploaded", 0)
        active_tasks = self.state.get("active_tasks", [])

        owasp_idx = self.state.get("owasp_last_index", 0)
        hf_idx = self.state.get("hf_last_index", 0)
        gh_page = self.state.get("github_page", 1)

        print()
        print("-" * 55)
        print("   OTOMASYON DURAKLATILDI")
        if "owasp" in active_tasks:
            print(f"   OWASP  : indeks {owasp_idx}")
        if "hf" in active_tasks:
            print(f"   HF     : indeks {hf_idx}")
        if "github" in active_tasks:
            print(f"   GitHub : sayfa {gh_page}")
        print(f"   Lokal  : {local_count} kayit")
        print(f"   Toplam : {total_uploaded} kayit yuklendi")
        print("-" * 55)
        print("   [1] Devam et (kaldigin yerden)")
        print("   [2] Yukle ve devam et")
        print("   [3] Platform degistir")
        print("   [4] Yukle ve cik")
        print("   [5] Cikis (ilerleme kaydedildi)")
        print("-" * 55)

        # Ctrl+C sinyalini gecici olarak yoksay (menu sirasinda)
        self._interrupted = False

        while True:
            try:
                choice = input("   Seciminiz [1-5]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return "exit"

            if choice == "1":
                print("   [*] Devam ediliyor...")
                return "continue"
            elif choice == "2":
                return "upload_continue"
            elif choice == "3":
                return "change_platform"
            elif choice == "4":
                return "upload_exit"
            elif choice == "5":
                return "exit"
            else:
                print("   [!] Gecersiz secim. 1-5 arasinda bir sayi girin.")

    # ══════════════════════════════════════════════════════════════════════
    #  Collection Methods (veri isleme mantigi ayni, sadece batch commit)
    # ══════════════════════════════════════════════════════════════════════

    def collect_from_hf(self, limit: int = 999999) -> None:
        logger.info(
            f"[Automator-HF] HF toplama basliyor. "
            f"Baslangic indeksi: {self.state['hf_last_index']}"
        )
        conn = get_db_connection()

        hf_platform_metadata = {"downloads": 0}
        try:
            from datasets import get_dataset_config_info
            ds_info = get_dataset_config_info(
                "google/code_x_glue_cc_defect_detection", "default"
            )
            if hasattr(ds_info, "download_size") and ds_info.download_size:
                hf_platform_metadata["downloads"] = ds_info.download_size
        except Exception:
            pass

        start_idx = self.state["hf_last_index"]
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "google/code_x_glue_cc_defect_detection",
                split="train", streaming=True
            )
            count = 0
            for idx, item in enumerate(ds):
                self._check_pause()

                # Ilerleme indeksini guncelle
                self.state["hf_last_index"] = idx
                if idx % 50 == 0:
                    self.save_state()

                if idx < start_idx:
                    continue

                raw_code = item["func"]
                if self._is_duplicate(raw_code):
                    continue

                processed = process_code(raw_code, source="hf")
                if not processed:
                    continue

                sec_ctx = auto_label_vulnerability(raw_code, language="cpp")
                cwe_id = (
                    sec_ctx["cwe_ids"][0]
                    if (sec_ctx.get("is_vulnerable") and sec_ctx.get("cwe_ids"))
                    else ("Unknown" if sec_ctx.get("is_vulnerable") else None)
                )

                insert_record("hf", raw_code, processed, cwe_id, conn=conn)
                self.hybrid_pipeline.ingest(
                    raw_code, "HUGGINGFACE", docstring="",
                    security_context=sec_ctx,
                    platform_metadata=hf_platform_metadata
                )

                count += 1
                # Batch commit: her N kayitta bir
                if count % BATCH_COMMIT_SIZE == 0:
                    conn.commit()

                if count >= limit:
                    break

        except AutomatorAction:
            conn.commit()
            raise
        except Exception as e:
            logger.error(f"[Automator-HF] Hata: {e}")
        finally:
            conn.commit()
            conn.close()
            self.save_state()

    def collect_from_owasp(self, limit: int = 999999,
                           target_repos: list = None) -> None:
        logger.info(
            f"[Automator-OWASP] OWASP toplama basliyor. "
            f"Baslangic indeksi: {self.state['owasp_last_index']}"
        )
        conn = get_db_connection()
        if not target_repos:
            target_repos = ["arichardson/juliet-test-suite-c"]

        start_idx = self.state["owasp_last_index"]
        try:
            count = 0
            for repo in target_repos:
                repo_url = f"https://api.github.com/repos/{repo}"
                repo_resp = self._api_request(repo_url)
                if repo_resp.status_code != 200:
                    continue
                default_branch = repo_resp.json().get(
                    "default_branch", "master"
                )

                tree_url = (
                    f"https://api.github.com/repos/{repo}/git/trees/"
                    f"{default_branch}?recursive=1"
                )
                tree_resp = self._api_request(tree_url)
                if tree_resp.status_code != 200:
                    continue

                tree_data = tree_resp.json().get("tree", [])
                for idx, item in enumerate(tree_data):
                    self._check_pause()

                    # Ilerleme indeksini guncelle
                    self.state["owasp_last_index"] = idx
                    if idx % 100 == 0:
                        self.save_state()

                    if idx < start_idx:
                        continue

                    if item["type"] == "blob":
                        filename = item["path"]
                        if not filename.endswith(
                            ('.c', '.cpp', '.cc', '.h', '.hpp')
                        ):
                            continue

                        raw_url = (
                            f"https://raw.githubusercontent.com/{repo}/"
                            f"{default_branch}/{filename}"
                        )
                        raw_code_resp = self._api_request(raw_url)
                        if raw_code_resp.status_code == 200:
                            raw_code = raw_code_resp.text
                            if self._is_duplicate(raw_code):
                                continue

                            processed = process_code(
                                raw_code, source="owasp"
                            )
                            if not processed:
                                continue

                            cwe_id = self._extract_cwe_from_context(
                                filename, raw_code
                            )
                            insert_record(
                                "owasp", raw_code, processed,
                                cwe_id, conn=conn
                            )
                            self.hybrid_pipeline.ingest(
                                raw_code, "OWASP", language="cpp",
                                cwe_hint=cwe_id
                            )

                            count += 1
                            # Batch commit
                            if count % BATCH_COMMIT_SIZE == 0:
                                conn.commit()

                            if count >= limit:
                                break

        except AutomatorAction:
            conn.commit()
            raise
        except Exception as e:
            logger.error(f"[Automator-OWASP] Hata: {e}")
        finally:
            conn.commit()
            conn.close()
            self.save_state()

    # ══════════════════════════════════════════════════════════════════════
    #  Task Runners
    # ══════════════════════════════════════════════════════════════════════

    def run_single_task(self, task: str) -> None:
        """Tek bir gorevi calistirir (owasp, hf veya github)."""
        task = task.lower()
        if task == "owasp":
            logger.info("[Automator] OWASP gorevi baslatiliyor...")
            self.collect_from_owasp(limit=999999)
            self.state["owasp_completed"] = True
            self.save_state()
            logger.info("[Automator] OWASP gorevi tamamlandi.")

        elif task == "hf":
            logger.info("[Automator] HF gorevi baslatiliyor...")
            self.collect_from_hf(limit=999999)
            self.state["hf_completed"] = True
            self.save_state()
            logger.info("[Automator] HF gorevi tamamlandi.")

        elif task == "github":
            logger.info("[Automator] GitHub sonsuz dongusu baslatiliyor...")
            self._run_github_loop()

        else:
            logger.error(f"[Automator] Bilinmeyen gorev: {task}")

    def _run_github_loop(self) -> None:
        """GitHub sonsuz arama dongusunu calistirir."""
        logger.info("[Automator] GitHub infinite search loop started.")
        while True:
            self._check_pause()
            page = self.state["github_page"]

            lang = "c" if page % 2 == 1 else "cpp"
            actual_page = (page + 1) // 2

            search_url = (
                f"https://api.github.com/search/repositories"
                f"?q=language:{lang}+pushed:>=2023-01-01"
                f"&sort=updated&order=desc"
                f"&page={actual_page}&per_page=30"
            )

            logger.info(
                f"[Automator] GitHub arama: sayfa={actual_page}, dil={lang}..."
            )
            resp = self._api_request(search_url)
            if resp.status_code != 200:
                logger.warning(
                    f"[Automator] GitHub API limiti veya hata "
                    f"(HTTP {resp.status_code}). 60s bekleniyor..."
                )
                time.sleep(60)
                continue

            items = resp.json().get("items", [])
            if not items:
                logger.info(
                    "[Automator] Sonuc yok. Sayfa 1'e donuluyor."
                )
                self.state["github_page"] = 1
                self.save_state()
                time.sleep(10)
                continue

            for repo in items:
                self._check_pause()
                repo_name = repo["full_name"]

                self.state["github_last_repo"] = repo_name
                self.save_state()

                logger.info(f"[Automator] Repo taraniyor: {repo_name}")
                try:
                    self.collect_from_github(
                        limit=50, target_repos=[repo_name]
                    )
                except Exception as e:
                    logger.error(
                        f"[Automator] Repo hatasi {repo_name}: {e}"
                    )

            next_page = page + 1
            if next_page > 20:
                logger.info(
                    "[Automator] Sayfa 20 limitine ulasildi. "
                    "Sayfa 1'e donuluyor."
                )
                next_page = 1

            self.state["github_page"] = next_page
            self.save_state()

    # ══════════════════════════════════════════════════════════════════════
    #  Main Interactive Loop
    # ══════════════════════════════════════════════════════════════════════

    def run_interactive(self) -> None:
        """Ana interaktif dongu. Menu gosterir, kullanici komutlarini isler."""
        # Baslangicta HF ile senkronize et
        self._sync_with_hf()

        while True:
            action = self._show_main_menu()

            if action == "exit":
                print("[*] Cikis yapiliyor...")
                break

            elif action == "upload":
                self._manual_upload()
                continue

            elif action == "continue":
                tasks = self.state.get("active_tasks", [])
                if not tasks:
                    print("[!] Onceki oturum bulunamadi. Platform secin.")
                    continue

            elif isinstance(action, list):
                tasks = action
                self.state["active_tasks"] = tasks
                self.save_state()

            else:
                continue

            # Secilen gorevleri calistir
            self.state["status"] = "RUNNING"
            self.save_state()

            print(f"\n[*] Gorevler baslatiliyor: {', '.join(tasks)}")
            print("[*] Durdurmak icin Ctrl+C basin.\n")

            try:
                for task in tasks:
                    self.run_single_task(task)
                print("\n[OK] Secilen gorevler tamamlandi.")
            except AutomatorAction as e:
                if e.action == "exit":
                    print("[*] Cikis yapiliyor...")
                    break
                elif e.action == "change_platform":
                    print("[*] Ana menuye donuluyor...")
                    continue


# ══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automator - Kesintisiz Otomasyon (Generation 2.9)"
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=["owasp", "hf", "github"],
        default=None,
        help=(
            "Dogrudan calistirilacak gorevler (menu atlanir). "
            "Ornekler: --task hf  |  --task hf owasp  |  --task github"
        ),
    )
    args = parser.parse_args()

    automator = AutomatedCollector()

    if args.task:
        # CLI modu: menu gostermeden dogrudan calistir
        automator.state["active_tasks"] = args.task
        automator.state["status"] = "RUNNING"
        automator.save_state()

        print(f"[*] CLI modu: {', '.join(args.task)}")
        print("[*] Durdurmak icin Ctrl+C basin.\n")

        try:
            for task in args.task:
                automator.run_single_task(task)
            print("[OK] Tamamlandi.")
        except AutomatorAction as e:
            if e.action == "upload_exit":
                pass
            print("[*] Otomasyon sonlandirildi.")
    else:
        # Interaktif mod: menu goster
        automator.run_interactive()
