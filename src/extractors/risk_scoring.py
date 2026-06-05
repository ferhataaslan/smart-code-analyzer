#!/usr/bin/env python3
"""
risk_scoring.py — Bağlamsal Risk Puanlaması (Contextual Risk Scoring)

3 katmanlı risk puanlama sistemi:
  Katman 1: Temel Zafiyet Skoru (CWE Base Score) — MITRE 2025 verileri
  Katman 2: Saldırı Yüzeyi Çarpanı (Attack Surface Multiplier)
  Katman 3: Varlık Kritiklik Çarpanı (Asset Criticality Multiplier)

Formül:
  final_calculated_risk = Base_Score × Attack_Surface × Asset_Criticality
"""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ============================================================================
# ADIM 1: Katman 1 — MITRE 2025 Güncel CWE Temel Zafiyet Skorları
# ============================================================================
CWE_BASE_SCORES: Dict[str, float] = {
    "CWE-79":  60.38,   # Cross-site Scripting (XSS)
    "CWE-89":  28.72,   # SQL Injection
    "CWE-352": 13.64,   # Cross-Site Request Forgery (CSRF)
    "CWE-862": 13.28,   # Missing Authorization
    "CWE-787": 13.00,   # Out-of-bounds Write
    "CWE-120": 12.00,   # Buffer Copy without Checking Size of Input
    "CWE-20":  11.00,   # Improper Input Validation
    "CWE-78":  10.50,   # OS Command Injection
    "CWE-416": 10.00,   # Use After Free
    "CWE-22":   9.80,   # Path Traversal
    "CWE-125":  9.50,   # Out-of-bounds Read
    "CWE-119":  9.00,   # Improper Restriction of Operations within Buffer
    "CWE-190":  8.50,   # Integer Overflow
    "CWE-200":  8.00,   # Exposure of Sensitive Information
    "CWE-122":  7.80,   # Heap-based Buffer Overflow
    "CWE-134":  7.50,   # Use of Externally-Controlled Format String
    "CWE-242":  7.20,   # Use of Inherently Dangerous Function
    "CWE-327":  7.00,   # Use of Broken Crypto Algorithm
    "CWE-330":  6.80,   # Use of Insufficiently Random Values
    "CWE-415":  6.50,   # Double Free
    "CWE-426":  6.20,   # Untrusted Search Path
    "CWE-427":  6.00,   # Uncontrolled Search Path Element
    "CWE-191":  5.80,   # Integer Underflow
    "CWE-476":  5.50,   # NULL Pointer Dereference
    "CWE-401":  5.20,   # Memory Leak
    "CWE-77":   8.80,   # Command Injection
    "CWE-94":   8.60,   # Code Injection
    "CWE-502":  8.40,   # Deserialization of Untrusted Data
    "CWE-918":  8.20,   # Server-Side Request Forgery (SSRF)
    "CWE-269":  7.80,   # Improper Privilege Management
    "CWE-434":  7.60,   # Unrestricted Upload of File
    "CWE-306":  7.40,   # Missing Authentication
    "CWE-798":  7.20,   # Use of Hard-coded Credentials
    "CWE-863":  7.00,   # Incorrect Authorization
    "CWE-611":  6.80,   # XXE
}

# Sözlükte bulunmayan CWE'ler için varsayılan taban skoru
_DEFAULT_BASE_SCORE: float = 5.0


def get_base_score(cwe_id: Optional[str], is_vulnerable: bool) -> float:
    """
    CWE ID'ye göre temel zafiyet skorunu döndürür.

    Args:
        cwe_id: "CWE-120" formatında CWE kimliği veya None
        is_vulnerable: Kodun zafiyet içerip içermediği

    Returns:
        Temel zafiyet skoru (float). Zafiyet yoksa 0.0
    """
    # Zafiyet yoksa skor 0.0
    if not is_vulnerable:
        return 0.0

    # CWE ID yoksa veya "Unknown" ise varsayılan skor
    if not cwe_id or cwe_id == "Unknown":
        return _DEFAULT_BASE_SCORE

    # Sözlükten skoru al, yoksa varsayılan
    return CWE_BASE_SCORES.get(cwe_id, _DEFAULT_BASE_SCORE)


# ============================================================================
# ADIM 2: Katman 2 — Saldırı Yüzeyi Çarpanı (Attack Surface Multiplier)
# ============================================================================
# Dış dünyadan veri alındığını gösteren riskli fonksiyonlar
_ATTACK_SURFACE_FUNCTIONS = frozenset({
    "recv", "recvfrom", "recvmsg",
    "socket", "bind", "listen", "accept", "connect",
    "getenv", "getenv_s",
    "scanf", "fscanf", "sscanf", "vscanf",
    "fgets", "gets", "getline",
    "fread",
    "cin",            # C++ stdin akışı
    "read",           # POSIX read
    "ReadFile",       # Windows API
    "WSARecv",        # Windows Socket
    "InternetReadFile",
    "argv",           # Komut satırı argümanları
})

# Derleme zamanında kullanılacak regex: fonksiyon çağrısı veya kelime eşleşmesi
_ATTACK_SURFACE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(fn) for fn in _ATTACK_SURFACE_FUNCTIONS) + r')\b'
)


def calculate_attack_surface(raw_code: str) -> float:
    """
    Kodda dış dünyadan veri alındığını gösteren riskli fonksiyonları arar.
    Taint analizini destekler.

    Args:
        raw_code: Ham C/C++ kaynak kodu

    Returns:
        1.0 → Riskli fonksiyon bulundu (dış veri girişi mevcut)
        0.5 → Riskli fonksiyon bulunamadı (düşük saldırı yüzeyi)
    """
    if not raw_code:
        return 0.5

    if _ATTACK_SURFACE_PATTERN.search(raw_code):
        return 1.0

    return 0.5


# ============================================================================
# ADIM 3: Katman 3 — Varlık Kritiklik Çarpanı (Asset Criticality Multiplier)
# ============================================================================
def calculate_criticality(
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Hedef kaynağın önemini belirler.

    Args:
        source: "github", "hf", "huggingface" veya "owasp"
        metadata: Platform-spesifik meta veriler
            - GitHub: {"stars": int}
            - HuggingFace: {"downloads": int}

    Returns:
        0.1 ~ 1.0 arası kritiklik çarpanı
    """
    if metadata is None:
        metadata = {}

    source_lower = source.lower()

    if source_lower == "github":
        # GitHub: API'den yıldız (star) sayısı
        stars = metadata.get("stars", 0)
        if not isinstance(stars, (int, float)):
            stars = 0
        # Formül: Min(stars, 10000) / 10000 * 0.9 + 0.1
        capped = min(stars, 10000)
        return (capped / 10000) * 0.9 + 0.1

    elif source_lower in ("hf", "huggingface"):
        # Hugging Face: dataset_info() ile aylık indirme sayısı
        downloads = metadata.get("downloads", 0)
        if not isinstance(downloads, (int, float)):
            downloads = 0
        # Formül: Min(downloads, 100000) / 100000 * 0.9 + 0.1
        capped = min(downloads, 100000)
        return (capped / 100000) * 0.9 + 0.1

    elif source_lower == "owasp":
        # OWASP: Varsayılan olarak 1.0 (en yüksek kritiklik)
        return 1.0

    else:
        # Bilinmeyen kaynak: varsayılan orta seviye
        logger.warning(f"[RiskScoring] Bilinmeyen kaynak: {source}. Varsayılan 0.5 kullanılıyor.")
        return 0.5


# ============================================================================
# Final Risk Hesaplama
# ============================================================================
def calculate_final_risk(
    cwe_id: Optional[str],
    is_vulnerable: bool,
    raw_code: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    3 risk katmanını çarparak final risk skoru hesaplar.

    Args:
        cwe_id: CWE kimliği veya None
        is_vulnerable: Kodun zafiyet içerip içermediği
        raw_code: Ham kaynak kodu
        source: Veri kaynağı
        metadata: Platform meta verileri

    Returns:
        {
            "base_score": float,
            "attack_surface": float,
            "asset_criticality": float,
            "final_calculated_risk": float,
            "cwe_id": str | None,
            "is_vulnerable": bool,
        }
    """
    base = get_base_score(cwe_id, is_vulnerable)
    surface = calculate_attack_surface(raw_code)
    criticality = calculate_criticality(source, metadata)

    final = base * surface * criticality

    return {
        "base_score": round(base, 4),
        "attack_surface": round(surface, 4),
        "asset_criticality": round(criticality, 4),
        "final_calculated_risk": round(final, 4),
        "cwe_id": cwe_id,
        "is_vulnerable": is_vulnerable,
    }


# ============================================================================
# CLI Demo
# ============================================================================
if __name__ == "__main__":
    import json

    # Test 1: Zafiyet içeren kod
    test_code_vuln = '''
    #include <stdio.h>
    #include <string.h>
    void handle(char *input) {
        char buf[64];
        char *data = getenv("USER_INPUT");
        strcpy(buf, data);
        printf("%s\\n", buf);
    }
    '''
    result = calculate_final_risk(
        cwe_id="CWE-120",
        is_vulnerable=True,
        raw_code=test_code_vuln,
        source="github",
        metadata={"stars": 5000},
    )
    print("=== Zafiyet İçeren Kod (CWE-120, GitHub 5000★) ===")
    print(json.dumps(result, indent=2))

    # Test 2: Temiz kod
    result_clean = calculate_final_risk(
        cwe_id=None,
        is_vulnerable=False,
        raw_code="int main() { return 0; }",
        source="owasp",
    )
    print("\n=== Temiz Kod (OWASP) ===")
    print(json.dumps(result_clean, indent=2))

    # Test 3: HF kaynaklı
    result_hf = calculate_final_risk(
        cwe_id="CWE-416",
        is_vulnerable=True,
        raw_code="void f() { free(ptr); use(ptr); }",
        source="hf",
        metadata={"downloads": 50000},
    )
    print("\n=== HF Kaynaklı (CWE-416, 50K downloads) ===")
    print(json.dumps(result_hf, indent=2))
