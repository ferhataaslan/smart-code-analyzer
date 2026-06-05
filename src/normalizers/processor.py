"""
processor.py — Merkezi Yönlendirici (Facade)

Bu dosya artık kendi başına normalizasyon YAPMAZ.
Tüm işi platform-spesifik modüllere delege eder:

  - "github"  → github_ast_structnorm.py
  - "hf"      → hf_crossnorm.py
  - "owasp"   → owasp_secnorm.py

collector.py ve diğer tüketici kodlar "from src.normalizers.processor import process_code"
çağrısını AYNEN kullanmaya devam eder; arka planda yeni algoritmalar çalışır.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ============================================================================
# Yeni modüllerden import
# ============================================================================
from src.normalizers.github_ast_structnorm import normalize_github_code, GitHubASTStructNorm
from src.normalizers.hf_crossnorm import normalize_hf_code, HFCrossNorm
from src.normalizers.owasp_secnorm import normalize_owasp_code, OWASPSecNorm


# ============================================================================
# Public API — Geriye Dönük Uyumlu (Backward-Compatible)
# ============================================================================
def process_code(code: str, source: str = "github", **kwargs) -> str:
    """
    Public API — collector.py bu fonksiyonu çağırır.

    Eski imza tamamen korunur:
        process_code(raw_code, source="github")  → str
        process_code(raw_code, source="hf")      → str
        process_code(raw_code, source="owasp")   → str

    Ek olarak kwargs ile yeni parametreler geçilebilir:
        process_code(raw_code, source="hf", docstring="...")
        process_code(raw_code, source="owasp", cwe_hint="CWE-89")
        process_code(raw_code, source="github", language="c")

    Returns:
        Normalize edilmiş kod (str). Eski davranışla birebir uyumlu.
    """
    if not code or not code.strip():
        return ""

    try:
        source_lower = source.lower()

        if source_lower == "github":
            language = kwargs.get("language", "cpp")
            result = normalize_github_code(code, language=language)
            return result.get("normalized_code", code)

        elif source_lower == "hf":
            docstring = kwargs.get("docstring", "")
            result = normalize_hf_code(code, docstring=docstring)
            return result.get("normalized_code", code)

        elif source_lower == "owasp":
            language = kwargs.get("language", "cpp")
            cwe_hint = kwargs.get("cwe_hint", "Unknown")
            result = normalize_owasp_code(
                code, language=language, cwe_hint=cwe_hint
            )
            return result.get("normalized_code", code)

        else:
            logger.warning(
                f"Bilinmeyen kaynak: '{source}'. "
                f"Varsayılan olarak GitHub algoritması kullanılıyor."
            )
            result = normalize_github_code(code)
            return result.get("normalized_code", code)

    except Exception as e:
        logger.error(f"[processor] Normalizasyon hatası (source={source}): {e}")
        return code


def process_code_full(code: str, source: str = "github", **kwargs) -> Dict[str, Any]:
    """
    Genişletilmiş API — normalize kod + meta-veri döner.

    Returns:
        Platform-spesifik sözlük (normalized_code, metadata, vb.)
    """
    if not code or not code.strip():
        return {"normalized_code": "", "applied_algorithm": f"{source}_empty"}

    try:
        source_lower = source.lower()

        if source_lower == "github":
            language = kwargs.get("language", "cpp")
            return normalize_github_code(code, language=language)

        elif source_lower == "hf":
            docstring = kwargs.get("docstring", "")
            return normalize_hf_code(code, docstring=docstring)

        elif source_lower == "owasp":
            language = kwargs.get("language", "cpp")
            cwe_hint = kwargs.get("cwe_hint", "Unknown")
            return normalize_owasp_code(
                code, language=language, cwe_hint=cwe_hint
            )

        else:
            logger.warning(f"Bilinmeyen kaynak: '{source}'. GitHub kullanılıyor.")
            return normalize_github_code(code)

    except Exception as e:
        logger.error(f"[processor] Full pipeline hatası: {e}")
        return {"normalized_code": code, "error": str(e)}


# ============================================================================
# Factory — Doğrudan sınıf erişimi gerekirse
# ============================================================================
_CLASS_MAP = {
    "github": GitHubASTStructNorm,
    "hf": HFCrossNorm,
    "owasp": OWASPSecNorm,
}


def get_processor(source: str = "github"):
    """Factory: source parametresine göre doğru normalizer sınıfını döner."""
    cls = _CLASS_MAP.get(source.lower(), GitHubASTStructNorm)
    return cls()
