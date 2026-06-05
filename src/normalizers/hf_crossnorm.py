#!/usr/bin/env python3
"""
hf_crossnorm.py — Hugging Face / CodeSearchNet Verileri İçin Çapraz Normalizasyon

Amaç: Doğal Dil (NL) ve Programlama Dili (PL) hizalamasını bozmadan,
MLM ve RTD eğitim görevleri için frekans dengelemesi yapmak.

Algoritma Adımları:
  1. Tree-Sitter AST ile identifier çıkarma (regex DEĞİL)
  2. Z-Score ile istatistiksel uç değer (outlier) tespiti
  3. AST-tabanlı birebir (exact match) ALLOW_LIST kontrolü
  4. Çift Yönlü Senkronize Yeniden Adlandırma (Bidirectional Synchronized Renaming)
  5. Optimize Edilmiş Semantik Parçalama (Semantic Chunking)
"""

import re
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================================
# Tree-Sitter Parser Kurulumu
# ============================================================================
try:
    import tree_sitter as ts
    import tree_sitter_c as tsc
    import tree_sitter_cpp as tscpp
    C_LANGUAGE = ts.Language(tsc.language())
    CPP_LANGUAGE = ts.Language(tscpp.language())
    _TS_AVAILABLE = True
except Exception:
    _TS_AVAILABLE = False
    logger.warning("[HFCrossNorm] Tree-sitter yüklenemedi, fallback aktif.")

# ============================================================================
# Yapılandırma Sabitleri
# ============================================================================
DEFAULT_Z_SCORE_THRESHOLD: float = 2.5
DEFAULT_MAX_CHUNK_TOKENS: int = 512

# C/C++ anahtar kelimeleri — bunlara dokunulmaz
# Ön işlemci direktifleri (#define, #include, #ifdef vb.) SİLİNMEZ, korunur.
# Sadece içlerindeki kullanıcı tanımlı identifier'lar normalize edilir.
CC_KEYWORDS: frozenset = frozenset({
    "int", "char", "float", "double", "void", "struct", "union", "enum",
    "typedef", "size_t", "bool", "const", "static", "extern", "volatile",
    "auto", "register", "signed", "unsigned", "long", "short", "inline",
    "sizeof", "if", "else", "while", "for", "do", "switch", "case",
    "break", "continue", "goto", "return", "default", "class", "namespace",
    "template", "typename", "virtual", "override", "public", "private",
    "protected", "new", "delete", "this", "nullptr", "true", "false",
    "try", "catch", "throw", "using", "noexcept", "constexpr",
    "malloc", "calloc", "realloc", "free", "memset", "memcpy", "memmove",
    "strcpy", "strncpy", "strcat", "strlen", "strcmp", "sprintf", "snprintf",
    "printf", "fprintf", "scanf", "fopen", "fclose", "fread", "fwrite",
    "socket", "bind", "listen", "accept", "connect", "send", "recv",
    "system", "exec", "fork", "exit", "popen", "main",
    # Büyük harfli sabitler (case-sensitive koruma)
    "NULL", "EOF", "TRUE", "FALSE",
    "EXIT_SUCCESS", "EXIT_FAILURE",
    "stdin", "stdout", "stderr", "errno",
    "std", "cout", "cin", "endl", "cerr", "string",
    "vector", "map", "set", "pair",
    # Ön işlemci anahtar kelimeleri
    "include", "define", "ifdef", "ifndef", "endif", "pragma",
    "defined", "elif", "undef", "once",
    # POSIX/Socket sabitleri
    "AF_INET", "AF_INET6", "SOCK_STREAM", "SOCK_DGRAM",
    "IPPROTO_TCP", "IPPROTO_UDP", "INADDR_ANY", "INVALID_SOCKET",
    "SOCKET", "SOCKET_ERROR", "SOL_SOCKET", "SO_REUSEADDR",
    "sockaddr_in", "sockaddr", "in_addr",
    "sin_family", "sin_port", "sin_addr", "s_addr",
    "BUFFER_SIZE", "MAX_PATH", "PATH_MAX", "BUFSIZ",
    "FILE", "HANDLE", "DWORD", "BOOL",
    "SEEK_SET", "SEEK_CUR", "SEEK_END",
    "assert",
    # Ağ/IO fonksiyonları
    "htons", "ntohs", "htonl", "ntohl", "inet_addr", "inet_ntoa",
    "read", "write", "open", "close", "ioctl", "select", "poll",
    "getenv", "setenv", "atoi", "atol", "atof", "strtol", "strtod",
    "perror", "abort", "signal",
})


# ============================================================================
# Adım 0: AST-Tabanlı Yorum Satırı Temizliği (Comment Stripping)
# ============================================================================
def _strip_comments_via_ast(code: str, language: str = "cpp") -> str:
    """
    Tree-Sitter AST ile koddan tüm yorum düğümlerini (comment) siler.
    String literal içindeki // veya /* dizilerine DOKUNMAZ.

    Args:
        code: Ham C/C++ kaynak kodu
        language: "c" veya "cpp"

    Returns:
        Yorumlardan temizlenmiş kod
    """
    if not _TS_AVAILABLE or not code or not code.strip():
        return code

    try:
        lang = CPP_LANGUAGE if language == "cpp" else C_LANGUAGE
        parser = ts.Parser(lang)
        source_bytes = code.encode("utf-8")
        tree = parser.parse(source_bytes)

        # Comment düğümlerini topla
        comments = []

        def _find_comments(node):
            if node.type == "comment":
                comments.append((node.start_byte, node.end_byte))
                return
            for child in node.children:
                _find_comments(child)

        _find_comments(tree.root_node)

        if not comments:
            return code

        # Sondan başa sil (byte pozisyonları kaymasın)
        buf = bytearray(source_bytes)
        for start, end in sorted(comments, reverse=True):
            buf[start:end] = b""

        result = buf.decode("utf-8", errors="ignore")

        # Boş satır temizliği
        lines = [ln for ln in result.split("\n") if ln.strip()]
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"[HFCrossNorm] Yorum temizleme hatası: {e}")
        return code


# ============================================================================
# Adım 1: AST-Tabanlı Identifier Çıkarma
# ============================================================================
def _extract_ast_identifiers(code: str, language: str = "cpp") -> List[Tuple[int, int, str, str]]:
    """
    Tree-Sitter AST ile koddan identifier, type_identifier ve field_identifier
    düğümlerini çıkarır. String literal, yorum ve #include dosya isimleri
    HARİÇ tutulur.

    Returns:
        [(start_byte, end_byte, name, node_type), ...]
    """
    if not _TS_AVAILABLE:
        return []

    lang = CPP_LANGUAGE if language == "cpp" else C_LANGUAGE
    parser = ts.Parser(lang)
    source_bytes = code.encode("utf-8")
    tree = parser.parse(source_bytes)

    query_str = """
        (identifier) @id
        (type_identifier) @type_id
        (field_identifier) @field_id
    """
    try:
        query = ts.Query(lang, query_str)
    except (TypeError, AttributeError):
        query = lang.query(query_str)

    cursor = ts.QueryCursor(query)
    matches = cursor.matches(tree.root_node)

    results = []
    seen: Set[int] = set()

    for _, caps in matches:
        for tag, nodes in caps.items():
            if isinstance(nodes, list):
                for nd in nodes:
                    if nd.id not in seen:
                        seen.add(nd.id)
                        name = source_bytes[nd.start_byte:nd.end_byte].decode("utf-8")
                        # #include içindeki dosya adlarını atla
                        parent = nd.parent
                        if parent and parent.type in ("preproc_include", "system_lib_string"):
                            continue
                        results.append((nd.start_byte, nd.end_byte, name, tag))
            else:
                nd = nodes
                if nd.id not in seen:
                    seen.add(nd.id)
                    name = source_bytes[nd.start_byte:nd.end_byte].decode("utf-8")
                    parent = nd.parent
                    if parent and parent.type in ("preproc_include", "system_lib_string"):
                        continue
                    results.append((nd.start_byte, nd.end_byte, name, tag))

    return results


def _detect_function_names(code: str, language: str = "cpp") -> Set[str]:
    """Tree-Sitter ile fonksiyon tanımları ve çağrılarını tespit eder."""
    if not _TS_AVAILABLE:
        return set()

    lang = CPP_LANGUAGE if language == "cpp" else C_LANGUAGE
    parser = ts.Parser(lang)
    source_bytes = code.encode("utf-8")
    tree = parser.parse(source_bytes)

    func_names: Set[str] = set()

    def walk(node):
        if node.type == "function_declarator":
            for c in node.children:
                if c.type == "identifier":
                    name = source_bytes[c.start_byte:c.end_byte].decode("utf-8")
                    if name not in CC_KEYWORDS:
                        func_names.add(name)
        if node.type == "call_expression":
            for c in node.children:
                if c.type == "identifier":
                    name = source_bytes[c.start_byte:c.end_byte].decode("utf-8")
                    if name not in CC_KEYWORDS:
                        func_names.add(name)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return func_names


# ============================================================================
# Adım 2: Z-Score Tabanlı Frekans Dengelemesi
# ============================================================================
@dataclass
class TokenFrequencyProfile:
    """Bir veri setindeki token frekanslarının istatistiksel profili."""
    frequencies: Dict[str, int] = field(default_factory=dict)
    mean: float = 0.0
    std_dev: float = 0.0
    outliers: Set[str] = field(default_factory=set)


class ZScoreOutlierDetector:
    """
    Token frekanslarını analiz eder, Z-Score ile istatistiksel
    uç değerleri (çok nadir / aşırı uzun spesifik değişken isimleri) tespit eder.
    """

    def __init__(self, z_threshold: float = DEFAULT_Z_SCORE_THRESHOLD) -> None:
        self._z_threshold = z_threshold

    def build_profile(self, identifier_names: List[str]) -> TokenFrequencyProfile:
        """Identifier listesinden frekans profili oluşturur."""
        profile = TokenFrequencyProfile()

        # CC_KEYWORDS exact match ile hariç tut
        identifiers = [t for t in identifier_names if t not in CC_KEYWORDS]

        if not identifiers:
            return profile

        profile.frequencies = dict(Counter(identifiers))
        freq_values = list(profile.frequencies.values())

        n = len(freq_values)
        profile.mean = sum(freq_values) / n
        variance = sum((f - profile.mean) ** 2 for f in freq_values) / n
        profile.std_dev = math.sqrt(variance) if variance > 0 else 0.0

        # Z-Score hesapla ve outlier'ları işaretle
        if profile.std_dev > 0:
            for token, freq in profile.frequencies.items():
                z_score = (freq - profile.mean) / profile.std_dev
                if z_score < -self._z_threshold or (
                    z_score < -1.0 and len(token) > 25
                ):
                    profile.outliers.add(token)

        return profile


# ============================================================================
# Adım 3: AST-Tabanlı Senkronize Yeniden Adlandırma
# ============================================================================
class BidirectionalSyncRenamer:
    """
    AST'den gelen identifier pozisyonlarını kullanarak byte-pozisyon tabanlı
    değiştirme yapar. Regex yerine AST exact-match kullanır.
    PL'de değiştirilen isimler NL (docstring) metninde de senkronize edilir.
    """

    def __init__(self) -> None:
        self._rename_map: Dict[str, str] = {}
        self._var_counter: int = 0
        self._func_counter: int = 0
        self._type_counter: int = 0

    def _get_normalized_name(self, original: str, category: str) -> str:
        """Kategoriye göre standart isim üretir."""
        if original in self._rename_map:
            return self._rename_map[original]

        if category == "func":
            self._func_counter += 1
            normalized = f"FUNC_{self._func_counter}"
        elif category == "type":
            self._type_counter += 1
            normalized = f"TYPE_{self._type_counter}"
        else:
            self._var_counter += 1
            normalized = f"VAR_{self._var_counter}"

        self._rename_map[original] = normalized
        return normalized

    def synchronized_rename(
        self, code: str, docstring: str,
        ast_identifiers: List[Tuple[int, int, str, str]],
        func_names: Set[str],
        outlier_map: Optional[Dict[str, str]] = None,
        language: str = "cpp",
    ) -> Tuple[str, str, Dict[str, str]]:
        """
        AST pozisyonlarına dayalı byte-exact değiştirme.
        String literal, yorum ve #include içerikleri DOKUNULMAZ.
        """
        self._rename_map = {}
        self._var_counter = 0
        self._func_counter = 0
        self._type_counter = 0

        # Outlier map'i entegre et
        if outlier_map:
            for orig, norm in outlier_map.items():
                self._rename_map[orig] = norm

        # Replacements listesi: (start_byte, end_byte, new_name)
        replacements: List[Tuple[int, int, str]] = []

        for start, end, name, tag in ast_identifiers:
            # CC_KEYWORDS exact match kontrolü
            if name in CC_KEYWORDS:
                continue

            # Outlier map'te varsa onu kullan
            if name in self._rename_map:
                replacements.append((start, end, self._rename_map[name]))
                continue

            # Kategoriye göre normalize et
            if tag == "id":
                if name in func_names:
                    new_name = self._get_normalized_name(name, "func")
                else:
                    new_name = self._get_normalized_name(name, "var")
            elif tag == "type_id":
                new_name = self._get_normalized_name(name, "type")
            elif tag == "field_id":
                new_name = self._get_normalized_name(name, "var")
            else:
                continue

            replacements.append((start, end, new_name))

        # Byte-pozisyon tabanlı değiştirme (sondan başa — pozisyon kaymasını önle)
        source_bytes = code.encode("utf-8")
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, new_name in replacements:
            source_bytes = source_bytes[:start] + new_name.encode("utf-8") + source_bytes[end:]

        renamed_code = source_bytes.decode("utf-8")

        # NL (docstring) senkronizasyonu — kelime sınırı korumalı regex
        renamed_docstring = docstring
        for original, normalized in sorted(
            self._rename_map.items(), key=lambda x: len(x[0]), reverse=True
        ):
            word_pattern = re.compile(r"\b" + re.escape(original) + r"\b")
            renamed_docstring = word_pattern.sub(normalized, renamed_docstring)

        return renamed_code, renamed_docstring, dict(self._rename_map)


# ============================================================================
# Adım 4: Optimize Edilmiş Semantik Parçalama (Semantic Chunking)
# ============================================================================
class SemanticChunker:
    """
    Token sayısı bağlam sınırını aşıyorsa, bölme işlemini rastgele
    karakterlerden değil, anlamsal cümle/blok sınırlarından yapar.
    """

    _FUNC_BOUNDARY = re.compile(
        r"^(?:(?:static|inline|extern|virtual|void|int|char|float|double|"
        r"long|short|unsigned|signed|bool|auto|size_t|struct|class|"
        r"(?:[A-Za-z_]\w*(?:::[A-Za-z_]\w*)?))\\s+)+"
        r"[A-Za-z_]\w*\s*\(",
        re.MULTILINE,
    )

    def __init__(self, max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS) -> None:
        self._max_tokens = max_tokens

    def _estimate_token_count(self, text: str) -> int:
        """Basit whitespace-based token sayısı tahmini (BPE'ye yaklaşık)."""
        words = text.split()
        return int(len(words) * 1.3)

    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """Metindeki anlamsal sınır noktalarını (satır indeksleri) bulur."""
        lines = text.split("\n")
        boundaries: List[int] = [0]

        brace_depth = 0
        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped and i > 0:
                boundaries.append(i)
                continue

            if self._FUNC_BOUNDARY.match(stripped):
                boundaries.append(i)

            open_count = stripped.count("{")
            close_count = stripped.count("}")
            brace_depth += open_count - close_count
            if brace_depth == 0 and close_count > 0:
                boundaries.append(i + 1)

        boundaries.append(len(lines))
        return sorted(set(boundaries))

    def chunk(self, text: str) -> List[str]:
        """Metni semantik sınırlardan böler."""
        estimated = self._estimate_token_count(text)
        if estimated <= self._max_tokens:
            return [text]

        lines = text.split("\n")
        boundaries = self._find_semantic_boundaries(text)

        chunks: List[str] = []
        current_chunk_lines: List[str] = []
        current_tokens = 0

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment_lines = lines[start:end]
            segment_text = "\n".join(segment_lines)
            segment_tokens = self._estimate_token_count(segment_text)

            if current_tokens + segment_tokens > self._max_tokens and current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
                current_tokens = 0

            current_chunk_lines.extend(segment_lines)
            current_tokens += segment_tokens

        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))

        return chunks


# ============================================================================
# Ana Sınıf: HFCrossNorm
# ============================================================================
class HFCrossNorm:
    """
    Hugging Face / CodeSearchNet verileri için çapraz normalizasyon motoru.

    Pipeline (AST-tabanlı):
      1. Tree-Sitter AST ile identifier çıkarma
      2. Z-Score outlier tespiti
      3. AST pozisyon-tabanlı bidirectional synchronized renaming (PL <-> NL)
      4. Semantic chunking
    """

    def __init__(
        self,
        z_threshold: float = DEFAULT_Z_SCORE_THRESHOLD,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    ) -> None:
        self._outlier_detector = ZScoreOutlierDetector(z_threshold=z_threshold)
        self._renamer = BidirectionalSyncRenamer()
        self._chunker = SemanticChunker(max_tokens=max_chunk_tokens)

    def process(
        self, code: str, docstring: str = "", language: str = "cpp"
    ) -> Dict[str, Any]:
        """
        Tam normalizasyon pipeline'ı.

        Args:
            code: PL (programlama dili) bloğu
            docstring: NL (doğal dil / açıklama) metni
            language: "c" veya "cpp"

        Returns:
            {
                "normalized_code": str,
                "normalized_docstring": str,
                "chunks": list[str],
                "nl_alignment": dict,
                "applied_algorithm": str,
            }
        """
        if not code or not code.strip():
            return {
                "normalized_code": "",
                "normalized_docstring": docstring,
                "chunks": [],
                "nl_alignment": {},
                "applied_algorithm": "hf_crossnorm",
            }

        try:
            # ── Adım 0: Yorum Satırı Temizliği (Comment Stripping) ──
            # Yorumlar model eğitiminde shortcut learning'e yol açar.
            # AST üzerinden comment düğümleri tespit edilip siliniyor.
            code = _strip_comments_via_ast(code, language)

            # ── Adım 1: AST ile identifier çıkarma ──
            ast_ids = _extract_ast_identifiers(code, language)
            func_names = _detect_function_names(code, language)

            # Sadece normalize edilecek identifier isimlerini topla
            normalizable_names = [
                name for _, _, name, _ in ast_ids
                if name not in CC_KEYWORDS
            ]

            # ── Adım 2: Z-Score Outlier Tespiti ──
            profile = self._outlier_detector.build_profile(normalizable_names)
            outlier_map: Dict[str, str] = {}
            counter = 0
            for name in sorted(profile.outliers):
                counter += 1
                outlier_map[name] = f"RARE_VAR_{counter}"

            # ── Adım 3: AST-tabanlı Bidirectional Renaming ──
            renamed_code, renamed_docstring, rename_map = (
                self._renamer.synchronized_rename(
                    code, docstring, ast_ids, func_names, outlier_map, language
                )
            )

            # ── Adım 4: Semantic Chunking ──
            chunks = self._chunker.chunk(renamed_code)

            # NL Alignment meta-verisi
            nl_alignment = {
                "rename_map": rename_map,
                "outlier_count": len(profile.outliers),
                "outlier_tokens": list(profile.outliers)[:20],
                "total_unique_identifiers": len(profile.frequencies),
                "frequency_mean": round(profile.mean, 4),
                "frequency_std_dev": round(profile.std_dev, 4),
                "chunk_count": len(chunks),
                "bidirectional_sync": True,
                "docstring_present": bool(docstring.strip()),
            }

            return {
                "normalized_code": renamed_code,
                "normalized_docstring": renamed_docstring,
                "chunks": chunks,
                "nl_alignment": nl_alignment,
                "applied_algorithm": "hf_crossnorm",
            }

        except Exception as e:
            logger.error(f"[HFCrossNorm] Pipeline hatası: {e}")
            return {
                "normalized_code": code,
                "normalized_docstring": docstring,
                "chunks": [code],
                "nl_alignment": {"error": str(e)},
                "applied_algorithm": "hf_crossnorm",
            }


# ============================================================================
# Public API
# ============================================================================
def normalize_hf_code(
    code: str, docstring: str = "",
    z_threshold: float = DEFAULT_Z_SCORE_THRESHOLD,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
) -> Dict[str, Any]:
    """
    Hugging Face veri seti kodunu normalize eder.

    Args:
        code: PL bloğu (C/C++ kaynak kodu)
        docstring: NL bloğu (fonksiyon açıklaması)
        z_threshold: Z-Score eşik değeri
        max_chunk_tokens: Maksimum chunk token sayısı

    Returns:
        Normalize edilmiş kod, docstring, chunk'lar ve alignment bilgisi
    """
    normalizer = HFCrossNorm(
        z_threshold=z_threshold,
        max_chunk_tokens=max_chunk_tokens,
    )
    return normalizer.process(code, docstring)


if __name__ == "__main__":
    import json

    sample_code = '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 256
#ifndef MY_HEADER_GUARD
#define MY_HEADER_GUARD

void process_authentication_request_handler(
    const char *user_authentication_token_value,
    int max_retry_count
) {
    char *session_buffer = (char *)malloc(BUFFER_SIZE);
    if (session_buffer == NULL) return;

    strcpy(session_buffer, user_authentication_token_value);
    for (int retry_index = 0; retry_index < max_retry_count; retry_index++) {
        printf("Attempt %d with token: %s\\n", retry_index, session_buffer);
    }
    free(session_buffer);
}
#endif
    '''

    sample_docstring = (
        "Processes the authentication request using "
        "user_authentication_token_value and retries up to max_retry_count times. "
        "Allocates a session_buffer for internal processing."
    )

    result = normalize_hf_code(sample_code, sample_docstring)
    print("=== Normalized Code ===")
    print(result["normalized_code"])
    print("\n=== Normalized Docstring ===")
    print(result["normalized_docstring"])
    print("\n=== NL Alignment ===")
    print(json.dumps(result["nl_alignment"], indent=2, ensure_ascii=False))
