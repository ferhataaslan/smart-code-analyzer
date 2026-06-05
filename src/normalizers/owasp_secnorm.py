#!/usr/bin/env python3
"""
owasp_secnorm.py — OWASP ve Güvenlik Verileri İçin Asimetrik Normalizasyon

Amaç: Güvenlik bağlamını, kirlilik (taint) yollarını ve zafiyet fonksiyonlarını
kesinlikle koruyarak asimetrik normalizasyon yapmak.

Algoritma Adımları:
  1. OWASP Top 10 Güvenlik Domain Semantiği Sözlüğü (Allow-List)
  2. Source→Sink taint yolu izleme ve ara taşıyıcı değişken standartlaştırma
  3. CWE / Source-Sink meta-veri çıktısı (JSON)

Kısıtlama: C/C++ odaklı. Bellek yönetimi ve zafiyet fonksiyonları tamamen
korunur.
"""

import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    import tree_sitter_c
    import tree_sitter_cpp
    from tree_sitter import Language, Parser
except ImportError as e:
    raise ImportError(
        f"tree-sitter bağımlılıkları eksik: {e}. "
        "Lütfen 'pip install tree-sitter tree-sitter-c tree-sitter-cpp' çalıştırın."
    )

logger = logging.getLogger(__name__)

# ============================================================================
# Dil Motorları
# ============================================================================
C_LANGUAGE = Language(tree_sitter_c.language())
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
_PARSERS = {"c": Parser(C_LANGUAGE), "cpp": Parser(CPP_LANGUAGE)}

# ============================================================================
# Adım 1: OWASP Top 10 Güvenlik Domain Semantiği Sözlüğü (Allow-List)
# ============================================================================
# Source (Kaynak): Dış girdi noktaları — kullanıcıdan / ağdan gelen veri
SOURCE_FUNCTIONS: frozenset = frozenset({
    # Ağ girdileri
    "recv", "recvfrom", "recvmsg", "read", "fread", "fgets",
    "getline", "gets", "scanf", "fscanf", "sscanf", "vscanf",
    # Ortam değişkenleri
    "getenv", "getenv_s",
    # Komut satırı / stdin
    "getchar", "fgetc", "getc", "gets_s",
    # Windows API
    "ReadFile", "ReadConsole", "GetCommandLine",
    "InternetReadFile", "WSARecv",
    # CGI / Web
    "cgi_param", "query_string",
    # Argüman vektörü
    "argv",
})

# Sink (Kuyu): Tehlikeli hedef fonksiyonlar — zafiyet noktaları
SINK_FUNCTIONS: frozenset = frozenset({
    # Bellek zafiyetleri (CWE-119, CWE-120, CWE-122)
    "strcpy", "strncpy", "strcat", "strncat", "sprintf", "vsprintf",
    "memcpy", "memmove", "memset", "bzero",
    "wcscpy", "wcsncpy", "wcscat", "wmemcpy", "wmemset",
    "gets",  # CWE-242: gets() asla güvenli değildir
    # OS Komut Enjeksiyonu (CWE-78)
    "system", "exec", "execl", "execlp", "execle", "execv", "execvp",
    "execvpe", "popen", "ShellExecute", "ShellExecuteEx",
    "CreateProcess", "CreateProcessA", "CreateProcessW",
    "WinExec",
    # SQL Enjeksiyonu (CWE-89)
    "mysql_query", "mysql_real_query", "sqlite3_exec",
    "PQexec", "PQexecParams",
    # Format String (CWE-134)
    "printf", "fprintf", "snprintf", "vprintf", "vfprintf",
    "syslog", "wprintf",
    # Dosya İşlemleri (CWE-22, CWE-73)
    "fopen", "open", "creat", "freopen",
    "CreateFile", "CreateFileA", "CreateFileW",
    # Bellek Yönetimi (CWE-415, CWE-416)
    "malloc", "calloc", "realloc", "free",
    "VirtualAlloc", "VirtualFree", "HeapAlloc", "HeapFree",
    "new", "delete",
    # Kriptografik (CWE-327, CWE-328)
    "MD5_Init", "MD5_Update", "MD5_Final",
    "SHA1_Init", "SHA1_Update", "SHA1_Final",
    "DES_ecb_encrypt", "DES_set_key",
    "EVP_des_ecb", "EVP_md5",
    "rand", "srand", "random",
    # Pointer / Integer Overflow (CWE-190, CWE-191)
    "atoi", "atol", "atof", "strtol", "strtoul",
    # Ağ Sink'leri
    "send", "sendto", "sendmsg", "write", "fwrite",
    "WSASend",
    # LoadLibrary (CWE-426, CWE-427)
    "LoadLibrary", "LoadLibraryA", "LoadLibraryW",
    "GetProcAddress", "dlopen", "dlsym",
})

# Tam koruma listesi: Source + Sink + C/C++ anahtar kelimeler
SECURITY_ALLOW_LIST: frozenset = frozenset(
    SOURCE_FUNCTIONS | SINK_FUNCTIONS | frozenset({
        # C/C++ Anahtar Kelimeler (normalizasyondan muaf)
        "int", "char", "float", "double", "void", "struct", "union", "enum",
        "typedef", "size_t", "ssize_t", "bool", "const", "static", "extern",
        "volatile", "auto", "register", "signed", "unsigned", "long", "short",
        "inline", "restrict", "sizeof", "alignof",
        "if", "else", "while", "for", "do", "switch", "case", "break",
        "continue", "goto", "return", "default",
        "class", "namespace", "template", "typename", "virtual", "override",
        "public", "private", "protected", "new", "delete", "this", "nullptr",
        "true", "false", "try", "catch", "throw", "using", "noexcept",
        "constexpr", "decltype", "mutable", "explicit",
        "NULL", "EOF", "stdin", "stdout", "stderr", "errno",
        "EXIT_SUCCESS", "EXIT_FAILURE", "main",
        "FILE", "HANDLE", "DWORD", "BOOL", "SOCKET",
        "AF_INET", "AF_INET6", "AF_UNIX", "SOCK_STREAM", "SOCK_DGRAM",
        "IPPROTO_TCP", "IPPROTO_UDP", "INADDR_ANY", "INVALID_SOCKET",
        "SOCKET_ERROR", "SOL_SOCKET", "SO_REUSEADDR",
        "sockaddr_in", "sockaddr", "in_addr",
        "sin_family", "sin_port", "sin_addr", "s_addr",
        "BUFFER_SIZE", "MAX_PATH", "PATH_MAX", "BUFSIZ",
        "O_RDONLY", "O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND",
        "SIGINT", "SIGTERM", "SIGKILL", "SIGHUP",
        "assert", "signal", "kill", "fork", "pipe", "dup", "dup2",
        "std", "cout", "cin", "cerr", "endl", "string", "vector",
    })
)

# CWE heuristic mapping: sink fonksiyonu → olası CWE
_CWE_HEURISTICS: Dict[str, str] = {
    "strcpy": "CWE-120", "strncpy": "CWE-120", "strcat": "CWE-120",
    "sprintf": "CWE-120", "vsprintf": "CWE-120", "gets": "CWE-242",
    "memcpy": "CWE-119", "memmove": "CWE-119",
    "system": "CWE-78", "popen": "CWE-78", "exec": "CWE-78",
    "execl": "CWE-78", "execv": "CWE-78",
    "CreateProcess": "CWE-78", "WinExec": "CWE-78",
    "mysql_query": "CWE-89", "sqlite3_exec": "CWE-89",
    "printf": "CWE-134", "fprintf": "CWE-134", "syslog": "CWE-134",
    "fopen": "CWE-22", "open": "CWE-22",
    "malloc": "CWE-122", "free": "CWE-415",
    "rand": "CWE-330", "srand": "CWE-330", "random": "CWE-330",
    "MD5_Init": "CWE-327", "DES_ecb_encrypt": "CWE-327",
    "atoi": "CWE-190", "atol": "CWE-190",
    "LoadLibrary": "CWE-426", "LoadLibraryA": "CWE-426",
    "dlopen": "CWE-426",
}


# ============================================================================
# Adım 2: Taint Analizi ve Taşıyıcı Değişken Standartlaştırma
# ============================================================================
@dataclass
class TaintPath:
    """Tek bir Source → Sink veri akış yolunu temsil eder."""
    source_func: str
    source_line: int
    sink_func: str
    sink_line: int
    carrier_vars: List[str] = field(default_factory=list)
    cwe_id: str = "Unknown"


class TaintFlowAnalyzer:
    """
    Basitleştirilmiş intra-prosedürel taint akış analizi.

    Kaynaktan Kuyuya giden yoldaki ara taşıyıcı değişkenleri tespit eder
    ve TAINT_1, TAINT_2, ... formatına dönüştürür.
    """

    def __init__(self) -> None:
        self._taint_set: Dict[str, int] = {}  # var_name → taint_id
        self._taint_counter: int = 0
        self._paths: List[TaintPath] = []

    def _next_taint_id(self) -> int:
        self._taint_counter += 1
        return self._taint_counter

    def analyze(
        self, node: Any, source_bytes: bytes, language: str = "cpp"
    ) -> Tuple[Dict[str, str], List[TaintPath]]:
        """
        AST üzerinde yürüyerek taint akışlarını analiz eder.

        Returns:
            (taint_rename_map, taint_paths)
        """
        self._taint_set = {}
        self._taint_counter = 0
        self._paths = []

        self._walk_for_taint(node, source_bytes)

        # Taint rename map oluştur
        rename_map: Dict[str, str] = {}
        for var_name, tid in self._taint_set.items():
            rename_map[var_name] = f"TAINT_{tid}"

        return rename_map, self._paths

    def _walk_for_taint(self, node: Any, sb: bytes) -> None:
        """AST üzerinde rekürsif taint analizi."""
        try:
            # Bildirim/atama: source fonksiyonundan dönen değeri izle
            if node.type == "declaration" or node.type == "init_declarator":
                self._check_source_assignment(node, sb)

            # Çağrı ifadesi: sink fonksiyonuna tainted veri geçişi kontrol
            if node.type == "call_expression":
                self._check_sink_call(node, sb)

            # Atama ifadesi: taint yayılımı
            if node.type == "assignment_expression":
                self._propagate_taint(node, sb)

            for child in node.children:
                self._walk_for_taint(child, sb)
        except Exception as e:
            logger.debug(f"[TaintAnalyzer] Walk hatası: {e}")

    def _check_source_assignment(self, node: Any, sb: bytes) -> None:
        """Source fonksiyonundan dönen değeri tainted olarak işaretle."""
        for child in node.children:
            if child.type == "init_declarator":
                self._check_source_assignment(child, sb)
                return

        # init_declarator: name = value
        decl = node.child_by_field_name("declarator") if hasattr(node, "child_by_field_name") else None
        value = node.child_by_field_name("value") if hasattr(node, "child_by_field_name") else None

        if decl is None or value is None:
            return

        # Value bir call_expression mi?
        call_node = value if value.type == "call_expression" else None
        if call_node is None:
            for c in value.children if hasattr(value, "children") else []:
                if c.type == "call_expression":
                    call_node = c
                    break

        if call_node is None:
            return

        callee = call_node.child_by_field_name("function")
        if callee is None:
            return

        func_name = sb[callee.start_byte:callee.end_byte].decode("utf-8", errors="ignore")
        if func_name in SOURCE_FUNCTIONS:
            # Declarator identifier'ını bul
            var_name = self._extract_identifier(decl, sb)
            if var_name and var_name not in SECURITY_ALLOW_LIST:
                tid = self._next_taint_id()
                self._taint_set[var_name] = tid
                line_no = node.start_point[0] + 1
                # Ön-kayıt: source bulundu ama henüz sink belli değil
                self._paths.append(TaintPath(
                    source_func=func_name,
                    source_line=line_no,
                    sink_func="",
                    sink_line=0,
                    carrier_vars=[var_name],
                ))

    def _check_sink_call(self, node: Any, sb: bytes) -> None:
        """Sink fonksiyonuna tainted argüman geçişini kontrol eder."""
        callee = node.child_by_field_name("function")
        if callee is None:
            return

        func_name = sb[callee.start_byte:callee.end_byte].decode("utf-8", errors="ignore")
        if func_name not in SINK_FUNCTIONS:
            return

        # Argümanları kontrol et
        args_node = node.child_by_field_name("arguments")
        if args_node is None:
            return

        for arg in args_node.children:
            arg_name = self._extract_identifier(arg, sb)
            if arg_name and arg_name in self._taint_set:
                line_no = node.start_point[0] + 1
                cwe = _CWE_HEURISTICS.get(func_name, "Unknown")

                # Mevcut incomplete path'i tamamla veya yeni path oluştur
                completed = False
                for path in self._paths:
                    if not path.sink_func and arg_name in path.carrier_vars:
                        path.sink_func = func_name
                        path.sink_line = line_no
                        path.cwe_id = cwe
                        completed = True
                        break

                if not completed:
                    self._paths.append(TaintPath(
                        source_func="unknown_source",
                        source_line=0,
                        sink_func=func_name,
                        sink_line=line_no,
                        carrier_vars=[arg_name],
                        cwe_id=cwe,
                    ))

    def _propagate_taint(self, node: Any, sb: bytes) -> None:
        """Atama ile taint yayılımını izler: lhs = rhs → rhs tainted ise lhs de tainted."""
        lhs = node.child_by_field_name("left")
        rhs = node.child_by_field_name("right")
        if lhs is None or rhs is None:
            return

        rhs_name = self._extract_identifier(rhs, sb)
        if rhs_name and rhs_name in self._taint_set:
            lhs_name = self._extract_identifier(lhs, sb)
            if lhs_name and lhs_name not in SECURITY_ALLOW_LIST:
                tid = self._next_taint_id()
                self._taint_set[lhs_name] = tid
                # Carrier listesine ekle
                for path in self._paths:
                    if rhs_name in path.carrier_vars:
                        path.carrier_vars.append(lhs_name)

    @staticmethod
    def _extract_identifier(node: Any, sb: bytes) -> Optional[str]:
        """Bir düğümden identifier metnini çıkarır."""
        if node.type == "identifier":
            return sb[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        for child in node.children:
            if child.type == "identifier":
                return sb[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
        return None


# ============================================================================
# Ana Sınıf: OWASPSecNorm
# ============================================================================
class OWASPSecNorm:
    """
    OWASP ve güvenlik verileri için asimetrik normalizasyon motoru.

    Pipeline:
      1. Security Allow-List koruması (Source/Sink dokunulmaz)
      2. Taint akış analizi + TAINT_N dönüşümü
      3. CWE / Source-Sink JSON meta-veri çıktısı
    """

    def __init__(self, language: str = "cpp") -> None:
        if language not in _PARSERS:
            raise ValueError(f"Desteklenmeyen dil: {language}. Seçenekler: c, cpp")
        self._language = language
        self._parser = _PARSERS[language]
        self._taint_analyzer = TaintFlowAnalyzer()

    def process(self, source_code: str, cwe_hint: str = "Unknown") -> Dict[str, Any]:
        """
        Tam normalizasyon pipeline'ı.

        Returns:
            {
                "normalized_code": str,
                "security_context": dict,
                "applied_algorithm": str,
            }
        """
        if not source_code or not source_code.strip():
            return {
                "normalized_code": "",
                "security_context": {},
                "applied_algorithm": "owasp_secnorm",
            }

        try:
            source_bytes = source_code.encode("utf-8")
            tree = self._parser.parse(source_bytes)

            # ── Adım 2: Taint Analizi ──
            taint_map, taint_paths = self._taint_analyzer.analyze(
                tree.root_node, source_bytes, self._language
            )

            # ── Adım 1 + 2: AST Walk ile asimetrik normalizasyon ──
            replacements: List[Tuple[int, int, str]] = []
            non_security_vars: Set[str] = set()
            var_counter = [0]

            self._asymmetric_walk(
                tree.root_node, source_bytes, taint_map,
                non_security_vars, var_counter, replacements
            )

            # Replacement uygula
            replacements.sort(key=lambda x: x[0], reverse=True)
            buf = bytearray(source_bytes)
            for start, end, txt in replacements:
                buf[start:end] = txt.encode("utf-8")

            normalized = buf.decode("utf-8", errors="ignore")
            lines = [ln for ln in normalized.split("\n") if ln.strip()]
            normalized_code = "\n".join(lines)

            # ── Adım 3: Güvenlik meta-verisi ──
            detected_cwe = self._detect_cwes(taint_paths, cwe_hint)
            security_context = {
                "cwe_ids": detected_cwe,
                "taint_paths": [
                    {
                        "source": p.source_func,
                        "source_line": p.source_line,
                        "sink": p.sink_func,
                        "sink_line": p.sink_line,
                        "carriers": p.carrier_vars,
                        "cwe": p.cwe_id,
                    }
                    for p in taint_paths
                    if p.sink_func  # Sadece tamamlanmış yollar
                ],
                "sources_detected": list({
                    p.source_func for p in taint_paths if p.source_func
                }),
                "sinks_detected": list({
                    p.sink_func for p in taint_paths if p.sink_func
                }),
                "taint_variables": {v: f"TAINT_{tid}" for v, tid in self._taint_analyzer._taint_set.items()},
                "non_security_vars_normalized": len(non_security_vars),
            }

            return {
                "normalized_code": normalized_code,
                "security_context": security_context,
                "applied_algorithm": "owasp_secnorm",
            }

        except Exception as e:
            logger.error(f"[OWASPSecNorm] Pipeline hatası: {e}")
            return {
                "normalized_code": source_code,
                "security_context": {"error": str(e)},
                "applied_algorithm": "owasp_secnorm",
            }

    def _asymmetric_walk(
        self,
        node: Any,
        sb: bytes,
        taint_map: Dict[str, str],
        non_sec_vars: Set[str],
        var_counter: List[int],
        reps: List[Tuple[int, int, str]],
    ) -> None:
        """
        Asimetrik AST yürüyüşü:
        - Source/Sink fonksiyonlarına DOKUNMA
        - Taint değişkenlerini TAINT_N yap
        - Geri kalan kullanıcı değişkenlerini VAR_N yap
        """
        # Yorum temizliği
        if node.type == "comment":
            reps.append((node.start_byte, node.end_byte, ""))
            return

        if node.type == "ERROR":
            return

        # Ön İşlemci Direktifleri — Yapıyı Koru, Tanımlayıcıları Normalize Et
        # Tüm preprocessor direktifleri (#define, #include, #ifdef, #ifndef,
        # #if, #endif vb.) AST çıktısında KORUNUR, SİLİNMEZ.
        # İçlerindeki tanımlayıcılar aşağıdaki identifier bloğunda
        # SECURITY_ALLOW_LIST exact-match kontrolünden geçirilir.
        if node.type == "preproc_include":
            # #include direktiflerini tamamen koru, dosya adları normalize edilmez
            return

        if node.type in ("preproc_def", "preproc_function_def", "preproc_call"):
            # Direktifi koru, içindeki tanımlayıcıları normalize et
            # Rekürsif yürüyüş ile çocuk düğümlerdeki identifier'lar
            # aşağıdaki identifier bloğunda yakalanır
            for child in node.children:
                try:
                    self._asymmetric_walk(
                        child, sb, taint_map, non_sec_vars, var_counter, reps
                    )
                except Exception as e:
                    logger.debug(f"[OWASPSecNorm] Preproc child walk hatası: {e}")
            return

        if node.type in ("preproc_if", "preproc_ifdef", "preproc_elif", "preproc_else"):
            # Blok direktifleri koru, koşul tanımlayıcılarını normalize et
            # return YOK — rekürsif yürüyüş devam eder, gövde kodu da işlenir
            pass

        # Identifier normalizasyonu
        if node.type == "identifier":
            name = sb[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

            # Allow-List kontrolü — Source/Sink/Keyword dokunulmaz
            if name in SECURITY_ALLOW_LIST:
                pass  # DOKUNMA
            elif name in taint_map:
                # Taint değişkeni → TAINT_N
                reps.append((node.start_byte, node.end_byte, taint_map[name]))
            else:
                # Güvenlik dışı değişken → VAR_N
                if name not in non_sec_vars:
                    var_counter[0] += 1
                    non_sec_vars.add(name)
                # Her seferinde aynı mapping'i kullanmak için set-based
                var_idx = sorted(non_sec_vars).index(name) + 1
                reps.append((node.start_byte, node.end_byte, f"VAR_{var_idx}"))

        # Type identifier
        if node.type == "type_identifier":
            tname = sb[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            if tname not in SECURITY_ALLOW_LIST:
                if tname not in non_sec_vars:
                    var_counter[0] += 1
                    non_sec_vars.add(tname)
                type_idx = sorted(non_sec_vars).index(tname) + 1
                reps.append((node.start_byte, node.end_byte, f"TYPE_{type_idx}"))

        # Field identifier (struct alanları: obj.field, ptr->field)
        # SECURITY_ALLOW_LIST ile birebir eşleşme kontrolü
        if node.type == "field_identifier":
            fname = sb[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            if fname in SECURITY_ALLOW_LIST:
                pass  # Whitelist'te → DOKUNMA
            elif fname in taint_map:
                # Taint değişkeni → TAINT_N
                reps.append((node.start_byte, node.end_byte, taint_map[fname]))
            else:
                # Güvenlik dışı alan → VAR_N
                if fname not in non_sec_vars:
                    var_counter[0] += 1
                    non_sec_vars.add(fname)
                field_idx = sorted(non_sec_vars).index(fname) + 1
                reps.append((node.start_byte, node.end_byte, f"VAR_{field_idx}"))

        # Rekürsif yürüyüş
        for child in node.children:
            try:
                self._asymmetric_walk(
                    child, sb, taint_map, non_sec_vars, var_counter, reps
                )
            except Exception as e:
                logger.debug(f"[OWASPSecNorm] Child walk hatası: {e}")

    def _detect_cwes(
        self, taint_paths: List[TaintPath], cwe_hint: str
    ) -> List[str]:
        """Tespit edilen CWE'leri birleştirir."""
        cwes: Set[str] = set()
        if cwe_hint and cwe_hint != "Unknown":
            cwes.add(cwe_hint)
        for path in taint_paths:
            if path.cwe_id and path.cwe_id != "Unknown":
                cwes.add(path.cwe_id)
        return sorted(cwes)


# ============================================================================
# Public API
# ============================================================================
def normalize_owasp_code(
    source_code: str,
    language: str = "cpp",
    cwe_hint: str = "Unknown",
) -> Dict[str, Any]:
    """
    OWASP / güvenlik veri seti kodunu normalize eder.

    Args:
        source_code: Ham C/C++ kaynak kodu
        language: "c" veya "cpp"
        cwe_hint: Bilinen CWE kimliği (opsiyonel)

    Returns:
        Normalize edilmiş kod, güvenlik bağlamı ve meta-veri
    """
    normalizer = OWASPSecNorm(language=language)
    return normalizer.process(source_code, cwe_hint=cwe_hint)


if __name__ == "__main__":
    sample = '''
    #include <stdio.h>
    #include <string.h>
    #include <stdlib.h>

    void vulnerable_function(int argc, char *argv[]) {
        char buffer[64];
        char *user_input = argv[1];  // SOURCE: dış girdi

        char temp_holder[128];
        strcpy(temp_holder, user_input);  // Taint yayılımı

        // SINK: buffer overflow
        strcpy(buffer, temp_holder);

        printf("Result: %s\\n", buffer);  // SINK: format string
    }

    int main(int argc, char *argv[]) {
        if (argc < 2) {
            printf("Usage: %s <input>\\n", argv[0]);
            return EXIT_FAILURE;
        }
        vulnerable_function(argc, argv);
        return EXIT_SUCCESS;
    }
    '''
    result = normalize_owasp_code(sample, language="c", cwe_hint="CWE-120")
    print("=== Normalized Code ===")
    print(result["normalized_code"])
    print("\n=== Security Context ===")
    print(json.dumps(result["security_context"], indent=2, ensure_ascii=False))
