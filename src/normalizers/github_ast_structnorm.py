#!/usr/bin/env python3
"""
github_ast_structnorm.py — GitHub Verileri İçin AST Tabanlı Yapısal Normalizasyon

Amaç: Yüksek varyanslı ve gürültülü (minified vb.) GitHub kodlarını
yapısal bir formata indirgemek.

Algoritma Adımları:
  1. tree-sitter ile full-fidelity AST oluşturma (C/C++)
  2. Capture-Avoiding Alpha-Conversion (kapsam-kaçınan değişken yeniden adlandırma)
  3. Allow-List ile standart kütüphane/anahtar kelime koruması
  4. AST → SBT (Structure-Based Traversal) dizisel serileştirme

Kısıtlama: C ve C++ odaklı. Makrolar ve pointer aritmetikleri semantiği
bozmadan ele alınır.
"""

import logging
import json
from typing import Optional, Dict, List, Set, Tuple, Any
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

_PARSERS = {
    "c": Parser(C_LANGUAGE),
    "cpp": Parser(CPP_LANGUAGE),
}

# ============================================================================
# Adım 3: Allow-List — C/C++ Standart Kütüphane + Anahtar Kelimeler
# Bu listedeki sembollere HİÇBİR normalizasyon uygulanmaz.
# ============================================================================
ALLOW_LIST: frozenset = frozenset({
    # ── C/C++ Anahtar Kelimeler ──
    "int", "char", "float", "double", "void", "struct", "union", "enum",
    "typedef", "size_t", "ssize_t", "bool", "const", "static", "extern",
    "volatile", "auto", "register", "signed", "unsigned", "long", "short",
    "inline", "restrict", "sizeof", "alignof", "typeof",
    "_Bool", "_Complex", "_Imaginary",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "ptrdiff_t", "intptr_t", "uintptr_t", "wchar_t",
    "class", "namespace", "template", "typename", "virtual", "override",
    "final", "public", "private", "protected", "friend", "operator",
    "new", "delete", "this", "nullptr", "true", "false", "throw",
    "try", "catch", "using", "noexcept", "constexpr", "decltype",
    "mutable", "explicit", "thread_local", "alignas",
    "if", "else", "while", "for", "do", "switch", "case", "break",
    "continue", "goto", "return", "default",
    # ── Bellek Yönetimi ──
    "malloc", "calloc", "realloc", "free",
    "memset", "memcpy", "memmove", "memcmp", "memchr", "bzero",
    # ── String İşlemleri ──
    "strcpy", "strncpy", "strcat", "strncat", "strlen", "strcmp", "strncmp",
    "sprintf", "snprintf", "vsprintf", "vsnprintf", "sscanf",
    "wcscpy", "wcsncpy", "wcslen", "wcscmp", "wcscat", "wmemcpy", "wmemset",
    "strstr", "strchr", "strrchr", "strtok",
    "strtol", "strtoul", "strtod", "strtoll", "atoi", "atof", "atol",
    # ── I/O ──
    "FILE", "fopen", "fclose", "fread", "fwrite", "fseek", "ftell",
    "fprintf", "fscanf", "fgets", "fputs", "printf", "puts",
    "getchar", "putchar", "fflush", "feof", "ferror", "perror", "rewind",
    "scanf", "getline", "setbuf", "setvbuf", "tmpfile", "tmpnam",
    "remove", "rename", "open", "close", "read", "write", "lseek",
    # ── Ağ (Socket) ──
    "socket", "bind", "listen", "accept", "connect", "send", "recv",
    "sendto", "recvfrom", "setsockopt", "getsockopt",
    "inet_addr", "inet_ntoa", "inet_pton", "inet_ntop",
    "htons", "htonl", "ntohs", "ntohl",
    "WSAStartup", "WSACleanup",
    # ── Sistem / Süreç ──
    "system", "exec", "execl", "execlp", "execle", "execv", "execvp",
    "execvpe", "fork", "exit", "_exit", "abort",
    "getenv", "setenv", "putenv", "popen", "pclose",
    "getpid", "getppid", "pipe", "dup", "dup2", "sleep", "usleep",
    "signal", "sigaction", "kill",
    # ── Windows API ──
    "HANDLE", "HMODULE", "DWORD", "WORD", "BYTE",
    "LoadLibrary", "LoadLibraryA", "LoadLibraryW",
    "GetProcAddress", "CreateProcess", "CreateFile",
    "VirtualAlloc", "VirtualProtect",
    "BOOL", "LPCSTR", "LPSTR", "LPVOID", "LPDWORD",
    # ── C++ STL ──
    "std", "cout", "cin", "cerr", "endl", "string", "vector", "map",
    "set", "list", "queue", "stack", "pair", "tuple", "array",
    "unique_ptr", "shared_ptr", "weak_ptr", "make_unique", "make_shared",
    "begin", "end", "size", "push_back", "emplace_back", "insert",
    "erase", "find", "sort", "swap", "move", "forward",
    # ── Sabitler ──
    "NULL", "EOF", "stdin", "stdout", "stderr", "errno",
    "TRUE", "FALSE", "SEEK_SET", "SEEK_CUR", "SEEK_END",
    "EXIT_SUCCESS", "EXIT_FAILURE",
    "assert", "main",
    # ── POSIX/Socket Sabitleri ──
    "AF_INET", "AF_INET6", "AF_UNIX", "SOCK_STREAM", "SOCK_DGRAM",
    "IPPROTO_TCP", "IPPROTO_UDP", "INADDR_ANY", "INVALID_SOCKET",
    "SOCKET_ERROR", "SOL_SOCKET", "SO_REUSEADDR",
    "sockaddr_in", "sockaddr", "in_addr",
    "sin_family", "sin_port", "sin_addr", "s_addr",
    "BUFFER_SIZE", "MAX_PATH", "PATH_MAX", "BUFSIZ",
    "O_RDONLY", "O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND",
    "SIGINT", "SIGTERM", "SIGKILL", "SIGHUP",
    "SOCKET",
})


# ============================================================================
# Adım 2: Kapsam-Kaçınan Alfa-Dönüşümü (Capture-Avoiding Alpha-Conversion)
# ============================================================================
@dataclass
class Scope:
    """Tek bir sözdizimsel kapsamı (scope) temsil eder."""
    bindings: Dict[str, str] = field(default_factory=dict)
    parent: Optional["Scope"] = None


class CaptureAvoidingSymbolTable:
    """
    Capture-Avoiding Alpha-Conversion için sembol tablosu.

    Kurallar:
    - Her yerel değişken kendi kapsamında VAR_N olarak yeniden adlandırılır.
    - Üst kapsamdaki bir değişkenin ismi alt kapsamda ezilmez (shadowing koruması).
    - Allow-List'teki sembollere dokunulmaz.
    - Fonksiyon isimleri FUNC_N, tipler TYPE_N olarak adlandırılır.
    """

    def __init__(self) -> None:
        self._root_scope = Scope()
        self._current_scope: Scope = self._root_scope
        self._var_counter: int = 0
        self._func_counter: int = 0
        self._type_counter: int = 0
        # Global haritalar: orijinal isim → normalize isim (scope-agnostic)
        self._func_map: Dict[str, str] = {}
        self._type_map: Dict[str, str] = {}

    def push_scope(self) -> None:
        """Yeni bir iç kapsam açar."""
        child = Scope(parent=self._current_scope)
        self._current_scope = child

    def pop_scope(self) -> None:
        """Mevcut kapsamı kapatır, üst kapsama döner."""
        if self._current_scope.parent is not None:
            self._current_scope = self._current_scope.parent

    def _lookup(self, name: str) -> Optional[str]:
        """Kapsamlar zincirinde yukarı doğru arama yapar."""
        scope = self._current_scope
        while scope is not None:
            if name in scope.bindings:
                return scope.bindings[name]
            scope = scope.parent
        return None

    def define_variable(self, original_name: str) -> str:
        """
        Mevcut kapsamda yeni bir değişken tanımlar.
        Eğer üst kapsamda aynı isim varsa, yeni bir VAR_N atar (capture-avoiding).
        """
        if original_name in ALLOW_LIST:
            return original_name

        # Mevcut kapsamda zaten tanımlıysa, mevcut bağlamayı döndür
        if original_name in self._current_scope.bindings:
            return self._current_scope.bindings[original_name]

        # Yeni bir benzersiz VAR_N üret
        self._var_counter += 1
        normalized = f"VAR_{self._var_counter}"
        self._current_scope.bindings[original_name] = normalized
        return normalized

    def resolve_variable(self, original_name: str) -> Optional[str]:
        """Bir değişkeni kapsam zincirine bakarak çözer."""
        if original_name in ALLOW_LIST:
            return original_name
        return self._lookup(original_name)

    def resolve_or_define_variable(self, original_name: str) -> str:
        """Çözer; bulamazsa mevcut kapsamda tanımlar."""
        if original_name in ALLOW_LIST:
            return original_name
        resolved = self._lookup(original_name)
        if resolved is not None:
            return resolved
        return self.define_variable(original_name)

    def resolve_function(self, original_name: str) -> str:
        """Fonksiyon isimlerini FUNC_N formatına dönüştürür."""
        if original_name in ALLOW_LIST:
            return original_name
        if original_name not in self._func_map:
            self._func_counter += 1
            self._func_map[original_name] = f"FUNC_{self._func_counter}"
        return self._func_map[original_name]

    def resolve_type(self, original_name: str) -> str:
        """Kullanıcı tanımlı tipleri TYPE_N formatına dönüştürür."""
        if original_name in ALLOW_LIST:
            return original_name
        if original_name not in self._type_map:
            self._type_counter += 1
            self._type_map[original_name] = f"TYPE_{self._type_counter}"
        return self._type_map[original_name]


# ============================================================================
# Ana Sınıf: GitHubASTStructNorm
# ============================================================================
class GitHubASTStructNorm:
    """
    GitHub kaynak kodları için AST tabanlı yapısal normalizasyon motoru.

    Pipeline:
      1. tree-sitter ile full-fidelity AST oluştur
      2. AST üzerinde yürüyerek capture-avoiding alpha-conversion uygula
      3. Allow-List koruması
      4. AST-SBT serileştirme
    """

    def __init__(self, language: str = "cpp") -> None:
        if language not in _PARSERS:
            raise ValueError(f"Desteklenmeyen dil: {language}. Seçenekler: c, cpp")
        self._language: str = language
        self._parser: Parser = _PARSERS[language]
        self._symbols: CaptureAvoidingSymbolTable = CaptureAvoidingSymbolTable()

    def _reset(self) -> None:
        """Her yeni kod parçası için sembol tablosunu sıfırlar."""
        self._symbols = CaptureAvoidingSymbolTable()

    # ── Adım 1: AST Oluşturma ────────────────────────────────────────────
    def _parse_to_ast(self, source_code: str) -> Any:
        """Kaynak kodu tree-sitter ile parse eder ve AST döner."""
        try:
            source_bytes = source_code.encode("utf-8")
            tree = self._parser.parse(source_bytes)
            return tree
        except Exception as e:
            logger.error(f"AST oluşturma hatası: {e}")
            raise

    # ── Adım 2 + 3: Alpha-Conversion Walk ────────────────────────────────
    def _collect_identifiers(
        self,
        node: Any,
        source_bytes: bytes,
        local_vars: Set[str],
        func_names: Set[str],
        custom_types: Set[str],
        replacements: List[Tuple[int, int, str]],
    ) -> None:
        """
        AST üzerinde DFS yürüyüşü yaparak:
        - Yorumları temizler
        - Makroları semantiği koruyarak ele alır
        - Fonksiyon, değişken ve tip isimlerini toplar
        - Kapsam giriş/çıkışlarını izler
        """
        # Yorum temizliği
        if node.type == "comment":
            replacements.append((node.start_byte, node.end_byte, ""))
            return

        # ERROR düğümlerini tolere et (GitHub wild code)
        if node.type == "ERROR":
            logger.debug(
                f"[GitHubAST] ERROR düğümü tolere edildi "
                f"(byte {node.start_byte}-{node.end_byte})"
            )
            return

        # ── Ön İşlemci Direktifleri — Yapıyı Koru, Tanımlayıcıları Normalize Et ──
        # Tüm preprocessor direktifleri (#define, #include, #ifdef, #ifndef,
        # #if, #endif vb.) AST çıktısında KORUNUR, SİLİNMEZ.
        # İçlerindeki tanımlayıcılar alpha-conversion aşamasında
        # ALLOW_LIST exact-match kontrolünden geçirilir.
        if node.type == "preproc_include":
            # #include direktiflerini tamamen koru, dosya adları normalize edilmez
            return

        if node.type in ("preproc_def", "preproc_function_def", "preproc_call"):
            # Direktifi koru, içindeki tanımlayıcıları alpha-conversion için topla
            for c in node.children:
                if c.type == "identifier":
                    name = source_bytes[c.start_byte:c.end_byte].decode("utf-8")
                    if name not in ALLOW_LIST:
                        local_vars.add(name)
                        self._symbols.define_variable(name)
            return

        if node.type in ("preproc_if", "preproc_ifdef", "preproc_elif", "preproc_else"):
            # Blok direktifleri koru, koşul tanımlayıcılarını topla
            for c in node.children:
                if c.type == "identifier":
                    name = source_bytes[c.start_byte:c.end_byte].decode("utf-8")
                    if name not in ALLOW_LIST:
                        local_vars.add(name)
                        self._symbols.define_variable(name)
            # return YOK — rekürsif yürüyüş devam eder, gövde kodu da işlenir

        # ── Kapsam İzleme ──
        if node.type == "compound_statement":
            self._symbols.push_scope()

        # ── Fonksiyon Tanımları ──
        if node.type == "function_declarator":
            try:
                decl = node.child_by_field_name("declarator")
                if decl and decl.type == "identifier":
                    name = source_bytes[decl.start_byte:decl.end_byte].decode("utf-8")
                    if name not in ALLOW_LIST:
                        func_names.add(name)
            except Exception as e:
                logger.debug(f"[GitHubAST] function_declarator hata: {e}")

        # ── Fonksiyon Çağrıları ──
        if node.type == "call_expression":
            try:
                callee = node.child_by_field_name("function")
                if callee and callee.type == "identifier":
                    name = source_bytes[callee.start_byte:callee.end_byte].decode("utf-8")
                    if name not in ALLOW_LIST:
                        func_names.add(name)
            except Exception as e:
                logger.debug(f"[GitHubAST] call_expression hata: {e}")

        # ── Özel Tip İsimleri ──
        if node.type == "type_identifier":
            tname = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
            if tname not in ALLOW_LIST:
                custom_types.add(tname)

        # ── Yerel Değişken Tanımları (pointer declarator dahil) ──
        if node.type in ("parameter_declaration", "declaration"):
            self._extract_declared_vars(node, source_bytes, local_vars)

        # Rekürsif yürüyüş
        for child in node.children:
            try:
                self._collect_identifiers(
                    child, source_bytes, local_vars,
                    func_names, custom_types, replacements
                )
            except Exception as e:
                logger.debug(f"[GitHubAST] Child walk hata (tolere): {e}")

        # Kapsam çıkışı
        if node.type == "compound_statement":
            self._symbols.pop_scope()

    def _extract_declared_vars(
        self, node: Any, source_bytes: bytes, local_vars: Set[str]
    ) -> None:
        """Bildirim düğümlerinden değişken isimlerini çıkarır."""
        def _find_ids(n: Any) -> None:
            if n.type == "identifier":
                name = source_bytes[n.start_byte:n.end_byte].decode("utf-8")
                if name not in ALLOW_LIST:
                    local_vars.add(name)
                    self._symbols.define_variable(name)
            else:
                for c in n.children:
                    if n.type == "init_declarator" and c == n.child_by_field_name("value"):
                        continue
                    _find_ids(c)

        for child in node.children:
            if child.type in (
                "init_declarator", "pointer_declarator",
                "array_declarator", "function_declarator",
                "reference_declarator", "identifier",
            ):
                _find_ids(child)

    def _apply_alpha_conversion(
        self,
        tree: Any,
        source_bytes: bytes,
        local_vars: Set[str],
        func_names: Set[str],
        custom_types: Set[str],
        replacements: List[Tuple[int, int, str]],
    ) -> None:
        """Query-based capture ile alpha-conversion uygular."""
        try:
            import tree_sitter as ts
            lang = CPP_LANGUAGE if self._language == "cpp" else C_LANGUAGE
            query_str = """
                (identifier) @id
                (type_identifier) @type_id
                (field_identifier) @field_id
                (string_literal) @str_lit
            """
            try:
                query = ts.Query(lang, query_str)
            except (TypeError, AttributeError):
                query = lang.query(query_str)

            cursor = ts.QueryCursor(query)
            matches = cursor.matches(tree.root_node)

            seen: Set[int] = set()
            flat: List[Tuple[Any, str]] = []
            for _, caps in matches:
                for tag, nodes in caps.items():
                    if isinstance(nodes, list):
                        for nd in nodes:
                            flat.append((nd, tag))
                    else:
                        flat.append((nodes, tag))

            for node, tag in flat:
                if node.id in seen:
                    continue
                seen.add(node.id)

                orig = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
                new_name = orig

                if tag == "id":
                    if orig in ALLOW_LIST:
                        continue
                    if orig in func_names:
                        new_name = self._symbols.resolve_function(orig)
                    else:
                        # Yerel değişkenler, makro isimleri (#define, #ifdef)
                        # ve diğer tüm tanımlayıcılar
                        new_name = self._symbols.resolve_or_define_variable(orig)
                elif tag == "type_id":
                    if orig in ALLOW_LIST:
                        continue
                    if orig in custom_types:
                        new_name = self._symbols.resolve_type(orig)
                elif tag == "field_id":
                    # field_identifier: struct alanları (obj.field, ptr->field)
                    # ALLOW_LIST ile birebir eşleşme kontrolü
                    if orig in ALLOW_LIST:
                        continue  # Whitelist'te → orijinal bırak
                    new_name = self._symbols.resolve_or_define_variable(orig)
                elif tag == "str_lit":
                    continue  # String literalleri koru

                if new_name != orig:
                    replacements.append((node.start_byte, node.end_byte, new_name))

        except Exception as e:
            logger.warning(f"[GitHubAST] Alpha-conversion query hatası: {e}")

    # ── Adım 4: AST-SBT Serileştirme ─────────────────────────────────────
    def _serialize_ast_sbt(self, node: Any, source_bytes: bytes) -> str:
        """
        AST'yi Structure-Based Traversal (SBT) formatında serileştirir.

        SBT formatı:
          ( node_type  child1_sbt  child2_sbt  ... ) node_type
        Yaprak düğümlerde:
          ( node_type  token_value ) node_type
        """
        node_label = node.type

        if node.child_count == 0:
            # Yaprak düğüm: token değerini ekle
            token_val = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            return f"( {node_label} {token_val} ) {node_label}"

        children_sbt: List[str] = []
        for child in node.children:
            children_sbt.append(self._serialize_ast_sbt(child, source_bytes))

        inner = " ".join(children_sbt)
        return f"( {node_label} {inner} ) {node_label}"

    # ── Ana Pipeline ─────────────────────────────────────────────────────
    def process(self, source_code: str) -> Dict[str, Any]:
        """
        Tam normalizasyon pipeline'ı çalıştırır.

        Returns:
            {
                "normalized_code": str,     # Alpha-converted temiz kod
                "ast_sbt_sequence": str,    # SBT serileştirilmiş AST
                "ast_metadata": dict,       # İstatistik meta-verisi
                "applied_algorithm": str,
            }
        """
        self._reset()

        if not source_code or not source_code.strip():
            return {
                "normalized_code": "",
                "ast_sbt_sequence": "",
                "ast_metadata": {},
                "applied_algorithm": "github_ast_structnorm",
            }

        try:
            # Adım 1: AST oluştur
            tree = self._parse_to_ast(source_code)
            source_bytes = source_code.encode("utf-8")

            local_vars: Set[str] = set()
            func_names: Set[str] = set()
            custom_types: Set[str] = set()
            replacements: List[Tuple[int, int, str]] = []

            # Adım 2-3: Walk + Alpha-Conversion
            self._collect_identifiers(
                tree.root_node, source_bytes,
                local_vars, func_names, custom_types, replacements
            )
            self._apply_alpha_conversion(
                tree, source_bytes,
                local_vars, func_names, custom_types, replacements
            )

            # Replacement'ları uygula (sondan başa — offset kayması önleme)
            replacements.sort(key=lambda x: x[0], reverse=True)
            buf = bytearray(source_bytes)
            for start, end, txt in replacements:
                buf[start:end] = txt.encode("utf-8")

            normalized = buf.decode("utf-8", errors="ignore")

            # Boş satır temizliği
            lines = [ln for ln in normalized.split("\n") if ln.strip()]
            normalized_code = "\n".join(lines)

            # Adım 4: Normalize edilmiş kodu yeniden parse edip SBT serileştir
            norm_bytes = normalized_code.encode("utf-8")
            norm_tree = self._parser.parse(norm_bytes)
            sbt_sequence = self._serialize_ast_sbt(norm_tree.root_node, norm_bytes)

            # Meta-veri
            ast_metadata = {
                "language": self._language,
                "total_nodes": self._count_nodes(norm_tree.root_node),
                "tree_depth": self._tree_depth(norm_tree.root_node),
                "variables_renamed": len(local_vars),
                "functions_renamed": len(func_names),
                "types_renamed": len(custom_types),
                "has_errors": norm_tree.root_node.has_error,
            }

            return {
                "normalized_code": normalized_code,
                "ast_sbt_sequence": sbt_sequence,
                "ast_metadata": ast_metadata,
                "applied_algorithm": "github_ast_structnorm",
            }

        except Exception as e:
            logger.error(f"[GitHubAST] Pipeline hatası: {e}")
            return {
                "normalized_code": source_code,
                "ast_sbt_sequence": "",
                "ast_metadata": {"error": str(e)},
                "applied_algorithm": "github_ast_structnorm",
            }

    # ── Yardımcı Metrikler ───────────────────────────────────────────────
    @staticmethod
    def _count_nodes(node: Any) -> int:
        count = 1
        for child in node.children:
            count += GitHubASTStructNorm._count_nodes(child)
        return count

    @staticmethod
    def _tree_depth(node: Any) -> int:
        if node.child_count == 0:
            return 1
        return 1 + max(GitHubASTStructNorm._tree_depth(c) for c in node.children)


# ============================================================================
# Public API
# ============================================================================
def normalize_github_code(
    source_code: str, language: str = "cpp"
) -> Dict[str, Any]:
    """
    GitHub ham kodunu normalize eder.

    Args:
        source_code: Ham C/C++ kaynak kodu
        language: "c" veya "cpp"

    Returns:
        Normalize edilmiş kod, SBT dizisi ve meta-veri içeren sözlük
    """
    normalizer = GitHubASTStructNorm(language=language)
    return normalizer.process(source_code)


if __name__ == "__main__":
    sample = '''
    #include <stdio.h>
    #include <stdlib.h>

    int compute_sum(int *data, int count) {
        int total = 0;
        for (int idx = 0; idx < count; idx++) {
            total += data[idx];
        }
        return total;
    }

    int main() {
        int *buffer = (int *)malloc(10 * sizeof(int));
        if (buffer == NULL) {
            printf("Allocation failed\\n");
            return EXIT_FAILURE;
        }
        int result = compute_sum(buffer, 10);
        printf("Sum: %d\\n", result);
        free(buffer);
        return EXIT_SUCCESS;
    }
    '''
    result = normalize_github_code(sample, language="c")
    print("=== Normalized Code ===")
    print(result["normalized_code"])
    print("\n=== AST Metadata ===")
    print(json.dumps(result["ast_metadata"], indent=2))
    print(f"\n=== SBT Sequence (ilk 300 karakter) ===")
    print(result["ast_sbt_sequence"][:300])
