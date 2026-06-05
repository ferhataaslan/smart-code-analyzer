#!/usr/bin/env python3
"""
dfg_extractor.py — Veri Akış Grafiği (Data Flow Graph - DFG) Çıkarıcı

Modelin "hangi değerin nereden geldiğini" öğrenebilmesi için kodun
veri akış şemasını Tree-Sitter AST üzerinden çıkarır.

Çıktı Formatı:
  [
    {"from": "getenv", "to": "user_input", "type": "assignment"},
    {"from": "user_input", "to": "strcpy", "type": "call_arg"},
    ...
  ]

Her kayıt:
  - from: Veri kaynağı (fonksiyon çağrısı veya değişken)
  - to: Veri hedefi (değişken veya fonksiyon)
  - type: İlişki tipi (assignment, call_arg, propagation)
"""

import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

try:
    import tree_sitter as ts
    import tree_sitter_c as tsc
    import tree_sitter_cpp as tscpp
    from tree_sitter import Language, Parser

    C_LANGUAGE = Language(tsc.language())
    CPP_LANGUAGE = Language(tscpp.language())
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False
    logger.warning("[DFGExtractor] Tree-sitter yüklenemedi.")


def _get_node_text(node: Any, source_bytes: bytes) -> str:
    """AST düğümünden metin çıkarır."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _find_identifier(node: Any, source_bytes: bytes) -> Optional[str]:
    """Düğüm veya çocuklarından ilk identifier metnini bulur."""
    if node.type == "identifier":
        return _get_node_text(node, source_bytes)
    for child in node.children:
        if child.type == "identifier":
            return _get_node_text(child, source_bytes)
    return None


def _find_function_name(call_node: Any, source_bytes: bytes) -> Optional[str]:
    """call_expression düğümünden fonksiyon adını çıkarır."""
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None
    if func_node.type == "identifier":
        return _get_node_text(func_node, source_bytes)
    # Qualified call: obj.method veya ns::func
    if func_node.type in ("field_expression", "qualified_identifier"):
        return _get_node_text(func_node, source_bytes)
    return None


def _extract_call_arguments(call_node: Any, source_bytes: bytes) -> List[str]:
    """call_expression düğümünden argüman isimlerini çıkarır."""
    args_node = call_node.child_by_field_name("arguments")
    if args_node is None:
        return []
    arg_names = []
    for child in args_node.children:
        if child.type == "identifier":
            arg_names.append(_get_node_text(child, source_bytes))
        elif child.type in ("string_literal", "number_literal", "char_literal"):
            continue  # Literal'ler atlanır
        else:
            # İç içe ifadelerden identifier çıkar
            ident = _find_identifier(child, source_bytes)
            if ident:
                arg_names.append(ident)
    return arg_names


class DFGExtractor:
    """
    Tree-Sitter AST üzerinde gezinerek veri akış grafiği çıkarır.

    İzlenen düğüm tipleri:
      - declaration / init_declarator: değişken = fonksiyon_çağrısı veya değişken
      - assignment_expression: lhs = rhs
      - call_expression: fonksiyon(argümanlar)
    """

    def __init__(self, language: str = "cpp") -> None:
        self._language = language
        self._edges: List[Dict[str, str]] = []

    def extract(self, source_code: str) -> List[Dict[str, str]]:
        """
        Kaynak kodun DFG'sini çıkarır.

        Args:
            source_code: Ham C/C++ kaynak kodu

        Returns:
            Edge listesi: [{"from": ..., "to": ..., "type": ...}, ...]
        """
        if not _TS_AVAILABLE or not source_code or not source_code.strip():
            return []

        self._edges = []

        try:
            lang = CPP_LANGUAGE if self._language == "cpp" else C_LANGUAGE
            parser = Parser(lang)
            source_bytes = source_code.encode("utf-8")
            tree = parser.parse(source_bytes)

            self._walk(tree.root_node, source_bytes)
        except Exception as e:
            logger.warning(f"[DFGExtractor] Çıkarım hatası: {e}")

        return self._edges

    def _walk(self, node: Any, sb: bytes) -> None:
        """AST üzerinde rekürsif yürüyüş."""
        try:
            # ── Bildirim + İlklendirme: int x = func() veya int x = y ──
            if node.type == "init_declarator":
                self._handle_init_declarator(node, sb)

            # ── Atama: x = func() veya x = y ──
            elif node.type == "assignment_expression":
                self._handle_assignment(node, sb)

            # ── Fonksiyon Çağrısı: func(arg1, arg2) ──
            elif node.type == "call_expression":
                self._handle_call(node, sb)

            # Rekürsif yürüyüş
            for child in node.children:
                self._walk(child, sb)

        except Exception as e:
            logger.debug(f"[DFGExtractor] Walk hatası: {e}")

    def _handle_init_declarator(self, node: Any, sb: bytes) -> None:
        """init_declarator: 'type var = value' yapısını işler."""
        decl = node.child_by_field_name("declarator")
        value = node.child_by_field_name("value")

        if decl is None or value is None:
            return

        var_name = _find_identifier(decl, sb)
        if not var_name:
            return

        if value.type == "call_expression":
            func_name = _find_function_name(value, sb)
            if func_name:
                self._edges.append({
                    "from": func_name,
                    "to": var_name,
                    "type": "assignment",
                })
        elif value.type == "identifier":
            src_name = _get_node_text(value, sb)
            self._edges.append({
                "from": src_name,
                "to": var_name,
                "type": "assignment",
            })

    def _handle_assignment(self, node: Any, sb: bytes) -> None:
        """assignment_expression: 'lhs = rhs' yapısını işler."""
        lhs = node.child_by_field_name("left")
        rhs = node.child_by_field_name("right")

        if lhs is None or rhs is None:
            return

        target = _find_identifier(lhs, sb)
        if not target:
            return

        if rhs.type == "call_expression":
            func_name = _find_function_name(rhs, sb)
            if func_name:
                self._edges.append({
                    "from": func_name,
                    "to": target,
                    "type": "assignment",
                })
        elif rhs.type == "identifier":
            src_name = _get_node_text(rhs, sb)
            self._edges.append({
                "from": src_name,
                "to": target,
                "type": "propagation",
            })

    def _handle_call(self, node: Any, sb: bytes) -> None:
        """call_expression: 'func(arg1, arg2)' yapısını işler."""
        func_name = _find_function_name(node, sb)
        if not func_name:
            return

        arg_names = _extract_call_arguments(node, sb)
        for arg in arg_names:
            self._edges.append({
                "from": arg,
                "to": func_name,
                "type": "call_arg",
            })


# ============================================================================
# Public API
# ============================================================================
def extract_dfg(source_code: str, language: str = "cpp") -> List[Dict[str, str]]:
    """
    Kaynak koddan DFG çıkarır.

    Args:
        source_code: Ham C/C++ kaynak kodu
        language: "c" veya "cpp"

    Returns:
        Edge listesi: [{"from": ..., "to": ..., "type": ...}, ...]
    """
    extractor = DFGExtractor(language=language)
    return extractor.extract(source_code)


# ============================================================================
# CLI Demo
# ============================================================================
if __name__ == "__main__":
    import json

    sample = '''
    #include <stdio.h>
    #include <string.h>
    #include <stdlib.h>

    void process(int argc, char *argv[]) {
        char *user_input = getenv("DATA");
        char buffer[128];
        strcpy(buffer, user_input);
        char *copy = buffer;
        printf("Result: %s\\n", copy);
    }
    '''
    edges = extract_dfg(sample, language="c")
    print("=== Data Flow Graph ===")
    print(json.dumps(edges, indent=2))
    print(f"\nToplam {len(edges)} kenar (edge) bulundu.")
