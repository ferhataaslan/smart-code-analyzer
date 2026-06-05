#!/usr/bin/env python3
"""
complexity_analyzer.py — AST-Based Big O Complexity Estimator for C/C++ Code

Analyzes C/C++ source code using tree-sitter AST to estimate time complexity:
- Counts nested loops (for, while, do-while) to determine polynomial degree (O(n), O(n^2), O(n^3)).
- Inspects loop condition/update steps to detect logarithmic scaling (O(log n), O(n log n)).
- Returns: "1", "log n", "n", "n log n", "n^2", "n^3", etc.
"""

import logging
from typing import Any, Dict, List, Optional, Set

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
    logger.warning("[ComplexityAnalyzer] Tree-sitter or languages not available.")


def get_loop_nodes(node: Any, loops_dict: Dict[Any, Optional[Any]], parent_loop: Optional[Any] = None) -> None:
    """
    Recursively travers AST to find loop statements and construct a parent-child relationship.
    """
    is_loop = node.type in ("for_statement", "while_statement", "do_statement")
    current_parent = parent_loop
    if is_loop:
        loops_dict[node] = parent_loop
        current_parent = node
    
    for child in node.children:
        get_loop_nodes(child, loops_dict, current_parent)


def has_logarithmic_update(loop_node: Any, source_bytes: bytes) -> bool:
    """
    Checks if a loop update or body contains division or bit shift assignments,
    indicating O(log n) scaling.
    Examples:
      - i /= 2
      - i >>= 1
      - i = i / 2
      - i = i >> 1
      - i *= 2
      - i <<= 1
    """
    log_operators = ("/=", ">>=", "*=", "<<=")
    log_keywords = ("/", ">>", "*", "<<")
    
    found_log = False
    
    def walk_check(node: Any) -> None:
        nonlocal found_log
        if found_log:
            return
        
        # Check assignment/update expressions within the loop
        if node.type in ("assignment_expression", "update_expression"):
            text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            # If any log operator is present
            for op in log_operators:
                if op in text:
                    found_log = True
                    return
            
            # Check if assignment uses division/shift/multiplication
            if node.type == "assignment_expression":
                right = node.child_by_field_name("right")
                if right:
                    right_text = source_bytes[right.start_byte:right.end_byte].decode("utf-8", errors="ignore")
                    for kw in log_keywords:
                        if kw in right_text:
                            found_log = True
                            return
                            
        for child in node.children:
            # Avoid descending into nested loops to check only current loop level updates
            if child.type not in ("for_statement", "while_statement", "do_statement"):
                walk_check(child)
                
    walk_check(loop_node)
    return found_log


def estimate_complexity(source_code: str, language: str = "cpp") -> str:
    """
    Estimates the Big O time complexity of a C/C++ code snippet using tree-sitter.
    
    Args:
        source_code: Raw C/C++ source code.
        language: "c" or "cpp"
        
    Returns:
        "1", "log n", "n", "n log n", "n^2", "n^3", etc.
    """
    if not _TS_AVAILABLE or not source_code or not source_code.strip():
        return "1"
        
    try:
        lang = CPP_LANGUAGE if language == "cpp" else C_LANGUAGE
        parser = Parser(lang)
        source_bytes = source_code.encode("utf-8")
        tree = parser.parse(source_bytes)
        
        # Find all loop nodes and construct nesting parent chains
        loops_dict: Dict[Any, Optional[Any]] = {}
        get_loop_nodes(tree.root_node, loops_dict)
        
        if not loops_dict:
            return "1"
            
        # Compute maximum nesting depth of loops
        loop_depths = {}
        for loop in loops_dict:
            depth = 1
            curr = loops_dict[loop]
            while curr is not None:
                depth += 1
                curr = loops_dict[curr]
            loop_depths[loop] = depth
            
        max_depth = max(loop_depths.values())
        
        # Check if any loop in the snippet has logarithmic steps
        has_log = any(has_logarithmic_update(loop, source_bytes) for loop in loops_dict)
        
        # Map loop features to Big O notation
        if max_depth == 1:
            return "log n" if has_log else "n"
        elif max_depth == 2:
            return "n log n" if has_log else "n^2"
        elif max_depth == 3:
            return "n^2 log n" if has_log else "n^3"
        else:
            return f"n^{max_depth}"
            
    except Exception as e:
        logger.warning(f"[ComplexityAnalyzer] Error estimating complexity: {e}")
        return "1"


if __name__ == "__main__":
    # Command line testing
    import sys
    
    samples = [
        ("Constant code", "void test() { int x = 1; }"),
        ("Single linear loop", "void test(int n) { for(int i=0; i<n; i++) { printf(\"%d\", i); } }"),
        ("Single logarithmic loop", "void test(int n) { for(int i=n; i>0; i/=2) { printf(\"%d\", i); } }"),
        ("Nested loops (Quadratic)", "void test(int n) { for(int i=0; i<n; i++) { for(int j=0; j<n; j++) { printf(\"%d\", i); } } }"),
        ("Linear + Logarithmic nested (n log n)", "void test(int n) { for(int i=0; i<n; i++) { for(int j=n; j>0; j>>=1) { printf(\"%d\", i); } } }")
    ]
    
    for desc, code in samples:
        print(f"{desc} -> Est Complexity: {estimate_complexity(code)}")
