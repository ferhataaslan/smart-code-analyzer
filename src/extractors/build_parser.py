import os
import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language

def build():
    print("[*] Building Tree-Sitter Language libraries if required...")
    
    # In tree-sitter >= 0.22, the pip packages (tree-sitter-c, tree-sitter-cpp)
    # come with pre-built bindings, so manual Language.build_library is usually not needed 
    # if you just do Language(tree_sitter_cpp.language()).
    # However, to ensure maximum compatibility across environments (Monster/Casper),
    # we simulate the build process or do a sanity check.
    
    try:
        lang_cpp = Language(tree_sitter_cpp.language())
        print(f"[+] Successfully loaded tree-sitter-cpp ABI version {lang_cpp.abi_version}")
    except Exception as e:
        print(f"[-] Failed to load tree-sitter-cpp: {e}")
        print("[-] Please ensure build-essential is installed.")
        exit(1)

    try:
        lang_c = Language(tree_sitter_c.language())
        print(f"[+] Successfully loaded tree-sitter-c ABI version {lang_c.abi_version}")
    except Exception as e:
        print(f"[-] Failed to load tree-sitter-c: {e}")
        print("[-] Please ensure build-essential is installed.")
        exit(1)

    print("[+] All Tree-Sitter parsers verified successfully.")

if __name__ == "__main__":
    build()

