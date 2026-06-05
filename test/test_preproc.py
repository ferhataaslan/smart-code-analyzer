#!/usr/bin/env python3
"""Preprocessor koruma mantigi dogrulama testi."""

from src.normalizers.github_ast_structnorm import normalize_github_code
from src.normalizers.owasp_secnorm import normalize_owasp_code
from src.normalizers.hf_crossnorm import normalize_hf_code

CODE = '''
#ifndef HDR_ATOMIC_H__
#define HDR_ATOMIC_H__

#include <stdio.h>
#include "local_header.h"

#define BUFFER_SIZE 256
#define MY_CUSTOM_MACRO 42
#define MAX(a,b) ((a)>(b)?(a):(b))

#ifdef _WIN32
void win_only_func() { printf("win"); }
#else
void unix_func() { printf("unix"); }
#endif

void process(char *data) {
    char buf[BUFFER_SIZE];
    strcpy(buf, data);
}
#endif
'''

def check(label, cond):
    icon = "OK" if cond else "FAIL"
    print(f"  [{icon}] {label}")
    return cond

print("=" * 70)
print("  TEST: github_ast_structnorm.py Preprocessor Koruma")
print("=" * 70)
r1 = normalize_github_code(CODE, language="c")
n1 = r1["normalized_code"]
print(n1)
print("-" * 70)

all_ok = True
all_ok &= check("#ifndef PRESERVED", "#ifndef" in n1)
all_ok &= check("#define PRESERVED", "#define" in n1)
all_ok &= check("#include <stdio.h> PRESERVED", "<stdio.h>" in n1)
all_ok &= check("#include local PRESERVED", "local_header" in n1)
all_ok &= check("#ifdef PRESERVED", "#ifdef" in n1)
all_ok &= check("#else PRESERVED", "#else" in n1)
all_ok &= check("#endif PRESERVED", "#endif" in n1)
all_ok &= check("BUFFER_SIZE kept (ALLOW_LIST)", "BUFFER_SIZE" in n1)
all_ok &= check("HDR_ATOMIC_H__ normalized (NOT in output)", "HDR_ATOMIC_H__" not in n1)
all_ok &= check("MY_CUSTOM_MACRO normalized", "MY_CUSTOM_MACRO" not in n1)
all_ok &= check("VAR_ present (macro renamed)", "VAR_" in n1)
all_ok &= check("strcpy preserved", "strcpy" in n1)
all_ok &= check("printf preserved", "printf" in n1)
all_ok &= check("Same VAR_N for same macro", n1.count("VAR_1") >= 2 or n1.count("VAR_2") >= 2)

print()
print("=" * 70)
print("  TEST: owasp_secnorm.py Preprocessor Koruma")
print("=" * 70)
r2 = normalize_owasp_code(CODE, language="c", cwe_hint="CWE-120")
n2 = r2["normalized_code"]
print(n2)
print("-" * 70)
all_ok &= check("#ifndef PRESERVED", "#ifndef" in n2)
all_ok &= check("#define PRESERVED", "#define" in n2)
all_ok &= check("#include <stdio.h> PRESERVED", "<stdio.h>" in n2)
all_ok &= check("strcpy preserved (SINK)", "strcpy" in n2)

print()
print("=" * 70)
print("  TEST: hf_crossnorm.py Preprocessor Koruma")
print("=" * 70)
r3 = normalize_hf_code(CODE, docstring="Processes data with buffer copy")
n3 = r3["normalized_code"]
print(n3)
print("-" * 70)
all_ok &= check("#ifndef PRESERVED", "#ifndef" in n3)
all_ok &= check("#define PRESERVED", "#define" in n3)
all_ok &= check("#include PRESERVED", "#include" in n3)
all_ok &= check("BUFFER_SIZE kept (CC_KEYWORDS)", "BUFFER_SIZE" in n3)
all_ok &= check("HDR_ATOMIC_H__ normalized", "HDR_ATOMIC_H__" not in n3)

print()
print("=" * 70)
if all_ok:
    print("  >>> ALL CHECKS PASSED <<<")
else:
    print("  !!! SOME CHECKS FAILED !!!")
print("=" * 70)
