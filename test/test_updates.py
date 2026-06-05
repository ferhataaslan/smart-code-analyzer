#!/usr/bin/env python3
"""4 guncellemeyi test eden script."""
import json, re

# Test 1: HF AST-tabanli normalizasyon
print("=" * 70)
print("  TEST 1: HF CrossNorm (AST-tabanli)")
print("=" * 70)
from src.normalizers.hf_crossnorm import normalize_hf_code

code1 = '''
#include <stdio.h>
#include "local_header.h"
#define BUFFER_SIZE 256
#ifndef MY_GUARD_H
#define MY_GUARD_H

void process_data(const char *user_input, int max_len) {
    char *buf = (char *)malloc(BUFFER_SIZE);
    if (buf == NULL) return;
    strncpy(buf, user_input, max_len);
    printf("Result: %s\\n", buf);
    free(buf);
}
#endif
'''
doc1 = "Processes user_input into buf with max_len limit"
r1 = normalize_hf_code(code1, doc1)
n1 = r1["normalized_code"]
print(n1)
print("-" * 70)

def ok(label, cond):
    print(f"  [{'OK' if cond else 'FAIL'}] {label}")
    return cond

all_ok = True
all_ok &= ok("stdio.h NOT renamed", "stdio" not in n1 or "#include <stdio.h>" in n1)
all_ok &= ok("#include <stdio.h> preserved", "<stdio.h>" in n1)
all_ok &= ok("#include local preserved", '"local_header.h"' in n1)
all_ok &= ok("BUFFER_SIZE preserved", "BUFFER_SIZE" in n1)
all_ok &= ok("#ifndef preserved", "#ifndef" in n1)
all_ok &= ok("#define preserved", "#define" in n1)
all_ok &= ok("#endif preserved", "#endif" in n1)
all_ok &= ok("malloc preserved", "malloc" in n1)
all_ok &= ok("strncpy preserved", "strncpy" in n1)
all_ok &= ok("printf preserved", "printf" in n1)
all_ok &= ok("free preserved", "free" in n1)
all_ok &= ok("NULL preserved", "NULL" in n1)
all_ok &= ok("FUNC_ present", "FUNC_" in n1)
all_ok &= ok("VAR_ present", "VAR_" in n1)
all_ok &= ok("user_input renamed", "user_input" not in n1)
all_ok &= ok("Docstring synced", "user_input" not in r1["normalized_docstring"])

# Test 2: CWE cikarma
print()
print("=" * 70)
print("  TEST 2: OWASP CWE Extraction")
print("=" * 70)
from src.database.collector import Collector
c = Collector()

# Dosya adindan
cwe1 = c._extract_cwe_from_context(
    "testcases/CWE114_Process_Control/CWE114_Process_Control__w32_01.c",
    "void test() {}"
)
all_ok &= ok(f"Filename CWE extraction: {cwe1}", cwe1 == "CWE-114")

# Yorum satirindan
cwe2 = c._extract_cwe_from_context(
    "some_file.c",
    "/* CWE: 120 Buffer Overflow */\nvoid bad() {}"
)
all_ok &= ok(f"Comment CWE extraction: {cwe2}", cwe2 == "CWE-120")

# CWE olmayan dosya
cwe3 = c._extract_cwe_from_context("utils.c", "int main() { return 0; }")
all_ok &= ok(f"No CWE returns None: {cwe3}", cwe3 is None)

# Test 3: Temiz kod etiketleme
print()
print("=" * 70)
print("  TEST 3: Clean code cwe_id = None (not 'Unknown')")
print("=" * 70)
from src.database.collector import auto_label_vulnerability
sec = auto_label_vulnerability("int main() { return 0; }", "c")
all_ok &= ok(f"Clean code is_vulnerable={sec['is_vulnerable']}", sec["is_vulnerable"] == False)

# Simule: temiz kod -> cwe_id logic
if sec.get("is_vulnerable") and sec.get("cwe_ids"):
    cwe_id = sec["cwe_ids"][0]
elif sec.get("is_vulnerable"):
    cwe_id = "Unknown"
else:
    cwe_id = None
all_ok &= ok(f"Clean code cwe_id is None: {cwe_id}", cwe_id is None)

# Test 4: cppcheck regex patterns
print()
print("=" * 70)
print("  TEST 4: CWE Regex Patterns")
print("=" * 70)
# cppcheck XML format
xml_line = '<error id="arrayIndexOutOfBounds" severity="error" msg="..." cwe="788">'
m1 = re.search(r'cwe="?(\d+)"?', xml_line)
all_ok &= ok(f"cppcheck XML cwe attr: CWE-{m1.group(1) if m1 else 'NONE'}", m1 and m1.group(1) == "788")

# flawfinder format
fw_line = "test.c:5:  [2] (buffer) strcpy:  (CWE-120)"
m2 = re.search(r'CWE-?(\d+)', fw_line)
all_ok &= ok(f"flawfinder CWE text: CWE-{m2.group(1) if m2 else 'NONE'}", m2 and m2.group(1) == "120")

# Juliet filename
fn = "CWE78_OS_Command_Injection__char_connect_socket_01.c"
m3 = re.search(r'CWE-?(\d+)', fn, re.IGNORECASE)
all_ok &= ok(f"Juliet filename: CWE-{m3.group(1) if m3 else 'NONE'}", m3 and m3.group(1) == "78")

print()
print("=" * 70)
if all_ok:
    print("  >>> ALL CHECKS PASSED <<<")
else:
    print("  !!! SOME CHECKS FAILED !!!")
print("=" * 70)
