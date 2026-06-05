#!/usr/bin/env python3
"""
Bölüm 2: Otomatize Edilmiş Canlı Test ve Doğrulama Script'i

3 platformdan (GitHub, HuggingFace, OWASP) gerçek veri çekerek:
1. Normalizasyon motorlarının doğruluğunu
2. Dedup mekanizmasının çalışmasını
3. Auto-labeling entegrasyonunu
4. Whitelist exact-match mantığını
test eder ve detaylı rapor üretir.
"""

import json
import sys
import os
import time
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("LIVE_TEST")

# Modülleri import et
from src.normalizers.processor import process_code, process_code_full
from src.normalizers.github_ast_structnorm import normalize_github_code, ALLOW_LIST
from src.normalizers.hf_crossnorm import normalize_hf_code, CC_KEYWORDS
from src.normalizers.owasp_secnorm import normalize_owasp_code, SECURITY_ALLOW_LIST
from src.database.db_integration import HybridDBPipeline
from src.database.collector import auto_label_vulnerability
from src.database.database import init_db, get_db_connection

# ============================================================================
# Test Verileri — Gerçek-dünya C/C++ kod örnekleri
# ============================================================================

# 10 GitHub-tarzı gerçek dünya örnekleri
GITHUB_SAMPLES = [
    {
        "name": "Redis dictAdd (hash table)",
        "code": '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct dictEntry {
    void *key;
    void *val;
    struct dictEntry *next;
} dictEntry;

int dictAdd(dictEntry **table, int size, const char *key, void *val) {
    unsigned long hash = 5381;
    const char *ptr = key;
    while (*ptr) { hash = ((hash << 5) + hash) + *ptr++; }
    int idx = hash % size;
    dictEntry *entry = (dictEntry *)malloc(sizeof(dictEntry));
    if (entry == NULL) return -1;
    entry->key = strdup(key);
    entry->val = val;
    entry->next = table[idx];
    table[idx] = entry;
    return 0;
}
''',
        "lang": "c"
    },
    {
        "name": "Linux kernel list_add",
        "code": '''
#include <stddef.h>

struct list_head {
    struct list_head *next, *prev;
};

static inline void __list_add(struct list_head *new_entry,
    struct list_head *prev, struct list_head *next) {
    next->prev = new_entry;
    new_entry->next = next;
    new_entry->prev = prev;
    prev->next = new_entry;
}

static inline void list_add(struct list_head *new_entry, struct list_head *head) {
    __list_add(new_entry, head, head->next);
}
''',
        "lang": "c"
    },
    {
        "name": "OpenSSL EVP digest",
        "code": '''
#include <stdio.h>
#include <string.h>
#include <openssl/evp.h>

int compute_sha256(const unsigned char *data, size_t len, unsigned char *out) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (ctx == NULL) return -1;
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1) goto err;
    if (EVP_DigestUpdate(ctx, data, len) != 1) goto err;
    unsigned int md_len;
    if (EVP_DigestFinal_ex(ctx, out, &md_len) != 1) goto err;
    EVP_MD_CTX_free(ctx);
    return (int)md_len;
err:
    EVP_MD_CTX_free(ctx);
    return -1;
}
''',
        "lang": "c"
    },
    {
        "name": "Socket server bind",
        "code": '''
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>

int start_server(int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sockfd < 0) { perror("socket"); return -1; }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); return -1;
    }
    listen(sockfd, 128);
    return sockfd;
}
''',
        "lang": "c"
    },
    {
        "name": "C++ vector sort",
        "code": '''
#include <vector>
#include <algorithm>
#include <iostream>

void process_data(std::vector<int>& data, int threshold) {
    std::sort(data.begin(), data.end());
    for (auto it = data.begin(); it != data.end(); ++it) {
        if (*it > threshold) {
            std::cout << "Above threshold: " << *it << std::endl;
        }
    }
}
''',
        "lang": "cpp"
    },
]

# 5 HuggingFace-tarzı örnekleri (fonksiyon + docstring)
HF_SAMPLES = [
    {
        "name": "Buffer copy function",
        "code": '''
void copy_user_buffer(const char *src_buffer, int max_length) {
    char *dest_buffer = (char *)malloc(max_length + 1);
    if (dest_buffer == NULL) return;
    strncpy(dest_buffer, src_buffer, max_length);
    dest_buffer[max_length] = '\\0';
    printf("Copied: %s\\n", dest_buffer);
    free(dest_buffer);
}
''',
        "docstring": "Copies src_buffer into a newly allocated dest_buffer up to max_length bytes."
    },
    {
        "name": "Auth token processor",
        "code": '''
int verify_authentication_token(const char *auth_token, int session_timeout) {
    if (auth_token == NULL) return -1;
    size_t token_length = strlen(auth_token);
    if (token_length < 8 || token_length > 256) return -1;
    char *session_cache = (char *)malloc(token_length + 1);
    strcpy(session_cache, auth_token);
    printf("Token verified, timeout: %d\\n", session_timeout);
    free(session_cache);
    return 0;
}
''',
        "docstring": "Verifies authentication_token length and caches it for session_timeout duration."
    },
    {
        "name": "File reader",
        "code": '''
int read_config_file(const char *filepath, char *output_buffer, int buffer_size) {
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL) { perror("fopen"); return -1; }
    size_t bytes_read = fread(output_buffer, 1, buffer_size - 1, fp);
    output_buffer[bytes_read] = '\\0';
    fclose(fp);
    return (int)bytes_read;
}
''',
        "docstring": "Reads config file at filepath into output_buffer with max buffer_size bytes."
    },
    {
        "name": "Network data sender",
        "code": '''
int transmit_payload(int socket_fd, const void *payload_data, size_t payload_length) {
    size_t total_sent = 0;
    while (total_sent < payload_length) {
        int sent = send(socket_fd, (const char *)payload_data + total_sent,
                        payload_length - total_sent, 0);
        if (sent < 0) return -1;
        total_sent += sent;
    }
    return (int)total_sent;
}
''',
        "docstring": "Transmits payload_data of payload_length bytes over socket_fd reliably."
    },
    {
        "name": "Memory pool allocator",
        "code": '''
typedef struct PoolBlock {
    struct PoolBlock *next_block;
    size_t block_size;
} PoolBlock;

void *pool_allocate(PoolBlock **free_list, size_t request_size) {
    PoolBlock *current_block = *free_list;
    while (current_block != NULL) {
        if (current_block->block_size >= request_size) {
            *free_list = current_block->next_block;
            return (void *)(current_block + 1);
        }
        current_block = current_block->next_block;
    }
    PoolBlock *new_block = (PoolBlock *)malloc(sizeof(PoolBlock) + request_size);
    if (new_block == NULL) return NULL;
    new_block->block_size = request_size;
    return (void *)(new_block + 1);
}
''',
        "docstring": "Allocates request_size bytes from a pool managed by free_list of PoolBlocks."
    },
]

# 5 OWASP/Juliet-tarzı güvenlik açığı örnekleri
OWASP_SAMPLES = [
    {
        "name": "CWE-120 Buffer Overflow",
        "code": '''
#include <stdio.h>
#include <string.h>
void cwe120_bad(char *user_input) {
    char fixed_buffer[64];
    strcpy(fixed_buffer, user_input);
    printf("Data: %s\\n", fixed_buffer);
}
''',
        "cwe": "CWE-120", "lang": "c"
    },
    {
        "name": "CWE-78 OS Command Injection",
        "code": '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void cwe78_bad(char *command_arg) {
    char cmd_buffer[256];
    sprintf(cmd_buffer, "ls %s", command_arg);
    system(cmd_buffer);
}
''',
        "cwe": "CWE-78", "lang": "c"
    },
    {
        "name": "CWE-416 Use After Free",
        "code": '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void cwe416_bad() {
    char *data_ptr = (char *)malloc(128);
    strcpy(data_ptr, "sensitive data");
    free(data_ptr);
    printf("UAF: %s\\n", data_ptr);
}
''',
        "cwe": "CWE-416", "lang": "c"
    },
    {
        "name": "CWE-134 Format String",
        "code": '''
#include <stdio.h>
void cwe134_bad(char *user_format_string) {
    printf(user_format_string);
}
''',
        "cwe": "CWE-134", "lang": "c"
    },
    {
        "name": "CWE-190 Integer Overflow",
        "code": '''
#include <stdio.h>
#include <stdlib.h>
void cwe190_bad(const char *size_str) {
    int num_elements = atoi(size_str);
    int *array = (int *)malloc(num_elements * sizeof(int));
    if (array == NULL) return;
    memset(array, 0, num_elements * sizeof(int));
    free(array);
}
''',
        "cwe": "CWE-190", "lang": "c"
    },
]

# ============================================================================
# Test Runner
# ============================================================================
def run_tests():
    # Clean up any existing test databases
    for db_file in ("test_data/dbs/document_store.db", "test_data/dbs/vector_store.db"):
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass

    results = {
        "github": [], "hf": [], "owasp": [],
        "dedup_test": {}, "whitelist_test": {}, "auto_label_test": {},
        "hf_case_test": {},
    }
    pipeline = HybridDBPipeline(doc_db_path="test_data/dbs/document_store.db", vec_db_path="test_data/dbs/vector_store.db")

    print("=" * 80)
    print("  BÖLÜM 2: OTOMATİZE EDİLMİŞ CANLI TEST")
    print("=" * 80)

    # ── TEST 1: GitHub Normalizasyon ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 1: GitHub AST Normalizasyon (5 örnek)")
    print("─" * 60)
    for i, sample in enumerate(GITHUB_SAMPLES):
        try:
            result = normalize_github_code(sample["code"], language=sample["lang"])
            norm = result.get("normalized_code", "")
            meta = result.get("ast_metadata", {})

            # Auto-label
            sec = auto_label_vulnerability(sample["code"], sample["lang"])

            # Hibrit DB'ye yaz
            eid = pipeline.ingest(sample["code"], "GITHUB", language=sample["lang"],
                                  security_context=sec)

            status = "✅ OK" if norm and eid else "❌ FAIL"
            results["github"].append({
                "name": sample["name"],
                "status": "OK" if norm and eid else "FAIL",
                "entity_id": eid[:8] if eid else "N/A",
                "nodes": meta.get("total_nodes", 0),
                "depth": meta.get("tree_depth", 0),
                "vars_renamed": meta.get("variables_renamed", 0),
                "funcs_renamed": meta.get("functions_renamed", 0),
                "has_errors": meta.get("has_errors", False),
                "auto_label": sec.get("is_vulnerable", False),
            })
            print(f"  [{i+1}] {status} {sample['name']} | "
                  f"Nodes:{meta.get('total_nodes',0)} Vars:{meta.get('variables_renamed',0)} "
                  f"Funcs:{meta.get('functions_renamed',0)} EID:{eid[:8] if eid else 'N/A'}")
        except Exception as e:
            results["github"].append({"name": sample["name"], "status": "ERROR", "error": str(e)})
            print(f"  [{i+1}] ❌ ERROR {sample['name']}: {e}")

    # ── TEST 2: HuggingFace Çapraz Normalizasyon ─────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 2: HuggingFace Çapraz Normalizasyon (5 örnek)")
    print("─" * 60)
    for i, sample in enumerate(HF_SAMPLES):
        try:
            result = normalize_hf_code(sample["code"], docstring=sample["docstring"])
            norm_code = result.get("normalized_code", "")
            norm_doc = result.get("normalized_docstring", "")
            alignment = result.get("nl_alignment", {})
            chunks = result.get("chunks", [])

            sec = auto_label_vulnerability(sample["code"], "cpp")
            eid = pipeline.ingest(sample["code"], "HUGGINGFACE",
                                  docstring=sample["docstring"], security_context=sec)

            status = "✅ OK" if norm_code and eid else "❌ FAIL"
            results["hf"].append({
                "name": sample["name"],
                "status": "OK" if norm_code and eid else "FAIL",
                "entity_id": eid[:8] if eid else "N/A",
                "rename_count": len(alignment.get("rename_map", {})),
                "chunk_count": len(chunks),
                "outlier_count": alignment.get("outlier_count", 0),
                "bidirectional": alignment.get("bidirectional_sync", False),
                "auto_label": sec.get("is_vulnerable", False),
            })
            print(f"  [{i+1}] {status} {sample['name']} | "
                  f"Renames:{len(alignment.get('rename_map',{}))} "
                  f"Chunks:{len(chunks)} Outliers:{alignment.get('outlier_count',0)} "
                  f"EID:{eid[:8] if eid else 'N/A'}")
        except Exception as e:
            results["hf"].append({"name": sample["name"], "status": "ERROR", "error": str(e)})
            print(f"  [{i+1}] ❌ ERROR {sample['name']}: {e}")

    # ── TEST 3: OWASP Güvenlik Normalizasyonu ────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 3: OWASP Güvenlik Normalizasyonu (5 örnek)")
    print("─" * 60)
    for i, sample in enumerate(OWASP_SAMPLES):
        try:
            result = normalize_owasp_code(sample["code"], language=sample["lang"],
                                          cwe_hint=sample["cwe"])
            norm = result.get("normalized_code", "")
            sec_ctx = result.get("security_context", {})

            eid = pipeline.ingest(sample["code"], "OWASP",
                                  language=sample["lang"], cwe_hint=sample["cwe"])

            detected_cwes = sec_ctx.get("cwe_ids", [])
            taint_count = len(sec_ctx.get("taint_paths", []))

            status = "✅ OK" if norm and eid else "❌ FAIL"
            results["owasp"].append({
                "name": sample["name"],
                "status": "OK" if norm and eid else "FAIL",
                "entity_id": eid[:8] if eid else "N/A",
                "expected_cwe": sample["cwe"],
                "detected_cwes": detected_cwes,
                "taint_paths": taint_count,
                "sinks_found": sec_ctx.get("sinks_detected", []),
            })
            cwe_match = sample["cwe"] in detected_cwes
            print(f"  [{i+1}] {status} {sample['name']} | "
                  f"CWE:{sample['cwe']}={'✅' if cwe_match else '⚠️'} "
                  f"Taints:{taint_count} Sinks:{sec_ctx.get('sinks_detected',[])} "
                  f"EID:{eid[:8] if eid else 'N/A'}")
        except Exception as e:
            results["owasp"].append({"name": sample["name"], "status": "ERROR", "error": str(e)})
            print(f"  [{i+1}] ❌ ERROR {sample['name']}: {e}")

    # ── TEST 4: Dedup Kontrolü ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 4: SHA-256 Deduplication Kontrolü")
    print("─" * 60)
    dup_code = GITHUB_SAMPLES[0]["code"]
    eid_dup = pipeline.ingest(dup_code, "GITHUB", language="c")
    is_deduped = (eid_dup == "")
    results["dedup_test"] = {"duplicate_detected": is_deduped}
    print(f"  Aynı kod tekrar gönderildi → Dedup: {'✅ Atlandı' if is_deduped else '❌ Tekrar yazıldı'}")

    # ── TEST 5: Whitelist Exact Match ────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 5: Whitelist Exact Match Doğrulama")
    print("─" * 60)
    wl_code = '''
void test_whitelist() {
    char *buf = (char *)malloc(64);
    strcpy(buf, "hello");
    printf("%s\\n", buf);
    free(buf);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
}
'''
    wl_result = normalize_github_code(wl_code, language="c")
    wl_norm = wl_result.get("normalized_code", "")
    checks = {
        "malloc_preserved": "malloc" in wl_norm,
        "strcpy_preserved": "strcpy" in wl_norm,
        "printf_preserved": "printf" in wl_norm,
        "free_preserved": "free" in wl_norm,
        "socket_preserved": "socket" in wl_norm,
        "AF_INET_preserved": "AF_INET" in wl_norm,
        "SOCK_STREAM_preserved": "SOCK_STREAM" in wl_norm,
        "htons_preserved": "htons" in wl_norm,
        "sin_family_preserved": "sin_family" in wl_norm,
        "sin_port_preserved": "sin_port" in wl_norm,
        "struct_preserved": "struct" in wl_norm,
        "sockaddr_in_preserved": "sockaddr_in" in wl_norm,
    }
    results["whitelist_test"] = checks
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")

    # ── TEST 6: HF Case-Sensitivity ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 6: HF Case-Sensitivity (NULL/EOF koruması)")
    print("─" * 60)
    cs_code = '''
int check_ptr(void *ptr) {
    if (ptr == NULL) return EOF;
    return EXIT_SUCCESS;
}
'''
    cs_result = normalize_hf_code(cs_code)
    cs_norm = cs_result.get("normalized_code", "")
    cs_checks = {
        "NULL_preserved": "NULL" in cs_norm,
        "EOF_preserved": "EOF" in cs_norm,
        "EXIT_SUCCESS_preserved": "EXIT_SUCCESS" in cs_norm,
    }
    results["hf_case_test"] = cs_checks
    for k, v in cs_checks.items():
        print(f"  {'✅' if v else '❌'} {k}")

    # ── TEST 7: Auto-Label ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TEST 7: Auto-Label (cppcheck/flawfinder)")
    print("─" * 60)
    vuln_code = '''
#include <string.h>
void overflow(char *input) { char buf[10]; strcpy(buf, input); }
'''
    al_result = auto_label_vulnerability(vuln_code, "c")
    results["auto_label_test"] = al_result
    tools_available = len(al_result.get("tool_findings", [])) > 0
    print(f"  is_vulnerable: {al_result.get('is_vulnerable')}")
    print(f"  cwe_ids: {al_result.get('cwe_ids', [])}")
    print(f"  tool_findings count: {len(al_result.get('tool_findings', []))}")
    if not tools_available:
        print(f"  ⚠️ cppcheck/flawfinder yüklü değil — Docker'da çalışacak")

    # ── İstatistikler ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  PIPELINE İSTATİSTİKLERİ")
    print("─" * 60)
    stats = pipeline.get_stats()
    print(f"  Vector Store: {json.dumps(stats.get('vector_store_counts', {}))}")

    # Sonuçları dosyaya yaz
    results["pipeline_stats"] = stats
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Detaylı sonuçlar: {output_path}")

    # Özet
    print("\n" + "=" * 80)
    print("  ÖZET")
    print("=" * 80)
    gh_ok = sum(1 for r in results["github"] if r.get("status") == "OK")
    hf_ok = sum(1 for r in results["hf"] if r.get("status") == "OK")
    ow_ok = sum(1 for r in results["owasp"] if r.get("status") == "OK")
    wl_ok = sum(1 for v in checks.values() if v)
    cs_ok = sum(1 for v in cs_checks.values() if v)
    print(f"  GitHub Normalizasyon:  {gh_ok}/5")
    print(f"  HF Normalizasyon:     {hf_ok}/5")
    print(f"  OWASP Normalizasyon:  {ow_ok}/5")
    print(f"  Whitelist Checks:     {wl_ok}/{len(checks)}")
    print(f"  Case-Sensitivity:     {cs_ok}/{len(cs_checks)}")
    print(f"  Dedup:                {'✅' if is_deduped else '❌'}")
    print(f"  Auto-Label:           {'✅ Araçlar mevcut' if tools_available else '⚠️ Docker gerekli'}")
    print("=" * 80)

    # Clean up test databases
    for db_file in ("test_data/dbs/document_store.db", "test_data/dbs/vector_store.db"):
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass

    return results


if __name__ == "__main__":
    run_tests()
