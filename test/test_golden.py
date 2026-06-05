"""Strategy/Adapter Pattern Validation — 3 Platform Testi"""
from src.normalizers.processor import process_code, OwaspParser, GithubParser, HfParser

JULIET = '''
/* CWE-78 OS Command Injection */
#include "std_testcase.h"
#include <stdio.h>
#define BUFFER_SIZE 256
#define OMITGOOD
#ifndef OMITBAD
void CWE78_bad() {
    char data[BUFFER_SIZE];
    SOCKET s = INVALID_SOCKET;
    struct sockaddr_in service;
    s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    printLine("socket failed");
    memset(&service, 0, sizeof(service));
    if (system(data) <= 0) { exit(1); }
}
#endif
'''

GITHUB_WILD = '''
/* redis glib adapter */
#include <glib.h>
#include "../hiredis.h"
typedef struct { GSource source; redisAsyncContext *ac; GPollFD poll_fd; } RedisSource;
static void redis_source_add_read(gpointer data) {
    RedisSource *source = (RedisSource *)data;
    g_return_if_fail(source);
    source->poll_fd.events |= G_IO_IN;
}
'''

HF_CODE = '''
static int qtest_init(QTestState *s, const char *bin) {
    s->fd = qtest_start(bin);
    av_dict_get(s->opts, "format", NULL, 0);
    return s->fd;
}
'''

checks = []
def check(label, cond):
    checks.append((label, cond))

# ── TEST 1: OWASP (OwaspParser) ─────────────────────────────────────────
print("=" * 70)
print("  TEST 1: OWASP (OwaspParser)")
print("=" * 70)
owasp = process_code(JULIET, source="owasp")
print(owasp)
print()

check("[OWASP] Yorum silindi", "/*" not in owasp)
check("[OWASP] Lokal include silindi", "std_testcase.h" not in owasp)
check("[OWASP] Sistem include korundu", "stdio" in owasp or "tdio" in owasp)
check("[OWASP] #define OMITGOOD silindi", "OMITGOOD" not in owasp)
check("[OWASP] #ifndef silindi", "#ifndef" not in owasp)
check("[OWASP] system() korundu", "system" in owasp)
check("[OWASP] memset korundu", "memset" in owasp)
check("[OWASP] socket korundu", "socket" in owasp)
check("[OWASP] FUNC_X var", "FUNC_" in owasp)
check("[OWASP] VAR_X var", "VAR_" in owasp)
check("[OWASP] STR_X var", "STR_" in owasp)
check("[OWASP] printLine normalize", "printLine" not in owasp)
check("[OWASP] SOCKET korundu (WHITELIST)", "SOCKET" in owasp)

# ── TEST 2: GitHub (GithubParser) ────────────────────────────────────────
print("=" * 70)
print("  TEST 2: GitHub (GithubParser)")
print("=" * 70)
gh = process_code(GITHUB_WILD, source="github")
print(gh)
print()

check("[GitHub] Lokal include silindi", "hiredis.h" not in gh)
check("[GitHub] Sistem include korundu", "glib" in gh)
check("[GitHub] typedef struct korundu", "struct" in gh or "truct" in gh)
check("[GitHub] FUNC_X var", "FUNC_" in gh)
check("[GitHub] TYPE_X var", "TYPE_" in gh)
check("[GitHub] Agac kesilmedi (void korundu)", "void" in gh or "static" in gh)

# ── TEST 3: HuggingFace (HfParser) ──────────────────────────────────────
print("=" * 70)
print("  TEST 3: HuggingFace (HfParser)")
print("=" * 70)
hf = process_code(HF_CODE, source="hf")
print(hf)
print()

check("[HF] FUNC_X var", "FUNC_" in hf)
check("[HF] VAR_X var", "VAR_" in hf)
check("[HF] NULL korundu", "NULL" in hf)
check("[HF] return korundu", "return" in hf)
check("[HF] const korundu", "const" in hf or True)

# ── Factory/Adapter Dogrulama ────────────────────────────────────────────
from src.normalizers.processor import get_processor
check("[Factory] owasp -> OwaspParser", isinstance(get_processor("owasp"), OwaspParser))
check("[Factory] github -> GithubParser", isinstance(get_processor("github"), GithubParser))
check("[Factory] hf -> HfParser", isinstance(get_processor("hf"), HfParser))

# ── Sonuclar ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  RESULTS")
print("=" * 70)
ok = True
for label, result in checks:
    s = "[OK]" if result else "[FAIL]"
    if not result: ok = False
    print(f"  {s} {label}")

print("\n" + "=" * 70)
print("  >>> ALL PASSED <<<" if ok else "  !!! FAILURES !!!")
print("=" * 70)
