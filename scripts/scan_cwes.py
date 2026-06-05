import json
from datasets import load_dataset

KNOWN_CWES = {
    "CWE-79", "CWE-89", "CWE-352", "CWE-862", "CWE-787", "CWE-120",
    "CWE-20", "CWE-78", "CWE-416", "CWE-22", "CWE-125", "CWE-119",
    "CWE-190", "CWE-200", "CWE-122", "CWE-134", "CWE-242", "CWE-327",
    "CWE-330", "CWE-415", "CWE-426", "CWE-427", "CWE-191", "CWE-476",
    "CWE-401", "CWE-77", "CWE-94", "CWE-502", "CWE-918", "CWE-269",
    "CWE-434", "CWE-306", "CWE-798", "CWE-863", "CWE-611"
}

print('Dataset yükleniyor (streaming)...')
ds = load_dataset('smart-code-analyzer-team/cpp-vulnerability-dataset', split='train', streaming=True)

all_cwes = set()

count = 0
for rec in ds:
    raw_sec = rec.get("security_context", "")
    if raw_sec and raw_sec.strip() not in ('', 'null', 'None'):
        try:
            sec = json.loads(raw_sec)
            cwe_ids = sec.get("cwe_ids", [])
            if isinstance(cwe_ids, str):
                cwe_ids = [cwe_ids]
            for cwe in cwe_ids:
                cwe_str = str(cwe).strip()
                if cwe_str and cwe_str != "Unknown":
                    all_cwes.add(cwe_str)
        except:
            pass
            
    count += 1
    if count % 10000 == 0:
        print(f"{count} kayit islendi...")

missing_cwes = all_cwes - KNOWN_CWES
print("\n--- Tarama Tamamlandi ---")
print(f"Toplam benzersiz CWE sayisi: {len(all_cwes)}")
print(f"Whitelist'te OLMAYAN CWE'ler: {missing_cwes}")

# Print code to easily copy-paste
missing_list = list(missing_cwes)
print("\nMissing list:", missing_list)
