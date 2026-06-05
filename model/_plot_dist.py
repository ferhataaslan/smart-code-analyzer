#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Veri seti dağılım grafikleri."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

print('Veri seti yükleniyor...')
ds = load_dataset(
    'smart-code-analyzer-team/cpp-vulnerability-dataset',
    split='train',
    token='YOUR_HUGGINGFACE_TOKEN_HERE'
)
n = len(ds)
print(f'Toplam: {n}')

# ── Veri toplama ──
vuln_labels = []  # 0=clean, 1=vuln_explicit, 2=vuln_recovered
comp_labels = []
risk_scores = []
cwe_counter = Counter()

COMP_MAP = {
    '1': 'O(1)', 'n': 'O(n)', 'log n': 'O(log n)',
    'n log n': 'O(n log n)', 'n^2': 'O(n²)', 'n^2 log n': 'O(n² log n)',
    'n^3': 'O(n³)', 'n^4': 'O(n⁴)', 'n^5': 'O(n⁵)',
    'n^6': 'O(n⁶)', 'n^7': 'O(n⁷)',
}

for row in ds:
    # Vulnerability
    s = row.get('security_context', '')
    try:
        sec = json.loads(s) if s and s.strip() not in ('', '{}') else {}
    except:
        sec = {}

    v = sec.get('is_vulnerable', None)
    cwe_ids = sec.get('cwe_ids', [])
    rs = sec.get('risk_scoring', {})
    fr = float(rs.get('final_calculated_risk', 0)) if isinstance(rs, dict) else 0

    if v is True:
        vuln_labels.append('Vulnerable (etiketli)')
    elif v is False:
        vuln_labels.append('Clean (etiketli)')
    elif cwe_ids or fr > 0:
        vuln_labels.append('Vulnerable (kurtarılan)')
    else:
        vuln_labels.append('Clean (kurtarılan)')

    if cwe_ids:
        for c in cwe_ids:
            cwe_counter[c] += 1

    risk_scores.append(fr)

    # Complexity
    c = row.get('complexity', '1')
    comp_labels.append(COMP_MAP.get(c, c))

# ══════════════════════════════════════════════════════════════
#  GRAFIKLER
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle('Smart Code Analyzer — Veri Seti Dağılım Analizi (69,299 kayıt)',
             fontsize=16, fontweight='bold', y=0.98)

colors_main = ['#E53935', '#FF7043', '#43A047', '#66BB6A']
colors_comp = ['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5',
               '#64B5F6', '#90CAF9', '#BBDEFB', '#E3F2FD', '#E8EAF6', '#F3E5F5']

# ── 1. Zafiyet Dağılımı (Pie) ──
ax = axes[0, 0]
vuln_counts = Counter(vuln_labels)
order = ['Vulnerable (etiketli)', 'Vulnerable (kurtarılan)', 'Clean (etiketli)', 'Clean (kurtarılan)']
sizes = [vuln_counts.get(k, 0) for k in order]
explode = (0.05, 0.03, 0.05, 0.03)
wedges, texts, autotexts = ax.pie(sizes, labels=order, colors=colors_main,
                                   autopct='%1.1f%%', startangle=90, explode=explode,
                                   textprops={'fontsize': 9})
ax.set_title('Zafiyet Dağılımı', fontsize=13, fontweight='bold')

# ── 2. Zafiyet Dağılımı (Bar - eğitim için) ──
ax = axes[0, 1]
total_vuln = vuln_counts.get('Vulnerable (etiketli)', 0) + vuln_counts.get('Vulnerable (kurtarılan)', 0)
total_clean = vuln_counts.get('Clean (etiketli)', 0) + vuln_counts.get('Clean (kurtarılan)', 0)
bars = ax.bar(['Vulnerable', 'Clean'], [total_vuln, total_clean],
              color=['#E53935', '#43A047'], edgecolor='white', linewidth=2)
ax.bar_label(bars, fmt='%d', fontsize=12, fontweight='bold')
ax.set_title('Eğitim İçin Sınıf Dengesi', fontsize=13, fontweight='bold')
ax.set_ylabel('Kayıt Sayısı')

# Ağırlık bilgisi ekle
w_vuln = n / (2 * total_vuln) if total_vuln else 0
w_clean = n / (2 * total_clean) if total_clean else 0
ax.text(0, total_vuln * 0.5, f'weight={w_vuln:.3f}', ha='center', va='center',
        fontsize=11, color='white', fontweight='bold')
ax.text(1, total_clean * 0.5, f'weight={w_clean:.3f}', ha='center', va='center',
        fontsize=11, color='white', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# ── 3. CWE Dağılımı (Top 12) ──
ax = axes[0, 2]
top_cwe = cwe_counter.most_common(12)
cwe_names = [c[0] for c in top_cwe]
cwe_vals = [c[1] for c in top_cwe]
bars = ax.barh(cwe_names[::-1], cwe_vals[::-1], color='#FF7043', edgecolor='white')
ax.bar_label(bars, fmt='%d', fontsize=9)
ax.set_title('CWE Dağılımı (Top 12)', fontsize=13, fontweight='bold')
ax.set_xlabel('Kayıt Sayısı')

# ── 4. Complexity Dağılımı ──
ax = axes[1, 0]
comp_counts = Counter(comp_labels)
comp_order = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)',
              'O(n² log n)', 'O(n³)', 'O(n⁴)', 'O(n⁵)', 'O(n⁶)', 'O(n⁷)']
comp_vals = [comp_counts.get(k, 0) for k in comp_order]
bars = ax.bar(comp_order, comp_vals, color=colors_comp, edgecolor='white', linewidth=1)
ax.bar_label(bars, fmt='%d', fontsize=8, rotation=45)
ax.set_title('Complexity (Big-O) Dağılımı', fontsize=13, fontweight='bold')
ax.set_ylabel('Kayıt Sayısı')
ax.set_yscale('log')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

# ── 5. Risk Score Histogramı ──
ax = axes[1, 1]
ax.hist(risk_scores, bins=50, color='#7B1FA2', edgecolor='white', alpha=0.85)
ax.axvline(x=np.mean(risk_scores), color='#FF9800', linestyle='--', linewidth=2,
           label=f'Ortalama: {np.mean(risk_scores):.2f}')
ax.axvline(x=5.0, color='#F44336', linestyle='--', linewidth=2,
           label='Yüksek Risk Eşiği (5.0)')
ax.set_title('Risk Score Dağılımı', fontsize=13, fontweight='bold')
ax.set_xlabel('Final Calculated Risk')
ax.set_ylabel('Kayıt Sayısı')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# ── 6. Class-Weight Önerisi ──
ax = axes[1, 2]
tasks = ['Vulnerability\n(2 sınıf)', 'Complexity\n(6 sınıf)', 'DataFlow\n(2 sınıf)']

# Vulnerability weights
vuln_w = [w_vuln, w_clean]
# Complexity weights (ilk 6 ana sınıf)
comp_main = [comp_counts.get(k, 1) for k in comp_order[:6]]
comp_total = sum(comp_main)
comp_w = [comp_total / (6 * c) if c > 0 else 0 for c in comp_main]

# Görselleştirme
x = np.arange(3)
vuln_ratio = total_vuln / (total_vuln + total_clean)
clean_ratio = total_clean / (total_vuln + total_clean)
dfg_has = 62639; dfg_no = 6660
dfg_ratio_has = dfg_has / (dfg_has + dfg_no)
dfg_ratio_no = dfg_no / (dfg_has + dfg_no)

ax.bar(x - 0.15, [vuln_ratio, comp_vals[0]/sum(comp_vals), dfg_ratio_has],
       0.3, label='Çoğunluk Sınıfı', color='#43A047', edgecolor='white')
ax.bar(x + 0.15, [clean_ratio, sum(comp_vals[1:])/sum(comp_vals), dfg_ratio_no],
       0.3, label='Azınlık Sınıfı', color='#E53935', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=10)
ax.set_ylabel('Oran')
ax.set_title('Görev Bazlı Sınıf Dengesizliği', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = r'c:\Users\aslan\OneDrive\Masaüstü\Akıllı Kod analizatörü G.2\Kodlar\model\dataset_distribution.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Grafik kaydedildi: {save_path}')

# Ağırlık önerisi yazdır
print(f'\n===== CLASS WEIGHT ÖNERİLERİ =====')
print(f'Vulnerability: vuln_weight={w_vuln:.4f}, clean_weight={w_clean:.4f}')
print(f'  ({total_vuln} vuln vs {total_clean} clean)')
print(f'\nComplexity sınıf ağırlıkları:')
for name, count, w in zip(comp_order[:6], comp_main, comp_w):
    print(f'  {name}: {count:>6} kayıt → weight={w:.4f}')
print(f'\nDataFlow: has={dfg_has}, no={dfg_no}')
dw1 = n / (2 * dfg_has)
dw0 = n / (2 * dfg_no)
print(f'  has_weight={dw1:.4f}, no_weight={dw0:.4f}')
