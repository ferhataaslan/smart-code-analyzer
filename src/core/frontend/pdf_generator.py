import os
from datetime import datetime
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'Akilli Kod Analizoru 2.0 - Teknik Analiz Raporu', 0, 1, 'C')
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Footer text
        date_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.cell(0, 10, f'Yapay Zeka Analiz Raporu | Tarih: {date_str} | Sayfa {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(analysis_results, output_path="report.pdf"):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Genel Durum
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Yapay Zeka Analizi (CodeBERT)', 0, 1)
    
    pdf.set_font("Arial", size=12)
    ai_res = analysis_results.get("ai_analysis", {})
    if ai_res:
        is_defective = ai_res.get("is_defective_predicted", False)
        severity = ai_res.get("severity_score", 0.0)
        
        status_text = "Hatali / Guvenlik Acigi Bulundu!" if is_defective else "Temiz / Guvenli"
        pdf.cell(0, 10, f'Durum: {status_text}', 0, 1)
        pdf.cell(0, 10, f'Ciddiyet Skoru: {severity} / 10', 0, 1)
        
        for obs in ai_res.get("observations", []):
            safe_obs = str(obs).encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, f'- {safe_obs}')
    else:
        pdf.cell(0, 10, 'Yapay zeka analizi yapilmadi (Devre Disi).', 0, 1)
        
    pdf.ln(5)

    # Statik Analiz
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Statik Analiz Sonuclari', 0, 1)
    
    pdf.set_font("Arial", size=12)
    static_res = analysis_results.get("static_analysis", {})
    metrics = static_res.get("metrics", {})
    pdf.cell(0, 10, f'Maks. Cyclomatic Complexity: {metrics.get("max_cyclomatic_complexity", 0)}', 0, 1)
    pdf.cell(0, 10, f'Satir Sayisi (NLOC): {metrics.get("nloc", 0)}', 0, 1)
    
    issues = static_res.get("issues", [])
    pdf.cell(0, 10, f'Toplam Bulunan Ihlal: {len(issues)}', 0, 1)
    for issue in issues:
        safe_issue = str(issue).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, f'* {safe_issue}')
    
    pdf.ln(5)

    # Performans Analizi
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Performans Analizi', 0, 1)
    
    pdf.set_font("Arial", size=12)
    perf_res = analysis_results.get("performance", {})
    warnings = perf_res.get("performance_warnings", 0)
    pdf.cell(0, 10, f'Performans Uyarisi Sayisi: {warnings}', 0, 1)
    for metric in perf_res.get("metrics", []):
        safe_metric = str(metric).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, f'* {safe_metric}')

    pdf.output(output_path)
    return output_path
