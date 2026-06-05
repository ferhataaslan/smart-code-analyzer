import streamlit as st
import requests
import plotly.graph_objects as go
import tempfile
import os
from src.core.frontend.pdf_generator import generate_pdf_report

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/analyze")

st.set_page_config(
    page_title="Akıllı Kod Analizörü 2.0",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
#  PREMIUM STYLING (Glassmorphism & Gradients)
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Arkaplan ve genel tema */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }

    /* Gorsel olarak zengin kartlar (Glassmorphism) */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Basliklar icin neon/gradient efekti */
    .gradient-text {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0px;
    }
    
    .sub-text {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Metrik kutulari icin override */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #00C9FF;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
#  HEADER
# ------------------------------------------------------------------
st.markdown('<h1 class="gradient-text">🌌 Akıllı Kod Analizörü 2.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Hugging Face GraphCodeBERT mimarisi ile güçlendirilmiş, yeni nesil güvenlik ve karmaşıklık analizi platformu.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
#  SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Analiz Motoru Ayarları")
    ai_enabled = st.checkbox("🧠 Yapay Zeka (GraphCodeBERT) Aktif", value=True)
    st.markdown("---")
    st.markdown("Bu panel, yüklediğiniz C/C++ kodlarını derin öğrenme modelleri ve `lizard` statik analiz motorları kullanarak inceler.")

# ------------------------------------------------------------------
#  MAIN INPUT AREA
# ------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📁 Dosya Yükle", "📝 Kodu Yapıştır"])

code_to_analyze = None

with tab1:
    uploaded_file = st.file_uploader("Bir C veya C++ dosyası yükleyin", type=["c", "cpp", "h"])
    if uploaded_file is not None:
        code_to_analyze = uploaded_file.getvalue().decode("utf-8")
        with st.expander("Görüntüle: " + uploaded_file.name):
            st.code(code_to_analyze, language="cpp")

with tab2:
    pasted_code = st.text_area("Analiz edilecek kodu buraya yapıştırın:", height=250)
    if pasted_code and not uploaded_file:
        code_to_analyze = pasted_code

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
#  ANALYSIS EXECUTION
# ------------------------------------------------------------------
if st.button("🚀 KODU ANALİZ ET", use_container_width=True, type="primary"):
    if not code_to_analyze:
        st.warning("⚠️ Lütfen analiz edilecek bir dosya yükleyin veya kod yapıştırın!")
    else:
        with st.spinner('Derin öğrenme modeli kodu tarıyor, lütfen bekleyin...'):
            try:
                response = requests.post(API_URL, json={
                    "code": code_to_analyze,
                    "ai_enabled": ai_enabled
                })
                
                if response.status_code == 200:
                    data = response.json()
                    st.balloons()
                    
                    st.markdown('<h2 style="margin-top: 40px; color: #fff;">📊 Analiz Sonuçları</h2>', unsafe_allow_html=True)
                    
                    # ------------------------------------------------------
                    # AI RESULTS
                    # ------------------------------------------------------
                    if ai_enabled and "ai_analysis" in data:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown("### 🧠 GraphCodeBERT Yapay Zeka Analizi")
                        
                        ai_data = data["ai_analysis"]
                        if "error" in ai_data:
                            st.error(f"AI Motoru Hatası: {ai_data['error']}")
                        else:
                            is_def = ai_data.get("is_defective_predicted", False)
                            if is_def:
                                st.error("🚨 DİKKAT: Yapay zeka bu kodda KRİTİK GÜVENLİK AÇIKLARI tespit etti!")
                            else:
                                st.success("✅ GÜVENLİ: Yapay zeka kodu temiz buldu.")
                                
                            # Ciddiyet Skoru Gauge
                            severity = ai_data.get("severity_score", 0.0)
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = severity,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Ciddiyet Skoru (10 Üzerinden)", 'font': {'color': '#fff'}},
                                gauge = {
                                    'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "white"},
                                    'bar': {'color': "#FF4B2B" if is_def else "#00C9FF"},
                                    'bgcolor': "rgba(0,0,0,0)",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps' : [
                                        {'range': [0, 3], 'color': "rgba(0, 201, 255, 0.2)"},
                                        {'range': [3, 7], 'color': "rgba(255, 255, 0, 0.2)"},
                                        {'range': [7, 10], 'color': "rgba(255, 75, 43, 0.3)"}
                                    ]
                                }
                            ))
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(l=20, r=20, t=30, b=20), height=250)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander("🔍 Tespit Edilen Detaylı Bulguları Önizle", expanded=True):
                                if ai_data.get("observations", []):
                                    for obs in ai_data.get("observations", []):
                                        st.markdown(f"- {obs}")
                                else:
                                    st.info("Herhangi bir bulgu raporlanmadı.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ------------------------------------------------------
                    # STATIC & PERFORMANCE
                    # ------------------------------------------------------
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown("### 🛠️ Statik Analiz (Lizard)")
                        static_data = data.get("static_analysis", {})
                        metrics = static_data.get("metrics", {})
                        
                        st.metric("Cyclomatic Complexity (Maks)", metrics.get("max_cyclomatic_complexity", 0))
                        st.metric("Satır Sayısı (NLOC)", metrics.get("nloc", 0))
                        st.metric("Fonksiyon Sayısı", metrics.get("function_count", 0))
                        
                        issues = static_data.get("issues", [])
                        if issues:
                            st.warning(f"{len(issues)} statik sorun tespit edildi.")
                            with st.expander("🛠️ Statik Sorunları Görüntüle"):
                                for issue in issues:
                                    st.markdown(f"- *{issue}*")
                        else:
                            st.info("Statik analizde kural ihlali bulunmadı.")
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    with col2:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown("### ⚡ Performans Özeti")
                        perf_data = data.get("performance", {})
                        st.metric("Hız/Performans Durumu", perf_data.get("status", "Bilinmiyor"))
                        st.metric("Uyarı Sayısı", perf_data.get("performance_warnings", 0))
                        
                        with st.expander("⚡ Performans Detayları"):
                            for metric in perf_data.get("metrics", []):
                                st.markdown(f"- {metric}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # ------------------------------------------------------
                    # RAW PREVIEW & PDF EXPORT
                    # ------------------------------------------------------
                    with st.expander("⚙️ Ham Analiz Verisi Önizleme (JSON)"):
                        st.json(data)

                    st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
                    st.markdown("### 📄 Raporlama İşlemleri")
                    st.markdown("Tüm analiz sonuçlarını profesyonel bir PDF formatında indirebilirsiniz.")
                    
                    with st.spinner("PDF Raporu Oluşturuluyor..."):
                        fd, path = tempfile.mkstemp(suffix=".pdf")
                        os.close(fd)
                        generate_pdf_report(data, path)
                        
                        with open(path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                            
                        st.download_button(
                            label="📥 Kapsamlı PDF Raporunu İndir",
                            data=pdf_bytes,
                            file_name="Akill_Kod_Analiz_Raporu.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        os.remove(path)
                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    st.error(f"API Sunucusu Hatası: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("API ile iletişim kurulamadı. Lütfen arka planda FastAPI sunucusunun ('uvicorn src.core.api:app --port 8000') çalıştığından emin olun.")
            except Exception as e:
                st.error(f"Beklenmeyen bir hata oluştu: {e}")
