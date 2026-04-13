import streamlit as st
import os

# ==========================================
# 1. PAGE CONFIG (SEO Meta) - HER ZAMAN İLK SIRADA!
# ==========================================
st.set_page_config(
    page_title="Hisse Senedi Analizi | BİST 50 Teknik Analiz - EUTA Borsa",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. PRECONNECT (Fontlardan önce - Kritik)
# ==========================================
st.html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
""")

# ==========================================
# 3. SEO META TAGS (Google için kritik) - 50 BİST Hissesi Dahil
# ==========================================
st.html("""
    <meta name="description" content="EUTA Borsa - BİST 50 popüler hisseleri için profesyonel teknik analiz. THYAO, GARAN, ASELS, KCHOL, SAHOL, EREGL, FROTO, BIMAS ve tüm BİST hisselerinin analizi, grafikleri, fibonacci seviyeleri ve al-sat sinyalleri.">
    <meta name="keywords" content="borsa, hisse senedi, teknik analiz, BİST, BIST 100, fibonacci, RSI, SMA, hisse analizi, eutaborsa, borsa istanbul, thyao, garanti, aselsan, THYAO, GARAN, ASELS, KCHOL, SAHOL, EREGL, FROTO, BIMAS, TCELL, ISCTR, YKBNK, AKBNK, HALKB, VAKBN, SISE, TUPRS, ENJSA, TOASO, EKGYO, PETKM, SASA, TAVHL, PGSUS, MGROS, AEFES, KRDMD, GUBRF, ASTOR, TTKOM, TRALT, ALARK, ARCLK, DOHOL, HEKTS, MAVI, CCOLA, CIMSA, DOAS, BRSAN, BTCIM, KONTR, KUYAS, MIATK, OYAKC, SOKM, TRMET, TSKB, VESTL, ZOREN, AGHOL">
    <meta name="author" content="Ercan USLU">
    <meta name="robots" content="index, follow">
    <meta name="language" content="tr">
    <meta name="revisit-after" content="1 days">

    <!-- Canonical URL -->
    <link rel="canonical" href="https://www.eutaborsa.com">

    <!-- Open Graph (Facebook, LinkedIn) -->
    <meta property="og:title" content="Hisse Senedi Analizi | BİST 50 Teknik Analiz - EUTA Borsa">
    <meta property="og:description" content="Profesyonel borsa analiz platformu. THYAO, GARAN, ASELS ve 50+ BİST hissesini analiz edin.">
    <meta property="og:url" content="https://www.eutaborsa.com">
    <meta property="og:type" content="website">
    <meta property="og:locale" content="tr_TR">
    <meta property="og:site_name" content="EUTA Borsa">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="EUTA Borsa - BİST Hisse Analiz Platformu">
    <meta name="twitter:description" content="THYAO, GARAN, ASELS ve 50+ BİST hissesi için teknik analiz">
    <meta name="twitter:creator" content="@ercanuslu">

    <!-- Schema.org JSON-LD -->
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": "EUTA Borsa",
        "url": "https://www.eutaborsa.com",
        "description": "BİST 50 popüler hisseleri için profesyonel teknik analiz platformu - THYAO GARAN ASELS KCHOL SAHOL",
        "author": {
            "@type": "Person",
            "name": "Ercan USLU"
        },
        "inLanguage": "tr-TR"
    }
    </script>
""")

# ==========================================
# 4. KRİTİK CSS (Sadece 35 satır - Inline)
# ==========================================
st.markdown("""
<style>
    /* Temel */
    .main { background: #0a0e14; color: #f0f6fc; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 0 !important; margin-top: 0 !important; }

    /* Header */
    .header-bar { position: fixed; top: 0; left: 0; right: 0; height: 40px; 
                  background: rgba(13,17,23,0.95); border-bottom: 1px solid #30363d;
                  display: flex; align-items: center; padding: 0 20px; z-index: 1000; }
    .header-title { font-weight: 700; color: #58a6ff; font-size: 15px; }

    /* Layout */
    .main-content { padding-top: 50px; max-width: 800px; margin: 0 auto; }
    .hero { text-align: center; padding: 20px; }
    .hero h1 { font-size: 26px; margin: 0; color: #f0f6fc; }
    .hero p { color: #8b949e; font-size: 12px; margin: 5px 0; }

    /* Grid */
    .menu-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; padding: 0 20px; }
    @media (max-width: 900px) { .menu-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 600px) { .menu-grid { grid-template-columns: 1fr; } }

    /* Cards */
    .menu-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; 
                 padding: 15px; text-align: center; transition: all 0.2s; }
    .menu-card:hover { border-color: #58a6ff; transform: translateY(-2px); }

    /* Gizle */
    #MainMenu, footer, header { visibility: hidden; }

    /* Buton */
    .stButton > button { background: #238636; color: white; border: none; 
                         border-radius: 6px; width: 100%; font-weight: 600; }

    /* SEO Content - Gizli ama indexlenebilir */
    .seo-content { position: absolute; left: -9999px; height: 0; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. HEADER
# ==========================================
st.markdown("""
    <div class="header-bar">
        <span style="font-size: 20px; margin-right: 8px;">📈</span>
        <span class="header-title">Ercan USLU - Hisse Analiz</span>
        <span style="margin-left: auto; color: #8b949e; font-size: 11px;">v1.0</span>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 6. HERO SECTION (SEO Dostu H1)
# ==========================================
st.html("""
    <div class="main-content">
        <div class="hero">
            <h1 style="font-size: 26px; margin: 0; color: #f0f6fc; font-weight: 700; line-height: 1.3;">
                Hisse Senedi Analizi
            </h1>
            <p style="color: #8b949e; font-size: 12px; margin: 8px 0; line-height: 1.5;">
                BİST 100 şirketleri için profesyonel teknik analiz platformu
            </p>
            <span style="display: inline-block; background: rgba(35,134,54,0.2); color: #7ee787; 
                         padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">
                🚀 v1.0 Beta
            </span>
        </div>
""")

# ==========================================
# 7. SEO CONTENT - 50 BİST Hissesi (Gizli ama indexlenebilir)
# ==========================================
st.html("""
    <div class="seo-content">
        <h2>BİST 50 Popüler Hisse Senedi Analizleri</h2>
        <p>EUTA Borsa platformunda analiz edilen Borsa İstanbul (BİST) hisse senetleri: 
        THYAO, GARAN, ASELS, KCHOL, SAHOL, EREGL, FROTO, BIMAS, TCELL, ISCTR, YKBNK, AKBNK, 
        HALKB, VAKBN, SISE, TUPRS, ENJSA, TOASO, EKGYO, PETKM, SASA, TAVHL, PGSUS, MGROS, 
        AEFES, KRDMD, GUBRF, ASTOR, TTKOM, TRALT, ALARK, ARCLK, DOHOL, HEKTS, MAVI, CCOLA, 
        CIMSA, DOAS, BRSAN, BTCIM, KONTR, KUYAS, MIATK, OYAKC, SOKM, TRMET, TSKB, VESTL, 
        ZOREN, AGHOL. Her hisse için teknik analiz, grafik, fibonacci seviyeleri, 
        RSI, MACD indikatörleri ve al-sat sinyalleri.</p>
        <ul>
            <li>THYAO (Türk Hava Yolları) hisse analizi ve grafik</li>
            <li>GARAN (Garanti Bankası) hisse yorum ve hedef fiyat</li>
            <li>ASELS (Aselsan) teknik analiz ve sinyaller</li>
            <li>KCHOL (Koç Holding) hisse analizi</li>
            <li>SAHOL (Sabancı Holding) teknik analiz</li>
            <li>EREGL (Ereğli Demir Çelik) hisse yorum</li>
            <li>BIMAS (BİM) teknik analiz</li>
            <li>TCELL (Turkcell) hisse analizi</li>
            <li>SISE (Şişe Cam) teknik analiz</li>
            <li>TUPRS (Tüpraş) hisse yorum</li>
        </ul>
        <p>BİST 30, BİST 50 ve BİST 100 endekslerindeki tüm hisselerin detaylı teknik analizleri 
        için EUTA Borsa platformunu kullanın. THYAO hisse analizi, GARAN teknik analiz, 
        ASELS hisse yorumları ve daha fazlası.</p>
    </div>
""")

# ==========================================
# 8. MENU CARDS
# ==========================================
st.markdown('<div class="menu-grid">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="menu-card">
            <div style="font-size: 32px; margin-bottom: 8px;">📊</div>
            <div style="font-size: 15px; font-weight: 700; color: #f0f6fc; margin-bottom: 5px;">Grafik Analizi</div>
            <div style="font-size: 10px; color: #8b949e; margin-bottom: 10px;">Teknik indikatörler ile detaylı analiz</div>
            <div style="display: flex; gap: 4px; justify-content: center; flex-wrap: wrap;">
                <span style="background: rgba(41,98,255,0.1); color: #58a6ff; padding: 2px 6px; 
                             border-radius: 4px; font-size: 9px;">SMA</span>
                <span style="background: rgba(41,98,255,0.1); color: #58a6ff; padding: 2px 6px; 
                             border-radius: 4px; font-size: 9px;">RSI</span>
                <span style="background: rgba(41,98,255,0.1); color: #58a6ff; padding: 2px 6px; 
                             border-radius: 4px; font-size: 9px;">Fibo</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Aç →", key="grafik_btn", use_container_width=True):
        st.switch_page("pages/01_Grafik.py")

with col2:
    st.markdown("""
        <div class="menu-card">
            <div style="font-size: 32px; margin-bottom: 8px;">🔍</div>
            <div style="font-size: 15px; font-weight: 700; color: #f0f6fc; margin-bottom: 5px;">Fibonacci Tarama</div>
            <div style="font-size: 10px; color: #8b949e; margin-bottom: 10px;">Tüm hisseleri fibonacci seviyelerine göre tara</div>
            <div style="display: flex; gap: 4px; justify-content: center; flex-wrap: wrap;">
                <span style="background: rgba(41,98,255,0.1); color: #58a6ff; padding: 2px 6px; 
                             border-radius: 4px; font-size: 9px;">Breakout</span>
                <span style="background: rgba(41,98,255,0.1); color: #58a6ff; padding: 2px 6px; 
                             border-radius: 4px; font-size: 9px;">Filtre</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Aç →", key="tarama_btn", use_container_width=True):
        st.switch_page("pages/02_Tarama.py")

with col3:
    st.markdown("""
        <div class="menu-card">
            <div style="font-size: 32px; margin-bottom: 8px;">📱</div>
            <div style="font-size: 15px; font-weight: 700; color: #f0f6fc; margin-bottom: 5px;">Mobil İzle</div>
            <div style="font-size: 10px; color: #8b949e; margin-bottom: 10px;">Mobil cihazdan canlı piyasa takibi</div>
            <div style="display: flex; gap: 4px; justify-content: center; flex-wrap: wrap;">
                <span style="background: rgba(41,98,255,0.1); color: #58a6ff; padding: 2px 6px; 
                             border-radius: 4px; font-size: 9px;">Canlı</span>
                <span style="background: rgba(41,98,255,0.1); color: #58a6ff; padding: 2px 6px; 
                             border-radius: 4px; font-size: 9px;">Favori</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Aç →", key="mobil_btn", use_container_width=True):
        st.switch_page("pages/04_Admin_Mobil.py")

st.markdown('</div></div>', unsafe_allow_html=True)

# ==========================================
# 9. FOOTER
# ==========================================
st.markdown("""
    <div style="text-align: center; padding: 20px; color: #6e7681; font-size: 10px; margin-top: 20px;">
        <div style="width: 50px; height: 1px; background: linear-gradient(90deg, transparent, #58a6ff, transparent); 
                    margin: 0 auto 8px auto;"></div>
        <p>© 2026 Ercan USLU - Hisse Analiz | eutaborsa.com</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 10. SIDEBAR (Kompakt) 
# ==========================================
st.sidebar.markdown("### 🚀 Hızlı Erişim")
st.sidebar.page_link("pages/01_Grafik.py", label="📊 Grafik Analizi")
st.sidebar.page_link("pages/02_Tarama.py", label="🔍 Fibonacci Tarama")
st.sidebar.page_link("pages/04_Admin_Mobil.py", label="📱 Mobil İzle")
st.sidebar.markdown("---")
st.sidebar.caption("🔗 **eutaborsa.com**\n\nTek port, çoklu sayfa yapısı")

# ==========================================
# 11. PRERENDER READY (Google indeksleme için)
# ==========================================
st.html("""
    <script>
        // Uygulama yüklendiğinde prerenderReady = true yap
        (function() {
            var checkReady = function() {
                if (document.readyState === 'complete') {
                    setTimeout(function() {
                        window.prerenderReady = true;
                        console.log('✅ Prerender ready: true');
                    }, 1500);
                } else {
                    setTimeout(checkReady, 100);
                }
            };
            checkReady();
        })();
    </script>
""")