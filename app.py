import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="Hisse Veri Analizi", layout="wide")

st.markdown("""
    <style>
        .main > div { padding-top: 0; }
        .stSidebar { background: #0d1117; }
        .stButton>button { width: 100%; }
        div[data-testid="stSidebarNav"] { display: none; }
        .block-container { max-width: 100%; padding: 0; margin: 0; }
        iframe { width: 100%; border: none; }
        .debug-section { border-top: 1px dashed rgba(48, 54, 61, 0.5); padding-top: 10px; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

dosya_yolu = r"C:\MTX_ASC_DATA\TumVeriler_1200.csv"

# CACHE TEMIZLEME BUTONU
if st.sidebar.button("🧹 Cache Temizle", type="secondary"):
    st.cache_data.clear()
    st.sidebar.success("Cache temizlendi!")


@st.cache_data(ttl=300)
def veri_yukle():
    df = pd.read_csv(dosya_yolu, sep=";", header=None, engine='c', on_bad_lines='skip',
                     names=["HisseAdi", "Tarih", "Acilis", "Yuksek", "Dusuk", "Kapanis", "Ortalama", "Hacim", "Tutar"],
                     encoding='utf-8')
    
    df["HisseAdi"] = df["HisseAdi"].astype(str).str.upper().str.strip()
    
    sayisal_kolonlar = ["Acilis", "Yuksek", "Dusuk", "Kapanis", "Ortalama", "Hacim", "Tutar"]
    for kolon in sayisal_kolonlar:
        df[kolon] = df[kolon].astype(str).str.replace(',', '.', regex=False)
        df[kolon] = pd.to_numeric(df[kolon], errors='coerce')
    
    df["Tarih"] = pd.to_datetime(df["Tarih"], errors='coerce')
    df = df.dropna(subset=["HisseAdi", "Tarih"])
    df = df[df["HisseAdi"].astype(str).str.strip() != ""]
    df = df.sort_values(["HisseAdi", "Tarih"], ascending=[True, True])
    df = df.reset_index(drop=True)
    
    df["Onceki_Kapanis"] = df.groupby("HisseAdi")["Kapanis"].shift(1)
    df["Renk"] = "gray"
    df.loc[df["Kapanis"] > df["Onceki_Kapanis"], "Renk"] = "green"
    df.loc[df["Kapanis"] < df["Onceki_Kapanis"], "Renk"] = "red"
    df["Degisim_Yuzde"] = ((df["Kapanis"] - df["Onceki_Kapanis"]) / df["Onceki_Kapanis"] * 100).round(2)
    df["Yil"] = df["Tarih"].dt.isocalendar().year
    df["Hafta"] = df["Tarih"].dt.isocalendar().week
    df["Ay"] = df["Tarih"].dt.month
    return df

def hesapla_rsi(seri, period=14):
    delta = seri.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def hesapla_bollinger(seri, period=20, std_dev=2):
    middle = seri.rolling(window=period).mean()
    std = seri.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def hesapla_sma(hisse_df, gun):
    if gun is None or gun < 1:
        return None
    return hisse_df.groupby("HisseAdi")["Kapanis"].transform(lambda x: x.rolling(window=int(gun)).mean())
def hesapla_dema(seri, period):
    """Double Exponential Moving Average"""
    ema1 = seri.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    dema = 2 * ema1 - ema2
    return dema


def hesapla_fibonacci(df, period=50):
    df = df.reset_index(drop=True)
    zirve = df["Yuksek"].rolling(window=period, min_periods=period).max()
    dip = df["Dusuk"].rolling(window=period, min_periods=period).min()
    fib_0236 = zirve - (zirve - dip) * 0.236
    fib_0382 = zirve - (zirve - dip) * 0.382
    fib_0500 = zirve - (zirve - dip) * 0.500
    fib_0618 = zirve - (zirve - dip) * 0.618
    fib_0764 = zirve - (zirve - dip) * 0.764
    return zirve, dip, fib_0236, fib_0382, fib_0500, fib_0618, fib_0764

def hesapla_haftalik(df):
    df = df.copy()
    haftalik = df.groupby(['HisseAdi', 'Yil', 'Hafta']).agg({
        'Tarih': 'first', 'Acilis': 'first', 'Yuksek': 'max',
        'Dusuk': 'min', 'Kapanis': 'last', 'Hacim': 'sum', 'Ortalama': 'mean'
    }).reset_index()
    haftalik = haftalik.sort_values(['HisseAdi', 'Tarih']).reset_index(drop=True)
    haftalik["Onceki_Kapanis"] = haftalik.groupby("HisseAdi")["Kapanis"].shift(1)
    haftalik["Degisim_Yuzde"] = ((haftalik["Kapanis"] - haftalik["Onceki_Kapanis"]) / haftalik["Onceki_Kapanis"] * 100).round(2)
    return haftalik

def hesapla_aylik(df):
    df = df.copy()
    aylik = df.groupby(['HisseAdi', 'Yil', 'Ay']).agg({
        'Tarih': 'first', 'Acilis': 'first', 'Yuksek': 'max',
        'Dusuk': 'min', 'Kapanis': 'last', 'Hacim': 'sum', 'Ortalama': 'mean'
    }).reset_index()
    aylik = aylik.sort_values(['HisseAdi', 'Tarih']).reset_index(drop=True)
    aylik["Onceki_Kapanis"] = aylik.groupby("HisseAdi")["Kapanis"].shift(1)
    aylik["Degisim_Yuzde"] = ((aylik["Kapanis"] - aylik["Onceki_Kapanis"]) / aylik["Onceki_Kapanis"] * 100).round(2)
    return aylik

# Session state başlatma
if 'indicators' not in st.session_state:
    st.session_state.indicators = []
if 'zaman_dilimi' not in st.session_state:
    st.session_state.zaman_dilimi = "Gunluk"
if 'secili_hisse' not in st.session_state:
    st.session_state.secili_hisse = None

df = veri_yukle()
hisse_listesi = sorted(df["HisseAdi"].unique())

if st.session_state.secili_hisse is None or st.session_state.secili_hisse not in hisse_listesi:
    st.session_state.secili_hisse = hisse_listesi[0]

# Sayfa düzeni
st.markdown("""
    <div style="background-color: #161b22; padding: 8px; border-bottom: 1px solid #30363d; text-align: center;">
        <span style="color: #8b949e; font-size: 11px; margin-right: 10px;">Zaman Dilimi:</span>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
with col2:
    gunluk_btn = st.button("GÜNLÜK", type="primary" if st.session_state.zaman_dilimi == "Gunluk" else "secondary", use_container_width=True)
with col3:
    haftalik_btn = st.button("HAFTALIK", type="primary" if st.session_state.zaman_dilimi == "Haftalik" else "secondary", use_container_width=True)
with col4:
    aylik_btn = st.button("AYLIK", type="primary" if st.session_state.zaman_dilimi == "Aylik" else "secondary", use_container_width=True)

if gunluk_btn:
    st.session_state.zaman_dilimi = "Gunluk"
    st.rerun()
elif haftalik_btn:
    st.session_state.zaman_dilimi = "Haftalik"
    st.rerun()
elif aylik_btn:
    st.session_state.zaman_dilimi = "Aylik"
    st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Hisse Seçimi")
    hisse_index = hisse_listesi.index(st.session_state.secili_hisse)
    hisse_secim = st.selectbox("", hisse_listesi, index=hisse_index, label_visibility="collapsed", key="hisse_selectbox")
    if hisse_secim != st.session_state.secili_hisse:
        st.session_state.secili_hisse = hisse_secim
        st.rerun()
    
    # İndikatör sayıları
    sma_count = len([i for i in st.session_state.indicators if i["tip"] == "SMA"])
    bb_count = len([i for i in st.session_state.indicators if i["tip"] == "BB"])
    rsi_count = len([i for i in st.session_state.indicators if i["tip"] == "RSI"])
    fib_count = len([i for i in st.session_state.indicators if i["tip"] == "FIB"])
    toplam_count = len(st.session_state.indicators)
    
    st.markdown("---")
    st.markdown("### 📈 İndikatörler")
    st.caption(f"SMA:{sma_count}/3 | BB:{bb_count}/1 | RSI:{rsi_count}/3 | FIB:{fib_count}/1 | Toplam:{toplam_count}")
    
    # İndikatör ekleme
    # DEMA sayısı kontrol (max 1 çift DEMA = 2 çizgi)
    dema_count = len([i for i in st.session_state.indicators if i["tip"] == "DEMA"])

    eklenebilir_tipler = []
    if sma_count < 3:
        eklenebilir_tipler.append("SMA")
    if dema_count < 1:  # Sadece 1 DEMA çifti eklenebilir
        eklenebilir_tipler.append("DEMA")
    if bb_count < 1:
        eklenebilir_tipler.append("Bollinger Bands")
    if rsi_count < 3:
        eklenebilir_tipler.append("RSI")
    if fib_count < 1:
        eklenebilir_tipler.append("Fibonacci")
    
    if len(eklenebilir_tipler) > 0 and toplam_count < 4:
        with st.expander("➕ İndikatör Ekle", expanded=False):
            ind_tipi = st.selectbox("Tip", eklenebilir_tipler)
            
            if ind_tipi == "SMA":
                mevcut_sma_periyotlar = [i["periyot"] for i in st.session_state.indicators if i["tip"] == "SMA"]
                periyot = st.number_input("Periyot", 5, 200, 20)
                if periyot in mevcut_sma_periyotlar:
                    st.warning(f"SMA{periyot} zaten ekli!")
                else:
                    renk = st.color_picker("Renk", "#00ff00")
                    kalinlik = st.slider("Kalınlık", 0.5, 3.0, 1.5)
                    onem = st.selectbox("Önem Sırası", [1, 2, 3], index=min(sma_count, 2))
                    opaklik = st.slider("Opaklık", 0.3, 1.0, 1.0)
                    if st.button("Ekle", type="primary"):
                        st.session_state.indicators.append({
                            "tip": "SMA", "periyot": int(periyot), "renk": renk,
                            "kalinlik": kalinlik, "onem": int(onem), "opaklik": opaklik
                        })
                        st.rerun()
                    
            elif ind_tipi == "Bollinger Bands":
                periyot = st.number_input("Periyot", 5, 100, 20)
                sapma = st.slider("Sapma", 0.5, 4.0, 2.0)
                kalinlik = st.slider("Kalınlık", 0.5, 3.0, 1.0)
                onem = st.selectbox("Önem Sırası", [1, 2, 3], index=0)
                opaklik = st.slider("Opaklık", 0.3, 1.0, 1.0)
                if st.button("Ekle", type="primary"):
                    st.session_state.indicators.append({
                        "tip": "BB", "periyot": int(periyot), "sapma": float(sapma),
                        "kalinlik": kalinlik, "onem": int(onem), "opaklik": opaklik
                    })
                    st.rerun()
                    
            elif ind_tipi == "DEMA":
                st.markdown("**Hızlı DEMA (Küçük Periyot)**")
                hizli_periyot = st.number_input("Hızlı Periyot", 2, 50, 9, key="dema_hizli")
                hizli_renk = st.color_picker("Hızlı Renk", "#FFFFFF", key="dema_hizli_renk")  # Beyaz default

                st.markdown("**Yavaş DEMA (Büyük Periyot)**")
                yavas_periyot = st.number_input("Yavaş Periyot", 5, 200, 21, key="dema_yavas")
                yavas_renk = st.color_picker("Yavaş Renk", "#0080FF", key="dema_yavas_renk")  # Mavi default

                if hizli_periyot >= yavas_periyot:
                    st.warning("Hızlı periyot yavaştan küçük olmalı!")
                else:
                    kalinlik = st.slider("Kalınlık", 0.5, 3.0, 1.5)
                    onem = st.selectbox("Önem Sırası", [1, 2, 3], index=0)
                    if st.button("Ekle", type="primary"):
                        # İki DEMA çizgisi ekle
                        st.session_state.indicators.append({
                            "tip": "DEMA", "periyot": int(hizli_periyot), "renk": hizli_renk,
                            "kalinlik": kalinlik, "onem": int(onem), "hizli": True
                        })
                        st.session_state.indicators.append({
                            "tip": "DEMA", "periyot": int(yavas_periyot), "renk": yavas_renk,
                            "kalinlik": kalinlik, "onem": int(onem), "hizli": False
                        })
                        st.rerun()

            elif ind_tipi == "RSI":
                mevcut_rsi_periyotlar = [i["periyot"] for i in st.session_state.indicators if i["tip"] == "RSI"]
                periyot = st.number_input("Periyot", 2, 50, 14)
                if periyot in mevcut_rsi_periyotlar:
                    st.warning(f"RSI{periyot} zaten ekli!")
                else:
                    onem = st.selectbox("Önem Sırası", [1, 2, 3], index=min(rsi_count, 2))
                    if st.button("Ekle", type="primary"):
                        st.session_state.indicators.append({
                            "tip": "RSI", "periyot": int(periyot), "onem": int(onem)
                        })
                        st.rerun()
            
            elif ind_tipi == "Fibonacci":
                periyot = st.number_input("FIBO Period (HHV/LLV)", 1, 1000, 50)
                onem = st.selectbox("Önem Sırası", [1, 2, 3], index=0)
                if st.button("Ekle", type="primary"):
                    st.session_state.indicators.append({
                        "tip": "FIB", "periyot": int(periyot), "onem": int(onem)
                    })
                    st.rerun()
    
    # Mevcut indikatörleri göster
    if st.session_state.indicators:
        st.markdown("---")
        for i, ind in enumerate(st.session_state.indicators):
            col1, col2 = st.columns([3, 1])
            with col1:
                if ind["tip"] == "SMA":
                    st.markdown(f"**SMA{ind['periyot']}** (Önem: {ind['onem']})")
                elif ind["tip"] == "BB":
                    st.markdown(f"**BB({ind['periyot']},{ind['sapma']})** (Önem: {ind['onem']})")
                elif ind["tip"] == "RSI":
                    st.markdown(f"**RSI({ind['periyot']})** (Önem: {ind['onem']})")
                elif ind["tip"] == "FIB":
                    st.markdown(f"**FIB({ind['periyot']})** (Önem: {ind['onem']})")
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.indicators.pop(i)
                    st.rerun()
        
        if st.button("Tümünü Temizle", type="secondary"):
            st.session_state.indicators = []
            st.rerun()
    
    # Debug modu
    st.markdown('<div class="debug-section">', unsafe_allow_html=True)
    gelismis_mod = st.checkbox("⚙️ Gelişmiş Mod", value=False)
    if gelismis_mod:
        st.caption("Debug bilgileri:")
        st.write(f"Toplam hisse: {len(hisse_listesi)}")
        st.write(f"Seçili hisse: {st.session_state.secili_hisse}")
        st.write(f"İndikatör sayısı: {len(st.session_state.indicators)}")
        if st.button("🔄 Cache Temizle"):
            st.cache_data.clear()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Veri hazırlama
zaman_dilimi = st.session_state.zaman_dilimi
hisse_secim = st.session_state.secili_hisse

if zaman_dilimi == "Haftalik":
    df_secili = hesapla_haftalik(df[df["HisseAdi"] == hisse_secim].copy())
elif zaman_dilimi == "Aylik":
    df_secili = hesapla_aylik(df[df["HisseAdi"] == hisse_secim].copy())
else:
    df_secili = df[df["HisseAdi"] == hisse_secim].copy()

hisse_tum_veri = df_secili.copy()

# DÜZELTME: Grafik için eskiden yeniye sırala (LightweightCharts böyle ister)
hisse_tum_veri = hisse_tum_veri.sort_values("Tarih", ascending=True)

if len(hisse_tum_veri) > 0:
    # İndikatörleri hesapla
    for ind in st.session_state.indicators:
        if ind["tip"] == "SMA":
            hisse_tum_veri[f"SMA_{ind['periyot']}"] = hesapla_sma(hisse_tum_veri, ind['periyot'])
        elif ind["tip"] == "DEMA":
            hisse_tum_veri[f"DEMA_{ind['periyot']}"] = hesapla_dema(hisse_tum_veri["Kapanis"], ind['periyot'])
        elif ind["tip"] == "BB":
            upper, middle, lower = hesapla_bollinger(hisse_tum_veri["Kapanis"], ind['periyot'], ind['sapma'])
            hisse_tum_veri[f"BB_U_{ind['periyot']}"] = upper
            hisse_tum_veri[f"BB_M_{ind['periyot']}"] = middle
            hisse_tum_veri[f"BB_L_{ind['periyot']}"] = lower
        elif ind["tip"] == "RSI":
            hisse_tum_veri[f"RSI_{ind['periyot']}"] = hesapla_rsi(hisse_tum_veri["Kapanis"], ind['periyot'])
        elif ind["tip"] == "FIB":
            zirve, dip, f0236, f0382, f0500, f0618, f0764 = hesapla_fibonacci(hisse_tum_veri, ind['periyot'])
            hisse_tum_veri[f"FIB_ZIRVE_{ind['periyot']}"] = zirve
            hisse_tum_veri[f"FIB_DIP_{ind['periyot']}"] = dip
            hisse_tum_veri[f"FIB_0236_{ind['periyot']}"] = f0236
            hisse_tum_veri[f"FIB_0382_{ind['periyot']}"] = f0382
            hisse_tum_veri[f"FIB_0500_{ind['periyot']}"] = f0500
            hisse_tum_veri[f"FIB_0618_{ind['periyot']}"] = f0618
            hisse_tum_veri[f"FIB_0764_{ind['periyot']}"] = f0764
    
    # JSON verileri hazırla
    fiyat_data = []
    ind_data = {}
    
    for ind in st.session_state.indicators:
        key = f"{ind['tip']}_{ind['periyot']}"
        ind_data[key] = []
    
    for _, row in hisse_tum_veri.iterrows():
        time_str = row['Tarih'].strftime('%Y-%m-%d')
        
        item = {
            'time': time_str,
            'open': round(row['Acilis'], 2),
            'high': round(row['Yuksek'], 2),
            'low': round(row['Dusuk'], 2),
            'close': round(row['Kapanis'], 2),
            'degisim': round(row.get('Degisim_Yuzde', 0), 2) if pd.notna(row.get('Degisim_Yuzde')) else 0
        }
        fiyat_data.append(item)
        
        for ind in st.session_state.indicators:
            if ind["tip"] == "SMA":
                key = f"SMA_{ind['periyot']}"
                val = row.get(key)
                if pd.notna(val):
                    ind_data[key].append({
                        'time': time_str, 'value': round(float(val), 2),
                        'renk': ind['renk'], 'kalinlik': ind['kalinlik'],
                        'onem': ind['onem'], 'opaklik': ind['opaklik']
                    })
            elif ind["tip"] == "DEMA":
                key = f"DEMA_{ind['periyot']}"
                val = row.get(key)
                if pd.notna(val):
                    ind_data[key].append({
                        'time': time_str, 'value': round(float(val), 2),
                        'renk': ind['renk'], 'kalinlik': ind['kalinlik'],
                        'onem': ind['onem']
                    })
            elif ind["tip"] == "DEMA":
                key = f"DEMA_{ind['periyot']}"
                val = row.get(key)
                if pd.notna(val):
                    ind_data[key].append({
                        'time': time_str, 'value': round(float(val), 2),
                        'renk': ind['renk'], 'kalinlik': ind['kalinlik'],
                        'onem': ind['onem']
                    })
            elif ind["tip"] == "RSI":
                key = f"RSI_{ind['periyot']}"
                val = row.get(key)
                if pd.notna(val):
                    ind_data[key].append({
                        'time': time_str, 'value': round(float(val), 2),
                        'onem': ind['onem']
                    })
    
    bb_data = []
    for ind in st.session_state.indicators:
        if ind["tip"] == "BB":
            periyot_str = str(int(ind['periyot']))
            for _, row in hisse_tum_veri.iterrows():
                upper_val = row.get(f"BB_U_{ind['periyot']}")
                middle_val = row.get(f"BB_M_{ind['periyot']}")
                lower_val = row.get(f"BB_L_{ind['periyot']}")
                
                if pd.notna(upper_val) and pd.notna(middle_val) and pd.notna(lower_val):
                    bb_data.append({
                        'time': row['Tarih'].strftime('%Y-%m-%d'),
                        'upper': round(float(upper_val), 2),
                        'middle': round(float(middle_val), 2),
                        'lower': round(float(lower_val), 2),
                        'kalinlik': float(ind['kalinlik']),
                        'onem': int(ind['onem']),
                        'opaklik': float(ind['opaklik']),
                        'periyot': periyot_str
                    })
    
    fib_data = []
    for ind in st.session_state.indicators:
        if ind["tip"] == "FIB":
            periyot_str = str(int(ind['periyot']))
            for _, row in hisse_tum_veri.iterrows():
                zirve_val = row.get(f"FIB_ZIRVE_{ind['periyot']}")
                dip_val = row.get(f"FIB_DIP_{ind['periyot']}")
                f0236_val = row.get(f"FIB_0236_{ind['periyot']}")
                f0382_val = row.get(f"FIB_0382_{ind['periyot']}")
                f0500_val = row.get(f"FIB_0500_{ind['periyot']}")
                f0618_val = row.get(f"FIB_0618_{ind['periyot']}")
                f0764_val = row.get(f"FIB_0764_{ind['periyot']}")
                
                if pd.notna(zirve_val) and pd.notna(dip_val):
                    fib_data.append({
                        'time': row['Tarih'].strftime('%Y-%m-%d'),
                        'zirve': round(float(zirve_val), 2),
                        'dip': round(float(dip_val), 2),
                        'f0236': round(float(f0236_val), 2) if pd.notna(f0236_val) else None,
                        'f0382': round(float(f0382_val), 2) if pd.notna(f0382_val) else None,
                        'f0500': round(float(f0500_val), 2) if pd.notna(f0500_val) else None,
                        'f0618': round(float(f0618_val), 2) if pd.notna(f0618_val) else None,
                        'f0764': round(float(f0764_val), 2) if pd.notna(f0764_val) else None,
                        'periyot': periyot_str,
                        'onem': int(ind['onem'])
                    })
    
    fiyat_json = json.dumps(fiyat_data)
    ind_json = json.dumps(ind_data)
    bb_json = json.dumps(bb_data)
    fib_json = json.dumps(fib_data)
    
    # HTML oluşturma
    ind_html = ""
    
    # SARİ FİYAT PANELİ
    price_panel_html = """
    <div id="price-panel">
        <div class="price-row">
            <span class="price-label">Tarih</span>
            <span class="price-value" id="info-tarih">-</span>
        </div>
        <div class="price-row">
            <span class="price-label">Açılış</span>
            <span class="price-value" id="info-acilis">-</span>
        </div>
        <div class="price-row">
            <span class="price-label">Yüksek</span>
            <span class="price-value" id="info-yuksek">-</span>
        </div>
        <div class="price-row">
            <span class="price-label">Düşük</span>
            <span class="price-value" id="info-dusuk">-</span>
        </div>
        <div class="price-row">
            <span class="price-label">Kapanış</span>
            <span class="price-value" id="info-kapanis">-</span>
        </div>
        <div class="price-row">
            <span class="price-label">%Değişim</span>
            <span class="price-value" id="info-degisim">-</span>
        </div>
    </div>
    """
    
    # İNDİKATÖR PANELLERİ
    dema_list = [i for i in st.session_state.indicators if i["tip"] == "DEMA"]
    sma_list = [i for i in st.session_state.indicators if i["tip"] == "SMA"]
    if sma_list:
        sma_rows = ""
        for ind in sorted(sma_list, key=lambda x: x['periyot']):
            sma_rows += f'<div class="ind-row"><span>SMA{ind["periyot"]}:</span><span id="sma-{ind["periyot"]}">-</span></div>'
        ind_html += f"""
        <div class="ind-panel sma-panel">
            <div class="ind-title">SMA ({len(sma_list)})</div>
            {sma_rows}
        </div>
        """
    
    bb_list = [i for i in st.session_state.indicators if i["tip"] == "BB"]
    if bb_list:
        ind = bb_list[0]
        periyot_str = str(int(ind['periyot']))
        ind_html += f"""
        <div class="ind-panel bb-panel">
            <div class="ind-title">Bollinger Bands ({periyot_str},{ind['sapma']})</div>
            <div class="ind-row"><span>Üst:</span><span id="bb-u-{periyot_str}">-</span></div>
            <div class="ind-row"><span>Orta:</span><span id="bb-m-{periyot_str}">-</span></div>
            <div class="ind-row"><span>Alt:</span><span id="bb-l-{periyot_str}">-</span></div>
        </div>
        """
    
    rsi_list = [i for i in st.session_state.indicators if i["tip"] == "RSI"]
    if rsi_list:
        rsi_rows = ""
        for ind in sorted(rsi_list, key=lambda x: x['periyot']):
            rsi_rows += f'<div class="ind-row"><span>RSI{ind["periyot"]}:</span><span id="rsi-{ind["periyot"]}">-</span></div>'
        ind_html += f"""
        <div class="ind-panel rsi-panel">
            <div class="ind-title">RSI ({len(rsi_list)})</div>
            {rsi_rows}
        </div>
        """
    
    fib_list = [i for i in st.session_state.indicators if i["tip"] == "FIB"]
    if fib_list:
        ind = fib_list[0]
        periyot_str = str(int(ind['periyot']))
        ind_html += f"""
        <div class="ind-panel fib-panel">
            <div class="ind-title">Fibonacci ({periyot_str})</div>
            <div class="ind-row"><span>Zirve:</span><span id="fib-zirve-{periyot_str}">-</span></div>
            <div class="ind-row"><span>23.6%:</span><span id="fib-0236-{periyot_str}">-</span></div>
            <div class="ind-row"><span>38.2%:</span><span id="fib-0382-{periyot_str}">-</span></div>
            <div class="ind-row"><span>50.0%:</span><span id="fib-0500-{periyot_str}">-</span></div>
            <div class="ind-row"><span>61.8%:</span><span id="fib-0618-{periyot_str}">-</span></div>
            <div class="ind-row"><span>76.4%:</span><span id="fib-0764-{periyot_str}">-</span></div>
            <div class="ind-row"><span>Dip:</span><span id="fib-dip-{periyot_str}">-</span></div>
        </div>
        """
    
    zaman_dilimi_label = zaman_dilimi.upper()[:1]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body, html {{
                background: #000;
                font-family: 'Segoe UI', sans-serif;
                height: 480px;
                width: 100%;
                overflow: hidden;
            }}
            #chart-container {{
                position: relative;
                width: 100%;
                height: 480px;
                display: flex;
                flex-direction: column;
                background: #000;
            }}
            #container-fiyat {{
                flex: 1;
                position: relative;
                min-height: 0;
            }}
            #time-badge {{
                position: absolute;
                top: 10px;
                right: 80px;
                background: rgba(0, 255, 0, 0.2);
                color: #00ff00;
                border: 1px solid #00ff00;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 700;
                z-index: 999999;
            }}
            #fullscreen-btn {{
                position: absolute;
                top: 10px;
                right: 10px;
                width: 32px;
                height: 32px;
                background: rgba(255,255,255,0.15);
                color: #fff;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                z-index: 9999999;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            #info-panels {{
                position: absolute;
                top: 45px;
                left: 10px;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                z-index: 999999;
                pointer-events: none;
            }}
            #price-panel {{
                background: rgba(255, 193, 7, 0.95);
                color: #000;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                pointer-events: auto;
                min-width: 140px;
            }}
            .price-row {{
                display: flex;
                justify-content: space-between;
                margin: 1px 0;
            }}
            .price-label {{ opacity: 0.7; font-size: 9px; }}
            .price-value {{ font-weight: 700; font-size: 11px; }}
            .positive {{ color: #006400; }}
            .negative {{ color: #8B0000; }}
            .ind-panel {{
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 10px;
                font-weight: 600;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                pointer-events: auto;
                min-width: 120px;
            }}
            .bb-panel {{ background: rgba(255, 235, 59, 0.95); color: #000; }}
            .dema-panel {{ background: rgba(255, 182, 193, 0.95); color: #000; }}  /* Pembe */
            .sma-panel {{ background: rgba(144, 238, 144, 0.95); color: #000; }}
            .rsi-panel {{ background: rgba(173, 216, 230, 0.95); color: #000; }}
            .fib-panel {{ background: rgba(255, 165, 0, 0.95); color: #000; }}
            .ind-title {{
                font-size: 9px;
                opacity: 0.7;
                margin-bottom: 3px;
                border-bottom: 1px solid rgba(0,0,0,0.2);
                padding-bottom: 2px;
                font-weight: 700;
            }}
            .ind-row {{
                display: flex;
                justify-content: space-between;
                margin: 1px 0;
            }}
        </style>
    </head>
    <body>
        <div id="chart-container">
            <div id="time-badge">{zaman_dilimi_label}</div>
            <button id="fullscreen-btn" onclick="toggleFullscreen()" title="Tam Ekran">⛶</button>
            <div id="info-panels">
                {price_panel_html}
                {ind_html}
            </div>
            <div id="container-fiyat"></div>
        </div>
        
        <script>
            var fiyatData = {fiyat_json};
            var indData = {ind_json};
            var bbData = {bb_json};
            var fibData = {fib_json};
            var currentIndex = fiyatData.length - 1;
            
            // İndikatörleri sırala
            var sortedIndicators = [];
            for (var k in indData) {{
                if (indData[k].length > 0) {{
                    var parts = k.split('_');
                    sortedIndicators.push({{
                        key: k,
                        data: indData[k],
                        onem: indData[k][0] && indData[k][0].onem ? indData[k][0].onem : 2,
                        tip: parts[0],
                        periyot: parts[1]
                    }});
                }}
            }}
            sortedIndicators.sort(function(a, b) {{ return a.onem - b.onem; }});
            
            function toggleFullscreen() {{
                var elem = document.getElementById('chart-container');
                if (!document.fullscreenElement) {{
                    elem.requestFullscreen();
                }} else {{
                    document.exitFullscreen();
                }}
            }}
            
            // Chart oluştur
            var chart = LightweightCharts.createChart(document.getElementById('container-fiyat'), {{
                width: document.getElementById('container-fiyat').clientWidth,
                height: document.getElementById('container-fiyat').clientHeight,
                layout: {{ background: {{ type: 'solid', color: '#000000' }}, textColor: '#d1d5db' }},
                grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Magnet,
                    vertLine: {{ color: '#00ff00', width: 1, style: 2 }},
                    horzLine: {{ color: '#00ff00', width: 1, style: 2 }}
                }},
                rightPriceScale: {{ borderColor: '#374151' }},
                timeScale: {{ borderColor: '#374151', timeVisible: false }}
            }});
            
            var series = chart.addCandlestickSeries({{
                upColor: '#22c55e', downColor: '#ef4444',
                borderUpColor: '#22c55e', borderDownColor: '#ef4444',
                wickUpColor: '#22c55e', wickDownColor: '#ef4444'
            }});
            series.setData(fiyatData);
            
            // FIBONACCI ÇİZGİLERİ
            var fibPeriyots = [];
            var seenFibPeriyots = {{}};
            for (var i = 0; i < fibData.length; i++) {{
                var p = fibData[i].periyot;
                if (!seenFibPeriyots[p]) {{
                    seenFibPeriyots[p] = true;
                    fibPeriyots.push(p);
                }}
            }}
            
            for (var i = 0; i < fibPeriyots.length; i++) {{
                var periyot = fibPeriyots[i];
                var fibItems = fibData.filter(f => f.periyot === periyot);
                if (fibItems.length === 0) continue;
                var lastFib = fibItems[fibItems.length - 1];
                if (!lastFib) continue;
                
                var levels = [
                    {{ name: 'Zirve', value: lastFib.zirve, color: '#ff6b6b' }},
                    {{ name: '0236', value: lastFib.f0236, color: '#ffd93d' }},
                    {{ name: '0382', value: lastFib.f0382, color: '#6bcf7f' }},
                    {{ name: '0500', value: lastFib.f0500, color: '#4dabf7' }},
                    {{ name: '0618', value: lastFib.f0618, color: '#da77f2' }},
                    {{ name: '0764', value: lastFib.f0764, color: '#ffa94d' }},
                    {{ name: 'Dip', value: lastFib.dip, color: '#ff6b6b' }}
                ];
                
                for (var j = 0; j < levels.length; j++) {{
                    var lvl = levels[j];
                    if (!lvl.value) continue;
                    var lineData = [];
                    for (var k = 0; k < fiyatData.length; k++) {{
                        lineData.push({{ time: fiyatData[k].time, value: lvl.value }});
                    }}
                    var line = chart.addLineSeries({{
                        color: lvl.color,
                        lineWidth: 1,
                        lastValueVisible: true,
                        title: lvl.name
                    }});
                    line.setData(lineData);
                }}
            }}
            
            // SMA ve DEMA ekle
            for (var i = 0; i < sortedIndicators.length; i++) {{
                var ind = sortedIndicators[i];
                if (ind.tip === 'SMA' || ind.tip === 'DEMA') {{
                    var lineData = [];
                    for (var j = 0; j < ind.data.length; j++) {{
                        lineData.push({{ time: ind.data[j].time, value: ind.data[j].value }});
                    }}
                    var width = ind.data[0] && ind.data[0].kalinlik ? ind.data[0].kalinlik : 1.5;
                    var color = ind.data[0] && ind.data[0].renk ? ind.data[0].renk : '#ffffff';
                    var line = chart.addLineSeries({{ 
                        color: color, 
                        lineWidth: width, 
                        lastValueVisible: false
                    }});
                    line.setData(lineData);
                }}
            }}
            
            // BB ekle
            if (bbData.length > 0) {{
                var opacity = bbData[0] && bbData[0].opaklik ? bbData[0].opaklik : 1;
                var width = bbData[0] && bbData[0].kalinlik ? bbData[0].kalinlik : 1;
                var color = 'rgba(255,255,255,' + opacity + ')';
                var upperData = [], middleData = [], lowerData = [];
                for (var i = 0; i < bbData.length; i++) {{
                    upperData.push({{ time: bbData[i].time, value: bbData[i].upper }});
                    middleData.push({{ time: bbData[i].time, value: bbData[i].middle }});
                    lowerData.push({{ time: bbData[i].time, value: bbData[i].lower }});
                }}
                var bbUpper = chart.addLineSeries({{ color: color, lineWidth: width, lastValueVisible: false }});
                var bbMiddle = chart.addLineSeries({{ color: color, lineWidth: width, lastValueVisible: false }});
                var bbLower = chart.addLineSeries({{ color: color, lineWidth: width, lastValueVisible: false }});
                bbUpper.setData(upperData);
                bbMiddle.setData(middleData);
                bbLower.setData(lowerData);
            }}
            
            // Bilgi panelini güncelle
            function updateInfo(index) {{
                if (index < 0 || index >= fiyatData.length) return;
                var d = fiyatData[index];
                currentIndex = index;
                
                // Fiyat bilgileri
                var dateParts = d.time.split('-');
                document.getElementById('info-tarih').textContent = dateParts[2] + '.' + dateParts[1] + '.' + dateParts[0].substr(2);
                document.getElementById('info-acilis').textContent = d.open.toFixed(2);
                document.getElementById('info-yuksek').textContent = d.high.toFixed(2);
                document.getElementById('info-dusuk').textContent = d.low.toFixed(2);
                document.getElementById('info-kapanis').textContent = d.close.toFixed(2);
                var degisimEl = document.getElementById('info-degisim');
                var degisim = d.degisim || 0;
                degisimEl.textContent = (degisim >= 0 ? '+' : '') + degisim.toFixed(2) + '%';
                degisimEl.className = 'price-value ' + (degisim >= 0 ? 'positive' : 'negative');
                
                // BB güncelle
                var bbPeriyots = [];
                var seenPeriyots = {{}};
                for (var i = 0; i < bbData.length; i++) {{
                    var p = bbData[i].periyot;
                    if (!seenPeriyots[p]) {{
                        seenPeriyots[p] = true;
                        bbPeriyots.push(p);
                    }}
                }}
                
                bbPeriyots.forEach(function(periyot) {{
                    var bbItem = null;
                    for (var i = 0; i < bbData.length; i++) {{
                        if (bbData[i].time == d.time && bbData[i].periyot == periyot) {{
                            bbItem = bbData[i];
                            break;
                        }}
                    }}
                    var uEl = document.getElementById('bb-u-' + periyot);
                    var mEl = document.getElementById('bb-m-' + periyot);
                    var lEl = document.getElementById('bb-l-' + periyot);
                    if (bbItem) {{
                        if (uEl) uEl.textContent = bbItem.upper.toFixed(2);
                        if (mEl) mEl.textContent = bbItem.middle.toFixed(2);
                        if (lEl) lEl.textContent = bbItem.lower.toFixed(2);
                    }}
                }});
                
                // Fibonacci güncelle
                for (var p = 0; p < fibPeriyots.length; p++) {{
                    var periyot = fibPeriyots[p];
                    var fibItem = null;
                    for (var i = 0; i < fibData.length; i++) {{
                        if (fibData[i].time == d.time && fibData[i].periyot == periyot) {{
                            fibItem = fibData[i];
                            break;
                        }}
                    }}
                    var zirveEl = document.getElementById('fib-zirve-' + periyot);
                    var f0236El = document.getElementById('fib-0236-' + periyot);
                    var f0382El = document.getElementById('fib-0382-' + periyot);
                    var f0500El = document.getElementById('fib-0500-' + periyot);
                    var f0618El = document.getElementById('fib-0618-' + periyot);
                    var f0764El = document.getElementById('fib-0764-' + periyot);
                    var dipEl = document.getElementById('fib-dip-' + periyot);
                    
                    if (fibItem) {{
                        if (zirveEl) zirveEl.textContent = fibItem.zirve.toFixed(2);
                        if (f0236El && fibItem.f0236) f0236El.textContent = fibItem.f0236.toFixed(2);
                        if (f0382El && fibItem.f0382) f0382El.textContent = fibItem.f0382.toFixed(2);
                        if (f0500El && fibItem.f0500) f0500El.textContent = fibItem.f0500.toFixed(2);
                        if (f0618El && fibItem.f0618) f0618El.textContent = fibItem.f0618.toFixed(2);
                        if (f0764El && fibItem.f0764) f0764El.textContent = fibItem.f0764.toFixed(2);
                        if (dipEl) dipEl.textContent = fibItem.dip.toFixed(2);
                    }}
                }}
                
                // SMA, DEMA ve RSI güncelle
                for (var i = 0; i < sortedIndicators.length; i++) {{
                    var ind = sortedIndicators[i];
                    var item = null;
                    for (var j = 0; j < ind.data.length; j++) {{
                        if (ind.data[j].time == d.time) {{
                            item = ind.data[j];
                            break;
                        }}
                    }}
                    if (ind.tip === 'SMA') {{
                        var el = document.getElementById('sma-' + ind.periyot);
                        if (el && item) el.textContent = item.value.toFixed(2);
                    }} else if (ind.tip === 'DEMA') {{
                        var el = document.getElementById('dema-' + ind.periyot);
                        if (el && item) el.textContent = item.value.toFixed(2);
                    }} else if (ind.tip === 'RSI') {{
                        var el = document.getElementById('rsi-' + ind.periyot);
                        if (el && item) el.textContent = item.value.toFixed(2);
                    }}
                }}
            }}
            
            // Crosshare hareketi
            chart.subscribeCrosshairMove(function(param) {{
                if (!param.time) return;
                var idx = -1;
                for (var i = 0; i < fiyatData.length; i++) {{
                    if (fiyatData[i].time === param.time) {{
                        idx = i;
                        break;
                    }}
                }}
                if (idx !== -1) updateInfo(idx);
            }});
            
            // Klavye kontrolü
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'ArrowLeft' && currentIndex > 0) {{
                    e.preventDefault();
                    currentIndex--;
                    chart.setCrosshairPosition({{ x: 0, y: 0 }}, fiyatData[currentIndex].time, series);
                    updateInfo(currentIndex);
                }} else if (e.key === 'ArrowRight' && currentIndex < fiyatData.length - 1) {{
                    e.preventDefault();
                    currentIndex++;
                    chart.setCrosshairPosition({{ x: 0, y: 0 }}, fiyatData[currentIndex].time, series);
                    updateInfo(currentIndex);
                }} else if (e.key === 'F11') {{
                    e.preventDefault();
                    toggleFullscreen();
                }}
            }});
            
            window.addEventListener('resize', function() {{
                var chartEl = document.getElementById('container-fiyat');
                if (chart && chartEl) chart.applyOptions({{ width: chartEl.clientWidth, height: chartEl.clientHeight }});
            }});
            
            chart.timeScale().fitContent();
            updateInfo(currentIndex);
        </script>
    </body>
    </html>
    """
    
    components.html(html, height=480, scrolling=False)
else:
    st.error("Veri bulunamadı!")
