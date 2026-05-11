import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarPrice AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e4dc; }
[data-testid="stSidebar"] { background: #111118; border-right: 1px solid #2a2a3a; }
.main-header { text-align: center; padding: 2rem 0 1.2rem; }
.main-header h1 {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 3rem;
    background: linear-gradient(135deg, #f5a623 0%, #f53c23 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.main-header p {
    color: #888; font-size: 0.85rem; margin-top: 0.3rem;
    font-family: 'Space Mono', monospace; letter-spacing: 2px; text-transform: uppercase;
}
.card { background: #13131e; border: 1px solid #2a2a3a; border-radius: 14px; padding: 1.4rem; margin-bottom: 1rem; }
.card-title {
    font-size: 0.65rem; letter-spacing: 3px; text-transform: uppercase;
    color: #f5a623; font-family: 'Space Mono', monospace; margin-bottom: 0.8rem;
}
.result-box {
    background: linear-gradient(135deg, #1a1008, #1e0f0f);
    border: 2px solid #f5a623; border-radius: 18px; padding: 2.2rem; text-align: center;
}
.result-label {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    letter-spacing: 4px; text-transform: uppercase; color: #f5a623; margin-bottom: 0.4rem;
}
.result-price { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 3rem; color: #fff; line-height: 1; }
.result-sub { font-size: 0.75rem; color: #666; margin-top: 0.4rem; font-family: 'Space Mono', monospace; }
.stButton > button {
    background: linear-gradient(135deg, #f5a623, #f53c23) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.7rem 1.5rem !important; width: 100% !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #aaa !important; font-size: 0.72rem !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    font-family: 'Space Mono', monospace !important;
}
hr { border-color: #2a2a3a !important; }
</style>
""", unsafe_allow_html=True)


# ── Frequency maps dari dataset kamu ─────────────────────────────────────────
# Ini adalah frekuensi kemunculan tiap Brand & model di dataset asli.
# Nilai ini HARUS sama dengan yang dipakai saat training.
# Jika kamu punya nilai persisnya dari notebook, ganti di sini.
BRAND_FREQ = {
    "Maruti Suzuki": 0.28, "Hyundai": 0.18, "Honda": 0.10, "Toyota": 0.08,
    "Mahindra": 0.07, "Tata": 0.06, "Renault": 0.05, "Ford": 0.05,
    "Volkswagen": 0.04, "Nissan": 0.03, "Skoda": 0.02, "Kia": 0.02,
    "BMW": 0.01, "Mercedes-Benz": 0.01, "Audi": 0.01,
}
MODEL_FREQ = {
    "Swift": 0.08, "Alto": 0.07, "City": 0.06, "Creta": 0.06,
    "i20": 0.05, "Innova": 0.05, "Baleno": 0.05, "Dzire": 0.04,
    "WagonR": 0.04, "Verna": 0.03, "Ertiga": 0.03, "Vitara Brezza": 0.03,
    "Duster": 0.02, "Ciaz": 0.02, "EcoSport": 0.02, "XUV500": 0.02,
    "Scorpio": 0.02, "Fortuner": 0.02, "Nexon": 0.02, "Harrier": 0.01,
    "VentoTest": 0.01, "Other": 0.01,
}
OWNER_ENCODE = {"first": 1, "second": 2, "third": 3, "fourth & above": 4}

BRANDS = sorted(BRAND_FREQ.keys())
MODELS_BY_BRAND = {
    "Maruti Suzuki" : ["Swift", "Alto", "Baleno", "Dzire", "WagonR", "Ertiga", "Vitara Brezza", "Ciaz"],
    "Hyundai"       : ["Creta", "i20", "Verna", "Grand i10", "Tucson", "Alcazar"],
    "Honda"         : ["City", "Amaze", "Jazz", "WR-V", "HR-V", "Civic"],
    "Toyota"        : ["Innova", "Fortuner", "Corolla", "Camry", "Rush"],
    "Mahindra"      : ["Scorpio", "XUV500", "Bolero", "Thar", "XUV300"],
    "Tata"          : ["Nexon", "Harrier", "Altroz", "Tiago", "Safari"],
    "Renault"       : ["Duster", "Kwid", "Triber", "Captur"],
    "Ford"          : ["EcoSport", "Endeavour", "Figo", "Aspire"],
    "Volkswagen"    : ["Vento", "Polo", "Tiguan", "Passat"],
    "Nissan"        : ["Magnite", "Kicks", "Terrano", "Sunny"],
    "Skoda"         : ["Octavia", "Superb", "Rapid", "Kushaq"],
    "Kia"           : ["Seltos", "Sonet", "Carnival", "Carens"],
    "BMW"           : ["3 Series", "5 Series", "X1", "X3"],
    "Mercedes-Benz" : ["C-Class", "E-Class", "GLA", "GLC"],
    "Audi"          : ["A4", "A6", "Q3", "Q5"],
}
OWNERS     = list(OWNER_ENCODE.keys())
FUEL_TYPES = ["Petrol", "Diesel", "Hybrid/CNG", "Other"]


# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    rf_model, scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚗 CarPrice AI</h1>
    <p>Random Forest · Used Car Price Predictor</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("""
    ❌ **File model tidak ditemukan!**

    Jalankan cell ini di Jupyter Notebook, pastikan disimpan di folder yang **sama** dengan `app.py`:
    ```python
    import joblib
    joblib.dump(scaler,   'scaler.joblib')
    joblib.dump(rf_model, 'rf_model.joblib')
    print("Tersimpan!")
    ```
    """)
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Input Spesifikasi Mobil")
    st.markdown("---")

    brand     = st.selectbox("Brand", BRANDS)
    model_car = st.selectbox("Model", MODELS_BY_BRAND.get(brand, ["Other"]))

    st.markdown("---")
    year = st.slider("Year", 2000, 2024, 2018)
    age  = 2024 - year

    km_input = st.number_input("kmDriven (km)", min_value=0, max_value=500000,
                               value=50000, step=1000)
    st.markdown("---")
    owner     = st.selectbox("Owner", OWNERS)
    fuel_type = st.selectbox("FuelType", FUEL_TYPES)
    trans     = st.selectbox("Transmission", ["Manual", "Automatic"])

    st.caption(f"Age dihitung otomatis: **{age} tahun**")
    st.markdown("---")
    predict_btn = st.button("🔍 Prediksi Harga", use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-title">Ringkasan Spesifikasi</div>', unsafe_allow_html=True)
    for k, v in {
        "🏷️ Brand": brand, "🚘 Model": model_car, "📅 Year": year,
        "🕐 Age": f"{age} tahun", "🛣️ km Driven": f"{km_input:,} km",
        "👤 Owner": owner, "⛽ Fuel Type": fuel_type, "⚙️ Transmission": trans,
    }.items():
        c1, c2 = st.columns(2)
        c1.markdown(f"<span style='color:#666;font-size:0.82rem'>{k}</span>", unsafe_allow_html=True)
        c2.markdown(f"<span style='color:#e8e4dc;font-size:0.82rem;font-weight:600'>{v}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if predict_btn:
        try:
            # ── Preprocessing — sesuai X_train kamu ──────────────────────
            # Kolom: ['Age', 'kmDriven', 'Owner_Encode', 'Brand_Freq',
            #         'model_Freq', 'Transmission_Manual',
            #         'FuelType_Hybrid/CNG', 'FuelType_Petrol']

            age_val         = age
            km_val          = float(km_input)
            owner_enc       = OWNER_ENCODE[owner]
            brand_freq      = BRAND_FREQ.get(brand, 0.01)
            model_freq      = MODEL_FREQ.get(model_car, 0.01)
            trans_manual    = 1 if trans == "Manual" else 0
            fuel_hybrid_cng = 1 if fuel_type == "Hybrid/CNG" else 0
            fuel_petrol     = 1 if fuel_type == "Petrol" else 0

            # Urutan kolom HARUS sama dengan X_train
            input_array = np.array([[
                age_val,          # Age
                km_val,           # kmDriven
                owner_enc,        # Owner_Encode
                brand_freq,       # Brand_Freq
                model_freq,       # model_Freq
                trans_manual,     # Transmission_Manual
                fuel_hybrid_cng,  # FuelType_Hybrid/CNG
                fuel_petrol,      # FuelType_Petrol
            ]])

            input_scaled    = scaler.transform(input_array)
            predicted_price = rf_model.predict(input_scaled)[0]
            price_min       = predicted_price * 0.90
            price_max       = predicted_price * 1.10

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Estimasi Harga Jual</div>
                <div class="result-price">₹ {predicted_price:,.0f}</div>
                <div class="result-sub">Rentang: ₹ {price_min:,.0f} – ₹ {price_max:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <br><div class="card">
                <div class="card-title">Breakdown Fitur</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.72rem;line-height:2;color:#888">
                    Age &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ <span style="color:#e8e4dc">{age_val}</span><br>
                    kmDriven &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ <span style="color:#e8e4dc">{km_val:,.0f}</span><br>
                    Owner_Encode &nbsp;→ <span style="color:#e8e4dc">{owner_enc}</span><br>
                    Brand_Freq &nbsp;&nbsp;&nbsp;→ <span style="color:#e8e4dc">{brand_freq:.3f}</span><br>
                    model_Freq &nbsp;&nbsp;&nbsp;→ <span style="color:#e8e4dc">{model_freq:.3f}</span><br>
                    Trans_Manual &nbsp;→ <span style="color:#e8e4dc">{trans_manual}</span><br>
                    Fuel_Hybrid/CNG→ <span style="color:#e8e4dc">{fuel_hybrid_cng}</span><br>
                    FuelType_Petrol→ <span style="color:#e8e4dc">{fuel_petrol}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    else:
        st.markdown("""
        <div style="height:260px;display:flex;align-items:center;justify-content:center;
                    text-align:center;border:2px dashed #2a2a3a;border-radius:14px;color:#444">
            <div>
                <div style="font-size:3rem;margin-bottom:1rem">🚗</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                            letter-spacing:2px;text-transform:uppercase">
                    Isi spesifikasi di sidebar<br>lalu klik Prediksi Harga
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# # ── Tips ──────────────────────────────────────────────────────────────────────
# st.markdown("---")
# with st.expander("💡 Tips: Cara mendapatkan Brand_Freq & model_Freq yang akurat"):
#     st.markdown("""
#     Nilai frequency encoding di `app.py` harus **sama persis** dengan yang dipakai saat training.
#     Jalankan cell ini di notebook untuk mendapatkan nilainya:

#     ```python
#     # Salin output ini ke BRAND_FREQ dan MODEL_FREQ di app.py
#     print("BRAND_FREQ =", df['Brand'].value_counts(normalize=True).to_dict())
#     print("MODEL_FREQ =", df['model'].value_counts(normalize=True).to_dict())
#     ```

#     Lalu ganti dictionary `BRAND_FREQ` dan `MODEL_FREQ` di bagian atas `app.py`.
#     """)