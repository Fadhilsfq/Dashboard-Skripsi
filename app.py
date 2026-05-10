import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import re
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Page Config
st.set_page_config(
    page_title="DepreScan | Mental Health Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2d2d5e;
        text-align: center;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #e0e6ff;
        margin: 0;
    }
    .hero-sub {
        color: #8892b0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #2a2d3e;
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #7c83fd;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8892b0;
        margin-top: 0.2rem;
    }

    /* Result cards */
    .result-card {
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-high {
        background: linear-gradient(135deg, #2d0d0d, #3d1515);
        border: 1px solid #8b1a1a;
    }
    .result-mid {
        background: linear-gradient(135deg, #2d1e00, #3d2800);
        border: 1px solid #b8860b;
    }
    .result-low {
        background: linear-gradient(135deg, #0d2d0d, #103010);
        border: 1px solid #1a7a1a;
    }

    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #a8b2d8;
        border-left: 3px solid #7c83fd;
        padding-left: 0.75rem;
        margin-bottom: 1rem;
    }

    /* Text area styling */
    .stTextArea textarea {
        background: #1a1d2e !important;
        border: 1px solid #3a3d5e !important;
        border-radius: 10px !important;
        color: #e0e6ff !important;
        font-size: 1rem !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd, #4facfe);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Disclaimer */
    .disclaimer {
        background: #1a1d2e;
        border: 1px solid #2d3a5e;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.82rem;
        color: #8892b0;
        margin-top: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #13151f;
        border-right: 1px solid #1e2130;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data & Model Loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Student Depression Dataset.csv")
    return df

@st.cache_resource
def train_model(df):
    data = df.copy()
    if 'id' in data.columns:
        data.drop(['id'], axis=1, inplace=True)
    data.dropna(inplace=True)

    # Feature engineering
    sleep_map = {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 2}
    data['Sleep_Score'] = data['Sleep Duration'].map(sleep_map).fillna(2)
    diet_map = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3, 'Others': 2}
    data['Diet_Score'] = data['Dietary Habits'].map(diet_map).fillna(2)
    data['Total_Pressure']         = data['Academic Pressure'] + data['Work Pressure'] + data['Financial Stress']
    data['Satisfaction_Index']     = data['Study Satisfaction'] + data['Job Satisfaction']
    data['Stress_Satisfaction_Ratio'] = data['Total_Pressure'] / (data['Satisfaction_Index'] + 1)
    data['Suicidal_Thoughts']      = data['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
    data['Family_History']         = data['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
    data.drop(['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
               'Sleep Duration', 'Dietary Habits'], axis=1, inplace=True)

    cat_features = ['Gender', 'City', 'Profession', 'Degree']
    for col in cat_features:
        data[col] = data[col].astype(str)

    X = data.drop('Depression', axis=1)
    y = data['Depression']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    model = CatBoostClassifier(
        iterations=800, depth=6, learning_rate=0.08,
        l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=0.5,
        border_count=128, loss_function='Logloss', eval_metric='Accuracy',
        verbose=0, random_seed=42
    )
    model.fit(X_train, y_train, cat_features=cat_features)

    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)

    return model, X_train.columns.tolist(), cat_features, acc, report, cm

def preprocess_input(user_input: dict) -> pd.DataFrame:
    row = user_input.copy()
    sleep_map = {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 2}
    diet_map  = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3, 'Others': 2}
    row['Sleep_Score']  = sleep_map.get(row.pop('Sleep Duration', 'Others'), 2)
    row['Diet_Score']   = diet_map.get(row.pop('Dietary Habits', 'Others'), 2)
    suicidal = row.pop('Have you ever had suicidal thoughts ?', 'No')
    family   = row.pop('Family History of Mental Illness', 'No')
    row['Suicidal_Thoughts'] = 1 if suicidal == 'Yes' else 0
    row['Family_History']    = 1 if family == 'Yes' else 0
    row['Total_Pressure']    = float(row.get('Academic Pressure', 3)) + float(row.get('Work Pressure', 3)) + float(row.get('Financial Stress', 3))
    row['Satisfaction_Index']= float(row.get('Study Satisfaction', 3)) + float(row.get('Job Satisfaction', 3))
    row['Stress_Satisfaction_Ratio'] = row['Total_Pressure'] / (row['Satisfaction_Index'] + 1)
    return pd.DataFrame([row])

# ─── Claude API Text Analysis ─────────────────────────────────────────────────
def analyze_text_claude(text: str) -> dict:
    """Call Claude API to analyze depression from free text."""
    system_prompt = """Anda adalah asisten analisis kesehatan mental berbasis AI yang terlatih untuk mengidentifikasi
indikator depresi dari teks bebas dalam Bahasa Indonesia maupun Inggris.

Tugas Anda:
1. Analisis teks yang diberikan untuk mendeteksi tanda-tanda depresi, kecemasan, atau tekanan mental
2. Berikan persentase risiko depresi (0–100%)
3. Identifikasi kata kunci / frasa yang menjadi sinyal
4. Kategorikan: RENDAH (<30%), SEDANG (30–70%), TINGGI (>70%)
5. Berikan rekomendasi singkat

Respons HANYA dalam format JSON berikut (tanpa teks lain):
{
  "risk_percentage": <angka 0-100>,
  "category": "<RENDAH|SEDANG|TINGGI>",
  "detected_signals": ["<signal1>", "<signal2>", ...],
  "emotional_tone": "<deskripsi singkat emosi dominan>",
  "recommendation": "<saran singkat 1-2 kalimat>",
  "confidence": <angka 0-100>
}"""

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": f"Analisis teks berikut:\n\n\"{text}\""}]
    }
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["content"][0]["text"]
        # Strip markdown fences if any
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DepreScan")
    st.markdown("---")
    page = st.radio("Navigasi", ["🔍 Analisis Teks", "📊 Prediksi Terstruktur", "📈 Eksplorasi Data", "ℹ️ Tentang Model"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#8892b0;'>
    ⚠️ <b>Disclaimer</b><br>
    Tools ini bukan pengganti diagnosis klinis. Jika Anda atau seseorang membutuhkan bantuan, hubungi profesional kesehatan mental atau hotline <b>119 ext 8</b>.
    </div>
    """, unsafe_allow_html=True)

# ─── Load Data & Model ────────────────────────────────────────────────────────
try:
    df_raw = load_data()
    model, feature_cols, cat_features, model_acc, report, cm = train_model(df_raw)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Gagal memuat dataset: {e}. Pastikan file `Student Depression Dataset.csv` ada di direktori yang sama.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – Analisis Teks
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔍 Analisis Teks":
    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>🧠 DepreScan</p>
        <p class='hero-sub'>Deteksi Risiko Depresi dari Teks Bebas menggunakan AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Ceritakan perasaan Anda</div>", unsafe_allow_html=True)

    col_input, col_info = st.columns([2, 1])

    with col_input:
        user_text = st.text_area(
            "",
            placeholder='Contoh: "Aku udah capek banget, rasanya nggak ada yang peduli sama aku..."\natau\n"I\'ve been feeling really hopeless lately, nothing seems to matter anymore."',
            height=180,
            label_visibility="collapsed"
        )
        analyze_btn = st.button("🔍 Analisis Sekarang", use_container_width=True)

    with col_info:
        st.markdown("""
        <div class='metric-card' style='margin-bottom:0.8rem;'>
            <div class='metric-label'>Didukung oleh</div>
            <div style='color:#7c83fd; font-weight:700; font-size:1.1rem;'>Claude Sonnet 4</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Bahasa yang didukung</div>
            <div style='color:#7c83fd; font-weight:700; font-size:1rem;'>🇮🇩 Bahasa Indonesia<br>🇺🇸 English</div>
        </div>
        """, unsafe_allow_html=True)

    # Example prompts
    st.markdown("**💡 Contoh ungkapan:**")
    ex_cols = st.columns(3)
    examples = [
        "ah capek banget, pengen istirahat selamanya",
        "udah nggak tau mau ngapain lagi hidupku",
        "ingin bunuh diri, rasanya semua sia-sia"
    ]
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(f'"{ex[:30]}..."', key=f"ex{i}"):
                st.session_state["example_text"] = ex
                st.rerun()

    if "example_text" in st.session_state:
        user_text = st.session_state.pop("example_text")

    # ── Analysis Result ─────────────────────────────────────────────────────
    if analyze_btn and user_text.strip():
        with st.spinner("⏳ Menganalisis teks..."):
            result = analyze_text_claude(user_text.strip())

        if "error" in result:
            st.error(f"Gagal menganalisis: {result['error']}")
        else:
            risk_pct = result.get("risk_percentage", 0)
            category = result.get("category", "RENDAH")
            signals  = result.get("detected_signals", [])
            tone     = result.get("emotional_tone", "-")
            rec      = result.get("recommendation", "-")
            conf     = result.get("confidence", 0)

            cat_class = {"RENDAH": "result-low", "SEDANG": "result-mid", "TINGGI": "result-high"}.get(category, "result-low")
            cat_color = {"RENDAH": "#4caf50", "SEDANG": "#ffc107", "TINGGI": "#f44336"}.get(category, "#4caf50")
            cat_icon  = {"RENDAH": "✅", "SEDANG": "⚠️", "TINGGI": "🚨"}.get(category, "✅")

            st.markdown(f"""
            <div class='result-card {cat_class}'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='font-size:1.8rem; font-weight:700; color:{cat_color};'>{risk_pct}%</span>
                        <span style='color:#8892b0; margin-left:0.5rem;'>Risiko Depresi</span>
                    </div>
                    <div style='font-size:2.5rem;'>{cat_icon}</div>
                </div>
                <div style='margin-top:0.8rem;'>
                    <span style='background:{cat_color}33; color:{cat_color}; padding:0.2rem 0.8rem; border-radius:20px; font-size:0.9rem; font-weight:600;'>
                        {category}
                    </span>
                    <span style='color:#8892b0; margin-left:1rem; font-size:0.85rem;'>Kepercayaan: {conf}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                # Progress bar visual
                fig, ax = plt.subplots(figsize=(4, 0.8))
                fig.patch.set_facecolor('#1e2130')
                ax.set_facecolor('#1e2130')
                ax.barh(0, 100, color='#2a2d3e', height=0.5)
                bar_color = '#f44336' if risk_pct > 70 else '#ffc107' if risk_pct > 30 else '#4caf50'
                ax.barh(0, risk_pct, color=bar_color, height=0.5)
                ax.set_xlim(0, 100); ax.axis('off')
                ax.text(risk_pct / 2, 0, f'{risk_pct}%', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
                plt.tight_layout(pad=0)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.markdown(f"**🎭 Nada Emosional:** {tone}")

            with col2:
                if signals:
                    st.markdown("**🔍 Sinyal yang Terdeteksi:**")
                    for s in signals:
                        st.markdown(f"- `{s}`")

            st.markdown(f"""
            <div style='background:#1a2030; border-left:3px solid #7c83fd; padding:1rem; border-radius:0 10px 10px 0; margin-top:1rem;'>
                <b style='color:#a8b2d8;'>💡 Rekomendasi</b><br>
                <span style='color:#ccd6f6;'>{rec}</span>
            </div>
            """, unsafe_allow_html=True)

            if category == "TINGGI":
                st.error("🚨 **Perhatian:** Teks menunjukkan risiko tinggi. Segera hubungi profesional kesehatan mental atau hubungi hotline **119 ext 8** (Kemenkes RI) atau **1500-454** (Into The Light Indonesia).")
            elif category == "SEDANG":
                st.warning("⚠️ **Perhatian:** Ada beberapa tanda yang perlu diperhatikan. Pertimbangkan untuk berbicara dengan seseorang yang Anda percaya atau konsultan kesehatan.")

    elif analyze_btn:
        st.info("Silakan masukkan teks terlebih dahulu.")

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <b>Disclaimer:</b> Analisis ini bersifat informatif dan tidak menggantikan diagnosis medis profesional.
        Selalu konsultasikan kondisi kesehatan mental Anda dengan tenaga profesional.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – Prediksi Terstruktur
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Prediksi Terstruktur":
    st.markdown("<div class='hero-banner'><p class='hero-title'>📊 Prediksi Berbasis Profil</p><p class='hero-sub'>Isi data profil untuk mendapatkan prediksi risiko depresi dari model CatBoost</p></div>", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model belum dimuat. Kembali ke halaman utama.")
    else:
        with st.form("structured_form"):
            st.markdown("<div class='section-header'>Data Demografis</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            gender     = c1.selectbox("Gender", ["Male", "Female"])
            age        = c2.number_input("Umur", 15, 60, 22)
            city       = c3.text_input("Kota", "Jakarta")

            c4, c5 = st.columns(2)
            profession = c4.selectbox("Profesi", ["Student", "Working Professional", "Others"])
            degree     = c5.text_input("Gelar / Jurusan", "B.Tech")

            st.markdown("<div class='section-header'>Akademik & Kerja</div>", unsafe_allow_html=True)
            c6, c7, c8 = st.columns(3)
            cgpa         = c6.slider("CGPA", 0.0, 4.0, 3.0, 0.1)
            acad_press   = c7.slider("Tekanan Akademik", 0, 5, 3)
            work_press   = c8.slider("Tekanan Kerja", 0, 5, 2)

            c9, c10, c11 = st.columns(3)
            study_sat   = c9.slider("Kepuasan Belajar", 0, 5, 3)
            job_sat     = c10.slider("Kepuasan Kerja", 0, 5, 3)
            work_hours  = c11.slider("Jam Belajar/Kerja/Hari", 0, 12, 7)

            st.markdown("<div class='section-header'>Gaya Hidup & Kesehatan</div>", unsafe_allow_html=True)
            c12, c13, c14 = st.columns(3)
            sleep   = c12.selectbox("Durasi Tidur", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
            diet    = c13.selectbox("Kebiasaan Makan", ["Unhealthy", "Moderate", "Healthy"])
            fin_str = c14.slider("Tekanan Finansial", 0, 5, 2)

            c15, c16 = st.columns(2)
            suicidal = c15.radio("Pernah punya pikiran bunuh diri?", ["No", "Yes"], horizontal=True)
            family   = c16.radio("Riwayat keluarga penyakit mental?", ["No", "Yes"], horizontal=True)

            submitted = st.form_submit_button("🔮 Prediksi Risiko Depresi", use_container_width=True)

        if submitted:
            user_row = {
                'Gender': gender, 'Age': age, 'City': city, 'Profession': profession,
                'Academic Pressure': acad_press, 'Work Pressure': work_press, 'CGPA': cgpa,
                'Study Satisfaction': study_sat, 'Job Satisfaction': job_sat,
                'Sleep Duration': sleep, 'Dietary Habits': diet, 'Degree': degree,
                'Have you ever had suicidal thoughts ?': suicidal,
                'Work/Study Hours': work_hours, 'Financial Stress': fin_str,
                'Family History of Mental Illness': family
            }
            df_input = preprocess_input(user_row)
            for col in cat_features:
                df_input[col] = df_input[col].astype(str)
            # Reindex to match training features
            for col in feature_cols:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[feature_cols]

            proba = model.predict_proba(df_input)[0]
            risk_pct = int(round(proba[1] * 100))
            pred_label = "Terindikasi Depresi" if proba[1] >= 0.5 else "Tidak Terindikasi Depresi"
            color = "#f44336" if proba[1] >= 0.5 else "#4caf50"

            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{color};'>{risk_pct}%</div><div class='metric-label'>Probabilitas Depresi</div></div>", unsafe_allow_html=True)
            rcol2.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{color};'>{int(round(proba[0]*100))}%</div><div class='metric-label'>Probabilitas Normal</div></div>", unsafe_allow_html=True)
            rcol3.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{color}; font-size:1.2rem;'>{pred_label}</div><div class='metric-label'>Hasil Prediksi</div></div>", unsafe_allow_html=True)

            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 1))
            fig.patch.set_facecolor('#1e2130')
            ax.set_facecolor('#1e2130')
            ax.barh(0, 100, color='#2a2d3e', height=0.6)
            ax.barh(0, risk_pct, color=color, height=0.6)
            ax.axvline(50, color='white', linewidth=1.5, linestyle='--', alpha=0.5)
            ax.text(risk_pct + 1, 0, f'{risk_pct}%', va='center', color='white', fontweight='bold')
            ax.set_xlim(0, 100); ax.axis('off')
            ax.set_title('Skor Risiko Depresi', color='#a8b2d8', pad=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            if proba[1] >= 0.5:
                st.error("🚨 Model mendeteksi risiko depresi yang signifikan. Disarankan untuk konsultasi dengan psikolog atau psikiater.")
            else:
                st.success("✅ Profil tidak menunjukkan indikasi depresi yang signifikan. Tetap jaga kesehatan mental Anda!")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – Eksplorasi Data
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Eksplorasi Data":
    st.markdown("<div class='hero-banner'><p class='hero-title'>📈 Eksplorasi Dataset</p><p class='hero-sub'>Student Depression Dataset – Visualisasi & Statistik</p></div>", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Dataset belum dimuat.")
    else:
        df = df_raw.dropna()
        n_total = len(df)
        n_dep   = df['Depression'].sum()
        pct_dep = n_dep / n_total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-val'>{n_total:,}</div><div class='metric-label'>Total Data</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#f44336;'>{n_dep:,}</div><div class='metric-label'>Terdepresi</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#4caf50;'>{n_total - n_dep:,}</div><div class='metric-label'>Tidak Terdepresi</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-val'>{pct_dep:.1f}%</div><div class='metric-label'>Prevalensi Depresi</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # Class Balance + Sleep vs Depression
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#1e2130')
        for ax in axes:
            ax.set_facecolor('#1e2130')
            ax.tick_params(colors='#8892b0')
            ax.spines[:].set_color('#2a2d3e')

        # Class balance
        counts = df['Depression'].value_counts()
        bars = axes[0].bar(['Tidak Depresi', 'Depresi'], counts.values, color=['#4caf50', '#f44336'])
        axes[0].set_title('Distribusi Kelas', color='#a8b2d8', fontsize=12)
        for b in bars:
            axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 50, str(int(b.get_height())), ha='center', color='white')

        # Sleep vs Depression
        sleep_dep = df.groupby(['Sleep Duration', 'Depression']).size().unstack(fill_value=0)
        sleep_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
        sleep_order = [s for s in sleep_order if s in sleep_dep.index]
        sleep_dep   = sleep_dep.reindex(sleep_order)
        x = np.arange(len(sleep_dep))
        w = 0.35
        axes[1].bar(x - w/2, sleep_dep.get(0, [0]*len(x)), width=w, color='#4caf50', label='Normal')
        axes[1].bar(x + w/2, sleep_dep.get(1, [0]*len(x)), width=w, color='#f44336', label='Depresi')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([s.replace(' hours', 'h').replace('Less than ', '<').replace('More than ', '>') for s in sleep_dep.index], color='#8892b0', fontsize=8)
        axes[1].set_title('Durasi Tidur vs Depresi', color='#a8b2d8', fontsize=12)
        axes[1].legend(facecolor='#2a2d3e', labelcolor='white')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Numeric distributions
        st.markdown("<div class='section-header'>Distribusi Fitur Numerik</div>", unsafe_allow_html=True)
        num_cols = ['Age', 'CGPA', 'Academic Pressure', 'Work/Study Hours', 'Financial Stress']
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        fig.patch.set_facecolor('#1e2130')
        for ax, col in zip(axes, num_cols):
            ax.set_facecolor('#1e2130')
            ax.tick_params(colors='#8892b0', labelsize=7)
            ax.spines[:].set_color('#2a2d3e')
            dep0 = df[df['Depression']==0][col].dropna()
            dep1 = df[df['Depression']==1][col].dropna()
            ax.hist(dep0, bins=15, alpha=0.6, color='#4caf50', label='Normal')
            ax.hist(dep1, bins=15, alpha=0.6, color='#f44336', label='Depresi')
            ax.set_title(col, color='#a8b2d8', fontsize=8)
        axes[0].legend(facecolor='#2a2d3e', labelcolor='white', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Suicidal thoughts
        st.markdown("<div class='section-header'>Pikiran Bunuh Diri & Riwayat Keluarga</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        fig.patch.set_facecolor('#1e2130')
        for ax, col, title in zip(axes,
            ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'],
            ['Pikiran Bunuh Diri vs Depresi', 'Riwayat Keluarga vs Depresi']):
            ax.set_facecolor('#1e2130')
            ax.tick_params(colors='#8892b0')
            ax.spines[:].set_color('#2a2d3e')
            ct = pd.crosstab(df[col], df['Depression'])
            ct.plot(kind='bar', ax=ax, color=['#4caf50', '#f44336'], legend=True)
            ax.set_title(title, color='#a8b2d8', fontsize=10)
            ax.set_xlabel('', color='#8892b0')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(['Normal', 'Depresi'], facecolor='#2a2d3e', labelcolor='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – Tentang Model
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ Tentang Model":
    st.markdown("<div class='hero-banner'><p class='hero-title'>ℹ️ Tentang Model</p><p class='hero-sub'>Arsitektur, Performa & Metodologi</p></div>", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model belum dimuat.")
    else:
        # Metrics
        prec0 = report['0']['precision']
        rec0  = report['0']['recall']
        f1_0  = report['0']['f1-score']
        prec1 = report['1']['precision']
        rec1  = report['1']['recall']
        f1_1  = report['1']['f1-score']

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-val'>{model_acc*100:.1f}%</div><div class='metric-label'>Akurasi Model</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-val'>{f1_1:.3f}</div><div class='metric-label'>F1-Score (Depresi)</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-val'>{prec1:.3f}</div><div class='metric-label'>Precision (Depresi)</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-val'>{rec1:.3f}</div><div class='metric-label'>Recall (Depresi)</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        col_cm, col_info = st.columns([1, 1])

        with col_cm:
            st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor('#1e2130')
            ax.set_facecolor('#1e2130')
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Normal', 'Depresi'],
                        yticklabels=['Normal', 'Depresi'],
                        linewidths=0.5, linecolor='#2a2d3e')
            ax.set_xlabel('Predicted', color='#a8b2d8')
            ax.set_ylabel('Actual', color='#a8b2d8')
            ax.tick_params(colors='#a8b2d8')
            ax.set_title('Confusion Matrix (Test Set)', color='#a8b2d8')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_info:
            st.markdown("<div class='section-header'>Arsitektur Pipeline</div>", unsafe_allow_html=True)
            st.markdown("""
            | Tahap | Detail |
            |---|---|
            | Dataset | Student Depression Dataset |
            | Split | 90% Train / 10% Test |
            | Preprocessing | Ordinal Encoding, Feature Engineering |
            | Model | CatBoost Classifier |
            | Optimasi | Bayesian Optimization (Optuna) |
            | Text AI | Claude Sonnet 4 (Anthropic) |
            """)

            st.markdown("<div class='section-header' style='margin-top:1rem;'>Fitur Engineered</div>", unsafe_allow_html=True)
            st.markdown("""
            - **Total_Pressure** = Akademik + Kerja + Finansial  
            - **Satisfaction_Index** = Kepuasan Belajar + Kerja  
            - **Stress_Satisfaction_Ratio** = Tekanan / (Kepuasan + 1)  
            - **Sleep_Score** & **Diet_Score** → Ordinal  
            - **Suicidal_Thoughts** & **Family_History** → Binary  
            """)

        # Feature importance
        st.markdown("<div class='section-header'>Feature Importance (Top 15)</div>", unsafe_allow_html=True)
        fi = pd.Series(model.get_feature_importance(), index=feature_cols).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#1e2130')
        ax.set_facecolor('#1e2130')
        colors = ['#7c83fd' if v < fi.max() * 0.7 else '#4facfe' for v in fi.values]
        ax.barh(fi.index, fi.values, color=colors)
        ax.tick_params(colors='#a8b2d8', labelsize=9)
        ax.spines[:].set_color('#2a2d3e')
        ax.set_title('Feature Importance', color='#a8b2d8')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
