import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DepreScan | Mental Health Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
        border: 1px solid #2d2d5e; text-align: center;
    }
    .hero-title { font-size: 2.4rem; font-weight: 700; color: #e0e6ff; margin: 0; }
    .hero-sub   { color: #8892b0; font-size: 1rem; margin-top: 0.5rem; }
    .metric-card {
        background: #1e2130; border-radius: 12px; padding: 1.2rem 1.4rem;
        border: 1px solid #2a2d3e; text-align: center; margin-bottom: 0.5rem;
    }
    .metric-val   { font-size: 2rem; font-weight: 700; color: #7c83fd; }
    .metric-label { font-size: 0.82rem; color: #8892b0; margin-top: 0.2rem; }
    .result-card  { border-radius: 14px; padding: 1.5rem; margin-top: 1rem; }
    .result-high  { background: linear-gradient(135deg,#2d0d0d,#3d1515); border:1px solid #8b1a1a; }
    .result-mid   { background: linear-gradient(135deg,#2d1e00,#3d2800); border:1px solid #b8860b; }
    .result-low   { background: linear-gradient(135deg,#0d2d0d,#103010); border:1px solid #1a7a1a; }
    .section-header {
        font-size: 1.15rem; font-weight: 600; color: #a8b2d8;
        border-left: 3px solid #7c83fd; padding-left: 0.75rem; margin-bottom: 1rem;
    }
    .stTextArea textarea {
        background: #1a1d2e !important; border: 1px solid #3a3d5e !important;
        border-radius: 10px !important; color: #e0e6ff !important; font-size: 1rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd, #4facfe);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
        width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .disclaimer {
        background: #1a1d2e; border: 1px solid #2d3a5e; border-radius: 10px;
        padding: 1rem; font-size: 0.82rem; color: #8892b0; margin-top: 1rem;
    }
    [data-testid="stSidebar"] { background: #13151f; border-right: 1px solid #1e2130; }
    .keyword-tag {
        display: inline-block; background: #2a2d4e; color: #a8b2d8;
        border-radius: 20px; padding: 0.15rem 0.6rem; font-size: 0.8rem; margin: 0.15rem;
        border: 1px solid #3a3d5e;
    }
    .keyword-tag.matched { background: #3d1515; color: #ff8080; border-color: #8b1a1a; }
</style>
""", unsafe_allow_html=True)

# ─── Keyword Bank (Rule-based Boosting) ───────────────────────────────────────
KEYWORD_BANK = {
    "bunuh_diri": {
        "keywords": ["bunuh diri", "ingin mati", "tidak mau hidup", "mengakhiri hidup",
                     "mati saja", "lebih baik mati", "pengen mati", "mau mati",
                     "suicide", "want to die", "end my life", "kill myself"],
        "weight": 0.45, "label": "Pikiran Bunuh Diri"
    },
    "putus_asa": {
        "keywords": ["tidak ada harapan", "putus asa", "hopeless", "sia-sia",
                     "percuma", "tidak berguna", "worthless", "useless",
                     "nggak ada gunanya", "hidup sia-sia", "nothing matters"],
        "weight": 0.30, "label": "Putus Asa"
    },
    "kelelahan": {
        "keywords": ["capek banget", "lelah", "exhausted", "burnout", "kelelahan",
                     "tidak kuat", "tidak sanggup", "overwhelmed", "kewalahan",
                     "sudah tidak bisa", "nggak kuat lagi", "so tired"],
        "weight": 0.20, "label": "Kelelahan Ekstrem"
    },
    "kesepian": {
        "keywords": ["sendirian", "lonely", "kesepian", "tidak ada yang peduli",
                     "diabaikan", "alone", "no one cares", "tidak dipedulikan",
                     "nggak ada yang ngerti", "feeling empty"],
        "weight": 0.25, "label": "Kesepian"
    },
    "cemas": {
        "keywords": ["cemas", "khawatir", "takut", "anxious", "anxiety",
                     "panik", "panic", "gelisah", "was-was", "nervous"],
        "weight": 0.15, "label": "Kecemasan"
    },
    "sedih": {
        "keywords": ["sedih", "menangis", "nangis terus", "sad", "depressed",
                     "depresi", "down", "murung", "galau", "heartbroken",
                     "patah hati", "hancur"],
        "weight": 0.20, "label": "Kesedihan"
    },
    "stres": {
        "keywords": ["stres", "stress", "tertekan", "under pressure", "beban",
                     "tekanan", "nggak tahan", "tidak tahan", "terbebani"],
        "weight": 0.15, "label": "Stres"
    },
    "tidak_semangat": {
        "keywords": ["malas", "tidak semangat", "nggak semangat", "demotivated",
                     "apatis", "apathy", "numb", "mati rasa", "hampa",
                     "kosong", "empty"],
        "weight": 0.18, "label": "Tidak Semangat"
    }
}

def rule_based_score(text: str):
    text_lower = text.lower()
    total_weight = 0.0
    matched = []
    for cat, info in KEYWORD_BANK.items():
        for kw in info["keywords"]:
            if kw in text_lower:
                total_weight += info["weight"]
                matched.append(info["label"])
                break
    score = min(total_weight / 1.0, 1.0)
    return score, list(set(matched))

# ─── Corpus Builder ────────────────────────────────────────────────────────────
def build_text_from_row(row) -> str:
    parts = []
    ap = row['Academic Pressure']
    if ap >= 4:   parts.append("tekanan akademik berat stres tugas menumpuk tidak sanggup menyerah")
    elif ap >= 3: parts.append("cukup tertekan kuliah sulit beban akademik")
    else:         parts.append("santai akademik tidak masalah nyaman belajar")

    sleep = str(row['Sleep Duration'])
    if 'Less than 5' in sleep:   parts.append("susah tidur insomnia begadang lelah kelelahan capek")
    elif '5-6' in sleep:         parts.append("kurang tidur sering mengantuk lelah")
    elif 'More than 8' in sleep: parts.append("tidur terus malas lesu tidak semangat apatis")
    else:                         parts.append("tidur cukup istirahat baik segar")

    if str(row['Have you ever had suicidal thoughts ?']).lower() == 'yes':
        parts.append("pernah ingin bunuh diri tidak ada harapan mengakhiri hidup menyerah putus asa mati")
    else:
        parts.append("semangat tidak ingin menyakiti diri sendiri harapan positif")

    fs = row['Financial Stress']
    if fs >= 4:   parts.append("sangat kesulitan finansial tidak punya uang masalah ekonomi berat cemas uang")
    elif fs >= 3: parts.append("khawatir keuangan agak kesulitan finansial")

    ss = row['Study Satisfaction']
    if ss <= 2:   parts.append("tidak puas gagal tidak berguna tidak berharga kecewa menyesal")
    elif ss >= 4: parts.append("puas senang kemajuan belajar bangga prestasi")

    diet = str(row['Dietary Habits'])
    if diet == 'Unhealthy': parts.append("makan tidak teratur skip makan tidak peduli kesehatan")
    elif diet == 'Healthy': parts.append("makan sehat teratur jaga kesehatan")

    if str(row['Family History of Mental Illness']).lower() == 'yes':
        parts.append("riwayat penyakit mental keluarga faktor genetik")

    wp = row['Work Pressure']
    if wp >= 4:   parts.append("burnout kelelahan kerja tidak kuat pekerjaan melelahkan")
    elif wp >= 3: parts.append("tekanan kerja berat capek pekerjaan")

    js = row['Job Satisfaction']
    if js <= 2: parts.append("tidak puas kerja frustasi jenuh tidak nyaman")

    wh = row.get('Work/Study Hours', 6)
    if wh >= 10: parts.append("kerja terlalu lama jam panjang tidak ada waktu istirahat")

    return " ".join(parts)

# ─── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Student Depression Dataset.csv")

# ─── Train Text Model (TF-IDF + CatBoost + Optuna) ────────────────────────────
@st.cache_resource
def train_text_model(_df):
    data = _df.dropna().copy()
    data['text'] = data.apply(build_text_from_row, axis=1)
    X_text = data['text']
    y      = data['Depression']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y, test_size=0.10, random_state=42, stratify=y
    )

    tfidf    = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_vec = tfidf.fit_transform(X_tr).toarray().astype("float32")
    X_te_vec = tfidf.transform(X_te).toarray().astype("float32")

    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 300, 800),
            "depth":               trial.suggest_int("depth", 4, 8),
            "learning_rate":       trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength":     trial.suggest_float("random_strength", 1e-3, 5, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "loss_function": "Logloss", "eval_metric": "Accuracy",
            "verbose": False, "random_seed": 42
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X_tr_vec, y_tr):
            m = CatBoostClassifier(**params)
            m.fit(X_tr_vec[tr_idx], y_tr.iloc[tr_idx])
            scores.append(accuracy_score(y_tr.iloc[val_idx], m.predict(X_tr_vec[val_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, show_progress_bar=False)

    best = study.best_params
    best.update({"loss_function": "Logloss", "eval_metric": "Accuracy",
                 "verbose": 0, "random_seed": 42})
    final_model = CatBoostClassifier(**best)
    final_model.fit(X_tr_vec, y_tr)

    y_pred = final_model.predict(X_te_vec)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True)
    cm     = confusion_matrix(y_te, y_pred)

    return final_model, tfidf, acc, report, cm, study.best_params, study.best_value

# ─── Train Structured Model ────────────────────────────────────────────────────
@st.cache_resource
def train_structured_model(_df):
    data = _df.dropna().copy()
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    sleep_map = {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 2}
    diet_map  = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3, 'Others': 2}
    data['Sleep_Score']  = data['Sleep Duration'].map(sleep_map).fillna(2)
    data['Diet_Score']   = data['Dietary Habits'].map(diet_map).fillna(2)
    data['Total_Pressure']            = data['Academic Pressure'] + data['Work Pressure'] + data['Financial Stress']
    data['Satisfaction_Index']        = data['Study Satisfaction'] + data['Job Satisfaction']
    data['Stress_Satisfaction_Ratio'] = data['Total_Pressure'] / (data['Satisfaction_Index'] + 1)
    data['Suicidal_Thoughts'] = data['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
    data['Family_History']    = data['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
    data.drop(['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
               'Sleep Duration', 'Dietary Habits'], axis=1, inplace=True)

    cat_features = ['Gender', 'City', 'Profession', 'Degree']
    for c in cat_features:
        data[c] = data[c].astype(str)

    X = data.drop('Depression', axis=1)
    y = data['Depression']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.08,
                           l2_leaf_reg=3.0, verbose=0, random_seed=42)
    m.fit(X_tr, y_tr, cat_features=cat_features)
    y_pred = m.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True)
    cm     = confusion_matrix(y_te, y_pred)
    return m, X_tr.columns.tolist(), cat_features, acc, report, cm

# ─── Predict Teks ─────────────────────────────────────────────────────────────
def predict_from_text(text: str, model, tfidf) -> dict:
    rule_score, matched_keywords = rule_based_score(text)
    vec   = tfidf.transform([text]).toarray().astype("float32")
    proba = model.predict_proba(vec)[0]
    catboost_score = float(proba[1])

    has_critical = any(
        kw in text.lower()
        for kw in ["bunuh diri", "ingin mati", "mati saja", "suicide", "want to die",
                   "kill myself", "end my life", "tidak mau hidup", "mengakhiri hidup"]
    )
    if has_critical:
        final_score = max(catboost_score * 0.5 + rule_score * 0.5, 0.80)
    else:
        final_score = catboost_score * 0.60 + rule_score * 0.40

    final_score = min(max(final_score, 0.0), 1.0)
    risk_pct    = int(round(final_score * 100))

    if risk_pct >= 70:   category = "TINGGI"
    elif risk_pct >= 35: category = "SEDANG"
    else:                category = "RENDAH"

    if category == "TINGGI":
        rec = "Sangat disarankan untuk segera berbicara dengan psikolog atau psikiater. Hubungi hotline 119 ext 8."
    elif category == "SEDANG":
        rec = "Pertimbangkan untuk berbagi perasaan dengan orang terpercaya atau konsultan kesehatan mental."
    else:
        rec = "Kondisi terlihat baik. Tetap jaga kesehatan mental dengan istirahat cukup dan aktivitas positif."

    return {
        "risk_percentage":  risk_pct,
        "category":         category,
        "catboost_score":   round(catboost_score * 100, 1),
        "rule_score":       round(rule_score * 100, 1),
        "detected_signals": matched_keywords,
        "recommendation":   rec,
        "has_critical":     has_critical,
    }

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DepreScan")
    st.markdown("---")
    page = st.radio("Navigasi", [
        "🔍 Analisis Teks",
        "📊 Prediksi Terstruktur",
        "📈 Eksplorasi Data",
        "ℹ️ Tentang Model"
    ])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#8892b0;'>
    ⚠️ <b>Disclaimer</b><br>
    Tools ini bukan pengganti diagnosis klinis. Jika membutuhkan bantuan hubungi hotline
    <b>119 ext 8</b> (Kemenkes RI).
    </div>
    """, unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}. Pastikan `Student_Depression_Dataset.csv` ada di folder yang sama.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analisis Teks
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Analisis Teks":
    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>DepreScan</p>
        <p class='hero-sub'>Deteksi Risiko Depresi dari Teks · CatBoost + Bayesian Optimization </p>
    </div>
    """, unsafe_allow_html=True)

    if "text_model" not in st.session_state:
        with st.spinner("⏳ Melatih model teks dengan CatBoost + Optuna... (±2 menit pertama kali)"):
            tm, tfidf_vec, tm_acc, tm_report, tm_cm, tm_params, tm_cv = train_text_model(df_raw)
            st.session_state["text_model"]  = tm
            st.session_state["tfidf_vec"]   = tfidf_vec
            st.session_state["tm_acc"]      = tm_acc
            st.session_state["tm_report"]   = tm_report
            st.session_state["tm_cm"]       = tm_cm
            st.session_state["tm_params"]   = tm_params

    tm      = st.session_state["text_model"]
    tfidf_v = st.session_state["tfidf_vec"]
    tm_acc  = st.session_state["tm_acc"]

    mc1, mc2 = st.columns(2)
    mc1.markdown(f"<div class='metric-card'><div class='metric-val'>{tm_acc*100:.1f}%</div><div class='metric-label'>Akurasi Model CatBoost + Bayesian Optimization</div></div>", unsafe_allow_html=True)
    mc2.markdown("<div class='metric-card'><div class='metric-val' style='font-size:1rem;'>CatBoost + Optuna</div><div class='metric-label'>Algoritma</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:1.2rem;'>Ceritakan perasaan Anda</div>", unsafe_allow_html=True)

    col_input, col_info = st.columns([2, 1])
    with col_input:
        user_text = st.text_area(
            "",
            placeholder='Ketik perasaan Anda...\nContoh: "ah capek banget, rasanya udah nggak ada harapan"\natau "I feel so hopeless and exhausted"',
            height=180,
            label_visibility="collapsed",
            key="main_text_input"
        )
        analyze_btn = st.button("🔍 Analisis Sekarang", use_container_width=True)

    with col_info:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Metode Analisis</div>
            <div style='color:#7c83fd; font-weight:600; font-size:0.9rem; margin-top:0.3rem;'>
                TF-IDF Vectorizer<br>+ CatBoost Classifier<br>+ Bayesian Optimization<br>+ Rule-based Boosting
            </div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Bahasa didukung</div>
            <div style='color:#7c83fd; font-weight:600; font-size:0.9rem; margin-top:0.3rem;'>
                Bahasa Indonesia<br>English
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("** Contoh ungkapan:**")
    ex_cols = st.columns(3)
    examples = [
        "ah capek banget, pengen istirahat selamanya",
        "udah nggak tau mau ngapain lagi hidupku, hampa",
        "ingin bunuh diri, rasanya semua sia-sia"
    ]
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(f'"{ex[:28]}..."', key=f"ex_{i}"):
                st.session_state["injected_text"] = ex

    if "injected_text" in st.session_state:
        user_text   = st.session_state.pop("injected_text")
        analyze_btn = True

    if analyze_btn and user_text and user_text.strip():
        result     = predict_from_text(user_text.strip(), tm, tfidf_v)
        risk_pct   = result["risk_percentage"]
        category   = result["category"]
        signals    = result["detected_signals"]
        rec        = result["recommendation"]
        cb_score   = result["catboost_score"]
        rule_score = result["rule_score"]

        cat_class = {"RENDAH": "result-low", "SEDANG": "result-mid", "TINGGI": "result-high"}[category]
        cat_color = {"RENDAH": "#4caf50",    "SEDANG": "#ffc107",    "TINGGI": "#f44336"}[category]
        cat_icon  = {"RENDAH": "✅",          "SEDANG": "⚠️",         "TINGGI": "🚨"}[category]

        st.markdown(f"""
        <div class='result-card {cat_class}'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='font-size:2.2rem; font-weight:700; color:{cat_color};'>{risk_pct}%</span>
                    <span style='color:#8892b0; margin-left:0.5rem;'>Risiko Depresi</span>
                </div>
                <div style='font-size:2.8rem;'>{cat_icon}</div>
            </div>
            <div style='margin-top:0.8rem;'>
                <span style='background:{cat_color}33; color:{cat_color}; padding:0.25rem 1rem;
                     border-radius:20px; font-weight:600;'>{category}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{cat_color};'>{risk_pct}%</div><div class='metric-label'>Skor Final (Ensemble)</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='metric-card'><div class='metric-val'>{cb_score}%</div><div class='metric-label'>CatBoost Score</div></div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='metric-card'><div class='metric-val'>{rule_score}%</div><div class='metric-label'>Rule-based Score</div></div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 0.9))
        fig.patch.set_facecolor('#1e2130')
        ax.set_facecolor('#1e2130')
        ax.barh(0, 100, color='#2a2d3e', height=0.6)
        ax.barh(0, risk_pct, color=cat_color, height=0.6)
        ax.axvline(35, color='#ffc107', linewidth=1, linestyle='--', alpha=0.6)
        ax.axvline(70, color='#f44336', linewidth=1, linestyle='--', alpha=0.6)
        ax.text(17, -0.55, 'RENDAH', color='#4caf50', fontsize=7, ha='center')
        ax.text(52, -0.55, 'SEDANG', color='#ffc107', fontsize=7, ha='center')
        ax.text(85, -0.55, 'TINGGI', color='#f44336', fontsize=7, ha='center')
        ax.set_xlim(0, 100); ax.axis('off')
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        col_sig, col_rec = st.columns(2)
        with col_sig:
            st.markdown("**🔍 Sinyal yang Terdeteksi:**")
            if signals:
                tags = "".join([f"<span class='keyword-tag matched'>{s}</span>" for s in signals])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.markdown("<span class='keyword-tag'>Tidak ada sinyal kritis</span>", unsafe_allow_html=True)
        with col_rec:
            st.markdown(f"""
            <div style='background:#1a2030; border-left:3px solid #7c83fd;
                  padding:1rem; border-radius:0 10px 10px 0;'>
                <b style='color:#a8b2d8;'>💡 Rekomendasi</b><br>
                <span style='color:#ccd6f6; font-size:0.9rem;'>{rec}</span>
            </div>
            """, unsafe_allow_html=True)

        if category == "TINGGI":
            st.error("🚨 **Perhatian Serius:** Risiko depresi tinggi terdeteksi. Segera hubungi **119 ext 8** (Kemenkes RI) atau **1500-454** (Into The Light Indonesia).")
        elif category == "SEDANG":
            st.warning("⚠️ Terdeteksi beberapa tanda yang perlu diperhatikan. Pertimbangkan berbicara dengan orang terpercaya atau profesional.")

    elif analyze_btn:
        st.info("Silakan masukkan teks terlebih dahulu.")

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <b>Disclaimer:</b> Analisis ini bersifat informatif menggunakan model machine learning
        dan tidak menggantikan diagnosis medis profesional. Selalu konsultasikan kondisi Anda dengan tenaga profesional.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prediksi Terstruktur
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Prediksi Terstruktur":
    st.markdown("<div class='hero-banner'><p class='hero-title'>📊 Prediksi Berbasis Profil</p><p class='hero-sub'>Isi data profil untuk prediksi risiko depresi menggunakan CatBoost</p></div>", unsafe_allow_html=True)

    if "struct_model" not in st.session_state:
        with st.spinner("⏳ Memuat model terstruktur..."):
            sm, feat_cols, cat_feats, sm_acc, sm_rep, sm_cm = train_structured_model(df_raw)
            st.session_state["struct_model"] = sm
            st.session_state["feat_cols"]    = feat_cols
            st.session_state["cat_feats"]    = cat_feats
            st.session_state["sm_acc"]       = sm_acc
            st.session_state["sm_rep"]       = sm_rep
            st.session_state["sm_cm"]        = sm_cm

    sm        = st.session_state["struct_model"]
    feat_cols = st.session_state["feat_cols"]
    cat_feats = st.session_state["cat_feats"]

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
        cgpa       = c6.slider("CGPA", 0.0, 4.0, 3.0, 0.1)
        acad_press = c7.slider("Tekanan Akademik (1-5)", 1, 5, 3)
        work_press = c8.slider("Tekanan Kerja (1-5)", 1, 5, 2)
        c9, c10, c11 = st.columns(3)
        study_sat  = c9.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
        job_sat    = c10.slider("Kepuasan Kerja (1-5)", 1, 5, 3)
        work_hours = c11.slider("Jam Belajar/Kerja per Hari", 1, 16, 7)

        st.markdown("<div class='section-header'>Gaya Hidup & Kesehatan</div>", unsafe_allow_html=True)
        c12, c13, c14 = st.columns(3)
        sleep   = c12.selectbox("Durasi Tidur", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        diet    = c13.selectbox("Kebiasaan Makan", ["Unhealthy", "Moderate", "Healthy"])
        fin_str = c14.slider("Tekanan Finansial (1-5)", 1, 5, 2)
        c15, c16 = st.columns(2)
        suicidal = c15.radio("Pernah punya pikiran bunuh diri?", ["No", "Yes"], horizontal=True)
        family   = c16.radio("Riwayat keluarga penyakit mental?", ["No", "Yes"], horizontal=True)

        submitted = st.form_submit_button("🔮 Prediksi Risiko Depresi", use_container_width=True)

    if submitted:
        sleep_map = {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4}
        diet_map  = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3}
        row = {
            'Gender': gender, 'Age': age, 'City': city, 'Profession': profession,
            'Academic Pressure': acad_press, 'Work Pressure': work_press, 'CGPA': cgpa,
            'Study Satisfaction': study_sat, 'Job Satisfaction': job_sat,
            'Degree': degree, 'Work/Study Hours': work_hours, 'Financial Stress': fin_str,
            'Sleep_Score': sleep_map.get(sleep, 2), 'Diet_Score': diet_map.get(diet, 2),
            'Suicidal_Thoughts': 1 if suicidal == 'Yes' else 0,
            'Family_History':    1 if family   == 'Yes' else 0,
        }
        row['Total_Pressure']            = acad_press + work_press + fin_str
        row['Satisfaction_Index']        = study_sat + job_sat
        row['Stress_Satisfaction_Ratio'] = row['Total_Pressure'] / (row['Satisfaction_Index'] + 1)
        df_in = pd.DataFrame([row])
        for c in cat_feats:
            df_in[c] = df_in[c].astype(str)
        for c in feat_cols:
            if c not in df_in.columns:
                df_in[c] = 0
        df_in = df_in[feat_cols]

        proba    = sm.predict_proba(df_in)[0]
        risk_pct = int(round(proba[1] * 100))
        color    = "#f44336" if proba[1] >= 0.5 else "#4caf50"
        label    = "⚠️ Terindikasi Depresi" if proba[1] >= 0.5 else "✅ Tidak Terindikasi"

        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{color};'>{risk_pct}%</div><div class='metric-label'>Probabilitas Depresi</div></div>", unsafe_allow_html=True)
        rc2.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#4caf50;'>{int(round(proba[0]*100))}%</div><div class='metric-label'>Probabilitas Normal</div></div>", unsafe_allow_html=True)
        rc3.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{color}; font-size:1.1rem;'>{label}</div><div class='metric-label'>Hasil Prediksi</div></div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 1))
        fig.patch.set_facecolor('#1e2130')
        ax.set_facecolor('#1e2130')
        ax.barh(0, 100, color='#2a2d3e', height=0.6)
        ax.barh(0, risk_pct, color=color, height=0.6)
        ax.axvline(50, color='white', linewidth=1.5, linestyle='--', alpha=0.4)
        ax.text(risk_pct + 1.5, 0, f'{risk_pct}%', va='center', color='white', fontweight='bold')
        ax.set_xlim(0, 100); ax.axis('off')
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if proba[1] >= 0.5:
            st.error("🚨 Model mendeteksi risiko depresi signifikan. Disarankan konsultasi dengan psikolog atau psikiater.")
        else:
            st.success("✅ Profil tidak menunjukkan indikasi depresi yang signifikan. Tetap jaga kesehatan mental!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Eksplorasi Data
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Eksplorasi Data":
    st.markdown("<div class='hero-banner'><p class='hero-title'>📈 Eksplorasi Dataset</p><p class='hero-sub'>Student Depression Dataset — Visualisasi & Statistik</p></div>", unsafe_allow_html=True)

    df      = df_raw.dropna()
    n_total = len(df)
    n_dep   = int(df['Depression'].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-val'>{n_total:,}</div><div class='metric-label'>Total Data</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#f44336;'>{n_dep:,}</div><div class='metric-label'>Terdepresi</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#4caf50;'>{n_total-n_dep:,}</div><div class='metric-label'>Tidak Terdepresi</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-val'>{n_dep/n_total*100:.1f}%</div><div class='metric-label'>Prevalensi Depresi</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#1e2130')
    for ax in axes:
        ax.set_facecolor('#1e2130'); ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2a2d3e')

    counts = df['Depression'].value_counts()
    bars = axes[0].bar(['Tidak Depresi','Depresi'], counts.values, color=['#4caf50','#f44336'])
    axes[0].set_title('Distribusi Kelas', color='#a8b2d8', fontsize=12)
    for b in bars:
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+50, str(int(b.get_height())), ha='center', color='white')

    sleep_dep = df.groupby(['Sleep Duration','Depression']).size().unstack(fill_value=0)
    so = [s for s in ['Less than 5 hours','5-6 hours','7-8 hours','More than 8 hours'] if s in sleep_dep.index]
    sleep_dep = sleep_dep.reindex(so)
    x = np.arange(len(sleep_dep)); w = 0.35
    axes[1].bar(x-w/2, sleep_dep.get(0,[0]*len(x)), width=w, color='#4caf50', label='Normal')
    axes[1].bar(x+w/2, sleep_dep.get(1,[0]*len(x)), width=w, color='#f44336', label='Depresi')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.replace(' hours','h').replace('Less than ','<').replace('More than ','>') for s in sleep_dep.index], color='#8892b0', fontsize=8)
    axes[1].set_title('Durasi Tidur vs Depresi', color='#a8b2d8', fontsize=12)
    axes[1].legend(facecolor='#2a2d3e', labelcolor='white')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='section-header'>Distribusi Fitur Numerik</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    fig.patch.set_facecolor('#1e2130')
    for ax, col in zip(axes, ['Age','CGPA','Academic Pressure','Work/Study Hours','Financial Stress']):
        ax.set_facecolor('#1e2130'); ax.tick_params(colors='#8892b0', labelsize=7); ax.spines[:].set_color('#2a2d3e')
        ax.hist(df[df['Depression']==0][col].dropna(), bins=15, alpha=0.6, color='#4caf50', label='Normal')
        ax.hist(df[df['Depression']==1][col].dropna(), bins=15, alpha=0.6, color='#f44336', label='Depresi')
        ax.set_title(col, color='#a8b2d8', fontsize=8)
    axes[0].legend(facecolor='#2a2d3e', labelcolor='white', fontsize=7)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='section-header'>Pikiran Bunuh Diri & Riwayat Keluarga vs Depresi</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor('#1e2130')
    for ax, col, title in zip(axes,
        ['Have you ever had suicidal thoughts ?','Family History of Mental Illness'],
        ['Pikiran Bunuh Diri vs Depresi','Riwayat Keluarga vs Depresi']):
        ax.set_facecolor('#1e2130'); ax.tick_params(colors='#8892b0'); ax.spines[:].set_color('#2a2d3e')
        pd.crosstab(df[col], df['Depression']).plot(kind='bar', ax=ax, color=['#4caf50','#f44336'])
        ax.set_title(title, color='#a8b2d8', fontsize=10); ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(['Normal','Depresi'], facecolor='#2a2d3e', labelcolor='white')
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Tentang Model
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang Model":
    st.markdown("<div class='hero-banner'><p class='hero-title'>ℹ️ Tentang Model</p><p class='hero-sub'>Arsitektur Pipeline, Performa & Metodologi</p></div>", unsafe_allow_html=True)

    for key, loader, keys_out in [
        ("text_model", lambda: train_text_model(df_raw),
         ["text_model","tfidf_vec","tm_acc","tm_report","tm_cm","tm_params","_"]),
        ("struct_model", lambda: train_structured_model(df_raw),
         ["struct_model","feat_cols","cat_feats","sm_acc","sm_rep","sm_cm"]),
    ]:
        if key not in st.session_state:
            with st.spinner("Memuat model..."):
                results = loader()
                for k, v in zip(keys_out, results):
                    if k != "_":
                        st.session_state[k] = v

    tm_acc    = st.session_state.get("tm_acc", 0)
    sm_acc    = st.session_state.get("sm_acc", 0)
    tm_report = st.session_state.get("tm_report", {})
    sm_rep    = st.session_state.get("sm_rep", {})
    tm_cm     = st.session_state.get("tm_cm", np.zeros((2,2)))
    sm_cm     = st.session_state.get("sm_cm", np.zeros((2,2)))
    sm        = st.session_state.get("struct_model")
    feat_cols = st.session_state.get("feat_cols", [])
    tm_params = st.session_state.get("tm_params", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-val'>{tm_acc*100:.1f}%</div><div class='metric-label'>Akurasi Model Teks</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-val'>{sm_acc*100:.1f}%</div><div class='metric-label'>Akurasi Model Profil</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-val'>{tm_report.get('1',{}).get('f1-score',0):.3f}</div><div class='metric-label'>F1 Model Teks</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-val'>{sm_rep.get('1',{}).get('f1-score',0):.3f}</div><div class='metric-label'>F1 Model Profil</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    col_cm, col_arch = st.columns(2)
    with col_cm:
        st.markdown("<div class='section-header'>Confusion Matrix — Model Teks</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        sns.heatmap(tm_cm.astype(int), annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal','Depresi'], yticklabels=['Normal','Depresi'],
                    linewidths=0.5, linecolor='#2a2d3e')
        ax.set_xlabel('Predicted', color='#a8b2d8'); ax.set_ylabel('Actual', color='#a8b2d8')
        ax.tick_params(colors='#a8b2d8'); ax.set_title('Text Model', color='#a8b2d8')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col_arch:
        st.markdown("<div class='section-header'>Arsitektur Pipeline</div>", unsafe_allow_html=True)
        st.markdown("""
        | Tahap | Detail |
        |---|---|
        | Input | Teks bebas (Indonesia / Inggris) |
        | Vectorizer | TF-IDF (500 fitur, bigram) |
        | Model | CatBoost Classifier |
        | Optimasi | Bayesian Optimization (Optuna, 15 trials) |
        | Boosting | Rule-based Keyword (8 kategori) |
        | Ensemble | 60% CatBoost + 40% Rule-based |
        | Output | Persentase risiko depresi |
        """)
        if tm_params:
            st.markdown("<div class='section-header' style='margin-top:1rem;'>Best Params (Optuna)</div>", unsafe_allow_html=True)
            for k, v in tm_params.items():
                st.markdown(f"- **{k}**: `{round(v,4) if isinstance(v,float) else v}`")

    if sm and feat_cols:
        st.markdown("<div class='section-header'>Feature Importance — Model Profil (Top 15)</div>", unsafe_allow_html=True)
        fi = pd.Series(sm.get_feature_importance(), index=feat_cols).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        ax.barh(fi.index, fi.values, color=['#4facfe' if v >= fi.max()*0.7 else '#7c83fd' for v in fi.values])
        ax.tick_params(colors='#a8b2d8', labelsize=9); ax.spines[:].set_color('#2a2d3e')
        ax.set_title('Feature Importance (Structured Model)', color='#a8b2d8')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='section-header'>Keyword Bank (Rule-based, 8 Kategori)</div>", unsafe_allow_html=True)
    kb_cols = st.columns(4)
    for i, (cat, info) in enumerate(KEYWORD_BANK.items()):
        with kb_cols[i % 4]:
            kw_tags = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in info['keywords'][:3]])
            st.markdown(f"""
            <div class='metric-card'>
                <div style='color:#7c83fd; font-weight:600;'>{info['label']}</div>
                <div style='color:#8892b0; font-size:0.75rem; margin-top:0.2rem;'>Bobot: {info['weight']}</div>
                <div style='margin-top:0.4rem;'>{kw_tags}</div>
            </div>
            """, unsafe_allow_html=True)
