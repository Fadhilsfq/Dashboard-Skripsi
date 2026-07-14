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
from sklearn.model_selection import train_test_split
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
</style>
""", unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Student Depression Dataset.csv")

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
                           l2_leaf_reg=3.0, verbose=0, random_seed=42, loss_function="Logloss")
    # Tambahkan eval_set untuk mendapatkan metrik loss per epoch
    m.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_te, y_te), early_stopping_rounds=50)
    
    y_pred = m.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True)
    cm     = confusion_matrix(y_te, y_pred)
    eval_res = m.evals_result_
    
    return m, X_tr.columns.tolist(), cat_features, acc, report, cm, eval_res

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DepreScan")
    st.markdown("---")
    page = st.radio("Navigasi", [
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
# PAGE 1 — Prediksi Terstruktur
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Prediksi Terstruktur":
    st.markdown("<div class='hero-banner'><p class='hero-title'>📊 Prediksi Berbasis Profil</p><p class='hero-sub'>Isi data profil untuk prediksi risiko depresi menggunakan CatBoost</p></div>", unsafe_allow_html=True)

    if "struct_model" not in st.session_state:
        with st.spinner("⏳ Memuat model terstruktur..."):
            sm, feat_cols, cat_feats, sm_acc, sm_rep, sm_cm, evals_res = train_structured_model(df_raw)
            st.session_state["struct_model"] = sm
            st.session_state["feat_cols"]    = feat_cols
            st.session_state["cat_feats"]    = cat_feats
            st.session_state["sm_acc"]       = sm_acc
            st.session_state["sm_rep"]       = sm_rep
            st.session_state["sm_cm"]        = sm_cm
            st.session_state["evals_res"]    = evals_res

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
# PAGE 2 — Eksplorasi Data
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
# PAGE 3 — Tentang Model
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang Model":
    st.markdown("<div class='hero-banner'><p class='hero-title'>ℹ️ Tentang Model</p><p class='hero-sub'>Arsitektur Pipeline, Performa & Metodologi</p></div>", unsafe_allow_html=True)

    if "struct_model" not in st.session_state:
        with st.spinner("Memuat model..."):
            sm, feat_cols, cat_feats, sm_acc, sm_rep, sm_cm, evals_res = train_structured_model(df_raw)
            st.session_state["struct_model"] = sm
            st.session_state["feat_cols"]    = feat_cols
            st.session_state["cat_feats"]    = cat_feats
            st.session_state["sm_acc"]       = sm_acc
            st.session_state["sm_rep"]       = sm_rep
            st.session_state["sm_cm"]        = sm_cm
            st.session_state["evals_res"]    = evals_res

    sm        = st.session_state.get("struct_model")
    sm_acc    = st.session_state.get("sm_acc", 0)
    sm_rep    = st.session_state.get("sm_rep", {})
    sm_cm     = st.session_state.get("sm_cm", np.zeros((2,2)))
    feat_cols = st.session_state.get("feat_cols", [])
    evals_res = st.session_state.get("evals_res", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-val'>{sm_acc*100:.1f}%</div><div class='metric-label'>Akurasi Model</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-val'>{sm_rep.get('macro avg',{}).get('precision',0)*100:.1f}%</div><div class='metric-label'>Precision (Macro)</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-val'>{sm_rep.get('macro avg',{}).get('recall',0)*100:.1f}%</div><div class='metric-label'>Recall (Macro)</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-val'>{sm_rep.get('macro avg',{}).get('f1-score',0)*100:.1f}%</div><div class='metric-label'>F1-Score (Macro)</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Visualisasi Kurva Loss & Bar Chart Metrik
    col_loss, col_metric = st.columns(2)
    
    with col_loss:
        st.markdown("<div class='section-header'>Grafik Kurva Loss (Training vs Validasi)</div>", unsafe_allow_html=True)
        if evals_res:
            try:
                train_loss = evals_res['learn']['Logloss']
                val_key = list(evals_res.keys())[1] if 'validation' not in evals_res else 'validation'
                val_loss = evals_res[val_key]['Logloss']
                epochs = range(1, len(train_loss) + 1)

                fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
                fig_loss.patch.set_facecolor('#1e2130')
                ax_loss.set_facecolor('#1e2130')
                ax_loss.plot(epochs, train_loss, label='Train Loss', color='#4facfe', linewidth=2)
                ax_loss.plot(epochs, val_loss, label='Validation Loss', color='#f44336', linewidth=2)
                ax_loss.set_xlabel('Iterations', color='#a8b2d8')
                ax_loss.set_ylabel('Logloss', color='#a8b2d8')
                ax_loss.tick_params(colors='#a8b2d8')
                ax_loss.spines[:].set_color('#2a2d3e')
                ax_loss.legend(facecolor='#2a2d3e', labelcolor='white')
                plt.tight_layout()
                st.pyplot(fig_loss, use_container_width=True)
                plt.close()
            except Exception as e:
                st.warning(f"Kurva loss tidak tersedia: {e}")
                
    with col_metric:
        st.markdown("<div class='section-header'>Barchart Perbandingan Metrik Akhir</div>", unsafe_allow_html=True)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        scores = [
            sm_acc,
            sm_rep.get('macro avg', {}).get('precision', 0),
            sm_rep.get('macro avg', {}).get('recall', 0),
            sm_rep.get('macro avg', {}).get('f1-score', 0)
        ]

        fig_metric, ax_metric = plt.subplots(figsize=(6, 4))
        fig_metric.patch.set_facecolor('#1e2130')
        ax_metric.set_facecolor('#1e2130')
        bars = ax_metric.bar(metrics, scores, color=['#4facfe', '#7c83fd', '#4caf50', '#f44336'])
        ax_metric.set_ylim(0, 1.1)
        ax_metric.set_ylabel('Score', color='#a8b2d8')
        ax_metric.tick_params(colors='#a8b2d8')
        ax_metric.spines[:].set_color('#2a2d3e')

        for bar in bars:
            height = bar.get_height()
            ax_metric.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5), textcoords="offset points", ha='center', color='white', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig_metric, use_container_width=True)
        plt.close()

    st.markdown("---")

    col_cm, col_arch = st.columns(2)
    with col_cm:
        st.markdown("<div class='section-header'>Confusion Matrix — Model Prediksi</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        sns.heatmap(sm_cm.astype(int), annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal','Depresi'], yticklabels=['Normal','Depresi'],
                    linewidths=0.5, linecolor='#2a2d3e')
        ax.set_xlabel('Predicted', color='#a8b2d8'); ax.set_ylabel('Actual', color='#a8b2d8')
        ax.tick_params(colors='#a8b2d8'); ax.set_title('CatBoost Structured Model', color='#a8b2d8')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col_arch:
        if sm and feat_cols:
            st.markdown("<div class='section-header'>Feature Importance (Top 10)</div>", unsafe_allow_html=True)
            fi = pd.Series(sm.get_feature_importance(), index=feat_cols).sort_values(ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            ax.barh(fi.index, fi.values, color=['#4facfe' if v >= fi.max()*0.7 else '#7c83fd' for v in fi.values])
            ax.tick_params(colors='#a8b2d8', labelsize=9); ax.spines[:].set_color('#2a2d3e')
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
