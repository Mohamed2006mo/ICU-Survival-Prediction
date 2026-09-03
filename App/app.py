import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="ICU Mortality Risk Predictor", page_icon="🏥", layout="wide"
)

# ---------------------------------------------------------------------------
# Theme — "Coolors" palette, forced light regardless of visitor's system/
# browser Dark mode setting. Explicit colors on every surface.
# Darkest #0D1B2A · Dark #1B263B · Steel #415A77 · Muted #778DA9 · Cream #E0E1DD
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
    --c-darkest: #0D1B2A;
    --c-dark: #1B263B;
    --c-steel: #415A77;
    --c-muted: #778DA9;
    --c-cream: #E0E1DD;
    --page-bg: #F4F5F3;
    --card-bg: #FFFFFF;
    --border: #D9DCD6;
    --accent: #E4572E;
}

/* Force the whole app surface to light, in both themes */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottomBlockContainer"] {
    background-color: var(--page-bg) !important;
    color: var(--c-dark) !important;
}
[data-testid="stHeader"] { background-color: transparent !important; }

/* Body text stays dark regardless of theme */
p, span, li, label, div { color: var(--c-dark); }

/* Headings */
h1, h2, h3, h4 { color: var(--c-darkest) !important; font-weight: 700 !important; }
h1 { border-bottom: 3px solid var(--accent); padding-bottom: 0.4rem; display: inline-block; }

/* Sidebar — always dark navy background, always cream text, in both themes */
section[data-testid="stSidebar"] {
    background-color: var(--c-darkest) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] markdown {
    color: var(--c-cream) !important;
}
section[data-testid="stSidebar"] h3 {
    color: var(--c-muted) !important;
    letter-spacing: 1.5px;
    font-size: 0.75rem !important;
    border: none;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background-color: rgba(255,255,255,0.08);
    border-radius: 6px;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

/* Metric cards — always white with dark text, regardless of theme */
div[data-testid="stMetric"] {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border);
    border-left: 5px solid var(--c-steel);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(13, 27, 42, 0.06);
}
div[data-testid="stMetric"] label { color: var(--c-steel) !important; font-weight: 600; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--c-darkest) !important; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background-color: var(--accent) !important;
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: background-color 0.15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background-color: #C8441E !important;
    color: white !important;
}
.stButton > button p, .stFormSubmitButton > button p { color: white !important; }

/* Containers / expanders / bordered result cards — always white */
div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--card-bg) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}

/* Text inputs / number inputs / selects — always light background */
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
[data-baseweb="select"] > div, [data-baseweb="input"] {
    background-color: var(--card-bg) !important;
    color: var(--c-dark) !important;
}

/* Alert boxes (info/success/error/warning) get rounder corners */
div[data-testid="stAlert"] { border-radius: 10px; }

/* Dataframe/table header */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] { color: var(--c-steel) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "xgb_icu_model.pkl"))
    num_imputer = joblib.load(os.path.join(base_dir, "num_imputer.pkl"))
    cat_imputer = joblib.load(os.path.join(base_dir, "cat_imputer.pkl"))
    with open(os.path.join(base_dir, "final_40_features.json")) as f:
        final_40_features = json.load(f)
    with open(os.path.join(base_dir, "chosen_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    with open(os.path.join(base_dir, "encoded_columns.json")) as f:
        encoded_columns = json.load(f)
    return model, num_imputer, cat_imputer, final_40_features, threshold, encoded_columns


model, num_imputer, cat_imputer, final_40_features, threshold, encoded_columns = load_artifacts()

num_cols = list(num_imputer.feature_names_in_)
num_defaults = dict(zip(num_cols, num_imputer.statistics_))

for col in num_cols:
    if col not in st.session_state:
        st.session_state[col] = float(num_defaults[col])
if "ventilated_status" not in st.session_state:
    st.session_state["ventilated_status"] = "No"

# ---------------------------------------------------------------------------
# Hardcoded reference numbers from the training notebook
# (XGBoost is the only model actually saved/deployed; these are for
#  reporting/comparison display only — not live predictions)
# ---------------------------------------------------------------------------
MODEL_METRICS = {
    "Logistic Regression": {"AUROC": 0.874, "AUPRC": 0.462},
    "Random Forest":       {"AUROC": 0.889, "AUPRC": 0.511},
    "XGBoost":             {"AUROC": 0.893, "AUPRC": 0.554},
}
CV_AUROC_FOLDS = [0.8908, 0.8883, 0.8897, 0.8918, 0.8906]
CV_AUPRC_FOLDS = [0.5367, 0.5316, 0.5481, 0.5431, 0.5391]
CONFUSION = {"tn": 12915, "fp": 3845, "fn": 256, "tp": 1327}

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 0.5rem 0 1.2rem 0;">
        <div style="font-size: 1.6rem;">🏥</div>
        <div style="font-size: 1.05rem; font-weight: 700; color: white; margin-top: 0.2rem;">
            ICU Risk Predictor
        </div>
        <div style="font-size: 0.75rem; color: #778DA9;">Graduation Project · Data Science</div>
    </div>
    <hr style="border-color: rgba(255,255,255,0.15); margin: 0 0 1rem 0;">
    """,
    unsafe_allow_html=True,
)

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

st.sidebar.markdown("### 🩺 CLINICAL TOOLS")
clinical_pages = ["Overview", "Data Exploration", "Prediction"]
clinical_choice = st.sidebar.radio(
    label="Clinical",
    options=clinical_pages,
    index=clinical_pages.index(st.session_state["active_page"]) if st.session_state["active_page"] in clinical_pages else None,
    label_visibility="collapsed",
    key="clinical_radio",
)

st.sidebar.markdown(
    "<hr style='border-color: rgba(255,255,255,0.15); margin: 1.2rem 0 0.8rem 0;'>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='font-size:0.7rem; color:#778DA9; letter-spacing:1px; margin-bottom:0.3rem;'>"
    "FOR DEVELOPERS / TECHNICAL REVIEWERS</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("### ⚙️ TECHNICAL DETAILS")
technical_pages = ["Model Performance", "All Models Comparison"]
technical_choice = st.sidebar.radio(
    label="Technical",
    options=technical_pages,
    index=technical_pages.index(st.session_state["active_page"]) if st.session_state["active_page"] in technical_pages else None,
    label_visibility="collapsed",
    key="technical_radio",
)

# Whichever radio group the user just clicked determines the active page
if clinical_choice != st.session_state.get("_last_clinical"):
    st.session_state["active_page"] = clinical_choice
    st.session_state["_last_clinical"] = clinical_choice
    st.session_state["_last_technical"] = None
elif technical_choice != st.session_state.get("_last_technical"):
    st.session_state["active_page"] = technical_choice
    st.session_state["_last_technical"] = technical_choice
    st.session_state["_last_clinical"] = None

page = st.session_state["active_page"]

# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "Overview":
    st.title("🏥 ICU Mortality Risk Predictor")
    st.caption("A clinical decision-support tool for early risk assessment")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients Analyzed", "91,713")
    c2.metric("Observed Mortality", "8.6%")
    c3.metric("Deaths Correctly Flagged", "84%")
    c4.metric("Validation", "5-Fold Tested")

    st.write("")
    st.markdown("#### About this tool")
    st.markdown(
        """
This tool estimates a patient's risk of in-hospital mortality based on admission data —
vital signs, lab results, and clinical scores collected during the first day in the ICU.

It was trained and validated on 91,713 real ICU patient records, and is designed to support
— not replace — clinical judgment.

**How to use it:**
- Go to the **Prediction** page
- Enter the patient's vitals, labs, and clinical scores (or upload a CSV)
- Get an instant risk estimate with the specific vitals driving that result

**Reliability:**
- Correctly flags 84% of patients who are at high risk of not surviving
- Tested for consistency across 5 different patient sample groups, with stable results each time
        """
    )

    st.info("⚕️ This tool provides a decision-support estimate, not a diagnosis. Always use clinical judgment alongside this output.")

# ===========================================================================
# PAGE 2 — DATA EXPLORATION
# ===========================================================================
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.caption("Key patterns found in the training data (91,713 ICU encounters)")
    st.divider()

    st.markdown("#### Class Imbalance")
    fig = go.Figure(go.Bar(
        x=["Survived", "Died"], y=[91.3, 8.6],
        marker_color=["#415A77", "#E4572E"], text=["91.3%", "8.6%"], textposition="outside"
    ))
    fig.update_layout(height=320, yaxis_title="Percentage (%)", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.info("A model predicting 'survived' for everyone would score ~91% accuracy while being clinically useless — this is why AUROC/AUPRC were used instead of accuracy throughout this project.")

    st.write("")
    st.markdown("#### Mortality Rate by Age Group")
    age_bins = ["15-23", "23-31", "31-38", "38-45", "45-53", "53-60", "60-67", "67-74", "74-82", "82-89"]
    mortality = [2.9, 2.8, 4.2, 4.2, 5.2, 6.5, 9.7, 9.4, 11.3, 13.7]
    fig2 = px.bar(x=age_bins, y=mortality, labels={"x": "Age Bin", "y": "Mortality Rate (%)"})
    fig2.update_traces(marker_color="#E4572E")
    fig2.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Mortality rises steadily with age — from ~3% under 30 to ~14% over 81.")

    st.write("")
    st.markdown("#### Top Features by Mutual Information")
    mi_feats = ["apache_4a_icu_death_prob", "apache_4a_hospital_death_prob", "apache_3j_diagnosis",
                "gcs_motor_apache", "gcs_total_score", "gcs_eyes_apache", "shock_index",
                "d1_sysbp_min", "d1_spo2_range", "spo2_resprate_ratio"]
    mi_scores = [0.073, 0.071, 0.039, 0.039, 0.036, 0.036, 0.030, 0.028, 0.021, 0.021]
    fig3 = px.bar(x=mi_scores, y=mi_feats, orientation="h", labels={"x": "Information Gain", "y": ""})
    fig3.update_traces(marker_color="#1B263B")
    fig3.update_layout(height=400, yaxis={"categoryorder": "total ascending"}, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Engineered features (shock_index, spo2_range, spo2_resprate_ratio) rank alongside raw clinical scores — confirming they carry real predictive signal.")

# ===========================================================================
# PAGE 3 — PREDICTION (the original form, unchanged logic)
# ===========================================================================
elif page == "Prediction":
    st.title("🏥 ICU Mortality Risk Predictor")
    st.caption(
        "Estimates in-hospital mortality risk from ICU admission data. "
        "Fields left blank default to typical (median) baseline values from the training data."
    )
    st.divider()

    st.markdown("### 📥 Auto-Fill Form from CSV")
    st.info("Upload a patient CSV. The form below will automatically fill with the data from the first row so you don't have to type it manually.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if not df.empty:
                row_data = df.iloc[0]
                for col in df.columns:
                    if col in st.session_state:
                        st.session_state[col] = float(row_data[col])
                if "ventilated_apache" in df.columns:
                    st.session_state["ventilated_status"] = "Yes" if int(row_data["ventilated_apache"]) == 1 else "No"
                st.success("✅ Form successfully auto-filled! You can now review or edit the inputs below before predicting.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    st.write("")

    with st.form("patient_form"):
        st.markdown("### 📋 1. Clinical Scores & GCS")
        c1, c2 = st.columns(2)
        with c1:
            apache_4a_icu_death_prob = st.number_input(
                "APACHE IVa ICU death probability", min_value=0.0, max_value=1.0,
                value=float(st.session_state["apache_4a_icu_death_prob"]), step=0.01)
            apache_2_diagnosis = st.number_input(
                "APACHE II diagnosis code", value=float(st.session_state["apache_2_diagnosis"]), step=1.0)
            gcs_motor_apache = st.slider("Motor response", 1, 6, int(st.session_state["gcs_motor_apache"]))
            gcs_verbal_apache = st.slider("Verbal response", 1, 5, int(st.session_state["gcs_verbal_apache"]))
        with c2:
            apache_4a_hospital_death_prob = st.number_input(
                "APACHE IVa hospital death probability", min_value=0.0, max_value=1.0,
                value=float(st.session_state["apache_4a_hospital_death_prob"]), step=0.01)
            apache_3j_diagnosis = st.number_input(
                "APACHE IIIj diagnosis code", value=float(st.session_state["apache_3j_diagnosis"]), step=1.0)
            gcs_eyes_apache = st.slider("Eye opening", 1, 4, int(st.session_state["gcs_eyes_apache"]))
            vent_options = ["No", "Yes"]
            vent_index = vent_options.index(st.session_state["ventilated_status"])
            ventilated_apache = st.selectbox("Ventilated on admission?", vent_options, index=vent_index)

        st.write("")
        st.markdown("### 🫀 2. Vitals — First Day (Day 1) Min/Max")
        c1, c2 = st.columns(2)
        with c1:
            d1_heartrate_max = st.number_input("Heart rate — max (bpm)", value=float(st.session_state["d1_heartrate_max"]))
            d1_sysbp_min = st.number_input("Systolic BP — min (mmHg)", value=float(st.session_state["d1_sysbp_min"]))
            d1_sysbp_noninvasive_min = st.number_input("Systolic BP (non-inv) — min", value=float(st.session_state["d1_sysbp_noninvasive_min"]))
            d1_mbp_min = st.number_input("Mean BP — min (mmHg)", value=float(st.session_state["d1_mbp_min"]))
            d1_mbp_noninvasive_min = st.number_input("Mean BP (non-inv) — min", value=float(st.session_state["d1_mbp_noninvasive_min"]))
            d1_diasbp_min = st.number_input("Diastolic BP — min (mmHg)", value=float(st.session_state["d1_diasbp_min"]))
            d1_diasbp_noninvasive_min = st.number_input("Diastolic BP (non-inv) — min", value=float(st.session_state["d1_diasbp_noninvasive_min"]))
        with c2:
            d1_heartrate_min = st.number_input("Heart rate — min (bpm)", value=float(st.session_state["d1_heartrate_min"]))
            d1_spo2_min = st.number_input("SpO2 — min (%)", value=float(st.session_state["d1_spo2_min"]))
            d1_spo2_max = st.number_input("SpO2 — max (%)", value=float(st.session_state["d1_spo2_max"]))
            d1_temp_min = st.number_input("Temperature — min (°C)", value=float(st.session_state["d1_temp_min"]))
            d1_temp_max = st.number_input("Temperature — max (°C)", value=float(st.session_state["d1_temp_max"]))
            h1_resprate_min = st.number_input("Resp. rate (hour 1) — min", value=float(st.session_state["h1_resprate_min"]))
            d1_resprate_min = st.number_input("Resp. rate (day 1) — min", value=float(st.session_state["d1_resprate_min"]))
            h1_resprate_max = st.number_input("Resp. rate (hour 1) — max", value=float(st.session_state["h1_resprate_max"]))

        st.write("")
        st.markdown("### 🧪 3. Laboratory Results")
        c1, c2 = st.columns(2)
        with c1:
            bun_apache = st.number_input("BUN (APACHE)", value=float(st.session_state["bun_apache"]))
            d1_bun_max = st.number_input("BUN — max", value=float(st.session_state["d1_bun_max"]))
            d1_bun_min = st.number_input("BUN — min", value=float(st.session_state["d1_bun_min"]))
            creatinine_apache = st.number_input("Creatinine (APACHE)", value=float(st.session_state["creatinine_apache"]))
            d1_creatinine_max = st.number_input("Creatinine — max", value=float(st.session_state["d1_creatinine_max"]))
            d1_creatinine_min = st.number_input("Creatinine — min", value=float(st.session_state["d1_creatinine_min"]))
        with c2:
            temp_apache = st.number_input("Temperature (APACHE)", value=float(st.session_state["temp_apache"]))
            d1_hco3_min = st.number_input("HCO3 — min", value=float(st.session_state["d1_hco3_min"]))
            d1_hco3_max = st.number_input("HCO3 — max", value=float(st.session_state["d1_hco3_max"]))
            d1_wbc_max = st.number_input("WBC — max", value=float(st.session_state["d1_wbc_max"]))
            d1_platelets_min = st.number_input("Platelets — min", value=float(st.session_state["d1_platelets_min"]))

        st.write("")
        submitted = st.form_submit_button("Predict Mortality Risk", type="primary", use_container_width=True)

    if submitted:
        row = dict(num_defaults)
        row.update({
            "apache_4a_icu_death_prob": apache_4a_icu_death_prob,
            "apache_4a_hospital_death_prob": apache_4a_hospital_death_prob,
            "apache_2_diagnosis": apache_2_diagnosis,
            "apache_3j_diagnosis": apache_3j_diagnosis,
            "gcs_motor_apache": gcs_motor_apache,
            "gcs_eyes_apache": gcs_eyes_apache,
            "gcs_verbal_apache": gcs_verbal_apache,
            "ventilated_apache": 1 if ventilated_apache == "Yes" else 0,
            "d1_heartrate_max": d1_heartrate_max,
            "d1_heartrate_min": d1_heartrate_min,
            "d1_sysbp_min": d1_sysbp_min,
            "d1_sysbp_noninvasive_min": d1_sysbp_noninvasive_min,
            "d1_mbp_min": d1_mbp_min,
            "d1_mbp_noninvasive_min": d1_mbp_noninvasive_min,
            "d1_diasbp_min": d1_diasbp_min,
            "d1_diasbp_noninvasive_min": d1_diasbp_noninvasive_min,
            "d1_spo2_min": d1_spo2_min,
            "d1_spo2_max": d1_spo2_max,
            "d1_temp_min": d1_temp_min,
            "d1_temp_max": d1_temp_max,
            "h1_resprate_min": h1_resprate_min,
            "d1_resprate_min": d1_resprate_min,
            "h1_resprate_max": h1_resprate_max,
            "bun_apache": bun_apache,
            "d1_bun_max": d1_bun_max,
            "d1_bun_min": d1_bun_min,
            "creatinine_apache": creatinine_apache,
            "d1_creatinine_max": d1_creatinine_max,
            "d1_creatinine_min": d1_creatinine_min,
            "temp_apache": temp_apache,
            "d1_hco3_min": d1_hco3_min,
            "d1_hco3_max": d1_hco3_max,
            "d1_wbc_max": d1_wbc_max,
            "d1_platelets_min": d1_platelets_min,
        })

        X_row = pd.DataFrame([row])[num_cols]
        X_row["shock_index"] = X_row["d1_heartrate_max"] / (X_row["d1_sysbp_min"] + 1e-5)
        X_row["spo2_resprate_ratio"] = X_row["d1_spo2_min"] / (X_row["h1_resprate_max"] + 1e-5)
        X_row["gcs_total_score"] = X_row["gcs_motor_apache"] + X_row["gcs_eyes_apache"] + X_row["gcs_verbal_apache"]
        X_row["d1_heartrate_range"] = X_row["d1_heartrate_max"] - X_row["d1_heartrate_min"]
        X_row["d1_temp_range"] = X_row["d1_temp_max"] - X_row["d1_temp_min"]
        X_row["d1_spo2_range"] = X_row["d1_spo2_max"] - X_row["d1_spo2_min"]
        X_row["d1_bun_max_log"] = np.log1p(X_row["d1_bun_max"])
        X_row["d1_creatinine_max_log"] = np.log1p(X_row["d1_creatinine_max"])

        X_final = X_row[final_40_features].fillna(0)
        proba = model.predict_proba(X_final)[:, 1][0]
        prediction = int(proba >= threshold)

        st.divider()
        st.markdown("## 📊 Prediction Result")
        st.caption("👇 This is the summary view — a clean shot for sharing or reporting.")

        result_container = st.container(border=True)
        with result_container:
            c1, c2 = st.columns([1, 1.2])

            with c1:
                if prediction == 1:
                    badge_color, badge_bg, badge_text = "#E4572E", "#FDEDE8", "⚠️ HIGH RISK"
                else:
                    badge_color, badge_bg, badge_text = "#415A77", "#E8EBF0", "✅ LOW RISK"

                st.markdown(
                    f"""
                    <div style="text-align:center; padding: 1.2rem 0;">
                        <div style="font-size: 3.5rem; font-weight: 800; color: #0D1B2A; line-height: 1;">
                            {proba:.1%}
                        </div>
                        <div style="font-size: 0.95rem; color: #415A77; margin-top: 0.3rem;">
                            Predicted Mortality Risk
                        </div>
                        <div style="display:inline-block; margin-top: 1rem; padding: 0.45rem 1.1rem;
                                    background-color:{badge_bg}; color:{badge_color};
                                    border-radius: 20px; font-weight: 700; font-size: 0.95rem;
                                    letter-spacing: 0.5px;">
                            {badge_text}
                        </div>
                        <div style="font-size: 0.8rem; color: #778DA9; margin-top: 0.8rem;">
                            Decision threshold: {threshold:.0%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown("**Key Patient Vitals**")

                def vital_row(label, value, unit, is_flagged, flag_note):
                    color = "#E4572E" if is_flagged else "#415A77"
                    icon = "🔴" if is_flagged else "🟢"
                    note = f"<span style='color:{color}; font-size:0.8rem;'> — {flag_note}</span>" if is_flagged else ""
                    st.markdown(
                        f"<div style='padding:0.35rem 0; border-bottom:1px solid #EEF1F5;'>"
                        f"{icon} <b>{label}:</b> {value}{unit} {note}</div>",
                        unsafe_allow_html=True,
                    )

                vital_row("Systolic BP (min)", f"{d1_sysbp_min:.0f}", " mmHg", d1_sysbp_min < 90, "low — possible hypotension")
                vital_row("Heart Rate (max)", f"{d1_heartrate_max:.0f}", " bpm", d1_heartrate_max > 120, "elevated")
                vital_row("SpO2 (min)", f"{d1_spo2_min:.0f}", "%", d1_spo2_min < 90, "low oxygenation")
                vital_row("Temperature (max)", f"{d1_temp_max:.1f}", " °C", d1_temp_max > 38.3, "fever")
                vital_row("GCS Total Score", f"{gcs_motor_apache + gcs_eyes_apache + gcs_verbal_apache:.0f}", " / 15", (gcs_motor_apache + gcs_eyes_apache + gcs_verbal_apache) < 13, "reduced consciousness")

            st.caption(
                "**Disclaimer:** This tool provides a decision-support estimate based on historical machine learning data, "
                "not a medical diagnosis. Always use clinical judgment alongside this output."
            )

# ===========================================================================
# PAGE 4 — MODEL PERFORMANCE (XGBoost only — the deployed model)
# ===========================================================================
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.caption("XGBoost — the deployed model")
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("AUROC", "0.893")
    c2.metric("AUPRC", "0.554")
    c3.metric("Decision Threshold", "0.40")

    st.write("")
    st.markdown("#### Confusion Matrix (Threshold = 0.40)")
    z = [[CONFUSION["tn"], CONFUSION["fp"]], [CONFUSION["fn"], CONFUSION["tp"]]]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=["Predicted: Survived", "Predicted: Died"], y=["Actual: Survived", "Actual: Died"],
        text=z, texttemplate="%{text:,}", colorscale="Blues", showscale=False
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Only 256 of 1,583 real deaths are missed — an 84% catch rate, at the cost of more false alarms among survivors.")

    st.write("")
    st.markdown("#### Cross-Validation Stability (5-Fold)")
    c1, c2 = st.columns(2)
    with c1:
        fig_cv1 = px.bar(x=[f"Fold {i+1}" for i in range(5)], y=CV_AUROC_FOLDS, labels={"x": "", "y": "AUROC"})
        fig_cv1.update_traces(marker_color="#1B263B")
        fig_cv1.update_layout(height=300, yaxis_range=[0.85, 0.92], margin=dict(l=20, r=20, t=30, b=20),
                               title=f"AUROC per Fold (mean {np.mean(CV_AUROC_FOLDS):.4f})")
        st.plotly_chart(fig_cv1, use_container_width=True)
    with c2:
        fig_cv2 = px.bar(x=[f"Fold {i+1}" for i in range(5)], y=CV_AUPRC_FOLDS, labels={"x": "", "y": "AUPRC"})
        fig_cv2.update_traces(marker_color="#E4572E")
        fig_cv2.update_layout(height=300, yaxis_range=[0.45, 0.60], margin=dict(l=20, r=20, t=30, b=20),
                               title=f"AUPRC per Fold (mean {np.mean(CV_AUPRC_FOLDS):.4f})")
        st.plotly_chart(fig_cv2, use_container_width=True)
    st.caption(f"Std. dev: AUROC ±{np.std(CV_AUROC_FOLDS):.4f}, AUPRC ±{np.std(CV_AUPRC_FOLDS):.4f} — low variance confirms the result is stable, not a lucky split.")

# ===========================================================================
# PAGE 5 — ALL MODELS COMPARISON (reference numbers, not live inference)
# ===========================================================================
elif page == "All Models Comparison":
    st.title("⚖️ All Models Comparison")
    st.caption("Reference results from the training notebook — only XGBoost is deployed live in this app")
    st.divider()

    st.warning("Logistic Regression and Random Forest were trained and evaluated during model selection, but their weights were not saved for deployment. The numbers below are reported from the training notebook for comparison only.")

    comp_df = pd.DataFrame(MODEL_METRICS).T.reset_index().rename(columns={"index": "Model"})
    fig = go.Figure()
    fig.add_trace(go.Bar(name="AUROC", x=comp_df["Model"], y=comp_df["AUROC"], marker_color="#1B263B"))
    fig.add_trace(go.Bar(name="AUPRC", x=comp_df["Model"], y=comp_df["AUPRC"], marker_color="#E4572E"))
    fig.update_layout(barmode="group", height=420, yaxis_range=[0, 1], margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comp_df.set_index("Model").style.format("{:.3f}"), use_container_width=True)
    st.caption("XGBoost wins on both metrics — especially AUPRC, which matters most given the 8.6% positive rate.")
