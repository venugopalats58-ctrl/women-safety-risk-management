import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Women Safety Risk Prediction",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "risk_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "CrimesOnWomenData[1].csv")

VERDICT_MAP = {
    "High": ("DANGER", "High reported crime in this location.", "🔴"),
    "Medium": ("CAUTION", "Moderate reported crime in this location.", "🟠"),
    "Low": ("SAFE", "Lower reported crime in this location.", "🟢"),
}

RECOMMENDATIONS = {
    "High": [
        "Avoid isolated areas, especially at night.",
        "Prefer well-lit and crowded locations.",
        "Keep emergency contacts ready.",
        "Use trusted transportation.",
        "Contact local authorities in an emergency.",
    ],
    "Medium": [
        "Stay aware of your surroundings.",
        "Prefer well-lit and populated areas.",
        "Avoid unnecessary travel through isolated locations.",
        "Keep emergency contacts available.",
    ],
    "Low": [
        "Continue normal safety precautions.",
        "Stay aware of your surroundings.",
        "Keep emergency contacts available.",
    ],
}

CRIME_LABELS = {
    "Rape": "Rape",
    "K&A": "Kidnapping & Abduction",
    "DD": "Dowry Deaths",
    "AoW": "Assault on Women",
    "AoM": "Assault on Modesty",
    "DV": "Domestic Violence",
    "WT": "Women Trafficking",
}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["State"] = df["State"].astype(str).str.strip()
    df["_state_key"] = df["State"].str.lower()
    return df


def run_prediction(bundle, feature_df):
    model = bundle["model"]
    le = bundle["label_encoder"]
    pred_code = model.predict(feature_df)[0]
    risk = le.inverse_transform([pred_code])[0]
    proba = model.predict_proba(feature_df)[0]
    prob_dict = dict(zip(le.classes_, np.round(proba * 100, 2)))
    return risk, round(float(proba.max()) * 100, 2), prob_dict


st.title("🛡️ Women Safety Risk Analysis")
st.caption("Location-based **DANGER / CAUTION / SAFE** prediction using historical crimes-against-women data.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`.")
    st.stop()

bundle = load_bundle()
feature_cols = bundle["features"]
df = load_data()

st.sidebar.header("Model Info")
st.sidebar.write(f"**Algorithm:** {bundle.get('model_name', 'Classifier')}")
st.sidebar.write(f"**Features:** {', '.join(feature_cols)}")

tab1, tab2 = st.tabs(["📍 Predict by Location (State & Year)", "🧮 Custom Crime Metrics Input"])

with tab1:
    if df is None:
        st.warning(
            "Dataset `CrimesOnWomenData[1].csv` not found in the project folder. "
            "Copy it into the folder to enable State/Year lookup, or use the **Custom Crime Metrics Input** tab."
        )
    else:
        states = sorted(df["State"].unique())
        col1, col2 = st.columns(2)
        with col1:
            selected_state = st.selectbox("Select State / UT", states)
        with col2:
            use_latest = st.checkbox("Use latest available year", value=True)
            selected_year = st.number_input(
                "Target Year",
                min_value=2001,
                max_value=2035,
                value=int(df["Year"].max()),
                step=1,
                disabled=use_latest,
            )

        if st.button("Analyze Location Risk", type="primary"):
            rows = df[df["_state_key"] == selected_state.strip().lower()]
            if use_latest:
                record = rows.sort_values("Year").iloc[-1]
                req_year = None
            else:
                record = rows.iloc[(rows["Year"] - int(selected_year)).abs().argsort().iloc[0]]
                req_year = int(selected_year)

            features = pd.DataFrame([record[feature_cols]], columns=feature_cols)
            risk, confidence, prob_dict = run_prediction(bundle, features)
            verdict, reason, icon = VERDICT_MAP[risk]

            if req_year is not None and req_year != int(record["Year"]):
                st.info(f"No record for **{req_year}**; used closest available year **{int(record['Year'])}**.")

            st.subheader(f"{icon} Prediction: {verdict} ({risk} Risk)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Location", record["State"])
            m2.metric("Data Year", int(record["Year"]))
            m3.metric("Confidence", f"{confidence}%")

            st.write(f"**Assessment:** {reason}")

            st.markdown("#### Safety Recommendations")
            for tip in RECOMMENDATIONS[risk]:
                st.markdown(f"- {tip}")

            st.markdown("#### Reported Crime Breakdown (Used by Model)")
            breakdown = pd.DataFrame({
                "Code": list(CRIME_LABELS.keys()),
                "Category": list(CRIME_LABELS.values()),
                "Reported Cases": [int(record[c]) for c in CRIME_LABELS.keys()],
            })
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

with tab2:
    st.write("Test the model directly by entering crime counts:")
    c1, c2, c3, c4 = st.columns(4)
    year_in = c1.number_input("Year", min_value=2001, max_value=2035, value=2021)
    rape_in = c2.number_input("Rape", min_value=0, value=500)
    ka_in = c3.number_input("K&A (Kidnapping)", min_value=0, value=600)
    dd_in = c4.number_input("DD (Dowry Deaths)", min_value=0, value=100)

    c5, c6, c7, c8 = st.columns(4)
    aow_in = c5.number_input("AoW (Assault)", min_value=0, value=1000)
    aom_in = c6.number_input("AoM (Modesty)", min_value=0, value=200)
    dv_in = c7.number_input("DV (Domestic Violence)", min_value=0, value=1500)
    wt_in = c8.number_input("WT (Trafficking)", min_value=0, value=10)

    if st.button("Predict Custom Input"):
        input_row = {
            "Year": year_in,
            "Rape": rape_in,
            "K&A": ka_in,
            "DD": dd_in,
            "AoW": aow_in,
            "AoM": aom_in,
            "DV": dv_in,
            "WT": wt_in,
        }
        features = pd.DataFrame([[input_row[col] for col in feature_cols]], columns=feature_cols)
        risk, confidence, prob_dict = run_prediction(bundle, features)
        verdict, reason, icon = VERDICT_MAP[risk]

        st.subheader(f"{icon} Prediction: {verdict} ({risk} Risk) — {confidence}% confidence")
        st.write(reason)
        st.bar_chart(pd.Series(prob_dict, name="Probability (%)"))