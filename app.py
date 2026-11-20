import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import credit_features  # noqa: F401  (the saved model refers to this module, so it must be importable)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "credit_model.pkl"
INFO_PATH = BASE_DIR / "model_info.json"

# ------------------------------
# 🎨 Streamlit Page Config
st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>💳 Creditworthiness Predictor</h1>
    <p style='text-align: center;'>Predict whether a customer is likely to have good or bad credit</p>
    <hr>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Educational demo trained on the 1994 German Credit dataset. "
    "It is not a lending tool and must not be used for real credit decisions."
)

if not (MODEL_PATH.exists() and INFO_PATH.exists()):
    st.error(
        "credit_model.pkl and model_info.json are required. Run credit_score_model_training.ipynb "
        "once and commit the files it creates."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_info():
    return json.loads(INFO_PATH.read_text())


model = load_model()
info = load_info()

# Friendly labels and starting values. Choices and numeric limits come from the training data
# (model_info.json), so the app only offers values the model has actually seen.
LABELS = {
    "checking_status": "Checking Account Status",
    "duration": "Duration (months)",
    "credit_history": "Credit History",
    "purpose": "Purpose",
    "credit_amount": "Credit Amount",
    "savings_status": "Savings Status",
    "employment": "Employment Duration",
    "installment_commitment": "Installment Commitment",
    "other_parties": "Other Parties",
    "residence_since": "Years at Current Residence",
    "property_magnitude": "Property",
    "age": "Age",
    "other_payment_plans": "Other Payment Plans",
    "housing": "Housing",
    "existing_credits": "Existing Credits",
    "job": "Job",
    "num_dependents": "Number of Dependents",
    "own_telephone": "Own Telephone",
}
DEFAULTS = {
    "checking_status": "no checking",
    "duration": 24,
    "credit_history": "critical/other existing credit",
    "purpose": "radio/tv",
    "credit_amount": 1500,
    "savings_status": "<100",
    "employment": "unemployed",
    "installment_commitment": 2,
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "real estate",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
}


def label(column: str) -> str:
    return LABELS.get(column, column.replace("_", " ").capitalize())


# ------------------------------
# 📝 Sidebar
st.sidebar.title("Input Parameters")
st.sidebar.markdown("Please fill the details:")


def user_input() -> pd.DataFrame:
    values = {}
    for column in info["inputs"]:
        if column in info["categorical"]:
            options = info["categorical"][column]
            default = DEFAULTS.get(column)
            index = options.index(default) if default in options else 0
            values[column] = st.sidebar.selectbox(label(column), options, index=index)
        else:
            spec = info["numeric"][column]
            low, high = spec["min"], spec["max"]
            start = min(max(DEFAULTS.get(column, spec["median"]), low), high)
            if spec["is_int"] and high - low <= 80:
                values[column] = st.sidebar.slider(
                    label(column), min_value=int(low), max_value=int(high), value=int(start)
                )
            elif spec["is_int"]:
                values[column] = st.sidebar.number_input(
                    label(column), min_value=int(low), max_value=int(high), value=int(start), step=50
                )
            else:
                values[column] = st.sidebar.number_input(
                    label(column), min_value=float(low), max_value=float(high), value=float(start)
                )
    return pd.DataFrame([values])[info["inputs"]]


# ------------------------------
# 🧠 Prediction (credit_per_month is created inside the saved pipeline)
input_df = user_input()

if st.button("🔍 Predict Creditworthiness"):
    p_good = float(model.predict_proba(input_df)[0, list(model.classes_).index(1)])
    threshold = float(info["threshold"])

    if p_good >= threshold:
        st.success(f"✅ Prediction: Good Creditworthiness (model score: {p_good:.2f})")
        st.balloons()
    else:
        st.error(f"❌ Prediction: Bad Creditworthiness (model score: {p_good:.2f})")

    st.caption(
        f"Decision threshold: {threshold:.2f}. The score is the model's estimated chance of a good loan, "
        f"and it is not a calibrated probability. The threshold is deliberately cautious because approving a "
        f"bad borrower costs {info['cost_bad_approved']}x more than rejecting a good one."
    )

# ------------------------------
# 📌 Footer
st.markdown(
    """
<hr>
<p style='text-align: center;'>Made with 💚 by Maheshwari Kalaskar | AI Credit Scoring App</p>
""",
    unsafe_allow_html=True,
)
