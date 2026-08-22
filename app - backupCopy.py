import streamlit as st
import pandas as pd
import joblib


# Load saved model
model = joblib.load("smote_fraud_detection_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# Title
st.title("💳 Credit Card Fraud Detection")

st.write(
    "Test the machine learning model using sample transactions "
    "from the Credit Card Fraud Detection dataset."
)


# ============================================================
# LEGITIMATE TRANSACTION
# ============================================================

legitimate_example = {
    "Time": 160760.000000,
    "V1": -0.674466,
    "V2": 1.408105,
    "V3": -1.110622,
    "V4": -1.328366,
    "V5": 1.388996,
    "V6": -1.308439,
    "V7": 1.885879,
    "V8": -0.614233,
    "V9": 0.311652,
    "V10": 0.650757,
    "V11": -0.857785,
    "V12": -0.229961,
    "V13": -0.199817,
    "V14": 0.266371,
    "V15": -0.046544,
    "V16": -0.741398,
    "V17": -0.605617,
    "V18": -0.392568,
    "V19": -0.162648,
    "V20": 0.394322,
    "V21": 0.080084,
    "V22": 0.810034,
    "V23": -0.224327,
    "V24": 0.707899,
    "V25": -0.135837,
    "V26": 0.045102,
    "V27": 0.533837,
    "V28": 0.291319,
    "Amount": 23.000000
}


# ============================================================
# FRAUDULENT TRANSACTION
# ============================================================

fraud_example = {
    "Time": 57007.000000,
    "V1": -1.271244,
    "V2": 2.462675,
    "V3": -2.851395,
    "V4": 2.324480,
    "V5": -1.372245,
    "V6": -0.948196,
    "V7": -3.065234,
    "V8": 1.166927,
    "V9": -2.268771,
    "V10": -4.881143,
    "V11": 2.255147,
    "V12": -4.686387,
    "V13": 0.652375,
    "V14": -6.174288,
    "V15": 0.594380,
    "V16": -4.849692,
    "V17": -6.536521,
    "V18": -3.119094,
    "V19": 1.715494,
    "V20": 0.560478,
    "V21": 0.652941,
    "V22": 0.081931,
    "V23": -0.221348,
    "V24": -0.523582,
    "V25": 0.224228,
    "V26": 0.756335,
    "V27": 0.632800,
    "V28": 0.250187,
    "Amount": 0.010000
}


# ============================================================
# FUNCTION FOR PREDICTION
# ============================================================

def make_prediction(transaction, actual_class):

    input_df = pd.DataFrame([transaction])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Potential Fraud Detected")
    else:
        st.success("✅ Transaction Appears Legitimate")

    st.write(f"Fraud Probability: **{probability:.2%}**")

    st.write(f"Actual Class: **{actual_class}**")


# ============================================================
# BUTTONS
# ============================================================

st.subheader("Test the Model")

col1, col2 = st.columns(2)

with col1:

    if st.button("🟢 Test Legitimate Transaction"):

        make_prediction(
            legitimate_example,
            0
        )


with col2:

    if st.button("🔴 Test Fraudulent Transaction"):

        make_prediction(
            fraud_example,
            1
        )