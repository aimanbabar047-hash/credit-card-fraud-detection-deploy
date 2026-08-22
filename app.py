import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("smote_fraud_detection_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "A machine learning application for detecting potentially "
    "fraudulent credit card transactions."
)

st.info(
    "Model: SMOTE + Logistic Regression | "
    "ROC-AUC: 0.9765 | Recall: 89.80%"
)

st.divider()


# ============================================================
# SAMPLE TRANSACTIONS
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
# PREDICTION FUNCTION
# ============================================================

def predict_transaction(transaction):

    input_df = pd.DataFrame([transaction])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("🔍 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Potential Fraud Detected")

    else:

        st.success("✅ Transaction Appears Legitimate")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with col2:

        if prediction == 1:

            st.metric(
                "Prediction",
                "Fraud"
            )

        else:

            st.metric(
                "Prediction",
                "Legitimate"
            )


# ============================================================
# SAMPLE TRANSACTION SECTION
# ============================================================

st.subheader("🧪 Test with Sample Transactions")

st.write(
    "Use the sample transactions below to quickly demonstrate "
    "how the model works."
)

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🟢 Test Legitimate Transaction",
        use_container_width=True
    ):

        predict_transaction(
            legitimate_example
        )


with col2:

    if st.button(
        "🔴 Test Fraudulent Transaction",
        use_container_width=True
    ):

        predict_transaction(
            fraud_example
        )


st.divider()


# ============================================================
# CUSTOM TRANSACTION SECTION
# ============================================================

st.subheader("✍️ Enter a Custom Transaction")

st.write(
    "Enter values for the 30 features used by the trained model."
)


# Create columns for input fields
col1, col2, col3 = st.columns(3)

custom_input = {}


# Time
with col1:

    custom_input["Time"] = st.number_input(
        "Time",
        value=0.0,
        format="%.6f"
    )


# V1 - V10
with col1:

    for feature in feature_names[1:11]:

        custom_input[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )


# V11 - V20
with col2:

    for feature in feature_names[11:21]:

        custom_input[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )


# V21 - V28
with col3:

    for feature in feature_names[21:29]:

        custom_input[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )


# Amount
with col3:

    custom_input["Amount"] = st.number_input(
        "Amount",
        value=0.0,
        min_value=0.0,
        format="%.2f"
    )


# Custom prediction button
if st.button(
    "🔍 Predict Custom Transaction",
    type="primary",
    use_container_width=True
):

    predict_transaction(custom_input)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Card Fraud Detection | Machine Learning Deployment Project"
)