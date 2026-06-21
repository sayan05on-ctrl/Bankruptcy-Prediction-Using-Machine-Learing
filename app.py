import streamlit as st
import pandas as pd
import joblib

# =====================================
# LOAD MODEL AND SCALER
# =====================================

model = joblib.load("models/bankruptcy_xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Bankruptcy Risk Prediction System")

st.write(
    """
    Upload a CSV file containing company financial indicators.
    The model will predict bankruptcy probability.
    """
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(df.head())

    try:

        scaled_data = scaler.transform(df)

        prediction = model.predict(
            scaled_data
        )

        probability = model.predict_proba(
            scaled_data
        )[:,1]

        result = df.copy()

        result["Prediction"] = prediction

        result["Bankruptcy Probability"] = probability

        result["Risk Level"] = result[
            "Bankruptcy Probability"
        ].apply(
            lambda x:
            "High Risk"
            if x >= 0.70
            else (
                "Medium Risk"
                if x >= 0.30
                else "Low Risk"
            )
        )

        st.subheader("Prediction Results")

        st.dataframe(result)

        csv = result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="bankruptcy_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(str(e))
