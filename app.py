import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return joblib.load("banking_rf_model.pkl")

model = load_model()

# ------------------ Page Config ------------------
st.set_page_config(page_title="💳 Banking Fraud Detection Dashboard", layout="wide")
st.title("💳 Banking Fraud Detection System")
st.write("Upload transaction data or enter manually to predict fraud detection using RandomForest.")

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.write("### 📄 Uploaded Data Preview")
        st.dataframe(data.head())

        # Run prediction
        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1]

        data["Prediction"] = preds
        data["Fraud_Probability"] = probs

        # ------------------ Results ------------------
        st.write("### ✅ Prediction Results")
        st.dataframe(
            data.style.apply(
                lambda row: ["background-color: #ffcccc" if row.Prediction == 1 else "" for _ in row],
                axis=1,
            )
        )

        # ------------------ Fraud vs Legit Pie ------------------
        st.write("### 🥧 Fraud vs Legit Breakdown")
        fraud_ratio = data["Prediction"].value_counts().reset_index()
        fraud_ratio.columns = ["Class", "Count"]
        fig_pie = px.pie(
            fraud_ratio,
            values="Count",
            names="Class",
            color="Class",
            color_discrete_map={0: "green", 1: "red"},
            hole=0.4,
            title="Fraud vs Legit Transactions"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # ------------------ Fraud Probability Distribution ------------------
        st.write("### 🔎 Fraud Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=data, x="Fraud_Probability", hue="Prediction", kde=True, bins=30, ax=ax)
        st.pyplot(fig)

        # ------------------ Feature Distribution Example ------------------
        if "Cred1" in data.columns:
            st.write("### 📈 Feature Distribution: Cred1")
            fig2, ax2 = plt.subplots()
            sns.kdeplot(data=data, x="Cred1", hue="Prediction", fill=True, ax=ax2)
            st.pyplot(fig2)

        # ------------------ Sidebar Filters ------------------
        st.sidebar.header("🔍 Filters")
        threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.8, 0.01)
        filtered_data = data[data["Fraud_Probability"] >= threshold]

        st.write(f"### 🚨 Transactions with Fraud Probability ≥ {threshold}")
        st.dataframe(filtered_data)

        # Search by Transaction ID if exists
        if "TransactionID" in data.columns:
            txn_id = st.sidebar.text_input("Search by Transaction ID")
            if txn_id:
                result = data[data["TransactionID"].astype(str) == txn_id]
                st.write("### 🎯 Search Result")
                st.dataframe(result)

        # ------------------ Correlation Heatmap ------------------
        st.write("### 🔗 Feature Correlation with Fraud")
        corr = data.corr(numeric_only=True)
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax_corr)
        st.pyplot(fig_corr)

        # ------------------ Fraud Trend Over Time ------------------
        if "TransactionTime" in data.columns:
            try:
                st.write("### ⏳ Fraud Trend Over Time")
                data["TransactionTime"] = pd.to_datetime(data["TransactionTime"], errors="coerce")
                time_trend = data.groupby(pd.Grouper(key="TransactionTime", freq="D"))["Prediction"].mean().reset_index()
                fig_time = px.line(time_trend, x="TransactionTime", y="Prediction", title="Fraud Rate Over Time")
                st.plotly_chart(fig_time, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not parse TransactionTime column: {e}")

        # ------------------ Download Options ------------------
        st.write("### ⬇️ Download Results")
        csv_all = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download All Predictions", csv_all, "predictions.csv", "text/csv")

        if not filtered_data.empty:
            csv_fraud = filtered_data.to_csv(index=False).encode("utf-8")
            st.download_button("Download High-Risk Fraud Cases", csv_fraud, "fraud_cases.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ------------------ Manual Input Section ------------------
st.write("---")
st.write("### 📝 Manual Input for Single Prediction")

feature_list = list(model.feature_names_in_)
feature_inputs = {}

for feature in feature_list:
    feature_inputs[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("🔮 Predict Fraud"):
    try:
        input_df = pd.DataFrame([feature_inputs])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"✅ Prediction: {'Fraud' if pred == 1 else 'Legit'} (Probability: {prob:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
