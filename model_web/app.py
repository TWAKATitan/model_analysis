import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import io

# 設定 Streamlit 頁面配置
st.set_page_config(
    page_title="MLOps - VotingClassifier 預測與可視化",
    page_icon="📊",
    layout="wide"
)

# 頁面標題
st.title("📊 MLOps - VotingClassifier 預測與可視化")
st.markdown("本應用可上傳 **CSV 檔案**，進行機器學習模型預測，並提供 **數據可視化分析**。")
st.markdown("---")  # 分隔線

# **1️⃣ 側邊欄：載入模型 & 上傳 CSV**
st.sidebar.header("📥 載入機器學習模型")
model_path = "model_web/stacking_model.joblib"
imputer_path = "model_web/imputer.joblib"
scaler_path = "model_web/scaler.joblib"


try:
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    st.sidebar.success("✅ 模型載入成功！")
except FileNotFoundError:
    st.sidebar.error("❌ 找不到模型，請先訓練並生成 `model.joblib`")

st.sidebar.header("📂 上傳 CSV 進行預測")
uploaded_file = st.sidebar.file_uploader("請選擇 CSV 檔案", type=["csv"])

# **2️⃣ 預設顯示內容（若未上傳 CSV）**
if uploaded_file is None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 範例數據格式")
        example_df = pd.DataFrame({
            "Feature1": [0.5, 1.2, 3.3],
            "Feature2": [2.1, 0.8, 4.5],
            "Feature3": [1.1, 3.0, 2.8]
        })
        st.table(example_df)

    with col2:
        st.subheader("📊 預測結果類別")
        st.markdown("""
        - **CN (0)** - 健康個體  
        - **MCI (1)** - 輕度認知障礙  
        - **AD (2)** - 阿茲海默症  
        """)

    st.markdown("---")
    st.subheader("📈 預測結果分佈範例")
    fig, ax = plt.subplots()
    ax.bar(["CN", "MCI", "AD"], [10, 20, 15], color=["skyblue", "lightcoral", "gold"])
    ax.set_xlabel("Predicted Group")
    ax.set_ylabel("Count")
    ax.set_title("Example Prediction Distribution")
    st.pyplot(fig)

# **3️⃣ 處理上傳的 CSV**
if uploaded_file is not None:
    # 讀取 CSV
    data = pd.read_csv(uploaded_file, na_values="--")
    
    st.subheader("📌 原始數據")
    st.dataframe(data)

    # **數據處理與預測**
    st.markdown("---")
    st.subheader("🎯 預測結果")

    # 提取特徵
    X = data.iloc[:, 1:]
    X["sum"] = X.sum(axis=1)

    # 缺失值處理
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    # 進行預測
    y_pred = model.predict(X_scaled)
    data["Predicted_Group"] = y_pred

    # 顯示預測結果
    st.dataframe(data[["Predicted_Group"]])

    # **下載預測結果**
    csv_output = data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下載預測結果", data=csv_output, file_name="predictions.csv", mime="text/csv")

    # **可視化區塊**
    st.markdown("---")
    st.subheader("📊 預測結果分析")

    pred_counts = data["Predicted_Group"].value_counts()
    pred_labels = pred_counts.index.map({0: "CN", 1: "MCI", 2: "AD"})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔹 預測結果分佈（長條圖）")
        fig_bar_chart, ax = plt.subplots()
        ax.bar(pred_labels, pred_counts.values, color=["skyblue", "lightcoral", "gold"])
        ax.set_xlabel("Predicted Group")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Result Distribution")
        buf = io.BytesIO()
        fig_bar_chart.savefig(buf, format="png")
        st.pyplot(fig_bar_chart)
        st.download_button("⬇️ 下載長條圖", data=buf.getvalue(), file_name="bar_chart.png", mime="image/png")

    with col2:
        st.markdown("### 🔹 預測結果比例（圓餅圖）")
        fig_pie_chart, ax = plt.subplots()
        ax.pie(pred_counts.values, labels=pred_labels, autopct="%1.1f%%", colors=["skyblue", "lightcoral", "gold"])
        ax.axis("equal")
        ax.set_title("Prediction Result Proportion")
        buf = io.BytesIO()
        fig_pie_chart.savefig(buf, format="png")
        st.pyplot(fig_pie_chart)
        st.download_button("⬇️ 下載圓餅圖", data=buf.getvalue(), file_name="pie_chart.png", mime="image/png")

    st.markdown("### 🔹 互動式長條圖（Plotly）")
    fig = px.bar(
        x=pred_labels,
        y=pred_counts.values,
        labels={"x": "Predicted Group", "y": "Count"},
        title="Prediction Result Distribution",
        color=pred_labels
    )
    buf = io.StringIO()
    fig.write_html(buf, include_plotlyjs=False)
    st.plotly_chart(fig)
    st.download_button("⬇️ 下載互動式長條圖", data=buf.getvalue(), file_name="interactive_bar_chart.html", mime="text/html")
