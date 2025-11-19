import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sleep Analysis Dashboard",
    page_icon="😴",
    layout="wide"
)

st.title("😴 Sleep Analysis & Explainability Dashboard")
st.write("Understand what affects your sleep using your ring data + SHAP explainability.")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("df_scaled.csv.gz", compression="gzip")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    explainer = shap.TreeExplainer(model)
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, explainer, feature_cols

df = load_data()
model, explainer, feature_cols = load_model()


# ---------- SIDEBAR ----------
st.sidebar.header("🔍 Select Date to Analyze")

unique_dates = sorted(list(set(df.index.date)))

selected_date = st.sidebar.date_input(
    "Choose a date from your dataset",
    value=unique_dates[0],
    min_value=unique_dates[0],
    max_value=unique_dates[-1]
)

st.sidebar.info("Pick a date to see your sleep score + factor contributions.")


# ---------- FILTER DATA ----------
df_day = df[df.index.date == selected_date]

if len(df_day) == 0:
    st.error("No data for this date.")
    st.stop()

# take first row of the day
x = df_day.iloc[[0]][feature_cols]

# ---------- PREDICTION ----------
pred_score = float(model.predict(x)[0])

st.subheader(f"📊 Predicted Sleep Score for {selected_date}")
st.metric(label="Sleep Score", value=f"{pred_score:.1f} / 100")

# ---------- SHAP EXPLANATION ----------
st.subheader("🔬 Factor Contributions (SHAP Values)")

shap_vals = explainer.shap_values(x)[0]
shap_abs = np.abs(shap_vals)
conf = shap_abs / shap_abs.sum() * 100

explanation_df = pd.DataFrame({
    "feature": feature_cols,
    "shap_value": shap_vals,
    "confidence (%)": conf
}).sort_values("confidence (%)", ascending=False)

st.dataframe(explanation_df, height=350)

# ---------- BAR CHART OF IMPORTANT FACTORS ----------
st.subheader("📈 Impact Breakdown")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(explanation_df["feature"], explanation_df["shap_value"])
ax.set_xlabel("SHAP Impact on Sleep Score")
st.pyplot(fig)

# ---------- NATURAL LANGUAGE INSIGHTS ----------
st.subheader("🧠 Personalized Insights")

def explain_row(row):
    insights = []

    if row["bedtime_proxy"] > 5:
        insights.append("You slept later than usual — this reduced your sleep quality.")

    if row["alcohol_proxy"] > 0:
        insights.append("Alcohol consumption was detected — this is known to reduce deep sleep.")

    if row["caffeine_proxy"] > 0:
        insights.append("Caffeine intake may have delayed your sleep onset.")

    if row["stress_proxy"] > 10:
        insights.append("High stress detected — likely contributed to restlessness.")

    if row["activity_proxy"] < 5:
        insights.append("Low physical activity may have reduced sleep pressure.")

    if len(insights) == 0:
        insights.append("No strong negatives detected — good recovery conditions!")

    return insights


insights = explain_row(df_day.iloc[0])

for point in insights:
    st.write("• " + point)

st.success("Done! Scroll up to explore visuals and explanations.")
