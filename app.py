import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sleep Analysis Dashboard", layout="wide")


# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("df_scaled.csv.gz")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    explainer = joblib.load("explainer.pkl")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return model, explainer, feature_cols


df = load_data()
model, explainer, feature_cols = load_model()

# Limit to the 5 proxy features ONLY
proxy_features = [
    "alcohol_proxy",
    "caffeine_proxy",
    "stress_proxy",
    "bedtime_proxy",
    "activity_proxy"
]

st.title("Sleep Analysis Dashboard")
st.markdown("""
This dashboard analyzes **sleep-quality signals** collected from Yuvan’s Ultrahuman Ring Air over ~1.5 years.

### What the raw data includes:
- Heart rate  
- Heart rate variability (HRV)  
- Motion & restlessness  
- Temperature trends  
- Breathing rate  
- SpO2  
- Steps  
- Behavioral proxies (alcohol, caffeine, stress, bedtime, activity)

All data is cleaned, aligned to 1-minute intervals, engineered, and fed into an ML model.  
Selecting a date will generate a **predicted sleep score** and explain **which lifestyle factors influenced it**.
""")

# ---------- SIDEBAR DATE PICKER ----------
unique_dates = sorted(df.index.date)
default_date = unique_dates[-1]

selected_date = st.sidebar.selectbox("Choose a date to analyze:", options=unique_dates, index=len(unique_dates)-1)


# ---------- GENERIC SUMMARY ON LOAD ----------
if selected_date is None:
    st.subheader("Sleep Score Insights")
    st.info("Select a date to view your sleep score analysis.")
    st.stop()


# ---------- FETCH DAY DATA ----------
df_day = df[df.index.date == selected_date]

if len(df_day) == 0:
    st.warning("No data available for this date.")
    st.stop()

# ---------- PREDICT SCORE ----------
X_day = df_day[feature_cols]
pred_score = float(model.predict(X_day.mean().to_frame().T)[0])


# ---------- SHAP FOR ONE ROW ----------
row_for_shap = X_day.iloc[[0]]
shap_values = explainer(row_for_shap)
shap_single = shap_values.values[0]

# Filter to only proxy features
proxy_indices = [feature_cols.index(f) for f in proxy_features]
proxy_shap_values = shap_single[proxy_indices]
proxy_names = [feature_cols[i] for i in proxy_indices]

# Confidence percentages
shap_abs = np.abs(proxy_shap_values)
confidence_scores = shap_abs / shap_abs.sum() * 100


# ---------- DOMAIN-AWARE DIRECTION ----------
def direction_override(feature, shap_value):
    negative = ["alcohol_proxy", "stress_proxy", "caffeine_proxy"]
    positive = ["activity_proxy", "bedtime_proxy"]

    if feature in negative:
        return "reduced"
    if feature in positive:
        return "improved"
    return "influenced"


def colored(word, meaning):
    if meaning == "improved":
        return f"<span style='color:green;font-weight:600'>{word}</span>"
    if meaning == "reduced":
        return f"<span style='color:red;font-weight:600'>{word}</span>"
    return word


# ---------- RESULTS ----------
st.subheader(f"Predicted Sleep Score for {selected_date}: **{pred_score:.1f}/100**")

# ---------- SHAP BREAKDOWN ----------
st.markdown("### Factor Contribution Breakdown")

for feature, shap_val, conf in zip(proxy_names, proxy_shap_values, confidence_scores):
    direction = direction_override(feature, shap_val)
    colored_word = colored(direction, direction)
    st.markdown(
        f"**{conf:.1f}%** → {feature.replace('_proxy','').title()} "
        f"**{colored_word}** your sleep score"
        , unsafe_allow_html=True
    )


# ---------- SHAP BEESWARM-LIKE CHART ----------
st.markdown("### Visual Contribution Chart")

fig, ax = plt.subplots(figsize=(6,4))
colors = ["green" if direction_override(f, v)=="improved" else "red" for f, v in zip(proxy_names, proxy_shap_values)]
ax.barh(proxy_names, proxy_shap_values, color=colors)
ax.set_xlabel("SHAP value (impact on prediction)")
ax.invert_yaxis()
st.pyplot(fig)


# ---------- INTERPRETATION ----------
st.markdown("### Interpretation")
st.write("""
These percentages represent how strongly each lifestyle factor influenced your sleep score.
Positive contributions help your sleep.  
Negative contributions reduce recovery or deep sleep quality.
""")


# ---------- SUGGESTIONS ----------
st.markdown("### Personalized Suggestions")

row = df_day.iloc[0]
suggestions = []

if row.get("alcohol_proxy", 0) > 0:
    suggestions.append("Alcohol-like physiology detected — this often suppresses HRV & deep sleep.")
if row.get("stress_proxy", 0) > 20:
    suggestions.append("Your body showed stress-like signals. Consider unwinding earlier.")
if row.get("caffeine_proxy", 0) > 0:
    suggestions.append("Caffeine-like stimulation detected before bedtime.")
if row.get("bedtime_proxy", 0) < df["bedtime_proxy"].median():
    suggestions.append("Your sleep timing was irregular; this can reduce restorative deep sleep.")
if row.get("activity_proxy", 0) < df["activity_proxy"].median():
    suggestions.append("Low daily activity detected — gentle workouts often improve sleep pressure.")

if not suggestions:
    suggestions.append("No strong negative lifestyle markers detected for this night.")

for s in suggestions:
    st.write("- " + s)