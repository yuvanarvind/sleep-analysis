import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Sleep Analysis Dashboard",
    layout="wide"
)

st.title("Sleep Analysis Dashboard")

# -------------------------------------------------------
# INTRO / DESCRIPTION
# -------------------------------------------------------
st.markdown("""
### Overview

This dashboard analyzes Yuvan’s sleep dataset collected using the Ultrahuman Ring Air over ~1.5 years.  
Some days are missing due to sensor charging gaps, but the dataset still includes hundreds of nights.

### Raw Signals Collected
- raw_hr – Heart rate  
- raw_hrv_2 – HRV  
- raw_motion – Motion intensity  
- respiratory_rate – Breathing BPM  
- spo2 – Oxygen saturation  
- steps – Step count  
- temp – Skin temperature  
- timestamp_epoch – Sensor timestamp  

### Engineered Features
- Rolling HR, HRV, motion, temperature, respiration  
- Stillness index  
- Slowing score  
- Awake flag  
- Sleep score (your computed version)  
- Lifestyle proxies (5 factors):  
  bedtime, caffeine, alcohol, stress, activity

### How to Use This App
Pick a date from the sidebar to analyze:
- Predicted sleep score  
- SHAP explainability  
- Factor breakdown and confidence  
- Personalized recommendations
""")

# -------------------------------------------------------
# LOAD DATA & MODEL
# -------------------------------------------------------
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

# -------------------------------------------------------
# SIDEBAR DATE PICKER (BUG FIXED)
# -------------------------------------------------------
st.sidebar.header("Select Date for Analysis")

unique_dates = sorted(list({ts.date() for ts in df.index}))

selected_date = st.sidebar.date_input(
    "Choose a date:",
    min_value=min(unique_dates),
    max_value=max(unique_dates)
)

if pd.Timestamp(selected_date).date() not in unique_dates:
    st.warning("No data available for the selected date.")
    st.stop()

# -------------------------------------------------------
# FILTER DATA FOR SELECTED DATE
# -------------------------------------------------------
df_day = df[df.index.date == pd.Timestamp(selected_date).date()]

if df_day.empty:
    st.error("No recordings available for this date.")
    st.stop()

x = df_day.iloc[[0]][feature_cols]

# -------------------------------------------------------
# PREDICT SCORE
# -------------------------------------------------------
pred_score = float(model.predict(x)[0])

st.subheader(f"Sleep Score Prediction for {selected_date}")
st.metric("Predicted Sleep Score", f"{pred_score:.1f} / 100")

# -------------------------------------------------------
# SHAP
# -------------------------------------------------------
st.subheader("Feature Contributions")

shap_values = explainer.shap_values(x)[0]

fig, ax = plt.subplots(figsize=(10, 4))
shap.summary_plot(
    shap_values.reshape(1, -1),
    x.values,
    feature_names=feature_cols,
    plot_type="dot",
    show=False
)
st.pyplot(fig)

# -------------------------------------------------------
# SUMMARY TEXT
# -------------------------------------------------------
st.subheader("Summary of Influencing Factors")

shap_abs = np.abs(shap_values)
confidence = shap_abs / shap_abs.sum() * 100

summary_df = pd.DataFrame({
    "feature": feature_cols,
    "shap": shap_values,
    "confidence": confidence
}).sort_values("confidence", ascending=False)

friendly = {
    "slowing_score": "your restlessness",
    "alcohol_proxy": "alcohol-like physiological effects",
    "respiratory_rate": "your breathing rate",
    "hrv_rolling": "your HRV",
    "temp_roll": "your body temperature",
    "hr_rolling": "your nighttime heart rate",
    "stress_proxy": "your stress level",
    "bedtime_proxy": "your bedtime",
    "caffeine_proxy": "caffeine or stimulation before sleep",
    "activity_proxy": "your physical activity"
}

def effect(v):
    if v > 0: return "improved"
    if v < 0: return "reduced"
    return "influenced"

st.markdown(f"Your predicted sleep score is **{pred_score:.1f}/100**.\n\nHere’s what influenced it:")

for _, row in summary_df.iterrows():
    label = friendly.get(row["feature"], row["feature"])
    st.markdown(f"- {row['confidence']:.1f}% confidence → {label} **{effect(row['shap'])}** your sleep score")

# -------------------------------------------------------
# INTERPRETATION
# -------------------------------------------------------
st.subheader("Interpretation")

st.markdown("""
Higher percentages reflect stronger influence—positive or negative—on your predicted sleep score.

Factors combine to affect your restorative quality:
- Stress spikes raise HR and reduce HRV  
- Alcohol reduces HRV and increases temperature  
- Caffeine delays sleep onset  
- Late bedtime shifts circadian alignment  
- Motion & stillness reflect sleep phase stability  
- Temperature and breathing reflect recovery  

These interactions give a more complete picture of why you felt good or drained the next day.
""")

# -------------------------------------------------------
# PERSONALIZED SUGGESTIONS
# -------------------------------------------------------
st.subheader("Personalized Suggestions")

row = df_day.iloc[0]
suggestions = []

if row["alcohol_proxy"] > 0:
    suggestions.append("Physiology matched alcohol-like patterns, which reduce HRV.")
if row["hrv_rolling"] < df["hrv_rolling"].median():
    suggestions.append("HRV was below your baseline, suggesting reduced recovery.")
if row["hr_rolling"] > df["hr_rolling"].median():
    suggestions.append("Nighttime HR was elevated, reducing deep sleep potential.")
if row["stress_proxy"] > 10:
    suggestions.append("Nighttime stress was detected. Try winding down earlier.")
if row["caffeine_proxy"] > 0:
    suggestions.append("Caffeine markers detected, which may delay deep sleep.")

if not suggestions:
    suggestions.append("No strong negative lifestyle markers detected this night.")

for s in suggestions:
    st.write("- " + s)
