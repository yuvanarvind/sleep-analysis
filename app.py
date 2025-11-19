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

This dashboard analyzes Yuvan’s personal sleep dataset collected using the Ultrahuman Ring Air over approximately 1.5 years.  
The dataset includes several continuous periods with some gaps but provides minute-level physiological sensor readings.

### Raw Signals Collected

The Ultrahuman Ring Air recorded the following raw data streams:
- **raw_hr** – Heart rate  
- **raw_hrv_2** – Heart rate variability  
- **raw_motion** – Motion intensity  
- **respiratory_rate** – Breathing rate (BPM)  
- **spo2** – Oxygen saturation  
- **steps** – Step count  
- **temp** – Skin temperature  
- **timestamp_epoch** – Epoch timestamp  

### Feature Engineering Performed

To understand the *drivers* of sleep quality rather than just the measurements, the following engineered features were added:
- Rolling averages for HR, HRV, motion, temperature, respiratory rate  
- Stillness index (motion stability)  
- Slowing score (HR + HRV relationship)  
- Awake flag  
- A modeled sleep score  
- Five lifestyle proxies:
  - bedtime proxy  
  - caffeine proxy  
  - alcohol proxy  
  - stress proxy  
  - activity proxy  

### How to Use the Dashboard

Select a specific day to explore:
- The predicted sleep score  
- The factor breakdown using SHAP explainability  
- A visual dot-plot showing contribution of each feature  
- A natural-language summary and personalized recommendations  

---
""")

# -------------------------------------------------------
# LOAD DATA + MODEL
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
# SIDEBAR DATE PICKER
# -------------------------------------------------------
st.sidebar.header("Select a Date")

unique_dates = sorted(list(set(df.index.date)))

selected_date = st.sidebar.selectbox(
    "Choose a date to analyze:",
    options=[None] + unique_dates,
    format_func=lambda x: "Select a date" if x is None else x
)

# Stop until the user selects a date
if selected_date is None:
    st.info("Please select a date from the left sidebar to view your sleep analysis.")
    st.stop()

# -------------------------------------------------------
# FILTER BY DATE
# -------------------------------------------------------
df_day = df[df.index.date == selected_date]

if len(df_day) == 0:
    st.error(f"No data available for {selected_date}.")
    st.stop()

# First entry of the day
x = df_day.iloc[[0]][feature_cols]

# -------------------------------------------------------
# PREDICT SLEEP SCORE
# -------------------------------------------------------
pred_score = float(model.predict(x)[0])

st.subheader(f"Sleep Score Prediction for {selected_date}")
st.metric(label="Predicted Sleep Score", value=f"{pred_score:.1f} / 100")

# -------------------------------------------------------
# SHAP VALUES
# -------------------------------------------------------
st.subheader("Feature Contributions (Explainability)")

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
# NATURAL LANGUAGE SUMMARY
# -------------------------------------------------------
st.subheader("Summary of Your Sleep Factors")

shap_abs = np.abs(shap_values)
confidence_scores = shap_abs / shap_abs.sum() * 100

summary_df = pd.DataFrame({
    "feature": feature_cols,
    "shap_value": shap_values,
    "confidence": confidence_scores
}).sort_values("confidence", ascending=False)

friendly_names = {
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

def impact_word(val):
    if val > 0: return "improved"
    if val < 0: return "reduced"
    return "influenced"

st.markdown(f"""
Your predicted sleep score for this night is **{pred_score:.1f}/100**.

Here’s what influenced your sleep the most:
""")

summary_lines = []
for _, row in summary_df.iterrows():
    f = row["feature"]
    human = friendly_names.get(f, f)
    conf = row["confidence"]
    eff = impact_word(row["shap_value"])
    summary_lines.append(f"- {conf:.1f}% confidence → {human} **{eff}** your sleep score")

st.markdown("\n".join(summary_lines))

# -------------------------------------------------------
# INTERPRETATION SECTION
# -------------------------------------------------------
st.subheader("Interpretation")

st.markdown("""
The percentages shown above represent how strongly each feature affected your predicted sleep score.  
Higher percentages indicate greater influence, whether positive or negative.

Sleep quality emerges from interactions between:
- Lifestyle inputs (caffeine, stress, alcohol, bedtime, activity)  
- Physiology (HR, HRV, temperature, breathing patterns, motion stillness)  
- Circadian alignment and recovery signals  

These combined factors help explain why certain nights feel more restorative than others.
""")

# -------------------------------------------------------
# PERSONALIZED SUGGESTIONS
# -------------------------------------------------------
st.subheader("Personalized Suggestions")

row = df_day.iloc[0]
suggestions = []

if row["alcohol_proxy"] > 0:
    suggestions.append("Alcohol-related physiological markers were present; these typically reduce HRV and deep sleep.")

if row["hrv_rolling"] < df["hrv_rolling"].median():
    suggestions.append("Lower-than-usual HRV suggests impaired recovery.")

if row["hr_rolling"] > df["hr_rolling"].median():
    suggestions.append("Elevated nighttime heart rate may have reduced deep-sleep potential.")

if row["stress_proxy"] > 10:
    suggestions.append("Stress signals were elevated at night, indicating your body did not fully settle.")

if row["caffeine_proxy"] > 0:
    suggestions.append("Caffeine was detected; it may have delayed sleep onset or increased restlessness.")

if len(suggestions) == 0:
    suggestions.append("No strong negative lifestyle markers detected for this sleep session.")

for s in suggestions:
    st.write("- " + s)
