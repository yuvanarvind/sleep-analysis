import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt
import datetime

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Sleep Analysis Dashboard",
    layout="wide"
)

st.title("Sleep Analysis Dashboard")

# -------------------------------------------------------
# INTRO / LANDING CONTENT
# -------------------------------------------------------
st.markdown(
    """
<div style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
  <h3>About this Dashboard</h3>
  <p>
    This dashboard analyzes Yuvan’s sleep data collected using the
    <strong>Ultrahuman Ring Air</strong> over roughly 1.5 years.
    The data covers multiple periods (with some gaps) at approximately
    one-minute resolution.
  </p>

  <h4>Raw sensor data from the ring</h4>
  <ul>
    <li><strong>raw_hr</strong> – heart rate</li>
    <li><strong>raw_hrv_2</strong> – heart rate variability</li>
    <li><strong>raw_motion</strong> – motion intensity</li>
    <li><strong>respiratory_rate</strong> – breathing rate (BPM)</li>
    <li><strong>spo2</strong> – blood oxygen saturation</li>
    <li><strong>steps</strong> – step count</li>
    <li><strong>temp</strong> – skin temperature</li>
    <li><strong>timestamp_epoch</strong> – sensor timestamp</li>
  </ul>

  <h4>Engineered features used for explainability</h4>
  <ul>
    <li>Rolling averages for HR, HRV, motion, temperature, respiratory rate</li>
    <li>Stillness index and a derived “slowing score”</li>
    <li>Awake flag and modeled sleep score</li>
    <li>Five lifestyle proxies: bedtime, caffeine, alcohol, stress, activity</li>
  </ul>

  <p>
    The goal is not just to report a score, but to explain
    <strong>why</strong> a night was good or bad, using SHAP values
    to attribute contributions across physiology and lifestyle.
  </p>

  <hr/>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# LOAD DATA & MODEL
# -------------------------------------------------------
@st.cache_data
def load_data():
    # Main dataset with engineered features
    df = pd.read_csv("df_scaled.csv.gz", compression="gzip")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


@st.cache_resource
def load_model_and_explainer():
    model = joblib.load("model.pkl")
    explainer = shap.TreeExplainer(model)
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, explainer, feature_cols


df = load_data()
model, explainer, feature_cols = load_model_and_explainer()

# -------------------------------------------------------
# GLOBAL SHAP FOR OVERALL BEHAVIOUR (BEESWARM)
# -------------------------------------------------------
@st.cache_resource
def compute_global_shap(df, feature_cols, max_samples=2000):
    X = df[feature_cols].dropna()
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
    shap_values = explainer.shap_values(X)
    return X, shap_values


with st.expander("Global feature importance across your data (SHAP summary)", expanded=True):
    X_global, shap_global = compute_global_shap(df, feature_cols)

    fig_global, ax_global = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_global,
        X_global.values,
        feature_names=feature_cols,
        plot_type="dot",
        show=False
    )
    st.pyplot(fig_global)


# -------------------------------------------------------
# SIDEBAR DATE PICKER
# -------------------------------------------------------
st.sidebar.header("Select Date for Nightly Analysis")

unique_dates = sorted(list({ts.date() for ts in df.index}))

# Default: 14 Nov 2025 if present, else the most recent date
default_date = datetime.date(2025, 11, 14)
if default_date not in unique_dates:
    default_date = max(unique_dates)

selected_date = st.sidebar.date_input(
    "Choose a date:",
    value=default_date,
    min_value=min(unique_dates),
    max_value=max(unique_dates)
)

sel_date = pd.Timestamp(selected_date).date()
if sel_date not in unique_dates:
    st.warning("No data available for the selected date.")
    st.stop()

# -------------------------------------------------------
# FILTER DATA FOR SELECTED DATE
# -------------------------------------------------------
df_day = df[df.index.date == sel_date]

if df_day.empty:
    st.error("No recordings available for this date.")
    st.stop()

x = df_day.iloc[[0]][feature_cols]

# -------------------------------------------------------
# PREDICT SLEEP SCORE
# -------------------------------------------------------
pred_score = float(model.predict(x)[0])

st.markdown("## Night-level analysis")
st.markdown(f"### Sleep score for {sel_date}  \n"
            f"<span style='font-size: 1.4rem;'><strong>{pred_score:.1f}/100</strong></span>",
            unsafe_allow_html=True)

# -------------------------------------------------------
# SHAP FOR SELECTED NIGHT (SINGLE ROW)
# -------------------------------------------------------
st.markdown("#### Feature contributions for this night")

shap_single = explainer.shap_values(x)[0]

fig, ax = plt.subplots(figsize=(8, 4))
# For a single row, shap.summary_plot still works with reshape
shap.summary_plot(
    shap_single.reshape(1, -1),
    x.values,
    feature_names=feature_cols,
    plot_type="dot",
    show=False
)
st.pyplot(fig)

# -------------------------------------------------------
# SUMMARY TEXT (CLEAR / HIGHLIGHTED)
# -------------------------------------------------------
st.markdown("#### Factor breakdown (confidence)")

shap_abs = np.abs(shap_single)
confidence = shap_abs / shap_abs.sum() * 100

summary_df = pd.DataFrame({
    "feature": feature_cols,
    "shap": shap_single,
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
    "activity_proxy": "your physical activity",
    "motion_roll": "your motion/stillness during sleep",
    "rr_roll": "your average breathing rate",
}

def effect(v):
    if v > 0:
        return "<span style='color: #1a7f37; font-weight:600;'>improved</span>"
    if v < 0:
        return "<span style='color: #b91c1c; font-weight:600;'>reduced</span>"
    return "<span style='color: #555;'>influenced</span>"

# st.markdown(
#     f"""
# <div style="padding: 0.5rem 0.75rem; border-left: 4px solid #4b8bbe; background-color: #f5f7fb; margin-bottom: 1rem;">
#   <p style="margin: 0 0 0.25rem 0;">
#     <strong>Predicted sleep score:</strong> {pred_score:.1f}/100
#   </p>
#   <p style="margin: 0;">
#     The numbers below show how strongly each factor contributed to this score.
#   </p>
# </div>
# """,
#     unsafe_allow_html=True,
# )

lines = []
for _, row in summary_df.iterrows():
    label = friendly.get(row["feature"], row["feature"])
    conf_str = f"{row['confidence']:.1f}%"
    eff = effect(row["shap"])
    lines.append(
        f"<li><strong>{conf_str}</strong> &rarr; {label} <strong>{eff}</strong> your sleep score</li>"
    )

st.markdown(
    "<ul style='margin-top: 0.2rem;'>" + "\n".join(lines) + "</ul>",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# INTERPRETATION + SUGGESTIONS
# -------------------------------------------------------
st.markdown("#### Interpretation")

st.markdown(
    """
The confidence percentages above indicate how strongly each factor pushed your sleep score up or down.
Higher percentages mean more influence—positive or negative.

Broadly:
- Stress patterns tend to raise heart rate and reduce HRV.
- Alcohol-like effects elevate temperature and suppress HRV.
- Caffeine near bedtime can delay sleep onset and increase restlessness.
- Late bedtime can misalign your circadian rhythm.
- Daytime activity builds sleep pressure and usually supports deeper sleep.
"""
)

st.markdown("#### Personalized suggestions")

row = df_day.iloc[0]
suggestions = []

if "alcohol_proxy" in row and row["alcohol_proxy"] > 0:
    suggestions.append("Alcohol-related patterns were present; these often reduce HRV and deep sleep quality.")
if "hrv_rolling" in row and row["hrv_rolling"] < df["hrv_rolling"].median():
    suggestions.append("HRV was below your typical baseline, indicating reduced recovery.")
if "hr_rolling" in row and row["hr_rolling"] > df["hr_rolling"].median():
    suggestions.append("Nighttime heart rate was elevated, which can limit deep sleep.")
if "stress_proxy" in row and row["stress_proxy"] > 10:
    suggestions.append("Nighttime stress markers were elevated. A calmer pre-bed routine may help.")
if "caffeine_proxy" in row and row["caffeine_proxy"] > 0:
    suggestions.append("Caffeine/stimulation was detected before sleep; consider limiting it later in the day.")

if not suggestions:
    suggestions.append("No strong negative lifestyle markers were detected for this night based on the proxies.")

for s in suggestions:
    st.write("- " + s)