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
    <li><strong>timestamp_epoch</strong> – raw sensor timestamp</li>
  </ul>

  <h4>Engineered features used for explainability</h4>
  <ul>
    <li>Rolling averages for HR, HRV, motion, temperature, respiratory rate</li>
    <li>Stillness index and a derived slowing score</li>
    <li>Predicted sleep score (model output)</li>
    <li>Five lifestyle proxies: bedtime, caffeine, alcohol, stress, activity</li>
  </ul>

  <p>
    The goal is not just to report a nightly score, but to explain
    <strong>why</strong> a night was good or bad using SHAP values
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
# GLOBAL SHAP PLOT
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
# SIDEBAR DATE SELECTION
# -------------------------------------------------------
st.sidebar.header("Select Date for Nightly Analysis")

unique_dates = sorted(list({ts.date() for ts in df.index}))

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
    st.warning("No data available for this date.")
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
# MODEL PREDICTION
# -------------------------------------------------------
pred_score = float(model.predict(x)[0])

st.markdown("## Night-level analysis")
st.markdown(
    f"### Sleep score for {sel_date}<br>"
    f"<span style='font-size: 1.4rem;'><strong>{pred_score:.1f}/100</strong></span>",
    unsafe_allow_html=True
)

# -------------------------------------------------------
# SHAP SINGLE-NIGHT VALUES
# -------------------------------------------------------
st.markdown("#### Visual Contribution Chart")

shap_single = explainer.shap_values(x)[0]

# Only proxies
proxy_features = [
    "alcohol_proxy",
    "caffeine_proxy",
    "stress_proxy",
    "bedtime_proxy",
    "activity_proxy"
]

proxy_indices = [feature_cols.index(f) for f in proxy_features]
proxy_shap_values = shap_single[proxy_indices]
proxy_names = [feature_cols[i] for i in proxy_indices]

# Domain-based direction
def direction(feature, shap_val):
    if feature in ["alcohol_proxy", "caffeine_proxy", "stress_proxy"]:
        return "reduced"
    if shap_val > 0:
        return "improved"
    if shap_val < 0:
        return "reduced"
    return "influenced"

fig, ax = plt.subplots(figsize=(6,4))
colors = ["green" if direction(f, v) == "improved" else "red"
          for f, v in zip(proxy_names, proxy_shap_values)]
ax.barh(proxy_names, proxy_shap_values, color=colors)
ax.set_xlabel("SHAP value (impact on prediction)")
ax.invert_yaxis()
st.pyplot(fig)

# -------------------------------------------------------
# FACTOR BREAKDOWN (CONFIDENCE)
# -------------------------------------------------------
st.markdown("#### Factor breakdown (confidence)")

shap_abs = np.abs(proxy_shap_values)
conf = shap_abs / shap_abs.sum() * 100

friendly = {
    "alcohol_proxy": "alcohol-like physiological effects",
    "caffeine_proxy": "caffeine or stimulation before sleep",
    "stress_proxy": "your stress level",
    "bedtime_proxy": "your bedtime",
    "activity_proxy": "your physical activity"
}

def color(word):
    if word == "improved":
        return "<span style='color:green;font-weight:600;'>improved</span>"
    if word == "reduced":
        return "<span style='color:red;font-weight:600;'>reduced</span>"
    return word

lines = []
for f, v, c in zip(proxy_names, proxy_shap_values, conf):
    eff = direction(f, v)
    lines.append(
        f"<li><strong>{c:.1f}%</strong> → {friendly[f]} "
        f"<strong>{color(eff)}</strong> your sleep score</li>"
    )

st.markdown("<ul>" + "\n".join(lines) + "</ul>", unsafe_allow_html=True)

# -------------------------------------------------------
# INTERPRETATION
# -------------------------------------------------------
st.markdown("#### Interpretation")

st.markdown(
    """
The percentages above show how strongly each lifestyle factor influenced your predicted sleep score.

Broad patterns:
- Stress increases heart rate and reduces HRV.
- Alcohol-like effects elevate temperature and suppress HRV.
- Caffeine delays sleep onset and increases restlessness.
- Irregular bedtime disrupts circadian rhythm.
- Daytime activity improves sleep drive and recovery.
"""
)

# -------------------------------------------------------
# GENERIC + PERSONALIZED SUGGESTIONS
# -------------------------------------------------------
st.markdown("#### Personalized suggestions")

generic_suggestions = [
    "Maintain a consistent bedtime to support your circadian rhythm.",
    "Lower nighttime heart rate usually improves deep sleep.",
    "Higher HRV indicates better recovery — breathing exercises can help.",
    "Avoid caffeine or stimulants 6–8 hours before sleep.",
    "A calming pre-bed routine reduces nighttime stress.",
    "Alcohol close to bedtime reduces HRV and deep sleep quality.",
    "Daytime activity improves sleep pressure and recovery."
]

row = df_day.iloc[0] if len(df_day) > 0 else None
personalized = []

if row is not None:
    if row.get("alcohol_proxy", 0) > 0:
        personalized.append("Alcohol-like physiology detected — this often reduces HRV and deep sleep.")
    if row.get("stress_proxy", 0) > 10:
        personalized.append("Stress-related patterns detected. A calming pre-bed routine may help.")
    if row.get("caffeine_proxy", 0) > 0:
        personalized.append("Caffeine-like stimulation detected. Avoid caffeine 6–8 hours before bedtime.")
    if row.get("bedtime_proxy", 0) < df["bedtime_proxy"].median(skipna=True):
        personalized.append("Your bedtime was irregular; a consistent schedule improves sleep quality.")
    if row.get("activity_proxy", 0) < df["activity_proxy"].median(skipna=True):
        personalized.append("Low daytime activity detected; light exercise can help deepen sleep.")

if not personalized:
    personalized.append("No strong negative lifestyle markers detected for this night.")

st.write(f"### Suggestions for {selected_date}:")
for s in personalized:
    st.write("- " + s)