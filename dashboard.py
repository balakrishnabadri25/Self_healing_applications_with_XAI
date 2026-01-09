import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import time
from self_healing import remedial_action, log_action, predict_message
import google.generativeai as genai

# =========================
# 🔐 Gemini API (HARDCODED)
# =========================
GEMINI_API_KEY = "AIzaSyC-ah0cOVfc9qdH_qt29FAyy7QbZCuOoMs"  
genai.configure(api_key=GEMINI_API_KEY)

def explain_anomaly(log_message: str) -> str:
    """
    One-line explanation using Gemini.
    """
    prompt = (
        f"Log: '{log_message}'. "
        "In one sentence, explain the possible system issue. "
        "If normal, say 'No issue.'"
    )
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No explanation available."
    except Exception:
        return "Explanation unavailable."

# =========================
# 📦 Load ML artifacts
# =========================
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/log_anomaly_model.pkl")
    return vectorizer, model

vectorizer, model = load_artifacts()

# =========================
# 🧾 Init actions log
# =========================
LOG_FILE = "actions_log.csv"
cols = ["timestamp", "message", "prediction", "action", "confidence"]

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

# =========================
# 🎛 Streamlit UI
# =========================
st.set_page_config(page_title="AIOps Self-Healing", layout="wide")
st.title("🤖 AIOps Self-Healing Simulator")
st.markdown(
    "Simulates application logs, detects anomalies using ML, "
    "and triggers automated remediation actions."
)

# Sidebar
st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Mode", ["Simulate Stream", "Manual Test"])
n_msgs = st.sidebar.slider("Messages per run", 1, 50, 20)
sleep = st.sidebar.slider("Delay (ms)", 0, 2000, 300)

# =========================
# 🔍 Prediction logic
# =========================
def predict_and_render(message, index, placeholder):
    pred, prob = predict_message(message, vectorizer, model)
    ts = datetime.utcnow().isoformat()

    with placeholder.container():
        if pred == 1:
            action = remedial_action(message)
            log_action(ts, message, pred, action, prob)
            st.markdown(f"**#{index+1} — {ts} — :red[ANOMALY]**")
            st.write(message)
            st.warning(f"Action: {action} | Confidence: {prob:.2f}")
            st.info(f"🔍 Why? {explain_anomaly(message)}")
        else:
            st.markdown(f"**#{index+1} — {ts} — :green[NORMAL]**")
            st.write(message)
            st.write(f"Confidence: {prob:.2f}")

# =========================
# ▶ Run modes
# =========================
if mode == "Simulate Stream":
    if st.sidebar.button("Start Simulation"):
        df_logs = pd.read_csv("data/sample_logs.csv")
        placeholder = st.empty()

        for i in range(n_msgs):
            msg = df_logs.sample(1).iloc[0]["message"]
            predict_and_render(msg, i, placeholder)
            time.sleep(sleep / 1000)

        st.success("Simulation completed.")
else:
    user_msg = st.sidebar.text_area(
        "Log message",
        value="ERROR - Database connection retrying"
    )
    if st.sidebar.button("Test Message"):
        placeholder = st.empty()
        predict_and_render(user_msg, 0, placeholder)

# =========================
# 📊 Action history
# =========================
st.header("Remediation Actions Log")
df_actions = pd.read_csv(LOG_FILE)
st.dataframe(df_actions.sort_values("timestamp", ascending=False).head(50))
