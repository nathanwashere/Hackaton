# app.py
import os
import streamlit as st
import pandas as pd

print("WORKING DIR =", os.getcwd())
print("--- APP.PY STARTED SUCCESSFULLY ---")

st.set_page_config(page_title="CAN IDS Analysis", layout="wide")
st.title("🛡️ SynCAN Intrusion Detector")

# -------------------------------------
# CACHING (TRAIN MODEL ONLY ONCE)
# -------------------------------------
@st.cache_resource
def load_detector():
    """Train the model ONCE and reuse it."""
    from ML import train_detector
    return train_detector()

def cached_attack_eval(_detector, _attack_zip):
    from ML import evaluate_attack_file
    return evaluate_attack_file(_detector, _attack_zip)


# -------------------------------------
# Session State Defaults
# -------------------------------------
if "detector" not in st.session_state:
    st.session_state.detector = None
if "attack_output" not in st.session_state:
    st.session_state.attack_output = {}

# -------------------------------------
# Attack options in sidebar
# -------------------------------------
attack_options = {
    "Plateau": "test_plateau.zip",
    "Continuous": "test_continuous.zip",
    "Playback": "test_playback.zip",
    "Suppress": "test_suppress.zip",
    "Flooding": "test_flooding.zip",
}

selected_attack = st.sidebar.selectbox("Select Attack Type:", list(attack_options.keys()))
zip_file_name = attack_options[selected_attack]

# -------------------------------------
# Main Buttons
# -------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("1️⃣ Train Model"):
        with st.spinner("Training Autoencoder... This will take some time for the first run."):
            st.session_state.detector = load_detector()
        st.success("✅ Model trained and ready!")

with col2:
    if st.button(f"2️⃣ Evaluate {selected_attack} Attack"):
        if st.session_state.detector is None:
            st.error("❌ Train the model first!")
        else:
            with st.spinner(f"Evaluating {selected_attack} attack..."):
                # Call the updated ML function and store dict
                from ML import evaluate_attack_file
                st.session_state.attack_output = cached_attack_eval(
                    st.session_state.detector, zip_file_name
                )
            st.success("✅ Evaluation complete!")

# -------------------------------------
# Display Attack Results
# -------------------------------------
if st.session_state.attack_output:
    result = st.session_state.attack_output
    
    st.subheader(f"Results for: {selected_attack} Attack")
    
    # Alert message
    if "alert_message" in result:
        st.text_area("Alert", result["alert_message"], height=150)

    # Top suspicious IDs
    if "top_ids" in result:
        st.write("Top suspicious IDs:", result["top_ids"])

    # Full per-ID DataFrame
    if "per_id" in result and isinstance(result["per_id"], pd.DataFrame):
        st.dataframe(result["per_id"])
