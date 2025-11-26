# app.py
import os
print("WORKING DIR =", os.getcwd())

import streamlit as st

# Import the necessary functions from your ML.py
from ML import train_detector, evaluate_attack_file 

print("--- APP.PY STARTED SUCCESSFULLY ---")

# --- GUI Layout Setup ---
st.set_page_config(page_title="CAN IDS Analysis")
st.title("🛡️ SynCAN Intrusion Detector")

# Initialize detector in session state so it persists after training
if 'detector' not in st.session_state:
    st.session_state.detector = None

# Sidebar for attack selection
attack_options = {
    "plateau": "test_plateau.zip",
    "continuous": "test_continuous.zip",
    "playback": "test_playback.zip",
    "suppress": "test_suppress.zip",
    "flooding": "test_flooding.zip",
}
selected_attack = st.sidebar.selectbox("Select Attack File to Evaluate:", list(attack_options.keys()))
zip_file_name = attack_options[selected_attack]


def run_training():
    """Wrapper function to train the model and save it to state."""
    with st.spinner("Training Autoencoder on Normal Data..."):
        # The function already exists in ML.py
        st.session_state.detector = train_detector()
    st.success("Training Complete! Model is ready for evaluation.")

def run_evaluation():
    """Wrapper function to evaluate the selected attack."""
    if st.session_state.detector is None:
        st.error("Model is not trained. Please click 'Train Model' first.")
        return
        
    with st.spinner(f"Evaluating {selected_attack} attack on {zip_file_name}..."):
        # The function now returns the formatted string
        alert_output = evaluate_attack_file(st.session_state.detector, zip_file_name)
    
    st.subheader(f"Results for: {selected_attack.capitalize()} Attack")
    
    # Display the result in a text area
    st.text_area(
        label="Suspicious ID Report",
        value=alert_output,
        height=300
    )
    st.success("Evaluation Done!")


# Main buttons in the app
if st.button("1. Train Model", key="train_btn"):
    run_training()

if st.button(f"2. Evaluate {selected_attack.capitalize()} Attack", key="eval_btn"):
    run_evaluation()