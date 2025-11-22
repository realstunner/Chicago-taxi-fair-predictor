import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Configuration & Model Loading ---
# Load the trained Random Forest model once when the app starts
@st.cache_resource
def load_model():
    try:
        # NOTE: Make sure the filename matches your saved model!
        model = joblib.load("taxi_model.joblib")
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'taxi_model.joblib' not found. Ensure it's in the same folder.")
        return None

model_rf = load_model()

# Define the exact feature names used during model training
FEATURE_NAMES = ['TRIP_MILES', 'TRIP_SECONDS', 'TRIP_START_HOUR', 
                 'RUSH_HOUR', 'AVERAGE_SPEED_MPH']

# --- Prediction Function ---
def predict_fare(miles, seconds, hour):
    # 1. Feature Engineering: Replicate training logic
    
    # RUSH_HOUR: 7-10 AM and 4-7 PM
    is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0

    # AVERAGE_SPEED_MPH: miles / (seconds / 3600)
    # Using 1e-6 to avoid division by zero
    average_speed = miles / (seconds / 3600 + 1e-6)

    # 2. Create the input array in the exact order the model expects
    input_data = [
        miles, 
        seconds, 
        hour, 
        is_rush_hour, 
        average_speed
    ]
    
    # 3. Convert to DataFrame (prevents Scikit-learn UserWarning)
    X_predict = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    
    # 4. Predict
    fare = model_rf.predict(X_predict)[0]
    return fare

# --- Streamlit UI ---

st.set_page_config(layout="centered", page_title="Taxi Fare Predictor")
st.title("🚕 Chicago Taxi Fare Predictor")
st.markdown("Enter trip details to estimate the fare using the trained LightGBM Regressor model.")

if model_rf is not None:
    
    # Input Widgets
    miles = st.slider("Trip Distance (Miles):", min_value=0.1, max_value=25.0, value=3.0, step=0.1)
    
    seconds = st.number_input("Trip Duration (Seconds):", min_value=10, max_value=7200, value=600, step=10)
    st.caption("600 seconds = 10 minutes")
    
    hour = st.slider("Start Hour (0-23):", min_value=0, max_value=23, value=17, step=1)
    
    # Button to trigger prediction
    if st.button("Calculate Predicted Fare", type="primary"):
        
        # Guard against zero or negative duration
        if seconds <= 0:
            st.warning("Duration must be greater than zero.")
        else:
            # Get the prediction
            predicted_fare = predict_fare(miles, seconds, hour)
            
            # Display the result
            st.success("---")
            st.subheader(f"💵 Predicted Fare Estimate:")
            st.metric(label="Estimated Fare (USD)", value=f"${predicted_fare:.2f}")

            # Optional: Show feature engineering result
            st.caption(f"Rush Hour Flag: {1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0}")
