import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
st.set_page_config(page_title="Accident Severity AI", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_assets():
    model          = joblib.load(os.path.join(BASE_DIR, 'accident_severity_model.pkl'))
    scaler         = joblib.load(os.path.join(BASE_DIR, 'data_scaler.pkl'))
    features       = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))
    mappings       = joblib.load(os.path.join(BASE_DIR, 'target_mappings.pkl'))
    state_city_map = joblib.load(os.path.join(BASE_DIR, 'state_city_map.pkl'))
    return model, scaler, features, mappings, state_city_map


try:
    model, scaler, features, mappings, state_city_map = load_assets()
except FileNotFoundError as e:
    st.error(f"Missing model file: {e}\n\nPlease ensure all .pkl files are in the app directory.")
    st.stop()

# --- APP UI ---
st.title("🛡️ US Accident Severity Predictor")
st.markdown("Fill in the details below to analyze the risk level of an accident.")

with st.form("prediction_form"):

    # ROW 1: Location
    st.subheader("📍 Location Details")
    col1, col2 = st.columns(2)
    with col1:
        all_states = sorted(list(state_city_map.keys()))
        state = st.selectbox("State", options=all_states)
    with col2:
        available_cities = sorted(list(state_city_map.get(state, ["Unknown"])))
        city = st.selectbox("City", options=available_cities)

    st.divider()

    # ROW 2: Weather
    st.subheader("🌤️ Weather & Atmospheric Conditions")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        weather = st.selectbox("Weather Condition", options=sorted(list(mappings['Weather_Condition'].keys())))
        temp    = st.slider("Temperature (°F)", -20, 120, 70)
    with w_col2:
        humidity = st.slider("Humidity (%)", 0, 100, 50)
        pressure = st.number_input("Pressure (in)", value=29.9, step=0.1)
    with w_col3:
        wind_speed = st.number_input("Wind Speed (mph)", value=5.0, step=1.0)
        wind_dir   = st.selectbox("Wind Direction", options=sorted(list(mappings['Wind_Direction'].keys())))

    st.divider()

    # ROW 3: Time & Road
    st.subheader("🛣️ Time & Road Factors")
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        hour  = st.slider("Hour (0-23)", 0, 23, 12)
        month = st.slider("Month (1-12)", 1, 12, 6)
    with r_col2:
        dist       = st.number_input("Distance Impacted (mi)", value=0.5, step=0.1, min_value=0.0)
        is_weekend = st.toggle("Is Weekend?")
    with r_col3:
        road_opts      = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                          'No_Exit', 'Railway', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
        selected_roads = st.multiselect("Road Features Nearby", options=road_opts)

    predict_btn = st.form_submit_button("⚡ Analyze Severity", use_container_width=True, type="primary")

# --- PREDICTION LOGIC ---
if predict_btn:
    input_dict = {f: 0 for f in features}

    # Target encoded averages
    input_dict['City_Sev_Avg']              = mappings['City'].get(city, 2.2)
    input_dict['State_Sev_Avg']             = mappings['State'].get(state, 2.2)
    input_dict['Weather_Condition_Sev_Avg'] = mappings['Weather_Condition'].get(weather, 2.2)
    input_dict['Wind_Direction_Sev_Avg']    = mappings['Wind_Direction'].get(wind_dir, 2.2)

    # Numeric & temporal
    input_dict['Temperature(F)']  = temp
    input_dict['Humidity(%)']     = humidity
    input_dict['Pressure(in)']    = pressure
    input_dict['Wind_Speed(mph)'] = wind_speed
    input_dict['Distance(mi)']    = dist
    input_dict['Hour']            = hour
    input_dict['Month']           = month
    input_dict['Is_Weekend']      = 1 if is_weekend else 0
    input_dict['Is_Night']        = 1 if (hour < 6 or hour > 20) else 0

    # Derived features
    input_dict['Log_Distance']         = np.log(dist + 0.1)
    input_dict['Road_Feature_Count']   = len(selected_roads)
    input_dict['Night_Low_Vis']        = 1 if (hour < 6 or hour > 20) else 0
    input_dict['High_Speed_Potential'] = 1 if dist > 1 else 0
    input_dict['Impact_Intensity']     = dist / 2  # duration unknown, use 1 min default

    # Road features
    for road in selected_roads:
        if road in input_dict:
            input_dict[road] = 1

    # Scale and predict
    input_df     = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)
    prob         = model.predict_proba(input_scaled)[0][1]

    # Results
    st.divider()
    st.subheader("🔍 Prediction Result")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if prob > 0.5:
            st.error("⚠️ HIGH SEVERITY RISK")
            st.metric("Probability of Severe Accident", f"{prob:.1%}")
            st.write("This accident is likely to result in **significant road closure (Level 3 or 4)**.")
        else:
            st.success("✅ LOW SEVERITY RISK")
            st.metric("Probability of Low Severity", f"{1 - prob:.1%}")
            st.write("This is likely a **minor incident with low traffic impact (Level 1 or 2)**.")

    with col_res2:
        st.progress(float(prob))
        st.caption(f"Severe risk score: {prob:.3f} | Threshold: 0.500")
