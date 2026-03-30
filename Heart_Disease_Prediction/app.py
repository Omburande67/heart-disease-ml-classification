import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk
import base64

# Load the models
try:
    logistic_model = pk.load(open('C:/Users/ombur/Desktop/ML_Projeccts/ML/Heart_Disease_Prediction/Heart_disease_LR_model.pkl', 'rb'))
    rf_model = pk.load(open('C:/Users/ombur/Desktop/ML_Projeccts/ML/Heart_Disease_Prediction/Heart_disease_RF_model.pkl', 'rb'))
    bagging_model = pk.load(open('C:/Users/ombur/Desktop/ML_Projeccts/ML/Heart_Disease_Prediction/Heart_disease_Bagging_model.pkl', 'rb'))
    xg_model = pk.load(open('C:/Users/ombur/Desktop/ML_Projeccts/ML/Heart_Disease_Prediction/Heart_disease_XGBoost_model.pkl', 'rb'))
except FileNotFoundError:
    st.error('Model files not found. Please check the paths.')
    st.stop()

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            color: white;
        }}
        .main-title {{
            text-align: left;
            font-size: 48px;  /* Increased size */
            font-weight: bold;
            color: #FFD700;
            text-shadow: 2px 2px 4px #000;
            margin-left: 20px;
        }}
        .input-section {{
            text-align: left;
            margin-left: 20px;
            font-size: 20px;  /* Increased input text size */
        }}
        .predict-button {{
            margin-left: 20px;
            margin-top: 20px;
            color: white;
            font-size: 24px;  /* Increased button text size */
            padding: 15px 30px;  /* Added padding for better look */
            border-radius: 5px;  /* Rounded corners */
        }}
        .metric-box {{
            background-color: rgba(0, 100, 0, 0.7);  /* Green background */
            border-radius: 10px;
            padding: 15px;
            margin: 10px;
            color: white;
            text-align: left;
            font-size: 18px;  /* Increased metric text size */
        }}
        .metric-row {{
            display: flex;
            justify-content: center;
        }}
        .footer {{
            text-align: left;
            margin-left: 20px;
            font-size: 22px;  /* Increased footer text size */
            font-weight: bold;
            color: #FFD700;
            text-shadow: 1px 1px 2px #000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background('C:/Users/ombur/Desktop/ML_Projeccts/ML/Heart_Disease_Prediction/img2.jpeg')

# Title styling
st.markdown("<div class='main-title'>Heart Disease Predictor</div>", unsafe_allow_html=True)

# Collecting user inputs
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

gender = st.selectbox('Choose Gender', ['Male', 'Female'])
gen = 1 if gender == 'Male' else 0

age = st.number_input("Enter Age", min_value=0, max_value=120, value=25)
currentSmoker = st.radio("Is the patient a current smoker?", ['Yes', 'No'])
cigsPerDay_disabled = currentSmoker == 'No'
cigsPerDay = st.number_input("Enter Cigarettes Per Day", min_value=0, max_value=100, value=0, step=1, disabled=cigsPerDay_disabled)
currentSmoker = 1 if currentSmoker == 'Yes' else 0

BPMeds = 1 if st.radio("Is the patient on BP medication?", ['Yes', 'No']) == 'Yes' else 0
prevalentStroke = 1 if st.radio("Has the patient had a stroke?", ['Yes', 'No']) == 'Yes' else 0
prevalentHyp = 1 if st.radio("Does the patient have hypertension?", ['Yes', 'No']) == 'Yes' else 0
diabetes = 1 if st.radio("Does the patient have diabetes?", ['Yes', 'No']) == 'Yes' else 0

totChol = st.number_input("Enter Total Cholesterol", min_value=100, max_value=600, value=200, step=1)
sysBP = st.number_input("Enter Systolic Blood Pressure (sysBP)", min_value=90, max_value=200, value=120, step=1)
diaBP = st.number_input("Enter Diastolic Blood Pressure (diaBP)", min_value=60, max_value=120, value=80, step=1)
BMI = st.number_input("Enter BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
heartRate = st.number_input("Enter Heart Rate", min_value=40, max_value=120, value=70, step=1)
glucose = st.number_input("Enter Glucose Level", min_value=50, max_value=250, value=100, step=1)

st.markdown("</div>", unsafe_allow_html=True)

# Predict Button
st.markdown("<div class='predict-button'>", unsafe_allow_html=True)
if st.button('Predict'):
    try:
        input_data = np.array([[gen, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, 
                                diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
        
        logistic_prediction = logistic_model.predict(input_data)
        rf_prediction = rf_model.predict(input_data)
        bagging_prediction = bagging_model.predict(input_data)
        xg_prediction = xg_model.predict(input_data)

        st.subheader('Prediction Results:')
        st.write("Logistic Regression Prediction: " + 
                 ('Heart Disease Detected' if logistic_prediction[0] == 1 else 'No Heart Disease'))
        st.write("Random Forest Prediction: " + 
                 ('Heart Disease Detected' if rf_prediction[0] == 1 else 'No Heart Disease'))
        st.write("Bagging Classifier Prediction: " + 
                 ('Heart Disease Detected' if bagging_prediction[0] == 1 else 'No Heart Disease'))
        st.write("XGBoost Prediction: " + 
                 ('Heart Disease Detected' if xg_prediction[0] == 1 else 'No Heart Disease'))
    except Exception as e:
        st.error(f"An error occurred: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# Display performance metrics in styled boxes
st.subheader('Model Performance Results:')
st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='metric-box'>
        <h4>Logistic Regression</h4>
        <p>Accuracy: 0.66</p>
        <p>Precision: 0.63</p>
        <p>Recall: 0.65</p>
        <p>F1 Score: 0.64</p>
    </div>
    <div class='metric-box'>
        <h4>Random Forest</h4>
        <p>Accuracy: 0.87</p>
        <p>Precision: 0.87</p>
        <p>Recall: 0.87</p>
        <p>F1 Score: 0.87</p>
    </div>
    """, unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='metric-box'>
        <h4>Bagging Classifier</h4>
        <p>Accuracy: 0.80</p>
        <p>Precision: 0.80</p>
        <p>Recall: 0.82</p>
        <p>F1 Score: 0.81</p>
    </div>
    <div class='metric-box'>
        <h4>XGBoost</h4>
        <p>Accuracy: 0.88</p>
        <p>Precision: 0.88</p>
        <p>Recall: 0.89</p>
        <p>F1 Score: 0.88</p>
    </div>
    """, unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class='footer'>
        Project created by Team Innovators<br>
        College: Vishwakarma Institute of Technology (VIT), Pune
    </div>
    """, unsafe_allow_html=True
)
