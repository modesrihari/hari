import streamlit as st
import pandas as pd
import pickle

# Load CSV and Model
@st.cache_data
def load_data():
    data = pd.read_csv('Health_Sleep_Statistics.csv')
    return data

@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load data and model
data = load_data()
model = load_model()

# Title
st.title("Health and Sleep Statistics Predictor")

# Show the data
if st.checkbox("Show data preview"):
    st.write(data.head())

# User Input for Prediction
st.header("Input Features for Prediction")

# Assuming the model uses these features, replace with the actual feature columns from your data
age = st.number_input("Age", min_value=0, max_value=100, value=30)
hours_of_sleep = st.slider("Hours of Sleep", min_value=0, max_value=24, value=8)
exercise_hours = st.number_input("Exercise Hours per Week", min_value=0, max_value=168, value=5)

# You can add more features based on your data

# Prepare input for prediction
input_features = pd.DataFrame({
    'Age': [age],
    'Hours of Sleep': [hours_of_sleep],
    'Exercise Hours per Week': [exercise_hours]
})

# Display input data
st.write("Input Data for Prediction:", input_features)

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)
    st.write(f"Prediction: {prediction[0]}")

