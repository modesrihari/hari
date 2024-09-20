import streamlit as st
import joblib
import numpy as np

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('model_joblib.pkl')  # Replace with your actual model path
    return model

def main():
    st.title("Your Project: Predictive Model with Streamlit")
    
    # Feature input fields (replace these with actual features)
    age = st.number_input('Age', min_value=0, max_value=100, value=25)
    height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
    weight = st.number_input('Weight (kg)', min_value=10, max_value=200, value=70)
    gender = st.selectbox('Gender', options=['Male', 'Female'])

    # Gender encoding (e.g., Male: 0, Female: 1)
    gender_encoded = 0 if gender == 'Male' else 1

    # Prepare the input data
    input_data = np.array([[age, height, weight, gender_encoded]])

    # Predict button
    if st.button('Predict'):
        model = load_model()  # Load the saved model
        prediction = model.predict(input_data)  # Make prediction
        st.success(f"Prediction: {prediction[0]}")

# Run the app
if __name__ == "__main__":
    main()
