import streamlit as st
import joblib
import pandas as pd
import time  

# Load the model
model = joblib.load('house_price_model.pkl')

# Apply custom styling
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #e6f2ff; /* Pale blue background */
        color: #0074a8; /* Cerulean blue text */
    }
    .stTextInput>label, .stNumberInput>label {
        color: #0074a8; /* Cerulean blue labels */
        font-weight: bold;
    }
    input {
        background-color: #ffffff !important; /* White background */
        color: #0074a8 !important; /* Cerulean blue text */
        border: 2px solid #0074a8 !important; 
        border-radius: 8px;
        padding: 8px;
    }
    .stButton>button {
        background-color: #0074a8 !important;
        color: white !important;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .stSpinner>div {
        color: #0074a8 !important; /* Cerulean blue spinner */
    }
    .result-box {
        background-color: #cce7ff; /* Light blue background */
        color: #0074a8; /* Cerulean blue text */
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #0074a8;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏡 House Price Prediction")

# Get user input
with st.form("input_form"):
    square_footage = st.text_input("Square Footage", placeholder="Enter square footage (e.g., 1500)")
    num_bedrooms = st.text_input("Number of Bedrooms", placeholder="Enter number of bedrooms (e.g., 3)")
    num_bathrooms = st.text_input("Number of Bathrooms", placeholder="Enter number of bathrooms (e.g., 2)")
    year_built = st.text_input("Year Built", placeholder="Enter year built (e.g., 1980)")
    lot_size = st.text_input("Lot Size (acres)", placeholder="Enter lot size in acres (e.g., 0.5)")
    garage_size = st.text_input("Garage Size (sq ft)", placeholder="Enter garage size in sq ft (e.g., 200)")
    neighborhood_quality = st.text_input("Neighborhood Quality (1-10)", placeholder="Enter quality score (e.g., 5)")

    submit = st.form_submit_button("Predict Price")

if submit:
    with st.spinner("🔄 Calculating..."):
        time.sleep(2)  # Simulating a delay

        # Convert input values
        try:
            input_data = {
                'Square_Footage': float(square_footage),
                'Num_Bedrooms': int(num_bedrooms),
                'Num_Bathrooms': int(num_bathrooms),
                'Year_Built': int(year_built),
                'Lot_Size': float(lot_size),
                'Garage_Size': float(garage_size),
                'Neighborhood_Quality': int(neighborhood_quality)
            }

            # Make prediction
            predicted_price = model.predict(pd.DataFrame([input_data]))[0]

            # Show result with custom styling
            st.markdown(f"""
                <div class="result-box">
                    💰 Predicted House Price: ${predicted_price:,.2f}
                </div>
            """, unsafe_allow_html=True)

        except ValueError:
            st.error("⚠️ Please enter valid numerical values for all fields.")
