import streamlit as st
import joblib
import pandas as pd
import time
import os
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
MODEL_PATH = Path("../models/house_price_model.joblib")
SCALER_PATH = Path("../models/scaler.joblib")

# Default values for the form
DEFAULT_VALUES = {
    'square_footage': 2000.0,
    'num_bedrooms': 3,
    'num_bathrooms': 2,
    'year_built': 2000,
    'lot_size': 0.25,
    'garage_size': 400.0,
    'neighborhood_quality': 7
}

# Theme configuration
def set_theme():
    # Load the custom CSS
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Add theme toggle in sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Theme toggle with custom styling
        st.markdown("""
            <style>
            /* Custom styling for the theme toggle */
            .theme-toggle {
                display: flex;
                align-items: center;
                padding: 10px;
                margin-bottom: 20px;
                border-radius: 10px;
            }
            .theme-label {
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: bold;
            }
            .theme-icon {
                font-size: 1.2em;
            }
            div[data-testid="stMarkdown"] {
                color: inherit;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create columns for the toggle layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            is_dark_theme = st.checkbox(
                "Switch Theme",
                value=True,
                help="Toggle between light and dark theme",
                key="theme_toggle"
            )
        
        # Show sun/moon icon based on theme
        with col2:
            if is_dark_theme:
                st.markdown("🌙")
            else:
                st.markdown("☀️")
        
        # Apply theme
        if is_dark_theme:
            st.markdown('''
                <style>
                    section[data-testid="stSidebar"] > div {
                        background-color: var(--dark-bg);
                        color: var(--dark-text);
                    }
                    .main {
                        background-color: var(--dark-bg);
                        color: var(--dark-text);
                    }
                    .stApp {
                        background-color: var(--dark-bg);
                        color: var(--dark-text);
                    }
                    .element-container, .stMarkdown, .stButton > button {
                        color: var(--dark-text);
                    }
                </style>
                ''', unsafe_allow_html=True)
            st.markdown('<div class="dark-theme">', unsafe_allow_html=True)
        else:
            st.markdown('''
                <style>
                    section[data-testid="stSidebar"] > div {
                        background-color: var(--light-bg);
                        color: var(--light-text);
                    }
                    .main {
                        background-color: var(--light-bg);
                        color: var(--light-text);
                    }
                    .stApp {
                        background-color: var(--light-bg);
                        color: var(--light-text);
                    }
                    .element-container, .stMarkdown, .stButton > button {
                        color: var(--light-text);
                    }
                </style>
                ''', unsafe_allow_html=True)
            st.markdown('<div class="light-theme">', unsafe_allow_html=True)
        
        # Add sidebar information with theme-aware styling
        st.markdown("---")
        st.markdown("""
            <style>
            .sidebar-info {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### About EstimateEstate")
        st.markdown("""
        EstimateEstate uses advanced machine learning to provide accurate house price predictions based on key property features.
        
        **Key Features:**
        - Square Footage 📏
        - Number of Bedrooms 🛏️
        - Number of Bathrooms 🚽
        - Year Built 📅
        - Lot Size 🌳
        - Garage Size 🚗
        - Neighborhood Quality ⭐
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### Help")
        with st.expander("How to use EstimateEstate"):
            st.markdown("""
            1. Adjust the default values as needed
            2. Click 'Get Estimate'
            3. Receive your property valuation
            
            Our AI model ensures accurate predictions based on market data.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

def load_models():
    """Load the model and scaler"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model files not found. Please ensure the model files are in the correct location: {e}")
        logger.error(f"Model loading error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logger.error(f"Unexpected error loading model: {e}")
        return None, None

def validate_input(input_data):
    """Validate user input"""
    try:
        # Convert and validate input values
        validated_data = {
            'Square_Footage': float(input_data['Square_Footage']),
            'Num_Bedrooms': int(input_data['Num_Bedrooms']),
            'Num_Bathrooms': int(input_data['Num_Bathrooms']),
            'Year_Built': int(input_data['Year_Built']),
            'Lot_Size': float(input_data['Lot_Size']),
            'Garage_Size': float(input_data['Garage_Size']),
            'Neighborhood_Quality': int(input_data['Neighborhood_Quality'])
        }
        
        # Additional validation rules
        if validated_data['Square_Footage'] <= 0:
            raise ValueError("Square footage must be positive")
        if validated_data['Num_Bedrooms'] <= 0:
            raise ValueError("Number of bedrooms must be positive")
        if validated_data['Num_Bathrooms'] <= 0:
            raise ValueError("Number of bathrooms must be positive")
        if validated_data['Year_Built'] < 1800 or validated_data['Year_Built'] > 2024:
            raise ValueError("Year built must be between 1800 and 2024")
        if validated_data['Lot_Size'] <= 0:
            raise ValueError("Lot size must be positive")
        if validated_data['Garage_Size'] < 0:
            raise ValueError("Garage size cannot be negative")
        if not 1 <= validated_data['Neighborhood_Quality'] <= 10:
            raise ValueError("Neighborhood quality must be between 1 and 10")
            
        return validated_data
    except ValueError as e:
        raise ValueError(f"Invalid input: {str(e)}")

# Load the model and scaler
model, scaler = load_models()

# Page configuration
st.set_page_config(
    page_title="EstimateEstate - AI Property Valuation",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply theme
set_theme()

# Main content container
with st.container():
    # Header section
    st.title("EstimateEstate 🏡")
    st.markdown("""
        <div class="header-description">
            Welcome to EstimateEstate - Your AI-Powered Property Valuation Tool
            <p class="subtitle">Get instant, accurate property valuations using our advanced machine learning model.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input form with columns for better organization
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            square_footage = st.number_input(
                "Square Footage 📏",
                min_value=500.0,
                max_value=10000.0,
                value=DEFAULT_VALUES['square_footage'],
                step=100.0,
                help="Enter the total square footage of the house (default: 2000 sq ft)"
            )
            num_bedrooms = st.number_input(
                "Number of Bedrooms 🛏️",
                min_value=1,
                max_value=10,
                value=DEFAULT_VALUES['num_bedrooms'],
                step=1,
                help="Enter the number of bedrooms (default: 3)"
            )
            num_bathrooms = st.number_input(
                "Number of Bathrooms 🚽",
                min_value=1,
                max_value=7,
                value=DEFAULT_VALUES['num_bathrooms'],
                step=1,
                help="Enter the number of bathrooms (default: 2)"
            )
            year_built = st.number_input(
                "Year Built 📅",
                min_value=1800,
                max_value=datetime.now().year,
                value=DEFAULT_VALUES['year_built'],
                step=1,
                help="Enter the year the house was built (default: 2000)"
            )
        
        with col2:
            lot_size = st.number_input(
                "Lot Size (acres) 🌳",
                min_value=0.1,
                max_value=5.0,
                value=DEFAULT_VALUES['lot_size'],
                step=0.1,
                help="Enter the lot size in acres (default: 0.25 acres)"
            )
            garage_size = st.number_input(
                "Garage Size (sq ft) 🚗",
                min_value=0.0,
                max_value=1500.0,
                value=DEFAULT_VALUES['garage_size'],
                step=50.0,
                help="Enter the garage size in square feet (default: 400 sq ft)"
            )
            neighborhood_quality = st.slider(
                "Neighborhood Quality ⭐",
                min_value=1,
                max_value=10,
                value=DEFAULT_VALUES['neighborhood_quality'],
                help="Rate the neighborhood quality from 1 (Poor) to 10 (Excellent) (default: 7)"
            )

        submit = st.form_submit_button("Get Estimate 🎯")

    if submit:
        if model is None or scaler is None:
            st.error("Model not loaded. Please check the logs for details.")
        else:
            with st.spinner("🔄 Calculating your property value..."):
                try:
                    # Prepare input data
                    input_data = {
                        'Square_Footage': square_footage,
                        'Num_Bedrooms': num_bedrooms,
                        'Num_Bathrooms': num_bathrooms,
                        'Year_Built': year_built,
                        'Lot_Size': lot_size,
                        'Garage_Size': garage_size,
                        'Neighborhood_Quality': neighborhood_quality
                    }
                    
                    # Validate input
                    validated_data = validate_input(input_data)
                    
                    # Create DataFrame and scale features
                    input_df = pd.DataFrame([validated_data])
                    scaled_input = scaler.transform(input_df)
                    
                    # Make prediction
                    predicted_price = model.predict(scaled_input)[0]
                    
                    # Show result in a nice card
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h2>EstimateEstate Valuation</h2>
                            <div class="price">💰 ${predicted_price:,.2f}</div>
                            <div class="estimate-date">Estimated on {datetime.now().strftime('%B %d, %Y')}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional information
                    with st.expander("See valuation details"):
                        st.markdown("""
                            ### How EstimateEstate calculated this value:
                            - Advanced machine learning model
                            - Property feature analysis
                            - Market data integration
                            
                            **Note:** This valuation is an estimate based on provided features and market data. 
                            For a more detailed assessment, consider consulting with a local real estate professional.
                        """)
                    
                except ValueError as e:
                    st.error(f"⚠️ {str(e)}")
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")
                    logger.error(f"Prediction error: {e}")

# Footer
st.markdown("""
<div class="footer">
    <p>EstimateEstate | AI-Powered Property Valuation | © 2024</p>
</div>
""", unsafe_allow_html=True) 