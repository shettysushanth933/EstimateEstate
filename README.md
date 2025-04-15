# EstimateEstate 🏡

EstimateEstate is an AI-powered property valuation tool that provides instant, accurate house price predictions using machine learning. Built with Python and Streamlit, it offers an intuitive interface for users to input property details and receive estimated valuations.

## Features

- 🤖 Advanced machine learning model for accurate price predictions
- 🎨 Modern, responsive UI with light/dark theme support
- 📊 Real-time property valuation
- 🔍 Comprehensive property feature analysis
- 📱 Mobile-friendly design

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EstimateEstate.git
cd EstimateEstate
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
cd House_Price_Prediction/app
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Input property details:
   - Square footage
   - Number of bedrooms
   - Number of bathrooms
   - Year built
   - Lot size
   - Garage size
   - Neighborhood quality

4. Click "Get Estimate" to receive your property valuation

## Project Structure

```
EstimateEstate/
├── House_Price_Prediction/
│   ├── app/
│   │   ├── app.py          # Main Streamlit application
│   │   └── style.css       # Custom styling
│   ├── models/             # Trained ML models
│   └── data/              # Dataset files
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Dependencies

- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- Joblib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Powered by scikit-learn
- Styled with custom CSS

## Contact

For any queries or suggestions, please open an issue in the repository.
