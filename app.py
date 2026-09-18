"""
Sydney Housing Price Prediction - Streamlit App
SIT307 8.1 Distinction Task

Run with:  streamlit run app.py
Requires:  house_price_model.pkl (saved from the notebook) in the same folder.
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sydney House Price Predictor", page_icon = "house")

#Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# Header
st.title("Sydney House Price Predictor")
st.write(
    "Enter a property's details to get an estimated sale price. "
    "The model was trained on 105 real sold properties from Bondi Beach, "
    "Epping and Liverpool. Estimates are a rough guide only, not a formal valuation."
)

# Inputs 
col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", ["Bondi Beach", "Epping", "Liverpool"])
    property_type = st.selectbox("Property type", ["House", "Unit"])
    bedrooms = st.number_input("Bedrooms", min_value = 0, max_value = 10, value = 2, step = 1)

with col2:
    bathrooms = st.number_input("Bathrooms", min_value = 0, max_value = 10, value = 1, step = 1)
    car_spaces = st.number_input("Car spaces", min_value = 0, max_value = 6, value = 1, step = 1)
    land_size = st.number_input(
        "Land size (sqm) - leave 0 for a unit",
        min_value = 0, max_value = 2000, value = 0, step = 10
    )

# Units have no meaningful land size, so pass missing (None) to match training
land_value = None if (property_type == "Unit" or land_size == 0) else land_size

# Predict
if st.button("Predict price"):
    X = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "car_spaces": car_spaces,
        "land_size_sqm": land_value,
        "suburb": suburb,
        "property_type": property_type,
    }])
    price = model.predict(X)[0]
    st.success(f"Estimated sale price: ${price:,.0f}")
    st.caption(
        "This estimate is based on suburb, property type and size only. "
        "It cannot account for views, renovations, condition or exact position, "
        "so treat luxury or unusual properties with extra caution."
    )

st.divider()
st.caption("SIT307 8.1 Distinction Task - Sydney Housing Price Prediction. "
           "Model: Gradient Boosting Regressor. Data collected from realestate.com.au.")
