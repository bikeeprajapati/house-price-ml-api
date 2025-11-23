import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"   # Point to FastAPI backend

st.title("🏡 House Price Prediction UI")
st.write("Enter details below to get predicted house price.")

# Input fields (match your ML model features)
bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0)
sqft_living = st.number_input("Living Area (sqft)", min_value=100, value=1500)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=500, value=5000)
floors = st.number_input("Floors", min_value=1.0, value=1.0)
condition = st.number_input("Condition (1–5)", min_value=1, max_value=5, value=3)
yr_built = st.number_input("Year Built", min_value=1800, value=1990)

if st.button("Predict Price"):
    payload = {
        "record": {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "condition": condition,
            "yr_built": yr_built
        }
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        st.success(f"Estimated Price: ${response.json()['predictions'][0]:,.2f}")
    else:
        st.error(f"Error: {response.text}")
