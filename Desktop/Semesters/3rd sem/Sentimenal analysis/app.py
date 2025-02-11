import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ Load the trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# ✅ Streamlit App Title
st.title("Car Price Prediction App 🚗💰")

# ✅ User Inputs for Prediction
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, value=20.0)
engine = st.number_input("Engine Capacity (CC)", min_value=500, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=30.0, value=80.0)
seats = st.number_input("Number of Seats", min_value=2, max_value=8, value=5)

# ✅ Categorical Inputs
fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# ✅ Convert user input to match model format
fuel_cols = {"Diesel": 1, "Petrol": 0, "CNG": 0, "LPG": 0}
seller_cols = {"Dealer": 0, "Individual": 1, "Trustmark Dealer": 0}
trans_cols = {"Manual": 1, "Automatic": 0}
owner_cols = {"First Owner": 0, "Second Owner": 1, "Third Owner": 0, "Fourth & Above Owner": 0, "Test Drive Car": 0}

input_data = np.array([
    year, km_driven, mileage, engine, max_power, seats,
    fuel_cols[fuel_type], fuel_cols.get("LPG", 0), fuel_cols.get("Petrol", 0),
    seller_cols[seller_type], seller_cols.get("Trustmark Dealer", 0),
    trans_cols[transmission],
    owner_cols[owner], owner_cols.get("Second Owner", 0),
    owner_cols.get("Test Drive Car", 0), owner_cols.get("Third Owner", 0)
]).reshape(1, -1)

# ✅ Make Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.write(f"**Predicted Selling Price: ₹{round(prediction[0], 2)}** 🎯")

columns = ["year", "km_driven", "mileage", "engine", "max_power", "seats",
           "fuel_Diesel", "fuel_LPG", "fuel_Petrol",
           "seller_type_Individual", "seller_type_Trustmark Dealer",
           "transmission_Manual",
           "owner_Fourth & Above Owner", "owner_Second Owner",
           "owner_Test Drive Car", "owner_Third Owner"]

input_df = pd.DataFrame(input_data, columns=columns)  # Add feature names
prediction = model.predict(input_df)
