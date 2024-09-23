import numpy as np
import pandas as pd
import re
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("tehranhouses.csv")

# Preprocess data
df['Area'] = df['Area'].apply(lambda x: re.sub(',', '', str(x)))
df["Area"] = pd.to_numeric(df["Area"], errors='coerce')
boolean_columns = ['Parking', 'Warehouse', 'Elevator']
for col in boolean_columns:
    df[col] = df[col].apply(lambda x: 1 if x in [True, 'True', 'true', 1] else (0 if x in [False, 'False', 'false', 0] else None))

df[boolean_columns] = df[boolean_columns].fillna(0).astype(int)
df['Price'] = df['Price'].fillna(0)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
df = df.drop(columns=['Price(USD)'])
df['Address'] = df['Address'].fillna('Unknown').astype(str)
df.dropna(inplace=True)

# Remove outliers
def lower_upper(x):
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

lower_area, upper_area = lower_upper(df['Area'])
lower_price, upper_price = lower_upper(df['Price'])
area_outliers = np.where(df['Area'] > upper_area)
price_outliers = np.where(df['Price'] > upper_price)
total_outliers = np.union1d(area_outliers, price_outliers)
total_outliers = [idx for idx in total_outliers if idx in df.index]
df = df.drop(index=total_outliers)

# Features and target
X = df[['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address']]
y = df['Price']
X = pd.get_dummies(X, columns=['Address'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Tehran House Price Predictor")
st.subheader("Get an estimate of house prices based on your inputs")

# Input fields
area = st.number_input("Enter Area (in square meters):", min_value=0, step=1)

room = st.number_input("Enter Number of Rooms:", min_value=0, step=1)

# Binary inputs as radio buttons
parking = st.radio("Parking Availability:", options=["No", "Yes"])
parking = 1 if parking == "Yes" else 0

warehouse = st.radio("Warehouse Availability:", options=["No", "Yes"])
warehouse = 1 if warehouse == "Yes" else 0

elevator = st.radio("Elevator Availability:", options=["No", "Yes"])
elevator = 1 if elevator == "Yes" else 0

# Address selection via dropdown
unique_addresses = df['Address'].unique()
unique_addresses_sorted = sorted(unique_addresses)  # Optional: Sort the addresses alphabetically

address = st.selectbox("Select Address:", options=["Select Address"] + list(unique_addresses_sorted))

# Predict price
if st.button("Predict Price"):
    if address == "Select Address":
        st.error("Please select a valid address.")
    else:
        user_data = pd.DataFrame({
            'Area': [area],
            'Room': [room],
            'Parking': [parking],
            'Warehouse': [warehouse],
            'Elevator': [elevator],
            'Address': [address]
        })

        user_data = pd.get_dummies(user_data, columns=['Address'], drop_first=True)

        # Ensure all necessary columns are present
        for col in X.columns:
            if col not in user_data.columns:
                user_data[col] = 0

        # Reorder columns to match the training data
        user_data = user_data[X.columns]

        predicted_price = model.predict(user_data)
        st.success(f"The predicted price is: {predicted_price[0]:,.0f} Tomans")

# Add additional styling
st.markdown("""
<style>
    /* Apply Times New Roman Bold to all text elements */
    body, label, div, span, p, h1, h2, h3, h4, h5, h6, button {
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
    }
    /* Style the title specifically */
    .stTitle {
        color: #4CAF50;
        font-size: 40px;
        text-align: center;
    }
    /* Style the subtitle */
    .stSubheader {
        font-size: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
