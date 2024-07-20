import yfinance as yf
import pandas as pd

# Function to fetch real-time data
def fetch_real_time_data(ticker):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="1d", interval="1m")
    return hist_data

# Example usage
ticker = "AAPL"
real_time_data = fetch_real_time_data(ticker)
print(real_time_data.tail())

# Function to preprocess the data
def preprocess_data(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data.dropna(inplace=True)
    return data

# Example usage
processed_data = preprocess_data(real_time_data)
print(processed_data.tail())

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Preparing the data for training
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30']
X = processed_data[features]
y = processed_data['Close'].shift(-1).dropna()
X = X.iloc[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

import streamlit as st

# Function to make real-time predictions
def predict_real_time(data, model):
    processed_data = preprocess_data(data)
    X_real_time = processed_data[features].iloc[-1:]
    prediction = model.predict(X_real_time)
    return prediction[0]

# Streamlit app
st.title("Real-Time Stock Price Prediction")

ticker = st.text_input("Enter Stock Ticker:", "AAPL")

if st.button("Predict"):
    real_time_data = fetch_real_time_data(ticker)
    prediction = predict_real_time(real_time_data, model)
    st.write(f"Predicted Next Closing Price: ${prediction:.2f}")

