import altair as alt
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import calendar
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
 
# Show the page title and description.
st.set_page_config(page_title="Predicting Energy Commodity Prices using Variants of LSTM Models", page_icon="🛢️")
st.title("🛢️ WTI Crude Oil Prices Dashboard")
st.write(
    """
    This app visualizes data from the WTI Futures Oil Prices.
    It shows the price of the WTI Crude Oil over the years.
    """
)

# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/crude oil WTI 1990 - 2024.csv", parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    return df

df = load_data()

st.sidebar.header("User Input")
start_date = st.sidebar.date_input("Select Start Date", min_value=pd.to_datetime("1990-01-01"), max_value=pd.to_datetime("2024-05-31"))
end_date = st.sidebar.date_input("Select End Date", min_value=pd.to_datetime("1990-01-01"), max_value=pd.to_datetime("2024-05-31"))

# filter data by selected date range
date_range_data = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Display the data
st.subheader(f"WTI Crude Oil Prices from {start_date} to {end_date}")
st.line_chart(date_range_data.set_index('Date')['Price'])

# Display average price for the selected date range
if not date_range_data.empty:
    avg_price = date_range_data['Price'].mean()
    st.write(f"Average price from {start_date} to {end_date}: $", avg_price)
    annual_return = date_range_data["Change %"].mean()*252*100
    st.write("Annual return is ", annual_return, "%")
    stdev = np.std(date_range_data["Change %"])*np.sqrt(252)*100
    st.write("Standard Deviation is ", stdev, "%")
else:
    st.write(f"No data available for the selected date range")

# create a new section for the bar chart
st.sidebar.header("Monthly Average Prices")
selected_year = st.sidebar.selectbox("Select Year for Monthly Averages", range(1990, 2024))

# Filter data for the selected year
year_data = df[df['Date'].dt.year == selected_year]

# Calculate monthly averages
monthly_avg = year_data.groupby(year_data['Date'].dt.month)['Price'].mean()

# Map month numbers to month names
monthly_avg.index = monthly_avg.index.map(lambda x: calendar.month_name[x])

# Convert index to a categorical type with the correct order
months = list(calendar.month_name[1:])
monthly_avg.index = pd.CategoricalIndex(monthly_avg.index, categories=months, ordered=True)
monthly_avg = monthly_avg.sort_index()

# Display the bar chart
st.subheader(f"Average Monthly WTI Crude Oil Prices for {selected_year}")
st.bar_chart(monthly_avg)

# Display raw data
st.subheader("Raw Data")
st.write(df)

################### PREDICTION #####################

def preprocess_data(data):
  feature_cols = ['Open', 'High', 'Low', 'Vol.', 'Change %']
  target_col = ['Price']
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data[feature_cols + [target_col]])

  x = []
  y = []

  for i in range(30, len(scaled_data)):
    x.append(scaled_data[i-30:i])
    y.append(scaled_data[i, 3])

def make_predictions(model, x):
  predictions = model.predict(x)
  return predictions

def inverse_transform_predictions(predictions, scaler):
    scaled_predictions = np.zeros((predictions.shape[0], 6))  # Assuming 6 feature columns
    scaled_predictions[:, -1] = predictions  # Assuming 'price' is the target column
    original_predictions = scaler.inverse_transform(scaled_predictions)
    return original_predictions[:, -1]

# Main function to run the Streamlit app

# Preprocess the data
x, y, scaler = preprocess_data(df)
st.subheader('Preprocessed Data')
st.write(f'X shape: {x.shape}')
st.write(f'y shape: {y.shape}')
    
# Load the trained model
model = tf.keras.models.load_model('best_model.h5')

# Make predictions
predictions = make_predictions(model, x)

# Inverse transform predictions
original_predictions = inverse_transform_predictions(predictions, scaler)
    
# Display predictions
st.subheader('Predictions')
st.line_chart(original_predictions)
    
# Comparison with actual data
actual_data = data['price'][30:].values  # Assuming 'price' is the target column
st.subheader('Actual vs Predicted')
comparison_df = pd.DataFrame({
        'Actual': actual_data,
        'Predicted': original_predictions
    }, index=data.index[30:])
st.line_chart(comparison_df)





# Function to prepare data for prediction
#def prepare_data_for_prediction(date, data, scaler):
    # Example of how to prepare data for prediction
    # This will depend on your model's input requirements
    # Here we just use the last available data point for simplicity
 #   last_known_data = df[df['Date'] < pd.to_datetime(date)].tail(1)
  #  if last_known_data.empty:
   #     st.write("Not enough data to make a prediction")
    #    return None
    #last_known_price = last_known_data['Price'].values
    #scaled_data = scaler.transform(last_known_price.reshape(-1, 1))
    #return np.array([scaled_data])

#if st.sidebar.button("Predict"):
 #   prediction_data = prepare_data_for_prediction(selected_date_for_prediction, df, scaler)
  #  if prediction_data is not None:
   #     prediction = model.predict(prediction_data)
    #    predicted_price = scaler.inverse_transform(prediction)[0][0]
     #   st.write(f"Predicted price for {selected_date_for_prediction}: ${predicted_price:.2f}")
