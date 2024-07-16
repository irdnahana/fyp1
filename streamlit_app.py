import altair as alt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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

# Filter data by selected date range
date_range_data = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Display the data
st.subheader(f"WTI Crude Oil Prices from {start_date} to {end_date}")
st.line_chart(date_range_data.set_index('Date')['Price'])

# Display average price for the selected date range
if not date_range_data.empty:
    avg_price = date_range_data['Price'].mean()
    st.write(f"Average price from {start_date} to {end_date}: $", avg_price)
    annual_return = date_range_data["Change %"].mean() * 252 * 100
    st.write("Annual return is ", annual_return, "%")
    stdev = np.std(date_range_data["Change %"]) * np.sqrt(252) * 100
    st.write("Standard Deviation is ", stdev, "%")
else:
    st.write(f"No data available for the selected date range")

# Create a new section for the bar chart
st.sidebar.header("Monthly Average Prices")
selected_year = st.sidebar.selectbox("Select Year for Monthly Averages", range(1990, 2024))

# Filter data for the selected year
year_data = df[df['Date'].dt.year == selected_year]

# Calculate monthly averages
monthly_avg = year_data.groupby(year_data['Date'].dt.month)['Price'].mean().reset_index()
monthly_avg.columns = ['Month', 'Price']

# Map month numbers to month names
monthly_avg['Month'] = monthly_avg['Month'].apply(lambda x: calendar.month_name[x])

# Ensure 'Month' column is in the correct order
months = list(calendar.month_name[1:])
monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=months, ordered=True)
monthly_avg = monthly_avg.sort_values('Month')

# Custom blue shades
custom_colors = [
    "#82EEFD", "#63C5DA", "#0492C2", "#2832C2", "#051094", 
    "#0A1172", "#281E5D", "#241571", "#0047AB", "#4169E1",
    "#5D3FD3", "#6495ED"
]

# Create a bar chart using Plotly Graph Objects
fig = go.Figure(data=[
    go.Bar(
        x=monthly_avg['Month'],
        y=monthly_avg['Price'],
        marker_color=custom_colors
    )
])

# Update layout to remove legend and improve appearance
fig.update_layout(
    showlegend=False,
    title=f"Average Monthly WTI Crude Oil Prices for {selected_year}",
    xaxis_title="Month",
    yaxis_title="Price"
)

# Display the bar chart in Streamlit
st.subheader(f"Average Monthly WTI Crude Oil Prices for {selected_year}")
st.plotly_chart(fig)
#st.bar_chart(monthly_avg)

# Display raw data
st.subheader("Raw Data")
st.write(df)

################### PREDICTION #####################

def preprocess_data(data):
    feature_cols = ['Open', 'High', 'Low', 'Vol.', 'Change %']
    target_col = 'Price'
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_cols + [target_col]])
    
    x = scaled_data[:, 1:]
    y = scaled_data[:, 0]
    
    return x, y, scaler

def create_sequence(data):
    sequences = []
    for i in range(len(data) - 30):
        sequences.append(data[i:i + 30])
    return np.array(sequences)

def reshape_for_lstm(x, y):
    x_reshaped = x.reshape((x.shape[0], x.shape[1], x.shape[2]))
    y_reshaped = y[:, -1]
    return x_reshaped, y_reshaped

def make_predictions(model, x):
    predictions = model.predict(x)
    return predictions

def inverse_transform_predictions(predictions, scaler):
    scaled_predictions = np.zeros((predictions.shape[0], 6))  # Assuming 6 feature columns
    scaled_predictions[:, -1] = predictions[:, 0]  # Assuming 'price' is the target column
    original_predictions = scaler.inverse_transform(scaled_predictions)
    return original_predictions[:, -1]

# Preprocess the data
x, y, scaler = preprocess_data(df)
x_seq = create_sequence(x)
y_seq = create_sequence(y)
x_lstm, y_lstm = reshape_for_lstm(x_seq, y_seq)

st.subheader('Preprocessed Data')
st.write(f'x shape: {x.shape}')
st.write(f'y shape: {y.shape}')

# Load the trained model
model = load_model('best_model.h5')

# Make predictions
predictions = make_predictions(model, x_lstm)

# Inverse transform predictions
original_predictions = inverse_transform_predictions(predictions, scaler)

# Display predictions
st.subheader('Predictions')
st.line_chart(original_predictions)

####################### PLOTTING NEW CHART ###########################

# convert array to dataframe
df_predictions = pd.DataFrame({'Predicted Price': original_predictions})
df_actual = pd.DataFrame({'Date': df['Date'], 'Actual Price': df['Price']})

# combine the dataframe
new_df = pd.concat([df_actual, df_predictions], axis=1)

st.subheader("Predicted Price Compared to Actual Price")
st.write(new_df)

# Comparison with actual data
st.subheader('Actual vs Predicted')
chart = alt.Chart(new_df).mark_line().encode(
    x='Date:T',  # T specifies temporal axis
    y=alt.Y('Actual Price', axis=alt.Axis(title='Value')),
    color=alt.value('blue')  # Set line color to blue for actual
).properties(
    title='Actual vs Predicted'
)

# Add the predicted line
predicted_line = alt.Chart(new_df).mark_line().encode(
    x='Date:T',
    y=alt.Y('Predicted Price', axis=alt.Axis(title='Value')),
    color=alt.value('purple')  # Set line color to purple for predicted
)

# Combine the charts
final_chart = chart + predicted_line

# Display the chart in Streamlit
st.altair_chart(final_chart, use_container_width=True)

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
