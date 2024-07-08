import altair as alt
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

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
    st.write(f"Average price from {start_date} to {end_date}: ${avg_price:.2f}")
else:
    st.write(f"No data available for the selected date range")

# Display raw data
st.subheader("Raw Data")
st.write(df)

# create a new section for the bar chart
st.sidebar.header("Monthly Average Prices")
selected_year = st.sidebar.selectbox("Select Year for Monthly Averages", range(1990, 2024))

# Filter data for the selected year
year_data = df[df['Date'].dt.year == selected_year]

# Calculate monthly averages
monthly_avg = year_data.groupby(year_data['Date'].dt.month)['Price'].mean()

# Display the bar chart
st.subheader(f"Average Monthly WTI Crude Oil Prices for {selected_year}")
st.bar_chart(monthly_avg)


############################################################################
#df["Date"] = pd.date_range('1990-1-1', periods=8760, freq='D')
#df = df.set_index(["Date"])

#data2 = df.loc[start_date:end_date]

fig = px.line(df, x = df['Date'], y = df['Price'], title="Price of Crude Oil over the Years")
st.plotly_chart(fig)

pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header("Price Movement")
    st.write(df)
    annual_return = df["Change %"].mean()*252*100
    st.write("Annual return is ", annual_return, "%")
    stdev = np.std(df["Change %"])*np.sqrt(252)*100
    st.write("Standard Deviation is ", stdev, "%")

with fundamental_data:
    st.write("Fundamental")

with news:
    st.write("News")
"""
# Show a multiselect widget with the genres using `st.multiselect`.
genres = st.multiselect(
    "Genres",
    df.genre.unique(),
    ["Action", "Adventure", "Biography", "Comedy", "Drama", "Horror"],
)


# Show a slider widget with the years using `st.slider`.
date = st.slider("Date", 1990, 2024, (2000, 2016))

# Filter the dataframe based on the widget input and reshape it.
df_filtered = df[df["Date"].between(date[0], date[1])]
df_reshaped = df_filtered.pivot_table(
    index="Date", values="Price", aggfunc="sum", fill_value=0
    #index="Date", columns="genre", values="gross", aggfunc="sum", fill_value=0
)
df_reshaped = df_reshaped.sort_values(by="Date", ascending=False)


# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_reshaped,
    use_container_width=True,
    column_config={"year": st.column_config.TextColumn("Date")},
)


# Display the data as an Altair chart using `st.altair_chart`.
df_chart = pd.melt(
    df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name="gross"
)
chart = (
    alt.Chart(df_chart)
    .mark_line()
    .encode(
        x=alt.X("year:N", title="Year"),
        y=alt.Y("gross:Q", title="Gross earnings ($)"),
        color="genre:N",
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)
"""
