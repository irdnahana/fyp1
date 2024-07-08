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
    return df

df = load_data()

st.sidebar.header("User Input")
year = st.sidebar.selectbox("Select Year", range(1990,2025))
month = st.sidebar.selectbox("Select Month", range(1,13))
date = st.sidebar.date_input("Select Date", min_value=pd.to_datetime("1990-01-01"), max_value=pd.to_datetime("2024-06-01"))

# filter by selected year
yearly_data = df[df['Date'].dt.year == year]
monthly_data = yearly_data[yearly_data['Date'].dt.month == month]

# show data for selected date
date_data = df[df['Date'] == pd.to_datetime(date)]

# display the data
st.subheader(f"WTI Crude Oil Prices for {year}")
st.line_chart(yearly_data.set_index('Date')['Price'])

# display average prices
if not monthly_data.empty:
    avg_yearly_price = yearly_data['Price'].mean()
    avg_monthly_price = monthly_data['Price'].mean()

    st.write(f"Average price for {year}: ${avg_yearly_price:.2f}")
    st.write(f"Average price for {month}/{year}: ${avg_monthyly_price:.2f}")

else:
    st.write(f"No data available for {month}/{year}")

# display price for selected date
if not date_data.empty:
    st.write(f"Price on {date}: ${date_data['Price'].values[0]:.2f}")

else:
    st.write(f"No data available for {date}")

# display raw data
st.subheader("Raw Data")
st.write(df)

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
