import altair as alt
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Show the page title and description.
st.set_page_config(page_title="Predicting Energy Commodity Prices using Variants of LSTM Models", page_icon="🛢️")
st.title("🛢️ Crude Oil Dashboard")
st.write(
    """
    This app visualizes data from the WTI Futures Oil Prices.
    It shows the price of the WTI Crude Oil over the years. Just 
    click on the widgets below to explore!
    """
)
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/crude oil WTI 1990 - 2024.csv")
    return df

df = load_data()

fig = px.line(df, x = df.index.between(start_date, end_date), y = df['Price'], title="Price of Crude Oil over the Years")
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
