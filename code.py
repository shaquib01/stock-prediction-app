# Import necessary libraries
import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.graph_objs as go

# Define the start date and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Start the web app
st.title("Stock Prediction App")

# Define the list of stocks
stocks = ("AAPL", "GOOG", "MSFT")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Define the prediction period
n_years = st.slider("Years of prediction", 1, 4)
period = n_years * 365

# Cache the data after downloading
@st.cache
def load_data(ticker):
    """Function to load data from Yahoo Finance"""
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

# Display raw data
st.subheader('Raw Data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    """Function to plot raw stock data"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Initialize and fit the Prophet model
m = Prophet()
m.fit(df_train)

# Make future dataframe
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast Data')
st.write(forecast.tail())

# Plot forecast data
st.write('Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Plot forecast components
st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)
