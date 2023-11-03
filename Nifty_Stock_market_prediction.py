# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from ta.trend import MACD
from ta.momentum import RSIIndicator

st.title('NSE Stocks Forecasting')

def load_data(stocks):
    if stocks is not None:
        with st.spinner("Loading data..."):
            data = pd.read_csv(stocks)
            data.rename(columns={
                'DATE ': 'date',
                'EXPIRY DATE ': 'expiry_date',
                'OPEN PRICE ': 'open',
                'HIGH PRICE ': 'high',
                'LOW PRICE ': 'low',
                'CLOSE PRICE ': 'close',
                'LAST PRICE ': 'last',
                'SETTLE PRICE ': 'settle',
                'Volume ': 'volume',
                'VALUE ': 'value',
                'PREMIUM VALUE ': 'premium_value',
                'OPEN INTEREST ': 'open_interest',
                'CHANGE IN OI ': 'change_in_OI'
            }, inplace=True)
            
            data['date'] = pd.to_datetime(data['date'], format='%d-%b-%Y')
            data['expiry_date'] = pd.to_datetime(data['expiry_date'], format='%d-%b-%Y')
            data['open'] = pd.to_numeric(data['open'], errors='coerce')
            data['high'] = pd.to_numeric(data['high'], errors='coerce')
            data['low'] = pd.to_numeric(data['low'], errors='coerce')
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
            data['value'] = pd.to_numeric(data['value'], errors='coerce')
            data['premium_value'] = pd.to_numeric(data['premium_value'], errors='coerce')
            data['open_interest'] = pd.to_numeric(data['open_interest'], errors='coerce')
            data['change_in_OI'] = pd.to_numeric(data['change_in_OI'], errors='coerce')
            
            return data
    return None

stocks = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])

if stocks is not None:
    data = load_data(stocks)
    if data is not None:
        st.dataframe(data.head())
    else:
        st.warning("Invalid or empty dataset. Please upload a valid dataset to proceed.")

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['date'], y=data['close'], name="stock_close"))
    fig.update_layout(autosize=False, width=1000, height=600)
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def forecast_for_days(data):
    days = st.slider('Days of predictions:', 1, 31)
    period = days

    data_pred = data[['date', 'close']]
    data_pred = data_pred.rename(columns={"date": "ds", "close": "y"})
    data_pred['y'] = data_pred['y'].str.replace(',', '', regex=True).astype(float)
    m2 = Prophet()
    m2.fit(data_pred)
    future = m2.make_future_dataframe(periods=period)
    forecast1 = m2.predict(future)

    fig3 = plot_plotly(m2, forecast1)
    st.write(f'Forecasting closing of stock value for a period of {days} days')
    st.plotly_chart(fig3)

    st.write("Component-wise forecast")
    fig4 = m2.plot_components(forecast1)
    st.write(fig4)

def forecast_for_years(data):
    st.title('Stock Forecast')

    years = st.slider('Years of prediction:', 1, 10)
    period = years * 365

    data_pred = data[['date', 'close']]
    data_pred = data_pred.rename(columns={"date": "ds", "close": "y"})
    data['close'] = data['close'].str.replace(',', '', regex=True).astype(float)
    data_pred['y'] = data_pred['y'].str.replace(',', '', regex=True).astype(float)


    def macd_plot():
        macd = MACD(data['close']).macd()
        st.write('Stock Moving Average Convergence Divergence (MACD) = Close')
        st.area_chart(macd)

    def rsi_plot():
        rsi = RSIIndicator(data['close']).rsi()
        st.write('Resistance Strength Indicator (RSI) = Close')
        st.line_chart(rsi)

    macd_plot()
    rsi_plot()

    m = Prophet()
    m.fit(data_pred)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    fig1 = plot_plotly(m, forecast)
    if st.checkbox('Show forecast data'):
        st.subheader('Forecast data')
        st.write(forecast)
        st.write(f'Forecasting closing of stock value for a period of {years} years')
        st.plotly_chart(fig1)

        st.write("Component-wise forecast")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

if stocks is not None and data is not None:
    page_names_to_funcs = {
        "Forecast for Days": forecast_for_days,
        "Forecast for Years": forecast_for_years,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page](data)
