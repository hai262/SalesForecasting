import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import holidays
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For ACF & PACF plots

warnings.filterwarnings('ignore')
# make new note
# -- Custom CSS for Background Image and Colored Text --
page_bg_img = """
<style>
body {
    background-image: url("https://taskdrive.com/wp-content/uploads/2023/06/sales-forecasting-2023.png");
    background-size: cover;
    background-attachment: fixed;
}
h1, h2, h3, h4, h5, h6 {
    color: #0c118a; /* Change heading color */
}
.stApp {
    background-color: rgba(255, 255, 255, 0);
    padding: 2rem;
    border-radius: 10px;
}
</style>
"""


# Streamlit UI Setup
st.set_page_config(layout="wide", page_title="📈 Advanced Sales Forecasting")
st.markdown(page_bg_img, unsafe_allow_html=True)

# Colored Title using HTML
st.markdown("<h1 style='text-align: center; color:#0c118a;'>🚀 AI-Powered Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color:#d5e809;'>Analyze historical sales data, understand trends, and forecast future sales using AI models (ARIMA, SARIMA, XGBoost, LSTM).</h3>", unsafe_allow_html=True)

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload your sales dataset (CSV format)", type=["csv"])

if uploaded_file:
    sales = pd.read_csv(uploaded_file, encoding='latin1')
    sales['Order Date'] = pd.to_datetime(sales['Order Date'])
    sales.set_index('Order Date', inplace=True)
    monthly_sales = sales['Sales'].resample('M').sum()

    # Display Data Overview in Sidebar
    st.sidebar.markdown("<h4 style='color:#1ABC9C;'>Data Overview</h4>", unsafe_allow_html=True)
    st.sidebar.write(f"📅 Date Range: {sales.index.min()} → {sales.index.max()}")
    st.sidebar.write(f"🔢 Total Entries: {sales.shape[0]}")

    st.subheader("📊 Step 1: Exploratory Data Analysis (EDA)")
    st.write("Understanding sales trends over time helps businesses optimize their inventory, marketing, and pricing strategies.")

    # Interactive Sales Trend Plot
    st.write("### 🔹 Monthly Sales Trend")
    fig_trend = px.line(monthly_sales, x=monthly_sales.index, y=monthly_sales.values, title="Monthly Sales Over Time")
    st.plotly_chart(fig_trend)
    st.write("**🔍 Insight:** Sales exhibit strong seasonal trends, with peaks occurring around holiday periods.")

    # Stationarity Check
    st.write("### 📊 Step 2: Checking Stationarity (ADF Test)")
    adf_test = adfuller(monthly_sales)
    st.write(f"📉 ADF Statistic: {adf_test[0]:.4f}")
    st.write(f"📊 p-value: {adf_test[1]:.4f}")
    if adf_test[1] < 0.05:
        st.success("✅ Data is **stationary**. No differencing needed.")
    else:
        st.warning("⚠️ Data is **non-stationary**. Differencing may be required.")

    # ADF Visualization: Compare ADF Statistic with Critical Values and include Autocorrelation Plot
    st.write("### 📊 ADF Visualization & Autocorrelation")
    crit_values = adf_test[4]
    crit_df = pd.DataFrame(list(crit_values.items()), columns=['Critical Level', 'Value'])
    fig_adf, ax_adf = plt.subplots(1, 2, figsize=(16, 4))
    ax_adf[0].bar(crit_df['Critical Level'], crit_df['Value'], color='skyblue', label="Critical Values")
    ax_adf[0].axhline(adf_test[0], color='red', linestyle='--', label=f"ADF Statistic: {adf_test[0]:.2f}")
    ax_adf[0].set_title("ADF Test: Critical Values vs ADF Statistic")
    ax_adf[0].legend()

    # Autocorrelation Plot using pandas
    autocorrelation_plot(sales['Sales'], ax=ax_adf[1])
    ax_adf[1].set_title("Autocorrelation Plot")
    st.pyplot(fig_adf)

    # ACF & PACF Visualization
    st.write("### 📊 ACF & PACF Analysis")
    max_lags = min(40, int(len(monthly_sales)/2) - 1)
    fig_acf, ax_acf = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(monthly_sales, lags=max_lags, ax=ax_acf[0])
    plot_pacf(monthly_sales, lags=max_lags, ax=ax_acf[1])
    ax_acf[0].set_title("ACF Plot")
    ax_acf[1].set_title("PACF Plot")
    st.pyplot(fig_acf)

    # Seasonal Decomposition
    st.subheader("📈 Step 3: Time-Series Decomposition")
    decomposition = seasonal_decompose(monthly_sales, model='additive')
    
    fig_dec, ax_dec = plt.subplots(4, 1, figsize=(12, 10))
    ax_dec[0].plot(decomposition.observed, color='black', label='Original')
    ax_dec[1].plot(decomposition.trend, color='blue', label='Trend')
    ax_dec[2].plot(decomposition.seasonal, color='green', label='Seasonality')
    ax_dec[3].plot(decomposition.resid, color='red', label='Residuals')
    for a in ax_dec:
        a.legend()
    st.pyplot(fig_dec)
    st.write("🔍 **Business Impact:** By understanding seasonal trends, businesses can optimize promotions and stock levels.")

    # Model Selection
    st.sidebar.header("⚙️ Auto Model Selection")
    auto_arima_model = auto_arima(monthly_sales, seasonal=True, m=12, trace=True)
    st.sidebar.write(f"✅ Best ARIMA Order: {auto_arima_model.order}")
    st.sidebar.write(f"✅ Best Seasonal Order: {auto_arima_model.seasonal_order}")

    # Forecasting Models
    st.subheader("🔮 Step 4: AI-Powered Sales Forecasting")

    # Train ARIMA
    arima_model = ARIMA(monthly_sales, order=auto_arima_model.order)
    arima_model_fit = arima_model.fit()
    arima_forecast = arima_model_fit.forecast(steps=12)

    # Train SARIMA
    sarima_model = SARIMAX(monthly_sales, order=auto_arima_model.order, seasonal_order=auto_arima_model.seasonal_order)
    sarima_model_fit = sarima_model.fit()
    sarima_forecast = sarima_model_fit.forecast(steps=12)

    # Train XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb_model.fit(np.arange(len(monthly_sales)).reshape(-1, 1), monthly_sales.values)
    xgb_forecast = xgb_model.predict(np.arange(len(monthly_sales), len(monthly_sales) + 12).reshape(-1, 1))

    # Train LSTM
    lstm_model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(np.arange(len(monthly_sales)).reshape(-1, 1), monthly_sales.values, epochs=50, batch_size=16, verbose=0)
    lstm_forecast = lstm_model.predict(np.arange(len(monthly_sales), len(monthly_sales) + 12).reshape(-1, 1))

    # Forecasting Plot
    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
    ax_forecast.plot(monthly_sales.index, monthly_sales.values, label="Actual Sales", color='black')
    ax_forecast.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), arima_forecast, label="ARIMA Forecast", linestyle='dashed', color='red')
    ax_forecast.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), sarima_forecast, label="SARIMA Forecast", linestyle='dashed', color='green')
    ax_forecast.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), xgb_forecast, label="XGBoost Forecast", linestyle='dashed', color='orange')
    ax_forecast.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), lstm_forecast, label="LSTM Forecast", linestyle='dashed', color='purple')
    ax_forecast.set_title("📊 AI Sales Forecasting: ARIMA vs SARIMA vs XGBoost vs LSTM")
    ax_forecast.set_xlabel("Date")
    ax_forecast.set_ylabel("Sales")
    ax_forecast.legend()
    st.pyplot(fig_forecast)

    st.subheader("📌 Business Insights & Recommendations")
    st.markdown("<p style='color:#16A085;'>📈 <strong>Inventory Management:</strong> Forecasting allows proactive restocking to avoid shortages.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#16A085;'>💡 <strong>Promotion Planning:</strong> Seasonal peaks indicate the best time for marketing campaigns.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#16A085;'>📊 <strong>Cost Reduction:</strong> AI-driven insights minimize overstocking & warehousing costs.</p>", unsafe_allow_html=True)

    st.download_button("📥 Download Forecast Data", monthly_sales.to_csv(index=False), "forecast.csv", "text/csv")
