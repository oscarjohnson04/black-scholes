import yfinance as yf 
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import streamlit as st
import datetime as dt

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

st.set_page_config(layout="wide")

st.title("Black-Scholes Options Pricing")

ticker_input = st.text_input("Enter Ticker", value="AAPL")
ticker = ticker_input.strip().upper()

df = yf.download(ticker, start, end)

#define variables
S = float(round((df['Close'].iloc[-1]), 2)) #base price
st.write("Latest closing price of chosen stock : ", S)

user_val = st.text_input("Enter the strike price", "0.01")
K = float(user_val)
r_percent = st.slider("Risk-free rate (%)", 0.0, 10.0, value=1.0, step=0.01, format="%.2f%%")
r = r_percent / 100
T = st.slider("Time to Maturity (in days)", 1, 365, value=240, step=1) / 365

vol_choice = st.radio("Select Volatility Type", ("Historical", "Custom"))

if vol_choice == "Historical":
    returns = df['Close'].pct_change().dropna()
    window2 = st.text_input("Enter the time window", "30")
    window = int(window2)
    rolling_std = returns.rolling(window=window).std()
    sigma_last = rolling_std.iloc[-1]
    sigma = sigma_last * np.sqrt(252)  # last value
    sigma_display = float(round((sigma), 4)) * 100
    sigma_display = str(sigma_display) + "%"
    st.write("Historical Volatility calculated from past ", window2," days: ", sigma_display)
else:
        # Let user enter custom volatility via slider
    sigma_percent = st.slider("Enter the volatility (%)", 0.0, 50.0, value=10.0, step=0.01, format="%.2f%%")
    sigma = sigma_percent / 100
  
option_type = st.radio("Select Option Type", ("Call", "Put"))

# Map it to the 'C' or 'P' needed by the function
option_type_code = "C" if option_type == "Call" else "P"

def black_scholes(S, K, T, r, sigma, type=option_type_code):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Price
    if option_type_code.upper() == "C":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
        rho = K*T*np.exp(-r*T)*norm.cdf(d2)
    elif option_type_code.upper() == "P":
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'C' or 'P'")

    # Greeks common to both
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return price, delta, gamma, vega, theta, rho

# =========================
# CALCULATE
price, delta, gamma, vega, theta, rho = black_scholes(S, K, T, r, sigma, option_type_code)

price = float(price)
delta = float(delta)
gamma = float(gamma)
vega = float(vega)
theta = float(theta)
rho = float(rho)

greeks = pd.DataFrame({
    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Value": [delta, gamma, vega, theta, rho]
})

st.write(f"{option_type} Option Price: {price:.2f}")
st.subheader("Option Greeks")
st.dataframe(greeks.set_index('Greek').style.format("{:.4f}"), use_container_width=True)
