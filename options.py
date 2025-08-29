import yfinance as yf 
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

user_val = st.text_input("Enter the strike price", "0")
K = float(user_val)
r_percent = st.slider("Risk-free rate (%)", 0.0, 10.0, value=1.0, step=0.01, format="%.2f%%")
r = r_percent / 100
sigma = st.slider("Enter the volatility (σ)", 0.0, 1.0, value=0.3, step=0.01, format="%.2f")
T = st.slider("Time to Maturity (in days)", 1, 365, value=240, step=1) / 365

option_type = st.radio("Select Option Type", ("Call", "Put"))

# Map it to the 'C' or 'P' needed by the function
option_type_code = "C" if option_type == "Call" else "P"

def black_scholes(r, S, K, T, sigma, type = option_type_code):
  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)
  try:
      if type.upper() == "C":
          price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
      elif type.upper() == "P":
          price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
      else:
          raise ValueError("option_type must be 'C' or 'P'")
      return price
  except Exception as e:
      st.write("Error:", e)
      return None

display_price = float(black_scholes(r, S, K, T, sigma, type=option_type_code))
st.write(f"{option_type} Option Price is: ", round((display_price), 2))
