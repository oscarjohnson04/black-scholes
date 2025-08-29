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
r=0.01 #risk free rate
S = df['Close'].iloc[-1] #base price
st.write(S)
K=200 #strike
T=240/365
sigma = 0.3 #volatility?

def black_scholes(r, S, K, T, sigma, type = "C"):
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

st.write("Option Price is: ", round(black_scholes(r, S, K, T, sigma, type = "C"), 2))
