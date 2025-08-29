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

ticker_input = st.text_input("Enter Ticker")
ticker = [t.strip().upper() for t in ticker_input]

df = yf.download(ticker, start, end, multi_level_index = False)

#define variables
r=0.01 #risk free rate
S = df['Close'].iloc[-1] #base price
K=40 #strike
T=240/365
sigma = 0.3 #volatility?

def black_scholes(r, S, K, T, sigma, type = "C"):
  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)
  try:
    if type == "C":
      price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "P":
      price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return price
  except:
    st.write("Please confirm all option parameters")

st.write("Option Price is: ", round(black_scholes(r, S, K, T, sigma, type = "C"), 2))
