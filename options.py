import yfinance as yf 
import numpy as np
from scipy import stats
from scipy.stats import norm
import streamlit as st

#define variables
r=0.01
S=30
K=40
T=240/365
sigma = 0.3

def black_scholes(r, S, K, T, sigma, type = "C"):
  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma*np.sqrt(T)
  try:
    if type == "C":
      price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    else type == "P":
      price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return price
  except:
    print("Please confirm all option parameters")

print("Option Price is: ", round(black_scholes(r, S, K, T, sigma, type = "C"), 2)
