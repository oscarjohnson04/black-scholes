import yfinance as yf 
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import streamlit as st
import datetime as dt
import plotly.graph_objects as go

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

st.set_page_config(layout="wide")

st.title("Options Pricing Models")

tab1, tab2 = st.tabs(["Black-Scholes Model", "Binomial Model"])
with tab1:
    ticker_input = st.text_input("Enter Ticker", value="AAPL", key="ticker_bs")
    ticker = ticker_input.strip().upper()
    
    df = yf.download(ticker, start, end)
    
    #define variables
    S = float(round((df['Close'].iloc[-1]), 2)) #base price
    st.write("Latest closing price of chosen stock : ", S)
    
    user_val = st.text_input("Enter the strike price", "0.01", key="strike_bs")
    K = float(user_val)
    r_percent = st.slider("Risk-free rate (%)", 0.0, 10.0, value=1.0, step=0.01, format="%.2f%%", key="interest_bs")
    r = r_percent / 100
    T = st.slider("Time to Maturity (in days)", 1, 365, value=240, step=1, key="time_bs") / 365
    
    vol_choice = st.radio("Select Volatility Type", ("Historical", "Custom"), key ="vol_bs")
    
    if vol_choice == "Historical":
        returns = df['Close'].pct_change().dropna()
        windowinput = st.text_input("Enter the time window", "30", key = "window_bs")
        window = int(windowinput)
        rolling_std = returns.rolling(window=window).std()
        sigma_last = rolling_std.iloc[-1]
        sigma = sigma_last * np.sqrt(252)  # last value
        sigma_display = float(round((sigma), 4)) * 100
        sigma_display = str(sigma_display) + "%"
        st.write("Historical Volatility calculated from past ", window," days: ", sigma_display)
    else:
            # Let user enter custom volatility via slider
        sigma_percent = st.slider("Enter the volatility (%)", 0.0, 50.0, value=10.0, step=0.01, format="%.2f%%", key="volcustom_bs")
        sigma = sigma_percent / 100
      
    option_type = st.radio("Select Option Type", ("Call", "Put"), key="type_bs")
    
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
    with st.expander("ℹ️ Option Greeks"):
        st.write("Delta: How much an option price changes from a $1 change in the underlying stock price")
        st.write("Gamma: How quickly delta changes when there is a change in the underlying stock price")
        st.write("Vega: How sensitive an option's price is to changes in expected market volatiltiy")
        st.write("Theta: How much value an option loses each day as it gets to closer to expiry")
        st.write("Rho: How sensitive an option's price is to changes in the risk-free rate")
    st.dataframe(greeks.set_index('Greek').style.format("{:.4f}"), use_container_width=True)
    
    st.header("P/L Analysis")
    
    # --- P/L Inputs ---
    pl_col1, pl_col2, pl_col3, pl_col4 = st.columns(4)
    
    with pl_col1:
        side = st.radio("Position", ("Long", "Short"), horizontal=True).lower()
    with pl_col2:
        contracts = int(st.number_input("Contracts", min_value=1, max_value=10000, value=1, step=1))
    with pl_col3:
        multiplier = int(st.number_input("Contract multiplier", min_value=1, max_value=10000, value=100, step=1))
    with pl_col4:
        # Use session state to prevent resetting
        if "entry_price" not in st.session_state:
            st.session_state.entry_price = round(price, 2)  # default = BS price
        entry_price = round(st.number_input(
            "Entry premium (per option)",
            min_value=0.0,
            value=st.session_state.entry_price,
            step=0.01,
            format="%.2f"
        ), 2)
        st.session_state.entry_price = entry_price
    
    # --- Current P/L Calculations ---
    if side == "long":
        current_pl_per_contract = price - entry_price
    else:  # short
        current_pl_per_contract = entry_price - price
    
    current_pl_total = current_pl_per_contract * contracts * multiplier
    
    # --- Display Metrics ---
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Current P/L (per contract)", f"{current_pl_per_contract:.2f}")
    with m2:
        st.metric("Current P/L (total)", f"{current_pl_total:,.2f}")
    
    # --- Breakeven ---
    if option_type_code == "C":
        breakeven = K + entry_price
    else:
        breakeven = K - entry_price
    st.caption(f"Breakeven at expiry (approx): {breakeven:.2f}")
    
    # --- Payoff at Expiry ---
    st.subheader("Payoff at Expiry")
    
    def payoff_at_expiration(S_T: np.ndarray, K: float, premium: float, option_type_code: str, side: str):
        option_type_code = option_type_code.upper()
        side = side.lower()
        if option_type_code == "C":
            intrinsic = np.maximum(S_T - K, 0.0)
        elif option_type_code == "P":
            intrinsic = np.maximum(K - S_T, 0.0)
        else:
            raise ValueError("Option type must be 'C' or 'P'")
        
        long_pl = intrinsic - premium
        short_pl = premium - intrinsic
        return long_pl if side == "long" else short_pl
    
    S_min = float(max(0.01, S * 0.5))
    S_max = float(S * 1.5)
    S_T_grid = np.linspace(S_min, S_max, 201)
    pl_expiry_per_contract = payoff_at_expiration(S_T_grid, K, entry_price, option_type_code, side)
    
    # --- Plot ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=S_T_grid, y=pl_expiry_per_contract, mode='lines', name='Payoff'))
    fig1.add_hline(y=0, line=dict(width=1))
    fig1.add_vline(x=breakeven, line=dict(width=1, dash='dash'))
    fig1.update_layout(
        title=f"{side.capitalize()} {option_type} – Payoff at Expiry",
        xaxis_title="Underlying price at expiry",
        yaxis_title="P/L per contract at expiry"
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    ticker_input2 = st.text_input("Enter Ticker", value="AAPL", key="ticker_bn")
    ticker2 = ticker_input2.strip().upper()
    
    df2 = yf.download(ticker2, start, end)
    
    #define variables
    S2 = float(round((df2['Close'].iloc[-1]), 2)) #base price
    st.write("Latest closing price of chosen stock : ", S2)
    
    user_val2 = st.text_input("Enter the strike price", "0.01", key="strike_bn")
    K2 = float(user_val2)
    r_percent2 = st.slider("Risk-free rate (%)", 0.0, 10.0, value=1.0, step=0.01, format="%.2f%%", key="interest_bn")
    r2 = r_percent2 / 100
    T2 = st.slider("Time to Maturity (in years)", 1, 50, value=5, step=1, key="time_bn") 
    N = st.slider("Number of time steps", 1, 50, value=5, step=1) 
    dt = T2/N
    vol_choice2 = st.radio("Select Volatility Type", ("Historical", "Custom"), key ="vol_bn")  
    if vol_choice2 == "Historical":
        returns2 = df2['Close'].pct_change().dropna()
        windowinput2 = st.text_input("Enter the time window", "30", key = "window_bn")
        window2 = int(windowinput2)
        rolling_std2 = returns2.rolling(window=window2).std()
        sigma_last2 = float(rolling_std2.iloc[-1])
        sigma2 = sigma_last2 * np.sqrt(252)
        sigma_display2 = round((sigma), 4) * 100
        sigma_display2 = str(sigma_display2) + "%"
        st.write("Historical Volatility calculated from past ", window2," days: ", sigma_display2)
    else:
        sigma_percent2 = st.slider("Enter the volatility (%)", 0.0, 50.0, value=10.0, step=0.01, format="%.2f%%", key="volcustom_bn")
        sigma2 = sigma_percent2 / 100
    returns2 = df2['Close'].pct_change().dropna()
    windowinput2 = st.text_input("Enter the time window", "30", key = "window_bn")
    window2 = int(windowinput2)
    rolling_std2 = returns2.rolling(window=window2).std()
    sigma_last2 = float(rolling_std2.iloc[-1])
    sigma2 = sigma_last2 * np.sqrt(252)
    u = np.exp(sigma2 * np.sqrt(dt))
    d = 1/u
    option_type2 = st.radio("Select Option Type", ("Call", "Put"), key="type_bn")
    option_type_code2 = "C" if option_type2 == "Call" else "P"

    def binomial_tree(K2, T2, S2, r2, N, u, d, option_type_code2):
        dt = T2/N
        q = (np.exp(r2*dt) - d) / (u-d)
        disc = np.exp(-r2*dt)
        ST = S2 * (u**np.arange(N, -1, -1)) * (d**np.arange(0, N+1, 1))
        if option_type_code2 == "C":
            C = np.maximum(ST - K2, 0.0)
        else:  # Put
            C = np.maximum(K2 - ST, 0.0)
        for i in np.arange(N,0,-1):
            C = disc * ( q * C[1:i+1] + (1-q) * C[0:i])

        return C[0]

    binom_price = binomial_tree(K2, T2, S2, r2, N, u, d, option_type_code2)
    st.write(f"{option_type2} Option Price: {binom_price:.2f}")
