import yfinance as yf 
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

st.set_page_config(layout="wide")

st.title("Options Pricing Models")

tab1, tab2, tab3 = st.tabs(["Black-Scholes Model", "Binomial Model", "Monte Carlo Simulation Model (European Option)"])
with tab1:
    ticker_input = st.text_input("Enter Ticker", value="AAPL", key="ticker_bs")
    ticker = ticker_input.strip().upper()
    
    df = yf.download(ticker, start, end, multi_level_index = False)
    
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
        sigma_display = float(round((sigma*100), 4))
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

    st.header("Results")
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

    st.subheader("3D Surface Plot of Option Price")

    surface_choice = st.radio(
        "Select dimension for surface plot",
        ("Underlying Price vs. Time to Maturity", "Underlying Price vs. Volatility"),
        key="surface_choice"
    )

    # Range of underlying prices
    S_range = np.linspace(S*0.5, S*1.5, 50)

    if surface_choice == "Underlying Price vs. Time to Maturity":
        T_range = np.linspace(0.01, 1, 50)  # up to 1 year
        S_grid, T_grid = np.meshgrid(S_range, T_range)
        Z = np.zeros_like(S_grid)

        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                price_tmp, _, _, _, _, _ = black_scholes(
                    S_grid[i, j], K, T_grid[i, j], r, sigma, option_type_code
                )
                Z[i, j] = price_tmp

        fig_surface = go.Figure(data=[go.Surface(
            z=Z, x=S_grid, y=T_grid, colorscale="Viridis"
        )])
        fig_surface.update_layout(
            scene=dict(
                xaxis_title="Underlying Price (S)",
                yaxis_title="Time to Maturity (T in years)",
                zaxis_title="Option Price"
            ),
            title=f"{option_type} Price Surface (S vs. T)"
        )
        st.plotly_chart(fig_surface, use_container_width=True)

    else:  # Underlying Price vs. Volatility
        sigma_range = np.linspace(0.01, 0.6, 50)  # 1% to 60%
        S_grid, sigma_grid = np.meshgrid(S_range, sigma_range)
        Z = np.zeros_like(S_grid)

        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                price_tmp, _, _, _, _, _ = black_scholes(
                    S_grid[i, j], K, T, r, sigma_grid[i, j], option_type_code
                )
                Z[i, j] = price_tmp

        fig_surface = go.Figure(data=[go.Surface(
            z=Z, x=S_grid, y=sigma_grid, colorscale="Plasma"
        )])
        fig_surface.update_layout(
            scene=dict(
                xaxis_title="Underlying Price (S)",
                yaxis_title="Volatility (σ)",
                zaxis_title="Option Price"
            ),
            title=f"{option_type} Price Surface (S vs. σ)"
        )
        st.plotly_chart(fig_surface, use_container_width=True)

with tab2:
    ticker_input2 = st.text_input("Enter Ticker", value="AAPL", key="ticker_bn")
    ticker2 = ticker_input2.strip().upper()
    
    df2 = yf.download(ticker2, start, end, multi_level_index = False)
    
    #define variables
    S2 = float(round((df2['Close'].iloc[-1]), 2)) #base price
    st.write("Latest closing price of chosen stock : ", S2)
    
    user_val2 = st.text_input("Enter the strike price", "0.01", key="strike_bn")
    K2 = float(user_val2)
    userh_val = st.text_input("Enter the barrier price", "0.01", key="barrier_bn")
    B = float(userh_val)
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
        sigma_display2 = round((sigma2*100), 4)
        sigma_display2 = str(sigma_display2) + "%"
        st.write("Historical Volatility calculated from past ", window2," days: ", sigma_display2)
    else:
        sigma_percent2 = st.slider("Enter the volatility (%)", 0.0, 50.0, value=10.0, step=0.01, format="%.2f%%", key="volcustom_bn")
        sigma2 = sigma_percent2 / 100

    u = np.exp(sigma2 * np.sqrt(dt))
    d = 1/u
    option_type2 = st.radio("Select Option Type", ("Call", "Put"), key="type_bn")
    option_type_code2 = "C" if option_type2 == "Call" else "P"
    barrier_type = st.radio("Select Barrier Type", ("Up-and-Out", "Down-and-Out"))

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
        
    st.header("Results")
    binom_price = binomial_tree(K2, T2, S2, r2, N, u, d, option_type_code2)
    st.write(f"{option_type2} Option Price: {binom_price:.2f}")

    def barrier_tree(K2, T2, S2, B, r2, N, u, d, option_type_code2, barrier_type):
        dt2 = T2 / N
        q2 = (np.exp(r2 * dt2) - d) / (u - d)
        disc2 = np.exp(-r2 * dt2)
    
        # Initialize option values at maturity
        ST2 = S2 * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1, 1))
        if option_type_code2 == "C":
            C2 = np.maximum(ST2 - K2, 0)
        else:
            C2 = np.maximum(K2 - ST2, 0)
    
        # Apply barrier condition at maturity
        if barrier_type == "Up-and-Out":
            C2[ST2 >= B] = 0
        elif barrier_type == "Down-and-Out":
            C2[ST2 <= B] = 0
    
        # Backward recursion
        for i in np.arange(N - 1, -1, -1):
            ST2 = S2 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
            C2[: i + 1] = disc2 * (q2 * C2[1 : i + 2] + (1 - q2) * C2[0 : i + 1])
            C2 = C2[:-1]
    
            # Barrier condition applied again at each node
            if barrier_type == "Up-and-Out":
                C2[ST2 >= B] = 0
            elif barrier_type == "Down-and-Out":
                C2[ST2 <= B] = 0
    
        return C2[0]

    def barrier_price(K2,T2,S2,B,r2,N,u,d,option_type_code2, barrier_type):
        if "Out" in barrier_type:
            return barrier_tree(K2,T2,S2,B,r2,N,u,d,option_type_code2, barrier_type)
        else:  # Knock-In = Vanilla - Knock-Out
            vanilla = binomial_tree(K2, T2, S2, r2, N, u, d, option_type_code2)
            if "Up" in barrier_type:
                kout = barrier_tree(K2,T2,S2,B,r2,N,u,d,option_type_code2, "Up-and-Out")
            else:
                kout = barrier_tree(K2,T2,S2,B,r2,N,u,d,option_type_code2, "Down-and-Out")
            return vanilla - kout
        
    display_barrier_price = barrier_price(K2,T2,S2,B,r2,N,u,d,option_type_code2, barrier_type)
    st.write(f"{option_type2} Barrier Option Price: {display_barrier_price:.2f}")

    def american_tree(K2,T2,S2,r2,N,u,d,option_type_code2):
        #precompute values
        dtUSA = T2/N
        qUSA = (np.exp(r2*dtUSA) - d)/(u-d)
        discUSA = np.exp(-r2*dtUSA)
    
        # initialise stock prices at maturity
        S_USA = S2 * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1))
    
        # option payoff
        if option_type_code2 == 'P':
            C_USA = np.maximum(0, K2 - S_USA)
        else:
            C_USA = np.maximum(0, S_USA - K2)

        # backward recursion through the tree
        for i in np.arange(N-1,-1,-1):
            S_USA = S2 * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
            C_USA[:i+1] = discUSA * ( qUSA*C_USA[1:i+2] + (1-qUSA)*C_USA[0:i+1] )
            C_USA = C_USA[:-1]
            if option_type_code2 == 'P':
                C_USA = np.maximum(C_USA, K2 - S_USA)
            else:
                C_USA = np.maximum(C_USA, S_USA - K2)
    
        return C_USA[0]

with tab3:
    ticker_input_mc = st.text_input("Enter Ticker", value="AAPL", key="ticker_mc")
    ticker_mc = ticker_input_mc.strip().upper()
    
    df_mc = yf.download(ticker_mc, start, end, multi_level_index = False)
    
    #define variables
    S_mc = float(round((df_mc['Close'].iloc[-1]), 2)) #base price
    st.write("Latest closing price of chosen stock : ", S_mc)
    
    user_val_mc = st.text_input("Enter the strike price", "0.01", key="strike_mc")
    K_mc = float(user_val_mc)
    r_percent_mc = st.slider("Risk-free rate (%)", 0.0, 10.0, value=1.0, step=0.01, format="%.2f%%", key="interest_mc")
    r_mc = r_percent_mc / 100
    T_mc = st.slider("Time to Maturity (in days)", 1, 365, value=240, step=1, key="time_mc") / 365
    M_mc = st.slider("Number of Simulations", 1, 1000, value=500, step=1, key="simulation_mc")
    N_mc = st.slider("Number of time steps", 1, 50, value=5, step=1, key="step_mc") 
    vol_choice_mc = st.radio("Select Volatility Type", ("Historical", "Custom"), key ="vol_mc")
    
    if vol_choice_mc == "Historical":
        returns = df_mc['Close'].pct_change().dropna()
        windowinput_mc = st.text_input("Enter the time window", "30", key = "window_mc")
        window_mc = int(windowinput_mc)
        rolling_std_mc = returns.rolling(window=window_mc).std()
        sigma_last_mc = rolling_std_mc.iloc[-1]
        sigma_mc = sigma_last_mc * np.sqrt(252)  # last value
        sigma_display_mc = float(round((sigma_mc*100), 4))
        sigma_display_mc = str(sigma_display_mc) + "%"
        st.write("Historical Volatility calculated from past ", window_mc," days: ", sigma_display_mc)
    else:
            # Let user enter custom volatility via slider
        sigma_percent_mc = st.slider("Enter the volatility (%)", 0.0, 50.0, value=10.0, step=0.01, format="%.2f%%", key="volcustom_mc")
        sigma_mc = sigma_percent_mc / 100

    entry_price_mc = round(st.number_input("Entry premium (per option)", min_value=0.0, value=3.25, step=0.01, format="%.2f"), 2)

    option_type_mc = st.radio("Select Option Type", ("Call", "Put"), key="type_mc")
    option_type_code_mc = "C" if option_type_mc == "Call" else "P"

    # Simulation parameters
    dt_mc = T_mc / N_mc
    nudt_mc = (r_mc - 0.5 * sigma_mc**2) * dt_mc
    volsdt_mc = sigma_mc * np.sqrt(dt_mc)
    lnS_mc = np.log(S_mc)
    
    # Monte Carlo Simulation
    Z_mc = np.random.normal(size=(N_mc, M_mc))
    delta_lnSt_mc = nudt_mc + volsdt_mc * Z_mc
    lnSt_mc = lnS_mc + np.cumsum(delta_lnSt_mc, axis=0)
    lnSt_mc = np.vstack((np.full((1, M_mc), lnS_mc), lnSt_mc))  # prepend S0
    ST_mc = np.exp(lnSt_mc)
    
    # Payoff (European Option)
    if option_type_code_mc == "C":
        CT_mc = np.maximum(ST_mc[-1] - K_mc, 0)
    else:
        CT_mc = np.maximum(K_mc - ST_mc[-1], 0)
    
    C0_mc = np.exp(-r_mc * T_mc) * np.mean(CT_mc)
    
    # Standard error
    standev = np.std(CT_mc, ddof=1)
    SE_mc = standev / np.sqrt(M_mc)

    st.header("Results")
    st.write(f"{option_type_mc} option value is {C0_mc:.2f} with SE +/- {SE_mc:.2f}")

    expected_payoff = np.mean(CT_mc)
    expected_profit = expected_payoff - entry_price_mc
    st.write(f"Expected Payoff: {expected_payoff:.2f}")
    st.write(f"Expected Profit (per option): {expected_profit:.2f}")

    prob_profit = np.mean(CT_mc > entry_price_mc)
    st.write(f"Probability of Profit: {prob_profit:.2%}")

    num_paths_to_plot = min(20, M_mc)

    fig_path = go.Figure()
    
    for i in range(num_paths_to_plot):
        fig_path.add_trace(go.Scatter(
            x=list(range(N_mc + 1)),
            y=ST_mc[:, i],
            mode="lines",
            line=dict(width=1),
            name=f"Path {i+1}",
            opacity=0.7
        ))
    
    fig_path.update_layout(
        title="Simulated Stock Price Paths (Monte Carlo)",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price",
        template="plotly_white",
        showlegend=False
    )
    
    st.plotly_chart(fig_path, use_container_width=True)

    fig_hist = px.histogram(
    x=ST_mc[-1], nbins=100, opacity=0.7,
    title="Distribution of Terminal Stock Prices",
    labels={"x": "Terminal Stock Price", "y": "Frequency"}
    )
    
    # Add strike line
    fig_hist.add_vline(
        x=K_mc, line_dash="dash", line_color="red",
        annotation_text="Strike", annotation_position="top right"
    )
    
    fig_hist.update_layout(
        template="plotly_white"
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
