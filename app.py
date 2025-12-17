import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Stock Analyzer & Scoring", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .stMetric label { font-size: 0.9rem !important; }
    .stMetric .metric-value { font-size: 1.3rem !important; }
    h1 { font-size: 1.8rem !important; margin-bottom: 1rem !important; }
    h2 { font-size: 1.3rem !important; margin-top: 0.5rem !important; }
    h3 { font-size: 1.1rem !important; }
    .element-container { margin-bottom: 0.3rem !important; }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def calc_ma(data, period):
    """Calculate moving average"""
    if len(data) < period:
        return None
    return data.tail(period).mean()

def calculate_stage(price, ma50, ma150, ma200):
    """Calculate market stage based on moving averages
    S2 (1.0): Price > 50MA > 150MA > 200MA
    S1 (0.5): Price > 50MA > 150MA, 150MA < 200MA
    S3 Strong (0.5): Price > 50MA, 50MA < 150MA > 200MA
    Other (0): All else
    """
    try:
        current_price = float(price)
        ma_50 = float(ma50)
        ma_150 = float(ma150)
        ma_200 = float(ma200)
        
        if current_price > ma_50 and ma_50 > ma_150 and ma_150 > ma_200:
            return "S2", 1.0
        elif current_price > ma_50 and ma_50 > ma_150 and ma_150 < ma_200:
            return "S1", 0.5
        elif current_price > ma_50 and ma_50 < ma_150 and ma_150 > ma_200:
            return "S3 Strong", 0.5
        else:
            return "Other", 0.0
    except:
        return "Error", 0.0

@st.cache_data(ttl=3600)
def fetch_market_data():
    """Fetch SPX and NDX data for Market Pulse"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=300)
        
        spx_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        ndx_data = yf.download('^NDX', start=start_date, end=end_date, progress=False)
        
        spx_close = spx_data['Close'].squeeze() if 'Close' in spx_data else None
        ndx_close = ndx_data['Close'].squeeze() if 'Close' in ndx_data else None
        
        return spx_close, ndx_close
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return None, None

def check_market_pulse(spx_data, ndx_data):
    """Check if SPX and NDX are in Stage 2 (good market condition)
    Returns score: 1.0 if both in S2, 0.5 if one in S2, 0 otherwise
    """
    scores = []
    
    for data in [spx_data, ndx_data]:
        if data is not None and len(data) >= 200:
            try:
                current_price = float(data.iloc[-1])
                ma_50 = calc_ma(data, 50)
                ma_150 = calc_ma(data, 150)
                ma_200 = calc_ma(data, 200)
                
                if ma_50 and ma_150 and ma_200:
                    _, score = calculate_stage(current_price, ma_50, ma_150, ma_200)
                    scores.append(score)
            except:
                scores.append(0)
    
    if len(scores) == 2:
        if scores[0] == 1.0 and scores[1] == 1.0:
            return 1.0, "Both SPX & NDX in S2 🟢"
        elif scores[0] >= 0.5 or scores[1] >= 0.5:
            return 0.5, "One index in good stage 🟡"
        else:
            return 0.0, "Market not favorable 🔴"
    return 0.0, "Error checking market"

def detect_key_bars(df):
    """Detect Key Bars in the stock data
    Key Bar criteria:
    - Daily volume > 30-day SMA volume
    - abs(% change from open to close) > 1.5%
    - Price makes 5-day new high
    
    Returns: DataFrame with key bar indicators and most recent key bar info
    """
    if df is None or len(df) < 30:
        return None, None
    
    df = df.copy()
    
    # Calculate 30-day SMA volume
    df['Volume_SMA30'] = df['Volume'].rolling(window=30).mean()
    
    # Calculate % change from open to close
    df['Open_Close_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    
    # Calculate 5-day high
    df['High_5D'] = df['High'].rolling(window=5).max()
    
    # Detect key bars
    df['Is_Key_Bar'] = (
        (df['Volume'] > df['Volume_SMA30']) &
        (abs(df['Open_Close_Change_Pct']) > 1.5) &
        (df['High'] >= df['High_5D'])
    )
    
    # Find most recent key bar in last 10 days
    recent_data = df.tail(10)
    recent_key_bars = recent_data[recent_data['Is_Key_Bar']]
    
    if len(recent_key_bars) > 0:
        most_recent_kb = recent_key_bars.iloc[-1]
        return df, most_recent_kb
    
    return df, None

def calculate_key_bar_score(df, recent_key_bar):
    """Calculate Key Bar score
    - 0.5 if key bar in last 10 days
    - +0.5 if current price <= 1.05x key bar close
    """
    if df is None or recent_key_bar is None:
        return 0.0, "No recent key bar"
    
    score = 0.5  # Key bar exists in last 10 days
    current_price = df['Close'].iloc[-1]
    kb_close = recent_key_bar['Close']
    
    details = f"Recent Key Bar found (Score: 0.5)"
    
    if current_price <= kb_close * 1.05:
        score += 0.5
        details += f"\nPrice near Key Bar (Score: +0.5)"
        details += f"\nCurrent: ${current_price:.2f} | Key Bar: ${kb_close:.2f} (max: ${kb_close*1.05:.2f})"
    else:
        details += f"\nPrice too far from Key Bar (Score: 0)"
        details += f"\nCurrent: ${current_price:.2f} | Key Bar: ${kb_close:.2f} (max: ${kb_close*1.05:.2f})"
    
    return score, details

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    """Fetch stock price and volume data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df is None or len(df) == 0:
            return None, "No price data available"
        
        return df, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker):
    """Fetch fundamental data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly financials
        quarterly_income = stock.quarterly_income_stmt
        quarterly_balance = stock.quarterly_balance_sheet
        quarterly_cashflow = stock.quarterly_cashflow
        
        # Get info for additional metrics
        info = stock.info
        
        return {
            'income': quarterly_income,
            'balance': quarterly_balance,
            'cashflow': quarterly_cashflow,
            'info': info
        }, None
    except Exception as e:
        return None, f"Error fetching fundamentals: {str(e)}"

def check_acceleration(values):
    """Check if values show acceleration (growth rate increasing)
    Returns True if at least 1 quarter shows acceleration
    """
    if values is None or len(values) < 2:
        return False
    
    # Calculate growth rates
    growth_rates = []
    for i in range(len(values) - 1):
        if values[i+1] != 0:
            growth = ((values[i] - values[i+1]) / abs(values[i+1])) * 100
            growth_rates.append(growth)
    
    # Check for acceleration (each quarter's growth > previous quarter's growth)
    accelerations = 0
    for i in range(len(growth_rates) - 1):
        if growth_rates[i] > growth_rates[i+1]:
            accelerations += 1
    
    return accelerations >= 1

def calculate_fundamental_scores(fund_data):
    """Calculate all 5 fundamental indicator scores"""
    scores = {
        'sales_growth': 0,
        'profit_margin': 0,
        'earnings': 0,
        'rule_of_40': 0,
        'roe': 0
    }
    
    details = {}
    
    if fund_data is None:
        return scores, details
    
    income = fund_data['income']
    balance = fund_data['balance']
    info = fund_data['info']
    
    # 1. Sales Growth Acceleration
    try:
        if 'Total Revenue' in income.index:
            revenue = income.loc['Total Revenue'].values[:4]  # Latest 4 quarters
            if check_acceleration(revenue):
                scores['sales_growth'] = 1
            details['sales'] = revenue
    except:
        pass
    
    # 2. Profit Margin Acceleration (using EBITDA or Operating Income)
    try:
        margin_metric = None
        if 'EBITDA' in income.index:
            ebitda = income.loc['EBITDA'].values[:4]
            revenue = income.loc['Total Revenue'].values[:4]
            margin_metric = (ebitda / revenue * 100) if len(revenue) == len(ebitda) else None
        elif 'Operating Income' in income.index:
            op_income = income.loc['Operating Income'].values[:4]
            revenue = income.loc['Total Revenue'].values[:4]
            margin_metric = (op_income / revenue * 100) if len(revenue) == len(op_income) else None
        
        if margin_metric is not None:
            if check_acceleration(margin_metric):
                scores['profit_margin'] = 1
            details['margin'] = margin_metric
    except:
        pass
    
    # 3. Earnings Acceleration (using Net Income or Diluted EPS)
    try:
        if 'Net Income' in income.index:
            earnings = income.loc['Net Income'].values[:4]
            if check_acceleration(earnings):
                scores['earnings'] = 1
            details['earnings'] = earnings
        elif 'Diluted EPS' in income.index:
            eps = income.loc['Diluted EPS'].values[:4]
            if check_acceleration(eps):
                scores['earnings'] = 1
            details['eps'] = eps
    except:
        pass
    
    # 4. Rule of 40 (Revenue Growth % + Profit Margin % >= 40%)
    try:
        revenue_growth = None
        profit_margin_pct = None
        
        if 'Total Revenue' in income.index:
            revenue = income.loc['Total Revenue'].values[:2]
            if len(revenue) >= 2 and revenue[1] != 0:
                revenue_growth = ((revenue[0] - revenue[1]) / abs(revenue[1])) * 100
        
        if 'EBITDA' in income.index and 'Total Revenue' in income.index:
            ebitda = income.loc['EBITDA'].values[0]
            revenue = income.loc['Total Revenue'].values[0]
            if revenue != 0:
                profit_margin_pct = (ebitda / revenue) * 100
        
        if revenue_growth is not None and profit_margin_pct is not None:
            rule_of_40_value = revenue_growth + profit_margin_pct
            details['rule_of_40'] = rule_of_40_value
            if rule_of_40_value >= 40:
                scores['rule_of_40'] = 1
    except:
        pass
    
    # 5. ROE or ROCE >= 17%
    try:
        roe = info.get('returnOnEquity', None)
        if roe is not None:
            roe_pct = roe * 100
            details['roe'] = roe_pct
            if roe_pct >= 17:
                scores['roe'] = 1
        else:
            # Try to calculate ROCE (EBIT / Capital Employed)
            if 'EBIT' in income.index and 'Total Assets' in balance.index and 'Current Liabilities' in balance.index:
                ebit = income.loc['EBIT'].values[0]
                total_assets = balance.loc['Total Assets'].values[0]
                current_liabilities = balance.loc['Current Liabilities'].values[0]
                capital_employed = total_assets - current_liabilities
                
                if capital_employed != 0:
                    roce = (ebit / capital_employed) * 100
                    details['roce'] = roce
                    if roce >= 17:
                        scores['roe'] = 1
    except:
        pass
    
    return scores, details

# ==================== MAIN APP ====================

st.title("📊 Stock Analyzer & Scoring System")
st.markdown("Comprehensive technical and fundamental analysis with scoring")

# Ticker input
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL").upper()
with col2:
    analyze_button = st.button("🔍 Analyze", type="primary")

if analyze_button or ticker:
    
    # Fetch market data for Market Pulse
    with st.spinner("Fetching market data..."):
        spx_data, ndx_data = fetch_market_data()
    
    # Fetch stock data
    with st.spinner(f"Fetching data for {ticker}..."):
        stock_df, error = fetch_stock_data(ticker)
        fund_data, fund_error = fetch_fundamental_data(ticker)
    
    if error:
        st.error(error)
        st.stop()
    
    # Display basic info
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${stock_df['Close'].iloc[-1]:.2f}")
        with col2:
            change = stock_df['Close'].iloc[-1] - stock_df['Close'].iloc[-2]
            change_pct = (change / stock_df['Close'].iloc[-2]) * 100
            st.metric("Change", f"{change_pct:.2f}%", delta=f"${change:.2f}")
        with col3:
            st.metric("Volume", f"{stock_df['Volume'].iloc[-1]:,.0f}")
        with col4:
            st.metric("Company", info.get('shortName', ticker))
    except:
        pass
    
    st.markdown("---")
    
    # ==================== TECHNICAL INDICATORS ====================
    st.header("🔧 Technical Indicators (Max: 3 points)")
    
    tech_scores = {}
    
    # 1. Stage 2
    st.subheader("1️⃣ Stage 2")
    with st.expander("ℹ️ Stage Definitions", expanded=False):
        st.markdown("""
        **S2** (1.0): Price > 50MA > 150MA > 200MA  
        **S1** (0.5): Price > 50MA > 150MA, 150MA < 200MA  
        **S3 Strong** (0.5): Price > 50MA, 50MA < 150MA > 200MA  
        **Other** (0): All else
        """)
    
    if len(stock_df) >= 200:
        current_price = float(stock_df['Close'].iloc[-1])
        ma_50 = calc_ma(stock_df['Close'], 50)
        ma_150 = calc_ma(stock_df['Close'], 150)
        ma_200 = calc_ma(stock_df['Close'], 200)
        
        if ma_50 and ma_150 and ma_200:
            stage, score = calculate_stage(current_price, ma_50, ma_150, ma_200)
            tech_scores['stage2'] = score
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"**Stage: {stage}**\n\nPrice: ${current_price:.2f} | 50MA: ${ma_50:.2f} | 150MA: ${ma_150:.2f} | 200MA: ${ma_200:.2f}")
            with col2:
                emoji = "🟢" if score == 1.0 else ("🟡" if score == 0.5 else "🔴")
                st.metric("Score", f"{score}/1.0", delta=emoji)
        else:
            tech_scores['stage2'] = 0
            st.warning("Unable to calculate moving averages")
    else:
        tech_scores['stage2'] = 0
        st.warning("Not enough data for Stage 2 calculation (need 200+ days)")
    
    st.markdown("---")
    
    # 2. Market Pulse
    st.subheader("2️⃣ Market Pulse")
    with st.expander("ℹ️ Market Pulse Info", expanded=False):
        st.markdown("""
        Checks if SPX and NDX indices are in favorable stages:
        - **1.0**: Both indices in Stage 2
        - **0.5**: At least one index in good stage
        - **0.0**: Market not favorable
        """)
    
    pulse_score, pulse_msg = check_market_pulse(spx_data, ndx_data)
    tech_scores['market_pulse'] = pulse_score
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"**{pulse_msg}**")
    with col2:
        emoji = "🟢" if pulse_score == 1.0 else ("🟡" if pulse_score == 0.5 else "🔴")
        st.metric("Score", f"{pulse_score}/1.0", delta=emoji)
    
    st.markdown("---")
    
    # 3. Key Bar
    st.subheader("3️⃣ Key Bar")
    with st.expander("ℹ️ Key Bar Definition", expanded=False):
        st.markdown("""
        **Key Bar Criteria:**
        - Daily volume > 30-day SMA volume
        - abs(% change from open to close) > 1.5%
        - Price makes 5-day new high
        
        **Scoring:**
        - 0.5: Key Bar in last 10 trading days
        - +0.5: Current price <= 1.05x Key Bar close
        """)
    
    df_with_kb, recent_kb = detect_key_bars(stock_df)
    kb_score, kb_details = calculate_key_bar_score(df_with_kb, recent_kb)
    tech_scores['key_bar'] = kb_score
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if recent_kb is not None:
            st.success(kb_details)
        else:
            st.warning("No Key Bar detected in last 10 trading days")
    with col2:
        emoji = "🟢" if kb_score == 1.0 else ("🟡" if kb_score == 0.5 else "🔴")
        st.metric("Score", f"{kb_score}/1.0", delta=emoji)
    
    # Technical Total Score
    total_tech_score = sum(tech_scores.values())
    st.markdown("---")
    st.markdown(f"### 📊 Technical Score: **{total_tech_score:.1f}/3.0**")
    
    # ==================== FUNDAMENTAL INDICATORS ====================
    st.header("💼 Fundamental Indicators (Max: 5 points)")
    
    if fund_error:
        st.error(fund_error)
        fund_scores = {k: 0 for k in ['sales_growth', 'profit_margin', 'earnings', 'rule_of_40', 'roe']}
        fund_details = {}
    else:
        fund_scores, fund_details = calculate_fundamental_scores(fund_data)
    
    # Display fundamental scores
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("1️⃣ Sales Growth Acceleration")
        if 'sales' in fund_details:
            revenue_data = fund_details['sales']
            st.write("**Last 4 Quarters Revenue:**")
            for i, val in enumerate(revenue_data):
                st.write(f"Q{i+1}: ${val:,.0f}")
        else:
            st.write("Data not available")
    with col2:
        emoji = "🟢" if fund_scores['sales_growth'] == 1 else "🔴"
        st.metric("Score", f"{fund_scores['sales_growth']}/1", delta=emoji)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("2️⃣ Profit Margin Acceleration")
        if 'margin' in fund_details:
            margin_data = fund_details['margin']
            st.write("**Last 4 Quarters Margin %:**")
            for i, val in enumerate(margin_data):
                st.write(f"Q{i+1}: {val:.2f}%")
        else:
            st.write("Data not available")
    with col2:
        emoji = "🟢" if fund_scores['profit_margin'] == 1 else "🔴"
        st.metric("Score", f"{fund_scores['profit_margin']}/1", delta=emoji)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("3️⃣ Earnings Acceleration")
        if 'earnings' in fund_details:
            earnings_data = fund_details['earnings']
            st.write("**Last 4 Quarters Net Income:**")
            for i, val in enumerate(earnings_data):
                st.write(f"Q{i+1}: ${val:,.0f}")
        elif 'eps' in fund_details:
            eps_data = fund_details['eps']
            st.write("**Last 4 Quarters EPS:**")
            for i, val in enumerate(eps_data):
                st.write(f"Q{i+1}: ${val:.2f}")
        else:
            st.write("Data not available")
    with col2:
        emoji = "🟢" if fund_scores['earnings'] == 1 else "🔴"
        st.metric("Score", f"{fund_scores['earnings']}/1", delta=emoji)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("4️⃣ Rule of 40")
        if 'rule_of_40' in fund_details:
            ro40 = fund_details['rule_of_40']
            st.write(f"**Rule of 40 Value: {ro40:.2f}%**")
            st.write("(Revenue Growth % + Profit Margin % ≥ 40%)")
        else:
            st.write("Data not available")
    with col2:
        emoji = "🟢" if fund_scores['rule_of_40'] == 1 else "🔴"
        st.metric("Score", f"{fund_scores['rule_of_40']}/1", delta=emoji)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("5️⃣ Return on Equity (ROE) or ROCE")
        if 'roe' in fund_details:
            roe_val = fund_details['roe']
            st.write(f"**ROE: {roe_val:.2f}%** (Target: ≥ 17%)")
        elif 'roce' in fund_details:
            roce_val = fund_details['roce']
            st.write(f"**ROCE: {roce_val:.2f}%** (Target: ≥ 17%)")
        else:
            st.write("Data not available")
    with col2:
        emoji = "🟢" if fund_scores['roe'] == 1 else "🔴"
        st.metric("Score", f"{fund_scores['roe']}/1", delta=emoji)
    
    # Fundamental Total Score
    total_fund_score = sum(fund_scores.values())
    st.markdown("---")
    st.markdown(f"### 📊 Fundamental Score: **{total_fund_score}/5.0**")
    
    # ==================== TOTAL SCORE ====================
    st.markdown("---")
    st.header("🎯 Total Score")
    
    total_score = total_tech_score + total_fund_score
    max_score = 8.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Technical", f"{total_tech_score:.1f}/3.0")
    with col2:
        st.metric("Fundamental", f"{total_fund_score}/5.0")
    with col3:
        percentage = (total_score / max_score) * 100
        color = "🟢" if percentage >= 70 else ("🟡" if percentage >= 50 else "🔴")
        st.metric("TOTAL", f"{total_score:.1f}/{max_score}", delta=f"{percentage:.0f}% {color}")
    
    # Rating
    if percentage >= 75:
        rating = "⭐⭐⭐⭐⭐ Excellent"
    elif percentage >= 60:
        rating = "⭐⭐⭐⭐ Good"
    elif percentage >= 45:
        rating = "⭐⭐⭐ Average"
    elif percentage >= 30:
        rating = "⭐⭐ Below Average"
    else:
        rating = "⭐ Poor"
    
    st.markdown(f"### Rating: {rating}")
    
    # ==================== CHARTS ====================
    st.markdown("---")
    st.header("📈 Price Chart with Moving Averages")
    
    # Create price chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])
    
    # Price and MAs
    fig.add_trace(go.Candlestick(
        x=stock_df.index,
        open=stock_df['Open'],
        high=stock_df['High'],
        low=stock_df['Low'],
        close=stock_df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Add moving averages
    if len(stock_df) >= 50:
        ma50_series = stock_df['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=stock_df.index, y=ma50_series, 
                                 name='50 MA', line=dict(color='blue', width=1)), row=1, col=1)
    
    if len(stock_df) >= 150:
        ma150_series = stock_df['Close'].rolling(window=150).mean()
        fig.add_trace(go.Scatter(x=stock_df.index, y=ma150_series, 
                                 name='150 MA', line=dict(color='orange', width=1)), row=1, col=1)
    
    if len(stock_df) >= 200:
        ma200_series = stock_df['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=stock_df.index, y=ma200_series, 
                                 name='200 MA', line=dict(color='red', width=1)), row=1, col=1)
    
    # Mark key bars
    if df_with_kb is not None:
        key_bar_dates = df_with_kb[df_with_kb['Is_Key_Bar']].index
        key_bar_prices = df_with_kb[df_with_kb['Is_Key_Bar']]['High']
        
        fig.add_trace(go.Scatter(
            x=key_bar_dates,
            y=key_bar_prices,
            mode='markers',
            name='Key Bar',
            marker=dict(color='green', size=10, symbol='star')
        ), row=1, col=1)
    
    # Volume
    colors = ['red' if stock_df['Close'].iloc[i] < stock_df['Open'].iloc[i] 
              else 'green' for i in range(len(stock_df))]
    
    fig.add_trace(go.Bar(x=stock_df.index, y=stock_df['Volume'], 
                         name='Volume', marker_color=colors), row=2, col=1)
    
    # Add volume SMA
    if len(stock_df) >= 30:
        vol_sma = stock_df['Volume'].rolling(window=30).mean()
        fig.add_trace(go.Scatter(x=stock_df.index, y=vol_sma, 
                                 name='30D Avg Vol', line=dict(color='orange', width=2)), row=2, col=1)
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("⚠️ This is for educational purposes only. Not financial advice.")
st.caption(f"📅 Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
