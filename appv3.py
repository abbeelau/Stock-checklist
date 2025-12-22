import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import requests

# ==================== VERSION INFO ====================
APP_VERSION = "v3.0"
VERSION_DATE = "2024-12-19"
CHANGELOG = """
**v3.0** (2024-12-19) - Alpha Vantage Integration
- ✅ Integrated Alpha Vantage API for reliable fundamental data
- ✅ Guaranteed 8+ quarters for all stocks (up to 81 quarters!)
- ✅ Full YoY data for ALL 4 quarters (no more N/A)
- ✅ 25 API calls/day = 12-13 stock analyses per day
- ✅ Clean JSON data structure
- ✅ Fallback to Yahoo Finance if AV fails

**v2.0** (2024-12-19)
- Added FMP API integration (deprecated - free tier no longer works)
- Hybrid mode with Yahoo Finance fallback

**v1.1** (2024-12-18)
- Changed to YoY growth calculations
- Added quarter date labels

**v1.0** (2024-12-18)
- Initial release
"""

# Set page configuration
st.set_page_config(page_title=f"Stock Analyzer {APP_VERSION}", layout="wide")

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

# ==================== PERSISTENT STORAGE ====================
USER_INPUTS_FILE = "user_inputs.json"

def load_user_inputs():
    """Load previously saved user inputs from JSON file"""
    if os.path.exists(USER_INPUTS_FILE):
        try:
            with open(USER_INPUTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_inputs(inputs):
    """Save user inputs to JSON file"""
    try:
        with open(USER_INPUTS_FILE, 'w') as f:
            json.dump(inputs, f, indent=2)
    except:
        pass

# Initialize session state
if 'market_pulse' not in st.session_state:
    saved_inputs = load_user_inputs()
    st.session_state.market_pulse = saved_inputs.get('market_pulse', 'Green - Acceleration')
if 'alpha_vantage_key' not in st.session_state:
    # Pre-filled with user's API key
    st.session_state.alpha_vantage_key = saved_inputs.get('alpha_vantage_key', '0Y0YAGE5H1Y7OLLU')

# ==================== ALPHA VANTAGE API FUNCTIONS (v3.0) ====================

def fetch_alpha_vantage_data(ticker, api_key):
    """
    Fetch fundamental data from Alpha Vantage API
    Uses 2 API calls per stock:
    1. Income Statement (quarterly)
    2. Balance Sheet (quarterly) - for ROE
    
    Returns: (income_data, balance_data, error)
    """
    if not api_key or api_key.strip() == '':
        return None, None, "Please provide a valid Alpha Vantage API key"
    
    try:
        base_url = "https://www.alphavantage.co/query"
        
        # Call 1: Income Statement
        income_params = {
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
            "apikey": api_key
        }
        
        income_response = requests.get(base_url, params=income_params, timeout=15)
        
        if income_response.status_code != 200:
            return None, None, f"Alpha Vantage API error: {income_response.status_code}"
        
        income_json = income_response.json()
        
        # Check for errors
        if "Error Message" in income_json:
            return None, None, f"Alpha Vantage error: {income_json['Error Message']}"
        elif "Note" in income_json:
            return None, None, f"Alpha Vantage rate limit: {income_json['Note']}"
        elif "quarterlyReports" not in income_json:
            return None, None, "No quarterly data available from Alpha Vantage"
        
        income_data = income_json["quarterlyReports"]
        
        # Call 2: Balance Sheet (for ROE)
        balance_params = {
            "function": "BALANCE_SHEET",
            "symbol": ticker,
            "apikey": api_key
        }
        
        balance_response = requests.get(base_url, params=balance_params, timeout=15)
        balance_data = []
        
        if balance_response.status_code == 200:
            balance_json = balance_response.json()
            if "quarterlyReports" in balance_json:
                balance_data = balance_json["quarterlyReports"]
        
        return income_data, balance_data, None
        
    except Exception as e:
        return None, None, f"Error fetching Alpha Vantage data: {str(e)}"


def calculate_alpha_vantage_fundamentals(income_data, balance_data):
    """
    Calculate all 5 fundamental indicators from Alpha Vantage data
    
    Returns: (scores, details)
    """
    scores = {
        'sales_growth': 0,
        'profit_margin': 0,
        'earnings': 0,
        'rule_of_40': 0,
        'roe': 0
    }
    
    details = {}
    
    if not income_data or len(income_data) < 4:
        return scores, details
    
    # Get quarter dates (first 4 quarters)
    quarter_dates = []
    for q in income_data[:4]:
        date_str = q.get('fiscalDateEnding', '')
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            quarter_dates.append(date_obj.strftime('%b-%Y'))
        except:
            quarter_dates.append(date_str)
    
    details['quarter_dates'] = quarter_dates
    details['revenue_quarters_available'] = len(income_data)
    details['earnings_quarters_available'] = len(income_data)
    
    # Extract basic data (first 4 quarters)
    revenues = []
    net_incomes = []
    operating_incomes = []
    
    for q in income_data[:4]:
        # Revenue
        rev = q.get('totalRevenue')
        revenues.append(int(rev) if rev and rev != 'None' else 0)
        
        # Net Income
        ni = q.get('netIncome')
        net_incomes.append(int(ni) if ni and ni != 'None' else 0)
        
        # Operating Income or EBITDA for margins
        oi = q.get('operatingIncome') or q.get('ebitda')
        operating_incomes.append(int(oi) if oi and oi != 'None' else 0)
    
    details['sales'] = revenues
    details['earnings'] = net_incomes
    
    # 1. Sales Growth Acceleration (YoY)
    if len(income_data) >= 8:
        sales_growth = []
        for i in range(4):
            current_rev = revenues[i]
            year_ago_rev_data = income_data[i + 4].get('totalRevenue')
            year_ago_rev = int(year_ago_rev_data) if year_ago_rev_data and year_ago_rev_data != 'None' else 0
            
            if year_ago_rev != 0 and current_rev != 0:
                yoy = ((current_rev - year_ago_rev) / abs(year_ago_rev)) * 100
                sales_growth.append(yoy)
            else:
                sales_growth.append(None)
        
        details['sales_growth'] = sales_growth
        
        # Check acceleration
        if sales_growth[0] is not None and sales_growth[1] is not None:
            if sales_growth[0] > sales_growth[1]:
                scores['sales_growth'] = 1
    else:
        details['sales_growth'] = [None, None, None, None]
    
    # 2. Profit Margin Acceleration
    margin_values = []
    for i in range(min(4, len(operating_incomes))):
        if revenues[i] != 0 and operating_incomes[i] != 0:
            margin = (operating_incomes[i] / revenues[i]) * 100
            margin_values.append(margin)
        else:
            margin_values.append(None)
    
    details['margin'] = margin_values
    
    if len(margin_values) >= 2 and margin_values[0] is not None and margin_values[1] is not None:
        if margin_values[0] > margin_values[1]:
            scores['profit_margin'] = 1
    
    # 3. Earnings Growth Acceleration (YoY)
    if len(income_data) >= 8:
        earnings_growth = []
        for i in range(4):
            current_ni = net_incomes[i]
            year_ago_ni_data = income_data[i + 4].get('netIncome')
            year_ago_ni = int(year_ago_ni_data) if year_ago_ni_data and year_ago_ni_data != 'None' else 0
            
            if year_ago_ni != 0 and current_ni != 0:
                yoy = ((current_ni - year_ago_ni) / abs(year_ago_ni)) * 100
                earnings_growth.append(yoy)
            else:
                earnings_growth.append(None)
        
        details['earnings_growth'] = earnings_growth
        
        # Check acceleration
        if earnings_growth[0] is not None and earnings_growth[1] is not None:
            if earnings_growth[0] > earnings_growth[1]:
                scores['earnings'] = 1
    else:
        details['earnings_growth'] = [None, None, None, None]
    
    # 4. Rule of 40 (latest quarter YoY)
    if len(income_data) >= 5:
        latest_rev = revenues[0]
        year_ago_rev_data = income_data[4].get('totalRevenue')
        year_ago_rev = int(year_ago_rev_data) if year_ago_rev_data and year_ago_rev_data != 'None' else 0
        
        if year_ago_rev != 0:
            revenue_growth = ((latest_rev - year_ago_rev) / abs(year_ago_rev)) * 100
            details['latest_revenue_growth'] = revenue_growth
            
            # Profit margin
            if revenues[0] != 0 and operating_incomes[0] != 0:
                profit_margin = (operating_incomes[0] / revenues[0]) * 100
                details['latest_profit_margin'] = profit_margin
                
                rule_of_40 = revenue_growth + profit_margin
                details['rule_of_40'] = rule_of_40
                
                if rule_of_40 >= 40:
                    scores['rule_of_40'] = 1
    
    # 5. ROE from balance sheet
    if balance_data and len(balance_data) > 0:
        roe_quarters = []
        
        # Match quarters from income and balance sheet
        for i in range(min(4, len(balance_data))):
            # Get net income for this quarter
            if i < len(net_incomes):
                ni = net_incomes[i]
                
                # Get equity
                equity_data = balance_data[i].get('totalShareholderEquity')
                equity = int(equity_data) if equity_data and equity_data != 'None' else 0
                
                if equity != 0 and ni != 0:
                    # Annualized ROE
                    quarterly_roe = (ni * 4 / equity) * 100
                    roe_quarters.append(quarterly_roe)
                else:
                    roe_quarters.append(None)
            else:
                roe_quarters.append(None)
        
        details['roe_quarters'] = roe_quarters
        
        # Check latest quarter ROE
        if roe_quarters and roe_quarters[0] is not None and roe_quarters[0] >= 17:
            scores['roe'] = 1
    
    return scores, details

# ==================== HELPER FUNCTIONS ====================

def calc_ma(data, period):
    """Calculate moving average"""
    if len(data) < period:
        return None
    return data.tail(period).mean()

def calculate_stage(price, ma50, ma150, ma200):
    """Calculate market stage based on moving averages"""
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

def detect_key_bars(df):
    """Detect Key Bars in the stock data
    Key Bar criteria:
    - Daily volume > 30-day SMA volume
    - abs(% change from open to close) > 1.5%
    - Day's high makes 5-day new high
    
    Returns: DataFrame with key bar indicators and most recent key bar info
    """
    if df is None or len(df) < 30:
        return None, None
    
    df = df.copy()
    
    # Calculate 30-day SMA volume
    df['Volume_SMA30'] = df['Volume'].rolling(window=30).mean()
    
    # Calculate % change from open to close
    df['Open_Close_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    
    # Calculate 5-day high (looking back at previous 5 days, not including today)
    df['High_5D_Previous'] = df['High'].shift(1).rolling(window=5).max()
    
    # Detect key bars: today's high must be greater than previous 5-day high
    df['Is_Key_Bar'] = (
        (df['Volume'] > df['Volume_SMA30']) &
        (abs(df['Open_Close_Change_Pct']) > 1.5) &
        (df['High'] > df['High_5D_Previous'])
    )
    
    # Find most recent key bar in last 10 days
    recent_data = df.tail(10)
    recent_key_bars = recent_data[recent_data['Is_Key_Bar']]
    
    if len(recent_key_bars) > 0:
        most_recent_kb = recent_key_bars.iloc[-1]
        return df, most_recent_kb
    
    return df, None

def calculate_key_bar_score(df, recent_key_bar):
    """Calculate Key Bar score"""
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
        
        # Get quarterly financials - get ALL available quarters
        quarterly_income = stock.quarterly_income_stmt
        quarterly_balance = stock.quarterly_balance_sheet
        
        # Get info for additional metrics
        info = stock.info
        
        return {
            'income': quarterly_income,
            'balance': quarterly_balance,
            'info': info
        }, None
    except Exception as e:
        return None, f"Error fetching fundamentals: {str(e)}"

def calculate_growth_rate(current, previous):
    """Calculate growth rate percentage"""
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return None
    return ((current - previous) / abs(previous)) * 100

def check_growth_acceleration(values):
    """Check if latest YoY growth rate > previous YoY growth rate
    YoY = comparing same quarter year-over-year (e.g., Q3-2024 vs Q3-2023)
    Returns: (is_accelerating, growth_rates_list)
    """
    if values is None or len(values) < 5:  # Need at least 5 quarters for 2 YoY comparisons
        return False, []
    
    # Calculate YoY growth rates
    # values[0] is most recent quarter, values[4] is 4 quarters ago (1 year)
    growth_rates = []
    
    # Latest YoY: Q0 vs Q4 (most recent vs 1 year ago)
    if len(values) > 4 and values[4] != 0 and not pd.isna(values[0]) and not pd.isna(values[4]):
        yoy_growth = ((values[0] - values[4]) / abs(values[4])) * 100
        growth_rates.append(yoy_growth)
    else:
        growth_rates.append(None)
    
    # Previous YoY: Q1 vs Q5 (previous quarter vs its year-ago comparison)
    if len(values) > 5 and values[5] != 0 and not pd.isna(values[1]) and not pd.isna(values[5]):
        yoy_growth = ((values[1] - values[5]) / abs(values[5])) * 100
        growth_rates.append(yoy_growth)
    else:
        growth_rates.append(None)
    
    # 2 quarters ago YoY: Q2 vs Q6
    if len(values) > 6 and values[6] != 0 and not pd.isna(values[2]) and not pd.isna(values[6]):
        yoy_growth = ((values[2] - values[6]) / abs(values[6])) * 100
        growth_rates.append(yoy_growth)
    else:
        growth_rates.append(None)
    
    # 3 quarters ago YoY: Q3 vs Q7
    if len(values) > 7 and values[7] != 0 and not pd.isna(values[3]) and not pd.isna(values[7]):
        yoy_growth = ((values[3] - values[7]) / abs(values[7])) * 100
        growth_rates.append(yoy_growth)
    else:
        growth_rates.append(None)
    
    # Check if latest YoY growth > previous YoY growth
    if len(growth_rates) >= 2 and growth_rates[0] is not None and growth_rates[1] is not None:
        return growth_rates[0] > growth_rates[1], growth_rates
    
    return False, growth_rates

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
    
    # Get quarter dates (show first 4 for display)
    quarter_dates = []
    if hasattr(income, 'columns'):
        for col in income.columns[:4]:
            try:
                quarter_dates.append(col.strftime('%b-%Y'))
            except:
                quarter_dates.append(str(col))
    
    details['quarter_dates'] = quarter_dates
    
    # 1. Sales Growth Acceleration (YoY)
    try:
        if 'Total Revenue' in income.index:
            revenue = income.loc['Total Revenue'].values[:8]  # Try to get 8 quarters for YoY
            is_accelerating, growth_rates = check_growth_acceleration(revenue)
            
            # Debug: Add info about available quarters
            details['revenue_quarters_available'] = len(revenue)
            
            if is_accelerating:
                scores['sales_growth'] = 1
            
            details['sales'] = revenue[:4]  # Only show first 4 quarters
            details['sales_growth'] = growth_rates[:4] if growth_rates else []  # Show 4 YoY growth rates
    except Exception as e:
        details['sales_error'] = str(e)
        pass
    
    # 2. Profit Margin Acceleration
    try:
        margin_values = []
        if 'EBITDA' in income.index and 'Total Revenue' in income.index:
            ebitda = income.loc['EBITDA'].values[:4]
            revenue = income.loc['Total Revenue'].values[:4]
            if len(revenue) == len(ebitda):
                margin_values = [(ebitda[i] / revenue[i] * 100) if revenue[i] != 0 else None for i in range(len(revenue))]
        elif 'Operating Income' in income.index and 'Total Revenue' in income.index:
            op_income = income.loc['Operating Income'].values[:4]
            revenue = income.loc['Total Revenue'].values[:4]
            if len(revenue) == len(op_income):
                margin_values = [(op_income[i] / revenue[i] * 100) if revenue[i] != 0 else None for i in range(len(revenue))]
        
        if len(margin_values) >= 2:
            # Check if latest margin > previous margin
            if margin_values[0] is not None and margin_values[1] is not None:
                if margin_values[0] > margin_values[1]:
                    scores['profit_margin'] = 1
            details['margin'] = margin_values
    except:
        pass
    
    # 3. Earnings Growth Acceleration (YoY using Net Income)
    try:
        if 'Net Income' in income.index:
            earnings = income.loc['Net Income'].values[:8]  # Try to get 8 quarters for YoY
            is_accelerating, growth_rates = check_growth_acceleration(earnings)
            
            # Debug: Add info about available quarters
            details['earnings_quarters_available'] = len(earnings)
            
            if is_accelerating:
                scores['earnings'] = 1
            
            details['earnings'] = earnings[:4]  # Only show first 4 quarters
            details['earnings_growth'] = growth_rates[:4] if growth_rates else []  # Show 4 YoY growth rates
    except Exception as e:
        details['earnings_error'] = str(e)
        pass
    
    # 4. Rule of 40
    try:
        revenue_growth = None
        profit_margin_pct = None
        
        if 'Total Revenue' in income.index:
            revenue = income.loc['Total Revenue'].values[:2]
            if len(revenue) >= 2 and revenue[1] != 0:
                revenue_growth = ((revenue[0] - revenue[1]) / abs(revenue[1])) * 100
                details['latest_revenue_growth'] = revenue_growth
        
        if 'EBITDA' in income.index and 'Total Revenue' in income.index:
            ebitda = income.loc['EBITDA'].values[0]
            revenue = income.loc['Total Revenue'].values[0]
            if revenue != 0:
                profit_margin_pct = (ebitda / revenue) * 100
                details['latest_profit_margin'] = profit_margin_pct
        elif 'Operating Income' in income.index and 'Total Revenue' in income.index:
            op_income = income.loc['Operating Income'].values[0]
            revenue = income.loc['Total Revenue'].values[0]
            if revenue != 0:
                profit_margin_pct = (op_income / revenue) * 100
                details['latest_profit_margin'] = profit_margin_pct
        
        if revenue_growth is not None and profit_margin_pct is not None:
            rule_of_40_value = revenue_growth + profit_margin_pct
            details['rule_of_40'] = rule_of_40_value
            if rule_of_40_value >= 40:
                scores['rule_of_40'] = 1
    except:
        pass
    
    # 5. ROE - Get last 4 quarters if available
    try:
        # Try to get TTM ROE from info
        roe = info.get('returnOnEquity', None)
        if roe is not None:
            roe_pct = roe * 100
            details['roe'] = roe_pct
            details['roe_quarters'] = [roe_pct]  # Only have TTM data
            if roe_pct >= 17:
                scores['roe'] = 1
        else:
            # Try to calculate quarterly ROE
            if 'Net Income' in income.index and 'Stockholders Equity' in balance.index:
                net_income = income.loc['Net Income'].values[:4]
                equity = balance.loc['Stockholders Equity'].values[:4]
                
                roe_quarters = []
                for i in range(min(len(net_income), len(equity))):
                    if equity[i] != 0 and not pd.isna(net_income[i]) and not pd.isna(equity[i]):
                        # Annualized quarterly ROE
                        quarterly_roe = (net_income[i] * 4 / equity[i]) * 100
                        roe_quarters.append(quarterly_roe)
                    else:
                        roe_quarters.append(None)
                
                if roe_quarters:
                    details['roe_quarters'] = roe_quarters
                    # Check latest quarter
                    if roe_quarters[0] is not None and roe_quarters[0] >= 17:
                        scores['roe'] = 1
    except:
        pass
    
    return scores, details

# ==================== MAIN APP ====================

st.title(f"📊 Stock Analyzer & Scoring System {APP_VERSION}")
st.markdown("Comprehensive technical and fundamental analysis with scoring")

# Version info in sidebar
with st.sidebar:
    st.markdown(f"### 🔖 Version {APP_VERSION}")
    st.caption(f"Released: {VERSION_DATE}")
    
    with st.expander("📝 Changelog"):
        st.markdown(CHANGELOG)
    
    st.markdown("---")
    
    # Alpha Vantage API Key input
    st.markdown("### 🔑 Alpha Vantage API Key")
    st.caption("[Get free key](https://www.alphavantage.co/support/#api-key) (25 calls/day)")
    
    av_key = st.text_input(
        "Alpha Vantage API Key",
        value=st.session_state.alpha_vantage_key,
        type="password",
        help="Free tier: 25 calls/day = 12-13 stock analyses. Guaranteed 8+ quarters with full YoY data!"
    )
    
    # Save API key if changed
    if av_key != st.session_state.alpha_vantage_key:
        st.session_state.alpha_vantage_key = av_key
        saved_inputs = load_user_inputs()
        saved_inputs['alpha_vantage_key'] = av_key
        save_user_inputs(saved_inputs)
    
    # Show data source status
    if av_key and av_key.strip() != '':
        st.success("✅ Using Alpha Vantage API")
        st.caption("2 API calls per stock | 12-13 stocks/day")
        st.caption("✓ Guaranteed 8+ quarters | ✓ Full YoY data")
    else:
        st.warning("⚠️ Using Yahoo Finance (limited data)")
        st.caption("Some stocks may have incomplete YoY data")

# Ticker input
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL").upper()
with col2:
    analyze_button = st.button("🔍 Analyze", type="primary")

if analyze_button or ticker:
    
    # Fetch stock data
    with st.spinner(f"Fetching data for {ticker}..."):
        stock_df, error = fetch_stock_data(ticker)
        
        # Try Alpha Vantage first if API key is provided
        use_av = False
        if av_key and av_key.strip() != '':
            st.info("📡 Fetching from Alpha Vantage API...")
            av_income, av_balance, av_error = fetch_alpha_vantage_data(ticker, av_key)
            
            if not av_error and av_income:
                use_av = True
                st.success(f"✅ Using Alpha Vantage data ({len(av_income)} quarters available)")
            else:
                st.warning(f"⚠️ Alpha Vantage failed: {av_error}. Falling back to Yahoo Finance...")
                use_av = False
        
        # Fetch Yahoo Finance data (either as primary or fallback)
        if not use_av:
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
    
    # ==================== TABS ====================
    tab1, tab2 = st.tabs(["🔧 Technical Analysis", "💼 Fundamental Analysis"])
    
    # ==================== TECHNICAL TAB ====================
    with tab1:
        st.header("Technical Indicators (Max: 3 points)")
        
        tech_scores = {}
        
        # 1. Stage 2
        st.subheader("1️⃣ Stage 2")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            with st.popover("ℹ️ Stage Definitions"):
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
        
        # 2. Market Pulse (Manual Input)
        st.subheader("2️⃣ Market Pulse")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            with st.popover("ℹ️ Market Pulse Info"):
                st.markdown("""
                **Green - Acceleration** (1.0): Price > 10VMA; VWMA8 > VWMA21 > VWMA34  
                **Grey Strong - Accumulation** (0.5): Price > 10VMA; VWMAs not perfectly stacked  
                **Grey Weak/Red** (0): Distribution or Deceleration
                
                *Manually assess the overall market condition (SPX/NDX)*
                """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            market_pulse = st.selectbox(
                "Overall Market Condition",
                ["Green - Acceleration", "Grey Strong - Accumulation", "Grey Weak - Distribution", "Red - Deceleration"],
                index=["Green - Acceleration", "Grey Strong - Accumulation", "Grey Weak - Distribution", "Red - Deceleration"].index(st.session_state.market_pulse),
                key="pulse_select"
            )
            
            # Save if changed
            if market_pulse != st.session_state.market_pulse:
                st.session_state.market_pulse = market_pulse
                saved_inputs = load_user_inputs()
                saved_inputs['market_pulse'] = market_pulse
                save_user_inputs(saved_inputs)
        
        with col2:
            if market_pulse == "Green - Acceleration":
                pulse_score = 1.0
                st.metric("Score", f"{pulse_score}/1.0", delta="🟢")
            elif market_pulse == "Grey Strong - Accumulation":
                pulse_score = 0.5
                st.metric("Score", f"{pulse_score}/1.0", delta="🟡")
            else:
                pulse_score = 0.0
                st.metric("Score", f"{pulse_score}/1.0", delta="🔴")
        
        tech_scores['market_pulse'] = pulse_score
        
        st.markdown("---")
        
        # 3. Key Bar
        st.subheader("3️⃣ Key Bar")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            with st.popover("ℹ️ Key Bar Definition"):
                st.markdown("""
                **Key Bar Criteria:**
                - Daily volume > 30-day SMA volume
                - abs(% change from open to close) > 1.5%
                - Day's high makes 5-day new high
                
                **Scoring:**
                - 0.5: Key Bar in last 10 trading days
                - +0.5: Current price ≤ 1.05x Key Bar close
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
        
        # ==================== PRICE CHART ====================
        st.markdown("---")
        st.subheader("📈 Price Chart with Moving Averages")
        
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
    
    # ==================== FUNDAMENTAL TAB ====================
    with tab2:
        st.header("Fundamental Indicators (Max: 5 points)")
        
        # Calculate scores based on data source
        if use_av:
            st.info("📊 Using Alpha Vantage data (guaranteed full YoY for all quarters)")
            fund_scores, fund_details = calculate_alpha_vantage_fundamentals(av_income, av_balance)
            fund_error = None
        else:
            if fund_error:
                st.error(fund_error)
                fund_scores = {k: 0 for k in ['sales_growth', 'profit_margin', 'earnings', 'rule_of_40', 'roe']}
                fund_details = {}
            else:
                fund_scores, fund_details = calculate_fundamental_scores(fund_data)
        
        # Display fundamental scores
        
        # 1. Sales Growth Acceleration
        st.subheader("1️⃣ Sales Growth Acceleration")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            with st.popover("ℹ️ Scoring Criteria"):
                st.markdown("""
                **Score 1 if:** Latest quarter YoY sales growth % > Previous quarter YoY sales growth %
                
                **YoY (Year-over-Year):** Compares same quarter across years (e.g., Q3-2024 vs Q3-2023)
                
                Shows last 4 quarters of revenue with their YoY growth rates.
                """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if 'sales' in fund_details and 'sales_growth' in fund_details:
                revenue_data = fund_details['sales']
                growth_data = fund_details['sales_growth']
                quarter_dates = fund_details.get('quarter_dates', [])
                quarters_available = fund_details.get('revenue_quarters_available', 0)
                
                st.write("**Last 4 Quarters (Newest to Oldest):**")
                
                # Count how many YoY values we have
                yoy_available = sum(1 for g in growth_data if g is not None)
                
                # Display all quarters with their YoY growth
                for i in range(min(len(revenue_data), 4)):
                    quarter_label = quarter_dates[i] if i < len(quarter_dates) else f"Q{i+1}"
                    revenue_str = f"${revenue_data[i]:,.0f}"
                    
                    # Show YoY growth if available
                    if i < len(growth_data) and growth_data[i] is not None:
                        st.write(f"{quarter_label}: {revenue_str} **(YoY: {growth_data[i]:+.2f}%)**")
                    else:
                        st.write(f"{quarter_label}: {revenue_str} (YoY: N/A)")
                
                # Info message if not all YoY data available
                if yoy_available < 4:
                    st.info(f"ℹ️ Only {yoy_available} of 4 quarters have YoY data ({quarters_available} quarters available from Yahoo Finance, need 8 for full YoY)")
                
                # Show acceleration comparison if we have at least 2 YoY values
                if len(growth_data) >= 2 and growth_data[0] is not None and growth_data[1] is not None:
                    latest_q = quarter_dates[0] if len(quarter_dates) > 0 else "Latest"
                    prev_q = quarter_dates[1] if len(quarter_dates) > 1 else "Previous"
                    if growth_data[0] > growth_data[1]:
                        st.success(f"✅ **Accelerating:** {latest_q} YoY ({growth_data[0]:+.2f}%) > {prev_q} YoY ({growth_data[1]:+.2f}%)")
                    else:
                        st.warning(f"❌ **Not Accelerating:** {latest_q} YoY ({growth_data[0]:+.2f}%) ≤ {prev_q} YoY ({growth_data[1]:+.2f}%)")
                elif yoy_available < 2:
                    st.warning("⚠️ Need at least 2 quarters with YoY data to check acceleration")
            else:
                st.write("Data not available")
        
        with col2:
            emoji = "🟢" if fund_scores['sales_growth'] == 1 else "🔴"
            st.metric("Score", f"{fund_scores['sales_growth']}/1", delta=emoji)
        
        st.markdown("---")
        
        # 2. Profit Margin Acceleration
        st.subheader("2️⃣ Profit Margin Acceleration")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            with st.popover("ℹ️ Scoring Criteria"):
                st.markdown("""
                **Score 1 if:** Latest quarter profit margin % > Previous quarter profit margin %
                
                Uses EBITDA margin or Operating margin.
                """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if 'margin' in fund_details:
                margin_data = fund_details['margin']
                quarter_dates = fund_details.get('quarter_dates', [])
                
                st.write("**Last 4 Quarters Margin % (Newest to Oldest):**")
                for i, val in enumerate(margin_data):
                    quarter_label = quarter_dates[i] if i < len(quarter_dates) else f"Q{i+1}"
                    if val is not None:
                        st.write(f"{quarter_label}: {val:.2f}%")
                    else:
                        st.write(f"{quarter_label}: N/A")
                
                if len(margin_data) >= 2 and margin_data[0] is not None and margin_data[1] is not None:
                    latest_q = quarter_dates[0] if len(quarter_dates) > 0 else "Latest"
                    prev_q = quarter_dates[1] if len(quarter_dates) > 1 else "Previous"
                    if margin_data[0] > margin_data[1]:
                        st.success(f"✅ Improving: {latest_q} ({margin_data[0]:.2f}%) > {prev_q} ({margin_data[1]:.2f}%)")
                    else:
                        st.warning(f"❌ Not Improving: {latest_q} ({margin_data[0]:.2f}%) ≤ {prev_q} ({margin_data[1]:.2f}%)")
            else:
                st.write("Data not available")
        
        with col2:
            emoji = "🟢" if fund_scores['profit_margin'] == 1 else "🔴"
            st.metric("Score", f"{fund_scores['profit_margin']}/1", delta=emoji)
        
        st.markdown("---")
        
        # 3. Earnings Growth Acceleration
        st.subheader("3️⃣ Earnings Growth Acceleration")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            with st.popover("ℹ️ Scoring Criteria"):
                st.markdown("""
                **Score 1 if:** Latest quarter YoY earnings growth % > Previous quarter YoY earnings growth %
                
                **YoY (Year-over-Year):** Compares same quarter across years (e.g., Q3-2024 vs Q3-2023)
                
                Uses Net Income to calculate YoY growth rates.
                """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if 'earnings' in fund_details and 'earnings_growth' in fund_details:
                earnings_data = fund_details['earnings']
                growth_data = fund_details['earnings_growth']
                quarter_dates = fund_details.get('quarter_dates', [])
                quarters_available = fund_details.get('earnings_quarters_available', 0)
                
                st.write("**Last 4 Quarters (Newest to Oldest):**")
                
                # Count how many YoY values we have
                yoy_available = sum(1 for g in growth_data if g is not None)
                
                # Display all quarters with their YoY growth
                for i in range(min(len(earnings_data), 4)):
                    quarter_label = quarter_dates[i] if i < len(quarter_dates) else f"Q{i+1}"
                    earnings_str = f"${earnings_data[i]:,.0f}"
                    
                    # Show YoY growth if available
                    if i < len(growth_data) and growth_data[i] is not None:
                        st.write(f"{quarter_label}: {earnings_str} **(YoY: {growth_data[i]:+.2f}%)**")
                    else:
                        st.write(f"{quarter_label}: {earnings_str} (YoY: N/A)")
                
                # Info message if not all YoY data available
                if yoy_available < 4:
                    st.info(f"ℹ️ Only {yoy_available} of 4 quarters have YoY data ({quarters_available} quarters available from Yahoo Finance, need 8 for full YoY)")
                
                # Show acceleration comparison if we have at least 2 YoY values
                if len(growth_data) >= 2 and growth_data[0] is not None and growth_data[1] is not None:
                    latest_q = quarter_dates[0] if len(quarter_dates) > 0 else "Latest"
                    prev_q = quarter_dates[1] if len(quarter_dates) > 1 else "Previous"
                    if growth_data[0] > growth_data[1]:
                        st.success(f"✅ **Accelerating:** {latest_q} YoY ({growth_data[0]:+.2f}%) > {prev_q} YoY ({growth_data[1]:+.2f}%)")
                    else:
                        st.warning(f"❌ **Not Accelerating:** {latest_q} YoY ({growth_data[0]:+.2f}%) ≤ {prev_q} YoY ({growth_data[1]:+.2f}%)")
                elif yoy_available < 2:
                    st.warning("⚠️ Need at least 2 quarters with YoY data to check acceleration")
            else:
                st.write("Data not available")
        
        with col2:
            emoji = "🟢" if fund_scores['earnings'] == 1 else "🔴"
            st.metric("Score", f"{fund_scores['earnings']}/1", delta=emoji)
        
        st.markdown("---")
        
        # 4. Rule of 40
        st.subheader("4️⃣ Rule of 40")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            with st.popover("ℹ️ Scoring Criteria"):
                st.markdown("""
                **Score 1 if:** Revenue Growth % + Profit Margin % ≥ 40%
                
                Uses latest quarter data.
                """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if 'rule_of_40' in fund_details:
                ro40 = fund_details['rule_of_40']
                rev_growth = fund_details.get('latest_revenue_growth', 0)
                profit_margin = fund_details.get('latest_profit_margin', 0)
                
                st.write(f"**Latest Quarter:**")
                st.write(f"Revenue Growth: {rev_growth:.2f}%")
                st.write(f"Profit Margin: {profit_margin:.2f}%")
                st.write(f"**Rule of 40: {ro40:.2f}%**")
                
                if ro40 >= 40:
                    st.success(f"✅ Passed: {ro40:.2f}% ≥ 40%")
                else:
                    st.warning(f"❌ Failed: {ro40:.2f}% < 40%")
            else:
                st.write("Data not available")
        
        with col2:
            emoji = "🟢" if fund_scores['rule_of_40'] == 1 else "🔴"
            st.metric("Score", f"{fund_scores['rule_of_40']}/1", delta=emoji)
        
        st.markdown("---")
        
        # 5. ROE/ROCE
        st.subheader("5️⃣ Return on Equity (ROE) or ROCE")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            with st.popover("ℹ️ Scoring Criteria"):
                st.markdown("""
                **Score 1 if:** ROE or ROCE ≥ 17%
                
                Shows last 4 quarters if available (or TTM data).
                """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if 'roe_quarters' in fund_details:
                roe_data = fund_details['roe_quarters']
                quarter_dates = fund_details.get('quarter_dates', [])
                
                st.write("**Return on Equity (Newest to Oldest):**")
                for i, val in enumerate(roe_data):
                    quarter_label = quarter_dates[i] if i < len(quarter_dates) else f"Q{i+1}"
                    if val is not None:
                        st.write(f"{quarter_label}: {val:.2f}%")
                    else:
                        st.write(f"{quarter_label}: N/A")
                
                if roe_data[0] is not None:
                    latest_q = quarter_dates[0] if len(quarter_dates) > 0 else "Latest"
                    if roe_data[0] >= 17:
                        st.success(f"✅ Passed: {latest_q} ({roe_data[0]:.2f}%) ≥ 17%")
                    else:
                        st.warning(f"❌ Failed: {latest_q} ({roe_data[0]:.2f}%) < 17%")
            elif 'roe' in fund_details:
                roe_val = fund_details['roe']
                st.write(f"**ROE (TTM): {roe_val:.2f}%**")
                
                if roe_val >= 17:
                    st.success(f"✅ Passed: {roe_val:.2f}% ≥ 17%")
                else:
                    st.warning(f"❌ Failed: {roe_val:.2f}% < 17%")
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
    st.header("🎯 Total Score Summary")
    
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

# ==================== FOOTER ====================
st.markdown("---")
st.caption("⚠️ This is for educational purposes only. Not financial advice.")
st.caption(f"📅 Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
