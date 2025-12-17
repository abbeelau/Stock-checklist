# 📊 Stock Analyzer & Scoring System

A comprehensive stock analysis tool that scores stocks based on technical and fundamental indicators.

## Features

### Technical Indicators (Max: 3 points)
1. **Stage 2** (1 point) - Analyzes price position relative to 50MA, 150MA, and 200MA
2. **Market Pulse** (1 point) - Checks if SPX and NDX indices are in favorable conditions
3. **Key Bar** (1 point) - Detects significant volume and price action patterns

### Fundamental Indicators (Max: 5 points)
1. **Sales Growth Acceleration** (1 point)
2. **Profit Margin Acceleration** (1 point)
3. **Earnings Acceleration** (1 point)
4. **Rule of 40** (1 point)
5. **ROE/ROCE >= 17%** (1 point)

## Setup Instructions

### Local Development

1. Clone this repository:
```bash
git clone <your-repo-url>
cd stock_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Deployment to Streamlit Cloud

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `stock-analyzer`
3. Set it to Public
4. Don't initialize with README (we already have files)

### Step 2: Push Code to GitHub

```bash
# Initialize git in your project folder
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Stock Analyzer app"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/stock-analyzer.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/stock-analyzer`
5. Set the main file path: `app.py`
6. Click "Deploy"

Your app will be live in a few minutes at: `https://YOUR_USERNAME-stock-analyzer.streamlit.app`

## Usage

1. Enter a stock ticker (e.g., AAPL, TSLA, MSFT)
2. Click "Analyze"
3. Review the technical and fundamental scores
4. Check the total score and rating
5. Analyze the price chart with key indicators

## Technical Indicator Details

### Stage 2
- **S2 (1.0)**: Price > 50MA > 150MA > 200MA (Best)
- **S1 (0.5)**: Price > 50MA > 150MA, 150MA < 200MA
- **S3 Strong (0.5)**: Price > 50MA, 50MA < 150MA > 200MA
- **Other (0)**: All other configurations

### Market Pulse
- **1.0**: Both SPX and NDX in Stage 2
- **0.5**: One index in good stage
- **0.0**: Market not favorable

### Key Bar Criteria
- Daily volume > 30-day SMA volume
- abs(% change from open to close) > 1.5%
- Price makes 5-day new high

**Scoring:**
- 0.5 points: Key Bar appears within last 10 trading days
- +0.5 points: Current price ≤ 1.05x Key Bar close

## Fundamental Indicator Details

### Acceleration Metrics
The app checks if the growth rate of each metric is increasing quarter-over-quarter. At least 1 quarter of acceleration is required to score a point.

### Rule of 40
Revenue Growth % + Profit Margin % should be ≥ 40%

### ROE/ROCE
Return on Equity or Return on Capital Employed should be ≥ 17%

## Data Source

All data is fetched from Yahoo Finance using the `yfinance` library.

## Rating System

- ⭐⭐⭐⭐⭐ Excellent: 75%+ (6+ points)
- ⭐⭐⭐⭐ Good: 60-74% (5-6 points)
- ⭐⭐⭐ Average: 45-59% (3.6-4.8 points)
- ⭐⭐ Below Average: 30-44% (2.4-3.5 points)
- ⭐ Poor: <30% (<2.4 points)

## Disclaimer

⚠️ This tool is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.

## License

MIT License
