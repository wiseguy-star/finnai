# Finance AI
First version of my Financial AI Assistant

Completed on 28 May 2025 20:25

## Getting started(v1.py)

```
pip install -r requirements.txt
pip list
```
To run
```
python -m streamlit run v1.py

          OR
streamlit run v1.py  
 ```

For testing:
```
python -m streamlit run v1test.py

          OR
streamlit run v1test.py   
```






## Comprehensive Beginner-Friendly Guide for stock market analysis.

### Introduction
These four Python scripts are tools for financial analysis and trading strategy evaluation, designed for beginners interested in stock and index markets. They are particularly useful for analyzing Indian (NIFTY, BSE) and U.S. (NYSE, NASDAQ) markets. Here’s what each does:

1. **Index Backtesting Script (`index_backtest.py`)**
   - Tests a 21-day vs. 50-day SMA crossover strategy on NIFTY, NYSE, NASDAQ, or BSE.
   - Outputs trade lists, performance metrics, and CSV files with historical data and trades.
   - Helps you evaluate if this strategy is profitable for index futures trading.

2. **Portfolio Analysis Script (`portfolio_analysis.py`)**
   - Analyzes a portfolio of stocks from a CSV file, assessing health, sentiment, and news.
   - Outputs a console report with a health summary, sentiment table, and news placeholders.
   - Useful for monitoring your stock investments.

3. **Stock Comparison Script (`stock_comparison.py`)**
   - Compares three stocks in the same sector (e.g., Indian banks) across fundamentals, financials, performance, outlook, technicals, and valuation.
   - Outputs an HTML report identifying the best investment.
   - Helps you choose the best stock to invest in.

4. **Index Analysis Script (`index_analysis.py`)**
   - Performs technical analysis on a single index (e.g., NIFTY) using indicators like MA, RSI, MACD, and Bollinger Bands.
   - Outputs an HTML report with a buy/hold/sell recommendation.
   - Useful for deciding whether to trade an index based on technical signals.

### Prerequisites
Before running the scripts, set up your computer:

1. **Install Python**:
   - Download from [python.org/downloads](https://www.python.org/downloads/) (version 3.11 or 3.12).
   - Windows: Check “Add Python to PATH” during installation.
   - Verify: Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:
     ```
     python --version
     ```
     Expected output: `Python 3.11.5` (or similar).

2. **Install a Code Editor**:
   - Download Visual Studio Code (VS Code) from [code.visualstudio.com](https://code.visualstudio.com/).
   - Install the Python extension in VS Code (Extensions > Search “Python” > Install Microsoft’s Python extension).

3. **Create a Project Folder**:
   - Create a folder (e.g., `StockAnalysis`) on your computer (e.g., `C:\StockAnalysis` or `~/StockAnalysis`).
   - Save the four scripts (`index_backtest.py`, `portfolio_analysis.py`, `stock_comparison.py`, `index_analysis.py`) in this folder.
   - Open the folder in VS Code: File > Open Folder > Select `StockAnalysis`.

4. **Install Libraries**:
   - Open a terminal in VS Code (Terminal > New Terminal).
   - Install required libraries:
     ```
     pip install yfinance pandas numpy pandas_ta
     ```
   - Verify: Run `pip list` to confirm `yfinance`, `pandas`, `numpy`, and `pandas_ta` are listed.

5. **Internet Connection**:
   - Ensure you’re online, as scripts fetch data from Yahoo Finance.

### Program 1: Index Backtesting Script (`index_backtest.py`)

#### What It Does
Tests a trading strategy where you buy index futures when the 21-day SMA crosses above the 50-day SMA and sell when it crosses below. Supports NIFTY (^NSEI), NYSE (^NYA), NASDAQ (^IXIC), and BSE (^BSESN). Starts with Rs. 2,00,000 (or equivalent) and trades 50 shares per lot.

#### How to Run
1. **Open the Script**:
   - In VS Code, open `index_backtest.py`.
2. **Run the Script**:
   - Open a terminal in VS Code.
   - Run for a specific index (e.g., NIFTY, 7 years):
     ```
     python index_backtest.py --index NIFTY --years 7
     ```
   - Other examples:
     - NYSE: `python index_backtest.py --index NYSE --years 5`
     - NASDAQ: `python index_backtest.py --index NASDAQ --years 7`
     - BSE: `python index_backtest.py --index BSE --years 7`
   - Default: NIFTY, 7 years if no arguments provided.
3. **What Happens**:
   - Downloads historical data for the chosen index from Yahoo Finance.
   - Calculates SMAs, generates signals, and simulates trades.
   - Saves two CSV files in `StockAnalysis`:
     - `<INDEX>_historical_data.csv` (e.g., `NIFTY_historical_data.csv`): Daily prices, SMAs, signals.
     - `<INDEX>_trades.csv` (e.g., `NIFTY_trades.csv`): Trade details.
   - Prints trade list, performance metrics, and profitability assessment.

#### Understanding the Output
- **Console Output**:
  - Example for NIFTY:
    ```
    Fetching NIFTY data from 2018-06-01 to 2025-06-01...
    Historical data with SMAs saved to 'NIFTY_historical_data.csv'

    ### List of Trades
    Entry: 2018-07-15 | Exit: 2018-09-10 | Type: Long | Entry Price: 11000.00 | Exit Price: 11500.00 | P&L: 25000.00
    ...

    ### Performance Metrics
    Total Trades: 50
    Profitable Trades: 28
    Loss Trades: 22
    Compounded Returns: 15.75%
    Total Profit/Loss: Rs. 31500.00 (15.75%)

    ### Strategy Profitability
    The strategy is profitable, generating a positive return over the backtest period.
    ```
  - **Explanation**:
    - **Trades**: Shows each trade’s entry/exit dates, type (Long/Short), prices, and profit/loss.
    - **Metrics**: Total trades, profitable/loss trades, and return on Rs. 2,00,000.
    - **Profitability**: Indicates if the strategy made money.
  - **Note**: P&L is in Rs. for simplicity; for NYSE/NASDAQ, interpret as USD or modify the script for currency.
- **CSV Files**:
  - **<INDEX>_historical_data.csv**:
    - Columns: `Date`, `Open`, `High`, `Low`, `Close`, `MA21`, `MA50`, `Signal`, `Position`.
    - Open in Excel to view data.
  - **<INDEX>_trades.csv**:
    - Columns: `Entry Date`, `Exit Date`, `Type`, `Entry Price`, `Exit Price`, `P&L`.

#### Troubleshooting
- **Error: No data found for ticker**:
  - Verify the index ticker (e.g., `^NYA` for NYSE).
  - Check internet connection and Yahoo Finance ([finance.yahoo.com](https://finance.yahoo.com)).
- **ModuleNotFoundError**:
  - Run `pip install yfinance pandas numpy`.
- **CSV files not created**:
  - Check write permissions in `StockAnalysis`.

### Program 2: Portfolio Analysis Script (`portfolio_analysis.py`)

#### What It Does
Analyzes a portfolio of stocks from a CSV file, providing health, sentiment, and news outlook (with placeholders for real data).

#### How to Prepare
1. **Create `portfolio.csv`**:
   - Use a text editor or Excel.
   - Add a `Stock` column with stock symbols (e.g., `.NS` for Indian stocks).
   - Example:
     ```
     Stock
     RELIANCE.NS
     TCS.NS
     HDFCBANK.NS
     ```
   - Save in `StockAnalysis`.
2. **Save as CSV**:
   - In Excel, choose “CSV (Comma-Separated Values)”.
   - In a text editor, name it `portfolio.csv`.

#### How to Run
1. **Open the Script**:
   - Open `portfolio_analysis.py` in VS Code.
2. **Run the Script**:
   - In a terminal:
     ```
     python portfolio_analysis.py --portfolio portfolio.csv
     ```
3. **What Happens**:
   - Reads `portfolio.csv`.
   - Simulates sentiment and expectations (random placeholders).
   - Prints a console report with health, sentiment table, review stocks, and news placeholders.

#### Understanding the Output
- **Console Output**:
  - Example:
    ```
    ### Portfolio Analysis

    #### Overall Portfolio Health
    The portfolio outlook is 'Reasonably Healthy with Areas Requiring Attention'...

    #### Stock Sentiment & Analyst Expectations Table
    | Stock        | News/Updates Sentiment | Analyst Expectations (General) |
    |--------------|------------------------|-------------------------------|
    | RELIANCE.NS  | Good                  | Positive/Buy                 |
    | TCS.NS       | Neutral               | Mixed/Hold                  |
    | HDFCBANK.NS  | Needs Review          | Mixed (Growth vs. Valuation) |

    ### Stocks Requiring Further Review
    **HDFCBANK.NS**
    - Focus Area 1: Profitability vs. Growth
      - Why review is needed: Assess if recent earnings support growth expectations.
    ...

    ### 10 Important News Pieces from the Last Week
    1. [Placeholder: Insert recent news relevant to portfolio stocks.]
       - Implication: [Placeholder: Explain impact on portfolio.]
    ...
    ```
  - **Explanation**:
    - **Health**: Portfolio condition based on sentiment.
    - **Table**: Sentiment (Good, Neutral, Needs Review) and expectations (Buy, Hold).
    - **Review**: Stocks needing attention.
    - **News**: Replace placeholders with real news from [moneycontrol.com](https://www.moneycontrol.com).

#### Troubleshooting
- **File not found**:
  - Ensure `portfolio.csv` is in `StockAnalysis`.
- **‘Stock’ column not found**:
  - Check `portfolio.csv` has a `Stock` column.
- **No output**:
  - Verify `pandas` and `numpy` are installed.

### Program 3: Stock Comparison Script (`stock_comparison.py`)

#### What It Does
Compares three stocks (e.g., `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`) in the same sector, generating an HTML report to identify the best investment.

#### How to Prepare
1. **Choose Stocks**:
   - Pick three stocks in the same sector (e.g., banking: `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`).
2. **Author Name**:
   - Decide on a name (e.g., “John Doe”).

#### How to Run
1. **Open the Script**:
   - Open `stock_comparison.py` in VS Code.
2. **Run the Script**:
   - In a terminal:
     ```
     python stock_comparison.py --stocks HDFCBANK.NS ICICIBANK.NS SBIN.NS --author "John Doe"
     ```
3. **What Happens**:
   - Fetches data from Yahoo Finance.
   - Generates `stock_comparison_report.html` in `StockAnalysis`.

#### Understanding the Output
- **HTML File**:
  - Open `stock_comparison_report.html` in a browser.
  - Example:
    ```
    Comparison Among HDFC Bank Ltd, ICICI Bank Ltd, & State Bank of India...
    Best Investment Opportunity
    | Category             | HDFC Bank Ltd | ICICI Bank Ltd | State Bank of India |
    |----------------------|---------------|----------------|---------------------|
    | Fundamental Analysis | 5             | 4              | 3                   |
    ...
    HDFC Bank Ltd is the best opportunity...
    ```
  - **Sections**: Scoring, fundamentals, financials, performance, outlook, technicals, valuation, and selection rationale.
  - **Placeholders**: Update `[Placeholder: ...]` with data from [screener.in](https://www.screener.in) or [trendlyne.com](https://www.trendlyne.com).
- **Save as PDF**:
  - In the browser, press Ctrl+P > Save as PDF.

#### Troubleshooting
- **No data for stock**:
  - Verify symbols (e.g., `HDFCBANK.NS`).
- **ModuleNotFoundError: pandas_ta**:
  - Run `pip install pandas_ta`.
- **HTML not generated**:
  - Check terminal errors and folder permissions.

### Program 4: Index Analysis Script (`index_analysis.py`)

#### What It Does
Analyzes a single index (e.g., NIFTY) using technical indicators, producing an HTML report with a buy/hold/sell recommendation.

#### How to Run
1. **Open the Script**:
   - Open `index_analysis.py` in VS Code.
2. **Run the Script**:
   - In a terminal:
     ```
     python index_analysis.py --ticker ^NSEI --author "John Doe"
     ```
   - Other indices:
     - NYSE: `--ticker ^NYA`
     - NASDAQ: `--ticker ^IXIC`
     - BSE: `--ticker ^BSESN`
   - Default: NIFTY (^NSEI) if no ticker provided.
3. **What Happens**:
   - Fetches 1 year of data from Yahoo Finance.
   - Calculates MA, RSI, MACD, and Bollinger Bands.
   - Generates `index_analysis_report.html`.

#### Understanding the Output
- **HTML File**:
  - Open `index_analysis_report.html` in a browser.
  - Example:
    ```
    Technical Analysis Report for ^NSEI by John Doe
    Summary
    Recommendation: Buy
    Technical Score: 4/5
    ...
    ```
  - **Sections**: Summary, indicators table, signals, and conclusion.
- **Save as PDF**:
  - Ctrl+P > Save as PDF.

#### Troubleshooting
- **No data for ticker**:
  - Verify ticker (e.g., `^NSEI`).
- **ModuleNotFoundError**:
  - Install `yfinance`, `pandas`, `pandas_ta`.

### General Tips
- **Start Simple**: Begin with `index_backtest.py` for NIFTY, then try others.
- **Real Data**: Replace placeholders in `portfolio_analysis.py` and `stock_comparison.py` with news from [moneycontrol.com](https://www.moneycontrol.com).
- **Errors**: Note error messages and check library installations or file paths.
- **Learn More**: Use [investopedia.com](https://www.investopedia.com) for financial terms.

### Example Workflow
1. **Setup**:
   - Install Python, VS Code, and libraries.
   - Create `StockAnalysis` folder with all scripts.
2. **Backtest NASDAQ**:
   - `python index_backtest.py --index NASDAQ --years 7`
3. **Analyze Portfolio**:
   - Create `portfolio.csv` with `RELIANCE.NS,TCS.NS,HDFCBANK.NS`.
   - `python portfolio_analysis.py --portfolio portfolio.csv`
4. **Compare Banks**:
   - `python stock_comparison.py --stocks HDFCBANK.NS ICICIBANK.NS SBIN.NS --author "John Doe"`
5. **Analyze NIFTY**:
   - `python index_analysis.py --ticker ^NSEI --author "John Doe"`

This guide should make using these scripts straightforward.

Common FAQs

    Can I use other indices/stocks?
        Yes, update tickers in index_backtest.py (add to INDEX_TICKERS) or use valid tickers in other scripts (e.g., ^GSPC for S&P 500).
    Why placeholders in reports?
        Scripts lack real-time news APIs. Manually add data from moneycontrol.com or screener.in.
    Is Yahoo Finance reliable?
        Good for testing, but less accurate than paid sources like Bloomberg. Upgrade for real-world use (see below).
    How do I automate trading?
        Integrate with brokers like Zerodha (India) or Interactive Brokers (U.S.) using APIs, but test thoroughly first.
    What if I get errors?
        Check library installations (pip install ...), file paths, and internet. Share error messages for help.
    Can I make money with these?
        Possible, but requires enhancements (below) and risk management. Markets are unpredictable.
    How do I learn financial terms?
        Use investopedia.com for terms like SMA, RSI, or P/E ratio.

Guide for Real-World Use and Profitability

To make these scripts competitive with BlackRock’s tools, enhance data quality, strategy sophistication, risk management, and execution. BlackRock uses proprietary data, advanced algorithms, and robust infrastructure. Here’s how to approach their level:
1. Improve Data Quality

    Current Issue: Reliance on yfinance can lead to missing or inaccurate data.
    Solutions:
        Paid Data Providers: Use Bloomberg Terminal, Refinitiv Eikon, or Quandl for high-quality data. Example: Subscribe to BSE’s data feed for accurate BSE data.
        APIs: Integrate Alpha Vantage or Tiingo for real-time data. Example:
        python
```
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='YOUR_API_KEY')
data, _ = ts.get_daily(symbol='^NSEI') 

```
Data Cleaning: Add checks for missing data or outliers in scripts.
python
```
        if data.isna().any().any():
            data = data.fillna(method='ffill')
```            

2. Enhance Trading Strategies

    Current Issue: Simple SMA crossover lacks robustness.
    Solutions:
        Multiple Indicators: Combine RSI, MACD, and Bollinger Bands in index_backtest.py. Example:
python
```
data['RSI'] = ta.rsi(data['Close'], length=14)
data['Signal'] = np.where((data['MA21'] > data['MA50']) & (data['RSI'] < 70), 1, 
                         np.where((data['MA21'] < data['MA50']) & (data['RSI'] > 30), -1, 0))
                         
```
Machine Learning: Use scikit-learn for predictive models. Example:
python
```
from sklearn.ensemble import RandomForestClassifier
X = data[['MA21', 'MA50', 'RSI']]
y = (data['Close'].shift(-1) > data['Close']).astype(int)
model = RandomForestClassifier()
model.fit(X[:-1], y[:-1])
```
Optimization: Optimize SMA periods using grid search.
python
```
        from itertools import product
        best_sharpe = -np.inf
        for short, long in product(range(10, 50, 5), range(50, 200, 10)):
            data['MA_short'] = calculate_sma(data, short)
            data['MA_long'] = calculate_sma(data, long)
            # Backtest and compute Sharpe ratio
```
3. Add Risk Management

    Current Issue: No stop-loss or position sizing.
    Solutions:
        Stop-Loss: Implement trailing stops in index_backtest.py.
        python
```
stop_loss = entry_price * 0.95  # 5% stop-loss
if row['Close'] < stop_loss and current_position > 0:
    # Exit trade
```    
Position Sizing: Use Kelly Criterion or fixed percentage.
python
```
lot_size = int(initial_capital * 0.02 / row['Close'])  # 2% risk
Portfolio Diversification: In portfolio_analysis.py, add correlation analysis.
```
python
```
        import seaborn as sns
        returns = pd.DataFrame({s: yf.Ticker(s).history()['Close'].pct_change() for s in portfolio['Stock']})
        corr = returns.corr()
        sns.heatmap(corr, annot=True)
```
4. Real-Time Execution

    Current Issue: Scripts are for analysis, not live trading.
    Solutions:
        Broker APIs: Integrate Zerodha Kite (India) or Interactive Brokers.
        python
```
from kiteconnect import KiteConnect
kite = KiteConnect(api_key='YOUR_KEY')
kite.set_access_token('YOUR_TOKEN')
kite.place_order(variety='regular', tradingsymbol='NIFTY', quantity=50, transaction_type='BUY')
```
Automation: Use schedule for daily runs.
python
```

        import schedule
        schedule.every().day.at("09:00").do(main)
```        

5. Match BlackRock’s Standards

    BlackRock’s Edge:
        Aladdin Platform: Integrates data, risk, and portfolio management.
        Proprietary Models: Advanced quant models and ESG integration.
        Scale: Real-time global market data and execution.
    Steps to Compete:
        Cloud Deployment: Host scripts on AWS/GCP for scalability.
        Database: Use PostgreSQL for storing historical data.
python
```
import psycopg2
conn = psycopg2.connect(dbname='stocks', user='user')
data.to_sql('prices', conn)
```  
ESG Integration: Add ESG scores in stock_comparison.py using Sustainalytics data.
Visualization: Use Plotly for interactive charts.
python
```
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Scatter(x=data.index, y=data['Close']))
        fig.show()
```        
        Team Expertise: Collaborate with quants and data scientists.


**#Conclusion**


These scripts are a strong starting point for financial analysis. With enhancements in data, strategies, risk management, and execution, they can approach professional tools’ capabilities. Focus on testing, learning, and integrating advanced features to maximize profitability while managing risks.
