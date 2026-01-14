import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import requests
from io import StringIO
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import streamlit as st
import json
import io

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# --- Constants ---
REFRESH_INTERVAL = 60  # seconds
LARGE_VALUE = 1e9  # Large finite value to replace infinities

# --- Indian Stock Exchanges Support ---
INDIAN_EXCHANGES = {
    'NSE': '.NS',
    'BSE': '.BO'
}

# --- Fetch INR to USD Exchange Rate ---
def fetch_inr_to_usd_rate():
    try:
        # Fetch USD/INR rate (how many INR per 1 USD)
        ticker = yf.Ticker("USDINR=X")
        data = ticker.history(period="1d")
        if data.empty or 'Close' not in data.columns:
            raise ValueError("Failed to fetch exchange rate data.")
        usd_to_inr = data['Close'].iloc[-1]  # e.g., 83.5 INR per 1 USD
        inr_to_usd = 1 / usd_to_inr  # e.g., 1 / 83.5 = 0.01198 USD per INR
        return inr_to_usd
    except Exception as e:
        st.warning(f"Error fetching INR to USD exchange rate: {e}. Using fallback rate of 0.012 USD/INR.")
        return 0.012  # Fallback rate as of recent data (approx)

# --- Fetch Stock Lists Dynamically ---
def fetch_stock_list(exchange):
    try:
        if exchange == 'US':
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                table = soup.find("table", {"id": "constituents"})
                if table:
                    symbols = []
                    for row in table.find("tbody").find_all("tr")[1:]:
                        symbol = row.find_all("td")[0].text.strip()
                        symbols.append(symbol)
                    if symbols:
                        return symbols[:50]
                raise Exception("Failed to parse S&P 500 table from Wikipedia.")
            except Exception as e:
                st.warning(f"Failed to fetch S&P 500 list from Wikipedia: {e}. Using fallback list.")
                default_list = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'V', 'WMT', 'DIS', 'KO',
                    'NVDA', 'META', 'BRK-B', 'UNH', 'MA', 'HD', 'PG', 'PFE', 'NFLX', 'INTC',
                    'CSCO', 'ADBE', 'CRM', 'QCOM', 'AMD', 'GILD', 'COST', 'SBUX', 'MCD', 'TGT',
                    'BA', 'MMM', 'GE', 'CAT', 'IBM', 'ORCL', 'NOW', 'LMT', 'SPGI', 'SYK',
                    'TJX', 'ZTS', 'AMT', 'PLD', 'BKNG', 'MDT', 'LOW', 'UPS', 'NKE', 'DHR'
                ]
                return default_list[:50]

        elif exchange == 'NSE':
            url = "https://raw.githubusercontent.com/Hpareek07/NSEData/master/ind_nifty500list.csv"
            response = requests.get(url)
            data = pd.read_csv(StringIO(response.text))
            symbols = data['Symbol'].tolist()
            return symbols[:50]

        elif exchange == 'BSE':
            url = "https://raw.githubusercontent.com/anirudh-hebbar/Indian-Finance/main/data/BSE500.csv"
            response = requests.get(url)
            data = pd.read_csv(StringIO(response.text))
            symbols = data['Symbol'].tolist()
            return symbols[:50]

        elif exchange == 'India':
            nse_symbols = fetch_stock_list('NSE')
            bse_symbols = fetch_stock_list('BSE')
            combined_symbols = list(set(nse_symbols + bse_symbols))
            return combined_symbols[:50]

        return []
    except Exception as e:
        st.error(f"Error fetching stock list for {exchange}: {e}. Using fallback list.")
        if exchange == 'US':
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'V', 'WMT', 'DIS', 'KO'][:50]
        elif exchange == 'NSE':
            return [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
                'SBIN', 'KOTAKBANK', 'HINDUNILVR', 'ITC', 'LT',
                'BAJFINANCE', 'MARUTI', 'ASIANPAINT', 'AXISBANK', 'TITAN',
                'SUNPHARMA', 'NESTLEIND', 'ULTRACEMCO', 'WIPRO', 'TECHM'
            ][:50]
        elif exchange == 'BSE':
            return [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
                'SBIN', 'KOTAKBANK', 'HINDUNILVR', 'ITC', 'LT',
                'BAJFINANCE', 'MARUTI', 'ASIANPAINT', 'AXISBANK', 'TITAN',
                'SUNPHARMA', 'NESTLEIND', 'ULTRACEMCO', 'WIPRO', 'TECHM',
                'HCLTECH', 'BHARTIARTL', 'M&M', 'POWERGRID', 'NTPC',
                'TATAMOTORS', 'ADANIENT', 'ADANIPORTS', 'COALINDIA', 'ONGC'
            ][:50]
        elif exchange == 'India':
            nse_fallback = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
                'SBIN', 'KOTAKBANK', 'HINDUNILVR', 'ITC', 'LT'
            ]
            bse_fallback = [
                'RELIANCE', 'TCS', "HDFCBANK", 'INFY', 'ICICIBANK',
                'SBIN', 'KOTAKBANK', 'HINDUNILVR', 'ITC', 'LT'
            ]
            combined_fallback = list(set(nse_fallback + bse_fallback))
            return combined_fallback[:50]
        return []

# --- Fetch S&P 500 Data for Threshold ---
def fetch_sp500_data():
    try:
        sp500_data = yf.download('^GSPC', period='1y')
        if 'Close' not in sp500_data.columns or sp500_data.empty:
            raise ValueError("S&P 500 data is insufficient or missing 'Close' column.")
        return sp500_data
    except Exception as e:
        st.error(f"Error fetching S&P 500 data: {e}. Using default threshold.")
        return None

# --- Extended Features for ML Model ---
def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_volatility(data):
    return data['Close'].pct_change().rolling(window=20).std()

def compute_technical_indicators(data):
    df = data.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Volatility'] = compute_volatility(df)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    df['Momentum'] = df['Close'].diff(10)
    df['Volume_Trend'] = df['Volume'].pct_change().rolling(window=5).mean()
    df['EPS_Growth'] = 0.0  # Placeholder; replace with actual data if available
    df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['INR_Volatility'] = 0.0  # Placeholder; replace if needed
    return df

# --- Fetch OHLCV data
def fetch_data(symbol, exchange='', retries=5, backoff=2):
    attempt = 0
    while attempt < retries:
        try:
            if exchange in ['NSE', 'BSE', 'India']:
                suffix = INDIAN_EXCHANGES.get('NSE') if exchange in ['NSE', 'India'] else INDIAN_EXCHANGES.get('BSE')
                stock = yf.Ticker(f"{symbol}{suffix}")
                df = stock.history(period="1y")
                if df.empty or len(df) < 20:
                    if exchange == 'India' and suffix == INDIAN_EXCHANGES['NSE']:
                        stock = yf.Ticker(f"{symbol}{INDIAN_EXCHANGES['BSE']}")
                        df = stock.history(period="1y")
                    if df.empty or len(df) < 20:
                        st.warning(f"Insufficient data for {symbol}: {len(df) if not df.empty else 0} rows")
                        return None
            else:
                stock = yf.Ticker(symbol)
                df = stock.history(period="1y")
                if df.empty or len(df) < 20:
                    st.warning(f"No data for {symbol}: {len(df) if not df.empty else 0} rows")
                    return None
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            if df.isna().any().any():
                st.warning(f"NaN values in data for {symbol} after filling")
                return None
            return df
        except Exception as e:
            attempt += 1
            if attempt == retries:
                st.error(f"Failed to fetch data for {symbol} after {retries} attempts: {e}")
                return None
            time.sleep(backoff * attempt)

def prepare_data(data, symbol, exchange='', threshold=0.01):
    try:
        st.write(f"Preparing data for {symbol} ({exchange})...")
        data = compute_technical_indicators(data)
        
        # Adjust threshold for US stocks using S&P 500 data
        if exchange == 'US':
            sp500_data = fetch_sp500_data()
            if sp500_data is not None:
                sp500_returns = sp500_data['Close'].pct_change().dropna()
                threshold = sp500_returns.std() if not sp500_returns.empty else 0.01
            else:
                threshold = 0.01
        
        data.dropna(inplace=True)
        if data.empty:
            st.error(f"Data preparation failed for {symbol}: Not enough valid data after dropping NaNs.")
            return None, None, None, None, None
        
        returns = data['Close'].shift(-1) / data['Close'] - 1
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()
        
        if returns.empty:
            st.error(f"Data preparation failed for {symbol}: No valid returns data after cleaning.")
            return None, None, None, None, None
        
        # Use finite values for bins to avoid numpy warning
        bins = [-LARGE_VALUE, -threshold, threshold, LARGE_VALUE]
        labels = ['Sell', 'Hold', 'Buy']
        # Explicitly convert bins to a numpy array with float64 dtype
        bins = np.array(bins, dtype=np.float64)
        st.write(f"Bins for {symbol}: {bins}")  # Debug statement to confirm bins
        
        # Apply pd.cut with the modified bins
        data['Trend'] = pd.cut(returns, bins=bins, labels=labels, include_lowest=True)
        data.dropna(subset=['Trend'], inplace=True)
        
        if data.empty:
            st.error(f"Data preparation failed for {symbol}: No data after binning trends.")
            return None, None, None, None, None
        
        st.write(f"Trend distribution for {symbol}: {data['Trend'].value_counts().to_dict()}")
        
        features = ['SMA_20', 'RSI', 'Volatility', 'MACD', 'Momentum', 
                    'Bollinger_Upper', 'Bollinger_Lower', 'Volume_Trend',
                    'EPS_Growth', 'Volume_Spike', 'INR_Volatility']
        if not all(col in data.columns for col in features):
            missing_cols = [col for col in features if col not in data.columns]
            st.error(f"Missing columns in data for {symbol}: {missing_cols}")
            return None, None, None, None, None
        
        X = data[features]
        y = data['Trend']
        
        if X.isnull().any().any() or y.isnull().any():
            st.error(f"Data preparation failed for {symbol}: NaN values in features or target.")
            return None, None, None, None, None
        
        # Encode the target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.to_numpy())  # Convert to numpy array explicitly
        st.write(f"Encoded labels for {symbol}: {list(zip(labels, label_encoder.transform(labels)))}")
        
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        return train_test_split(X_scaled, y_encoded, test_size=0.2, shuffle=False), scaler, features, imputer, label_encoder
    except Exception as e:
        st.error(f"Error in prepare_data for {symbol}: {str(e)}")
        return None, None, None, None, None

def train_model(data, symbol, exchange=''):
    try:
        st.write(f"Training model for {symbol} ({exchange})...")
        splits, scaler, features, imputer, label_encoder = prepare_data(data, symbol, exchange)
        if splits is None:
            st.error(f"Data preparation failed for {symbol}: Not enough valid data to train the model.")
            return None, None, None, None
        X_train, _, y_train, _ = splits
        if len(np.unique(y_train)) < 2:
            st.error(f"Insufficient class variety in training data for {symbol}: {np.unique(y_train)}")
            return None, None, None, None
        # Ensure y_train is a numpy array
        if not isinstance(y_train, np.ndarray):
            st.error(f"y_train is not a numpy array for {symbol}. Type: {type(y_train)}")
            return None, None, None, None
        st.write(f"y_train shape for {symbol}: {y_train.shape}, unique values: {np.unique(y_train)}")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        st.write(f"Model trained successfully for {symbol}.")
        return model, scaler, imputer, label_encoder
    except Exception as e:
        st.error(f"Model training failed for {symbol}: {str(e)}")
        return None, None, None, None

def predict_trend(model, scaler, imputer, label_encoder, latest_data, features, symbol):
    try:
        st.write(f"Predicting trend for {symbol}...")
        latest_data = compute_technical_indicators(latest_data.tail(20))
        if len(latest_data) < 1:
            st.error(f"Not enough data to predict trend for {symbol}.")
            return None
        if not all(col in latest_data.columns for col in features):
            missing_cols = [col for col in features if col not in latest_data.columns]
            st.error(f"Missing columns in latest data for {symbol}: {missing_cols}")
            return None
        X = latest_data[features].iloc[-1:].values
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        # Map prediction back to labels
        trend_label = label_encoder.inverse_transform([prediction])[0]
        
        # Create confidence dictionary
        classes = label_encoder.classes_
        confidence = {cls: proba[i] for i, cls in enumerate(classes)}
        
        st.write(f"Prediction for {symbol}: {trend_label}, Confidence: {confidence}")
        return {
            "trend": trend_label,
            "confidence": confidence
        }
    except Exception as e:
        st.error(f"Error predicting trend for {symbol}: {str(e)}")
        return None

# --- Asset Info + Market Cap Classification ---
def get_asset_info(symbol, exchange='US'):
    try:
        if exchange in ['NSE', 'BSE', 'India']:
            suffix = INDIAN_EXCHANGES.get('NSE') if exchange in ['NSE', 'India'] else INDIAN_EXCHANGES.get('BSE')
            stock = yf.Ticker(f"{symbol}{suffix}")
            info = stock.info
            if not info or 'symbol' not in info:
                if exchange == 'India' and suffix == INDIAN_EXCHANGES['NSE']:
                    stock = yf.Ticker(f"{symbol}{INDIAN_EXCHANGES['BSE']}")
                    info = stock.info
            currency = info.get('currency', 'INR')
        else:
            stock = yf.Ticker(symbol)
            info = stock.info
            currency = info.get('currency', 'USD')
        return {
            "sector": info.get('sector', "N/A"),
            "exchange": info.get('exchange', exchange),
            "name": info.get('longName', "N/A"),
            "marketCap": info.get('marketCap', 0),
            "currency": currency,
            "pe_ratio": info.get('trailingPE', 0),
            "price_change_30d": info.get('previousClose', 0) / info.get('open', 1) - 1 if info.get('open', 1) != 0 else 0,
            "returnOnEquity": info.get('returnOnEquity', 0),
            "debtToEquity": info.get('debtToEquity', 0)
        }
    except Exception as e:
        st.warning(f"Failed to fetch asset info for {symbol}: {e}. Using default values.")
        return {
            "sector": "N/A",
            "exchange": exchange,
            "name": "N/A",
            "marketCap": 0,
            "currency": "USD" if exchange == 'US' else "INR",
            "pe_ratio": 0,
            "price_change_30d": 0,
            "returnOnEquity": 0,
            "debtToEquity": 0
        }

def classify_market_cap(marketCap):
    if marketCap == 0: return "Unknown"
    if marketCap > 200e9: return "Mega Cap"
    elif marketCap > 10e9: return "Large Cap"
    elif marketCap > 2e9: return "Mid Cap"
    elif marketCap > 300e6: return "Small Cap"
    else: return "Micro Cap"

def compute_risk_score(rsi, volatility):
    try:
        risk_rsi = abs(rsi - 50) / 50
        risk_vol = volatility * 100
        return round((risk_rsi + risk_vol) * 50, 2)
    except:
        return 0.0

# --- Sentiment Analysis Using Yahoo Finance News ---
def fetch_news_sentiment(symbol, exchange='US'):
    try:
        base_symbol = symbol
        if exchange in ['NSE', 'BSE', 'India']:
            base_symbol = symbol
        url = f"https://finance.yahoo.com/quote/{base_symbol}/news/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = soup.find_all("h3", class_="Mb(5px)")
        if not headlines:
            return 0.0
        
        headline_texts = [headline.get_text(strip=True) for headline in headlines[:5]]
        if not headline_texts:
            return 0.0
        
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = []
        for title in headline_texts:
            score = sia.polarity_scores(title)['compound']
            sentiment_scores.append(score)
        
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    
    except Exception as e:
        st.error(f"Error fetching news sentiment for {symbol}: {e}")
        return 0.0

# --- Portfolio Management ---
def initialize_portfolio():
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now() - timedelta(seconds=REFRESH_INTERVAL + 1)
    if 'previous_total_value' not in st.session_state:
        st.session_state.previous_total_value = 0
    if 'json_loaded' not in st.session_state:
        st.session_state.json_loaded = False  # Flag to track if JSON has been loaded

def save_portfolio():
    pass  # In a browser environment, we provide a download button

def load_portfolio(uploaded_file):
    if uploaded_file is not None:
        try:
            portfolio = json.load(uploaded_file)
            if not isinstance(portfolio, list):
                raise ValueError("Portfolio file is not a valid list")
            return portfolio
        except Exception as e:
            st.error(f"Error loading portfolio: {e}. Please ensure the file is a valid JSON list.")
            return []
    return []

# --- Smart AI-based Recommendations ---
def recommend_assets(top_n=10, exchange='US'):
    symbols = fetch_stock_list(exchange)
    if not symbols:
        st.error(f"No symbols fetched for {exchange}.")
        return [], "No symbols available."

    symbols = symbols[:50]  # Process 50 assets as requested
    recommendations = []
    total_processed = 0
    failed_data_fetch = 0
    failed_model = 0
    failed_trend = 0
    failed_buy_filter = 0
    failed_buy_confidence = 0
    failed_momentum = 0
    failed_pe_ratio = 0
    failed_stocks = []

    for i, symbol in enumerate(symbols):
        try:
            st.write(f"Processing {symbol} ({i+1}/{len(symbols)}) for {exchange}...")
            total_processed += 1
            data = fetch_data(symbol, exchange)
            if data is None:
                failed_data_fetch += 1
                failed_stocks.append((symbol, "Data fetch failed"))
                continue

            close_price = data['Close'].iloc[-1]
            model, scaler, imputer, label_encoder = train_model(data, symbol, exchange)
            if model is None:
                failed_model += 1
                failed_stocks.append((symbol, "Model training failed"))
                continue
            _, _, features, _, _ = prepare_data(data, symbol, exchange)
            if features is None:
                failed_model += 1
                failed_stocks.append((symbol, "Data preparation failed"))
                continue
            trend_result = predict_trend(model, scaler, imputer, label_encoder, data, features, symbol)
            if trend_result is None:
                failed_trend += 1
                failed_stocks.append((symbol, "Trend prediction failed"))
                continue

            asset_info = get_asset_info(symbol, exchange)
            
            momentum_6m = data['Close'].pct_change(periods=126).iloc[-1] if len(data) >= 126 else 0
            momentum_12m = data['Close'].pct_change(periods=252).iloc[-1] if len(data) >= 252 else 0
            pe_ratio = asset_info["pe_ratio"]

            if trend_result["trend"] != "Buy":
                failed_buy_filter += 1
                failed_stocks.append((symbol, f"Trend is {trend_result['trend']}, not Buy"))
                continue
            buy_confidence = trend_result["confidence"].get("Buy", 0)
            if buy_confidence <= 0.5:
                failed_buy_confidence += 1
                failed_stocks.append((symbol, f"Buy confidence {buy_confidence*100:.2f}% <= 50%"))
                continue
            if not (momentum_6m > -0.10 or momentum_12m > -0.10):  # Allow up to -10% momentum
                failed_momentum += 1
                failed_stocks.append((symbol, f"Momentum_6M {momentum_6m*100:.2f}%, Momentum_12M {momentum_12m*100:.2f}%"))
                continue
            if not (0 < pe_ratio < 50):
                failed_pe_ratio += 1
                failed_stocks.append((symbol, f"PE ratio {pe_ratio:.2f} not in range 0–50"))
                continue

            rsi = compute_rsi(data['Close']).iloc[-1]
            volatility = data['Close'].pct_change().std()
            risk_score = compute_risk_score(rsi, volatility)

            momentum_score = (momentum_6m + momentum_12m) / 2 if momentum_6m and momentum_12m else 0

            roe = asset_info.get('returnOnEquity', 0) or 0
            debt_to_equity = asset_info.get('debtToEquity', 0) or 0
            quality_score = (roe / 100) - (debt_to_equity / 100) if roe and debt_to_equity else 0

            sentiment_score = fetch_news_sentiment(symbol, exchange)

            pe_score = 1 / (1 + pe_ratio) if pe_ratio > 0 else 0.5

            sector = asset_info["sector"]
            sector_boost = 1.2 if sector in ["Technology", "Information Technology", "Energy"] else 1.0

            score = (
                trend_result["confidence"].get("Buy", 0) * 0.3 +
                (1 - trend_result["confidence"].get("Sell", 0)) * 0.15 +
                momentum_score * 0.2 +
                quality_score * 0.15 +
                sentiment_score * 0.1 +
                pe_score * 0.1 +
                (1 - risk_score/100) * 0.1
            ) * sector_boost

            recommendations.append({
                "Symbol": symbol,
                "Price": round(close_price, 2),
                "Currency": asset_info["currency"],
                "Trend": trend_result["trend"],
                "Buy %": round(trend_result["confidence"].get("Buy", 0) * 100, 2),
                "Hold %": round(trend_result["confidence"].get("Hold", 0) * 100, 2),
                "Sell %": round(trend_result["confidence"].get("Sell", 0) * 100, 2),
                "RiskScore": risk_score,
                "Sector": asset_info["sector"],
                "MarketCap": classify_market_cap(asset_info["marketCap"]),
                "Exchange": asset_info["exchange"],
                "Score": round(score * 100, 2),
                "PE_Ratio": round(pe_ratio, 2),
                "Momentum_6M": round(momentum_6m * 100, 2),
                "Momentum_12M": round(momentum_12m * 100, 2),
                "Sentiment": round(sentiment_score, 2),
                "Quality_Score": round(quality_score, 2)
            })

            time.sleep(0.5)

        except Exception as e:
            failed_stocks.append((symbol, f"General error: {str(e)}"))
            continue

    debug_info = (
        f"Debugging Information for {exchange}\n"
        f"- Total stocks processed: {total_processed}\n"
        f"- Failed data fetch: {failed_data_fetch}\n"
        f"- Failed model training: {failed_model}\n"
        f"- Failed trend prediction: {failed_trend}\n"
        f"- Failed 'Buy' filter: {failed_buy_filter}\n"
        f"- Failed Buy confidence (>50%): {failed_buy_confidence}\n"
        f"- Failed momentum filter: {failed_momentum}\n"
        f"- Failed PE ratio filter (0 < PE < 50): {failed_pe_ratio}\n"
        f"- Stocks passing all filters: {len(recommendations)}\n"
        f"- Failed stocks and reasons:\n"
    )
    for symbol, reason in failed_stocks:
        debug_info += f"  - {symbol}: {reason}\n"

    if not recommendations:
        st.error(f"No recommendations could be generated for {exchange}.")
        return [], debug_info

    recommendations.sort(key=lambda x: x["Score"], reverse=True)
    return recommendations[:top_n], debug_info

# --- Analyze Asset Function ---
def analyze_asset(symbol, exchange):
    try:
        data = fetch_data(symbol, exchange)
        if data is None or len(data) < 20:
            return "Insufficient data for analysis. At least 20 days of data are required.", None, None

        data = compute_technical_indicators(data)
        if 'SMA_20' not in data.columns or 'RSI' not in data.columns:
            return "Failed to compute technical indicators.", None, None

        data.dropna(inplace=True)
        if data.empty:
            return "No valid data remains after processing.", None, None

        close_price = data['Close'].iloc[-1]
        model, scaler, imputer, label_encoder = train_model(data, symbol, exchange)
        if model is None:
            return "Failed to train model.", None, None
        _, _, features, _, _ = prepare_data(data, symbol, exchange)
        if features is None:
            return "Failed to prepare data for prediction.", None, None
        trend_result = predict_trend(model, scaler, imputer, label_encoder, data, features, symbol)
        if trend_result is None:
            return "Failed to predict trend.", None, None
        asset_info = get_asset_info(symbol, exchange)

        rsi = data['RSI'].iloc[-1]
        volatility = data['Volatility'].iloc[-1]
        risk_score = compute_risk_score(rsi, volatility)

        result = (
            f"Latest Price: {asset_info.get('currency', 'N/A')} {close_price:.2f}\n\n"
            f"AI Recommendation: {trend_result['trend']}\n"
            f"Confidence: Buy={round(trend_result['confidence'].get('Buy', 0)*100, 2)}%, "
            f"Hold={round(trend_result['confidence'].get('Hold', 0)*100, 2)}%, "
            f"Sell={round(trend_result['confidence'].get('Sell', 0)*100, 2)}%\n"
            f"Risk Score: {risk_score:.2f}\n"
            f"Sector: {asset_info['sector']}, Exchange: {asset_info['exchange']}\n"
            f"Market Cap: {classify_market_cap(asset_info['marketCap'])}"
        )

        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(data.index, data['Close'], label='Close Price')
        ax1.plot(data.index, data['SMA_20'], label='SMA 20', linestyle="--")
        ax1.set_title('Price and SMA')
        ax1.legend()

        fig2, ax2 = plt.subplots(figsize=(6, 2))
        ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
        ax2.axhline(70, color='red', linestyle="--", alpha=0.5)
        ax2.axhline(30, color='green', linestyle="--", alpha=0.5)
        ax2.set_title('RSI')
        ax2.legend()

        return result, fig1, fig2

    except Exception as e:
        return f"Error analyzing asset: {str(e)}", None, None

# --- Add to Portfolio Function ---
def add_to_portfolio(symbol, quantity, exchange):
    try:
        if not symbol:
            return "Symbol cannot be empty."
        quantity = int(quantity)
        if quantity < 1:
            return "Quantity must be at least 1."

        data = fetch_data(symbol, exchange)
        if data is None or len(data) < 20:
            return "No data found for this symbol."

        price = data['Close'].iloc[-1]
        value = price * quantity
        model, scaler, imputer, label_encoder = train_model(data, symbol, exchange)
        if model is None:
            return "Model training failed."
        _, _, features, _, _ = prepare_data(data, symbol, exchange)
        if features is None:
            return "Failed to prepare data for prediction."
        trend_result = predict_trend(model, scaler, imputer, label_encoder, data, features, symbol)
        if trend_result is None:
            return "Trend prediction failed."
        asset_info = get_asset_info(symbol, exchange)

        rsi = compute_rsi(data['Close']).iloc[-1]
        volatility = data['Close'].pct_change().std()
        risk_score = compute_risk_score(rsi, volatility)

        st.session_state.portfolio.append({
            "Symbol": symbol,
            "Price": round(price, 2),
            "Qty": quantity,
            "PurchasePrice": round(price, 2),
            "CurrentPrice": round(price, 2),
            "PurchaseValue": round(value, 2),
            "CurrentValue": round(value, 2),
            "Profit/Loss": 0.0,
            "Profit/Loss %": 0.0,
            "Trend": trend_result["trend"],
            "confidence": trend_result["confidence"],
            "RiskScore": risk_score,
            "Sector": asset_info["sector"],
            "MarketCap": classify_market_cap(asset_info["marketCap"]),
            "Exchange": asset_info["exchange"],
            "Currency": asset_info["currency"]
        })
        save_portfolio()
        return f"{symbol} added to portfolio!"

    except Exception as e:
        return f"Error adding asset: {str(e)}"

# --- Remove from Portfolio Function ---
def remove_from_portfolio(symbol):
    st.session_state.portfolio = [asset for asset in st.session_state.portfolio if asset['Symbol'] != symbol]
    save_portfolio()
    return f"{symbol} removed from portfolio!"

# --- Clear Portfolio Function ---
def clear_portfolio():
    st.session_state.portfolio = []
    save_portfolio()
    st.session_state.json_loaded = False  # Reset the flag when clearing portfolio
    return "Portfolio cleared!"

# --- Auto-Refresh Portfolio Function ---
def auto_refresh_portfolio():
    if not st.session_state.portfolio or (datetime.now() - st.session_state.last_refresh).total_seconds() < REFRESH_INTERVAL:
        return

    updated_portfolio = []
    for asset in st.session_state.portfolio:
        try:
            data = fetch_data(asset["Symbol"], asset["Exchange"])
            if data is None:
                updated_portfolio.append(asset)
                continue

            latest_price = data['Close'].iloc[-1]
            updated_value = latest_price * asset["Qty"]
            model, scaler, imputer, label_encoder = train_model(data, asset["Symbol"], asset["Exchange"])
            if model is None:
                updated_portfolio.append(asset)
                continue
            _, _, features, _, _ = prepare_data(data, asset["Symbol"], asset["Exchange"])
            if features is None:
                updated_portfolio.append(asset)
                continue
            trend_result = predict_trend(model, scaler, imputer, label_encoder, data, features, asset["Symbol"])
            if trend_result is None:
                updated_portfolio.append(asset)
                continue
            
            rsi = compute_rsi(data['Close']).iloc[-1]
            volatility = data['Close'].pct_change().std()
            risk_score = compute_risk_score(rsi, volatility)

            profit_loss = (latest_price - asset["PurchasePrice"]) * asset["Qty"]
            profit_loss_pct = ((latest_price - asset["PurchasePrice"]) / asset["PurchasePrice"] * 100) if asset["PurchasePrice"] != 0 else 0

            updated_asset = {
                "Symbol": asset["Symbol"],
                "Price": asset["Price"],
                "Qty": asset["Qty"],
                "PurchasePrice": asset["PurchasePrice"],
                "CurrentPrice": round(latest_price, 2),
                "PurchaseValue": asset["PurchaseValue"],
                "CurrentValue": round(updated_value, 2),
                "Profit/Loss": round(profit_loss, 2),
                "Profit/Loss %": round(profit_loss_pct, 2),
                "Trend": trend_result["trend"],
                "confidence": trend_result["confidence"],
                "RiskScore": risk_score,
                "Sector": asset["Sector"],
                "MarketCap": asset["MarketCap"],
                "Exchange": asset["Exchange"],
                "Currency": asset["Currency"]
            }
            updated_portfolio.append(updated_asset)
        except Exception as e:
            st.error(f"Error updating asset {asset['Symbol']}: {e}")
            updated_portfolio.append(asset)

    st.session_state.portfolio = updated_portfolio
    save_portfolio()
    st.session_state.last_refresh = datetime.now()

# --- Streamlit App ---
st.title("Financial AI Assistant")

# Initialize session state
initialize_portfolio()

# Sidebar for navigation and portfolio file management
st.sidebar.header("Portfolio Management")
st.sidebar.warning("**Important**: Portfolio resets on refresh. Download `portfolio.json` to save your portfolio before refreshing or closing the app!")

# File uploader for portfolio
uploaded_file = st.sidebar.file_uploader("Upload portfolio.json", type="json", key="portfolio_uploader")

# Load portfolio only if it hasn't been loaded yet
if uploaded_file and not st.session_state.json_loaded:
    loaded_portfolio = load_portfolio(uploaded_file)
    # Append loaded portfolio to existing portfolio (instead of overwriting)
    current_portfolio = st.session_state.portfolio
    # Avoid duplicates by checking symbols
    existing_symbols = {asset["Symbol"] for asset in current_portfolio}
    for asset in loaded_portfolio:
        if asset["Symbol"] not in existing_symbols:
            current_portfolio.append(asset)
            existing_symbols.add(asset["Symbol"])
    st.session_state.portfolio = current_portfolio
    st.session_state.json_loaded = True  # Set flag to prevent reloading
    st.sidebar.success("Portfolio loaded successfully!")

# Option to clear the uploaded file
if st.session_state.json_loaded:
    if st.sidebar.button("Clear Uploaded Portfolio"):
        st.session_state.json_loaded = False
        # Reset the file uploader
        st.session_state.portfolio_uploader = None
        st.sidebar.success("Uploaded portfolio cleared. You can now upload a new file.")

# Download current portfolio
if st.session_state.portfolio:
    portfolio_json = json.dumps(st.session_state.portfolio, indent=4)
    st.sidebar.download_button(
        label="Download portfolio.json",
        data=portfolio_json,
        file_name="portfolio.json",
        mime="application/json"
    )

page = st.sidebar.selectbox("Select Page", ["Analyze Asset", "My Portfolio", "Recommended Portfolio"])

if page == "Analyze Asset":
    st.header("Analyze Asset")
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    with col2:
        exchange = st.selectbox("Select Exchange", ['NASDAQ', 'NYSE', 'NSE', 'BSE'])
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            result, fig1, fig2 = analyze_asset(symbol, exchange)
            st.write(result)
            if fig1:
                st.pyplot(fig1)
            if fig2:
                st.pyplot(fig2)

elif page == "My Portfolio":
    st.header("My Portfolio")
    
    # Auto-refresh portfolio
    auto_refresh_portfolio()

    # Manual refresh button
    if st.button("Refresh Portfolio"):
        auto_refresh_portfolio()
        st.rerun()

    # Display portfolio
    portfolio_df = pd.DataFrame(st.session_state.portfolio)
    if portfolio_df.empty:
        st.write("Your portfolio is empty.")
    else:
        # Fetch INR to USD exchange rate
        inr_to_usd = fetch_inr_to_usd_rate()
        st.write(f"Using INR to USD exchange rate: 1 INR = {inr_to_usd:.5f} USD")

        # Convert values to USD where necessary
        total_purchase_value_usd = 0.0
        total_current_value_usd = 0.0
        for _, row in portfolio_df.iterrows():
            if row['Currency'] == 'INR':
                total_purchase_value_usd += row['PurchaseValue'] * inr_to_usd
                total_current_value_usd += row['CurrentValue'] * inr_to_usd
            else:  # Assume USD if not INR
                total_purchase_value_usd += row['PurchaseValue']
                total_current_value_usd += row['CurrentValue']

        total_pl_usd = total_current_value_usd - total_purchase_value_usd
        total_pl_pct = (total_pl_usd / total_purchase_value_usd * 100) if total_purchase_value_usd != 0 else 0
        
        summary = (
            f"Total Purchase Value: USD {total_purchase_value_usd:.2f}\n"
            f"Total Current Value: USD {total_current_value_usd:.2f}\n"
            f"Total Profit/Loss: USD {total_pl_usd:.2f} ({total_pl_pct:.2f}%)\n"
        )
        if st.session_state.previous_total_value != 0:
            perf_change = ((total_current_value_usd - st.session_state.previous_total_value) / st.session_state.previous_total_value) * 100
            summary += f"Portfolio Performance (since last refresh): {perf_change:.2f}%\n"
        st.session_state.previous_total_value = total_current_value_usd

        st.write(summary)
        # Display individual assets with their original currencies
        st.dataframe(portfolio_df[['Symbol', 'Price', 'Qty', 'CurrentValue', 'Profit/Loss', 'Profit/Loss %', 'Trend', 'Sector', 'Exchange', 'Currency']])

    # Add new asset
    st.subheader("Add New Asset to Portfolio")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_symbol = st.text_input("Symbol (e.g., AAPL or RELIANCE)")
    with col2:
        new_quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
    with col3:
        new_exchange = st.selectbox("Exchange", ['NASDAQ', 'NYSE', 'NSE', 'BSE'], key="add_exchange")
    if st.button("Add to Portfolio"):
        result = add_to_portfolio(new_symbol, new_quantity, new_exchange)
        st.write(result)

    # Remove asset
    st.subheader("Remove Asset from Portfolio")
    if st.session_state.portfolio:
        remove_symbol = st.selectbox("Select Symbol to Remove", [asset['Symbol'] for asset in st.session_state.portfolio])
        if st.button("Remove Asset"):
            result = remove_from_portfolio(remove_symbol)
            st.write(result)
    else:
        st.write("No assets to remove.")

    # Clear portfolio
    st.subheader("Clear Portfolio")
    if st.button("Clear Portfolio"):
        result = clear_portfolio()
        st.write(result)

elif page == "Recommended Portfolio":
    st.header("Recommended Portfolio")
    exchange = st.selectbox("Select Market", ['US', 'NSE', 'BSE', 'India'])
    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations (this may take a few minutes)..."):
            recommendations, debug_info = recommend_assets(top_n=10, exchange=exchange)
            st.session_state['recommendations'] = recommendations
            st.session_state['debug_info'] = debug_info

    if 'recommendations' in st.session_state and st.session_state['recommendations']:
        rec_df = pd.DataFrame(st.session_state['recommendations'])
        st.dataframe(rec_df[['Symbol', 'Price', 'Buy %', 'Hold %', 'Sell %', 'PE_Ratio', 'RiskScore', 'Momentum_6M', 'Momentum_12M', 'Sentiment', 'Quality_Score', 'Sector', 'Score']])
        
        st.subheader("Debug Information")
        st.write(st.session_state['debug_info'])

        st.subheader("Add Recommended Stock to Portfolio")
        selected_symbol = st.selectbox("Select Stock to Add", rec_df['Symbol'].tolist())
        quantity = st.number_input("Quantity to Add", min_value=1, value=1, step=1)
        if st.button("Add Selected to Portfolio"):
            asset = next((asset for asset in st.session_state['recommendations'] if asset["Symbol"] == selected_symbol), None)
            if not asset:
                st.error("Selected stock not found in recommendations.")
            else:
                st.session_state.portfolio.append({
                    "Symbol": asset["Symbol"],
                    "Price": asset["Price"],
                    "Qty": quantity,
                    "PurchasePrice": asset["Price"],
                    "CurrentPrice": asset["Price"],
                    "PurchaseValue": asset["Price"] * quantity,
                    "CurrentValue": asset["Price"] * quantity,
                    "Profit/Loss": 0.0,
                    "Profit/Loss %": 0.0,
                    "Trend": asset["Trend"],
                    "confidence": {
                        "Buy": asset["Buy %"] / 100,
                        "Hold": asset["Hold %"] / 100,
                        "Sell": asset["Sell %"] / 100
                    },
                    "RiskScore": asset["RiskScore"],
                    "Sector": asset["Sector"],
                    "MarketCap": asset["MarketCap"],
                    "Exchange": asset["Exchange"],
                    "Currency": asset["Currency"]
                })
                save_portfolio()
                st.write(f"{asset['Symbol']} added to portfolio!")
    else:
        st.write("No recommendations available. Click 'Generate Recommendations' to see top stocks.")

if __name__ == '__main__':
    st.write(f"App running on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")