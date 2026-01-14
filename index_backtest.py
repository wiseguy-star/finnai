import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import argparse

# Define index tickers
INDEX_TICKERS = {
    'NIFTY': '^NSEI',
    'NYSE': '^NYA',
    'NASDAQ': '^IXIC',
    'BSE': '^BSESN'
}

def fetch_data(ticker, start_date, end_date):
    """Fetch historical data for the given ticker."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        if data.empty:
            print(f"No data found for {ticker}.")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return data['Close'].rolling(window=window).mean()

def generate_signals(data):
    """Generate buy/sell signals based on SMA crossover."""
    data['Signal'] = 0
    data.loc[data['MA21'] > data['MA50'], 'Signal'] = 1
    data.loc[data['MA21'] < data['MA50'], 'Signal'] = -1
    data['Position'] = data['Signal'].diff()
    return data

def backtest_strategy(data, initial_capital=200000, lot_size=50):
    """Backtest the trading strategy."""
    positions = pd.DataFrame(index=data.index)
    positions['Position'] = data['Position']
    positions['Close'] = data['Close']
    
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Holdings'] = 0
    portfolio['Cash'] = initial_capital
    portfolio['Total'] = initial_capital
    
    trades = []
    current_position = 0
    entry_price = 0
    entry_date = None
    
    for date, row in data.iterrows():
        if row['Position'] == 2:  # Buy signal
            if current_position == 0:
                entry_price = row['Close']
                entry_date = date
                current_position = lot_size
                portfolio.loc[date, 'Holdings'] = current_position * entry_price
                portfolio.loc[date, 'Cash'] -= portfolio.loc[date, 'Holdings']
            elif current_position < 0:
                # Exit short position
                exit_price = row['Close']
                pnl = (entry_price - exit_price) * lot_size
                trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Short', 
                              'Entry Price': entry_price, 'Exit Price': exit_price, 'P&L': pnl})
                portfolio.loc[date, 'Cash'] += pnl
                current_position = lot_size
                entry_price = row['Close']
                entry_date = date
                portfolio.loc[date, 'Holdings'] = current_position * entry_price
                portfolio.loc[date, 'Cash'] -= portfolio.loc[date, 'Holdings']
        elif row['Position'] == -2:  # Sell signal
            if current_position == 0:
                entry_price = row['Close']
                entry_date = date
                current_position = -lot_size
                portfolio.loc[date, 'Holdings'] = current_position * entry_price
                portfolio.loc[date, 'Cash'] -= portfolio.loc[date, 'Holdings']
            elif current_position > 0:
                # Exit long position
                exit_price = row['Close']
                pnl = (exit_price - entry_price) * lot_size
                trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Long', 
                              'Entry Price': entry_price, 'Exit Price': exit_price, 'P&L': pnl})
                portfolio.loc[date, 'Cash'] += pnl
                current_position = -lot_size
                entry_price = row['Close']
                entry_date = date
                portfolio.loc[date, 'Holdings'] = current_position * entry_price
                portfolio.loc[date, 'Cash'] -= portfolio.loc[date, 'Holdings']
        portfolio.loc[date, 'Total'] = portfolio.loc[date, 'Cash'] + (current_position * row['Close'] if current_position != 0 else 0)
    
    return portfolio, trades

def main():
    parser = argparse.ArgumentParser(description='Index SMA Crossover Backtest')
    parser.add_argument('--index', type=str, default='NIFTY', choices=INDEX_TICKERS.keys(), help='Index to backtest (NIFTY, NYSE, NASDAQ, BSE)')
    parser.add_argument('--years', type=int, default=7, help='Years of historical data')
    args = parser.parse_args()
    
    index = args.index
    ticker = INDEX_TICKERS[index]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.years)
    
    # Fetch and save data
    print(f"Fetching {index} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    data = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if data is None:
        return
    
    data['MA21'] = calculate_sma(data, 21)
    data['MA50'] = calculate_sma(data, 50)
    data = data.dropna()
    data.to_csv(f'{index}_historical_data.csv')
    print(f"Historical data with SMAs saved to '{index}_historical_data.csv'")
    
    # Generate signals and backtest
    data = generate_signals(data)
    portfolio, trades = backtest_strategy(data)
    
    # Export trades to CSV
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(f'{index}_trades.csv', index=False)
    
    # Trade list and metrics
    print("\n### List of Trades")
    for trade in trades:
        print(f"Entry: {trade['Entry Date']} | Exit: {trade['Exit Date']} | Type: {trade['Type']} | "
              f"Entry Price: {trade['Entry Price']:.2f} | Exit Price: {trade['Exit Price']:.2f} | P&L: {trade['P&L']:.2f}")
    
    total_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if trade['P&L'] > 0)
    loss_trades = sum(1 for trade in trades if trade['P&L'] < 0)
    total_pnl = portfolio['Total'].iloc[-1] - 200000
    compounded_returns = (portfolio['Total'].iloc[-1] / 200000 - 1) * 100
    
    print("\n### Performance Metrics")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {profitable_trades}")
    print(f"Loss Trades: {loss_trades}")
    print(f"Compounded Returns: {compounded_returns:.2f}%")
    print(f"Total Profit/Loss: Rs. {total_pnl:.2f} ({compounded_returns:.2f}%)")
    
    print("\n### Strategy Profitability")
    if total_pnl > 0:
        print("The strategy is profitable, generating a positive return over the backtest period.")
    else:
        print("The strategy is not profitable, resulting in a net loss over the backtest period.")

if __name__ == "__main__":
    main()