import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

# Define index tickers
index_tickers = {
    'NASDAQ': '^IXIC',
    'NYSE': '^NYA',
    'NIFTY': '^NSEI',
    'BSE': '^BSESN'
}

def dataset_introduction(data, index_name):
    print(f"1. Introduction to the Dataset - {index_name}")
    print("-" * 50)
    print(f"Dataset: {index_name} Index")
    print(f"Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total Trading Days: {len(data)}")
    print("Features:")
    print("- Date: Trading date")
    print("- Open: Opening price")
    print("- High: Highest price of the day")
    print("- Low: Lowest price of the day")
    print("- Close: Closing price")
    print("- Adj Close: Adjusted closing price")
    print("- Volume: Trading volume")
    print("- Year: Year of the trading day")
    print("- Month: Month of the trading day (1-12)")
    print("- MonthName: Short month name (e.g., Jan)")
    print("\n")

def summary_statistics(data, index_name):
    print(f"2. Summary Statistics - {index_name}")
    print("-" * 50)
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    total_return = (end_price - start_price) / start_price * 100
    years = (data.index[-1] - data.index[0]).days / 365.25
    cagr = ((end_price / start_price) ** (1 / years) - 1) * 100 if years > 0 else 0
    avg_price = data['Close'].mean()
    high_price = data['Close'].max()
    low_price = data['Close'].min()
    daily_returns = data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    print(f"Start Price: {start_price:.2f}")
    print(f"End Price: {end_price:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"CAGR: {cagr:.2f}%")
    print(f"All-time High: {high_price:.2f}")
    print(f"All-time Low: {low_price:.2f}")
    print(f"Average Price: {avg_price:.2f}")
    print(f"Volatility (Annualized): {volatility:.2f}%")
    print("\n")

def trend_analysis(data, index_name):
    print(f"3. Trend Analysis - {index_name}")
    print("-" * 50)
    print("Identifying key trends over the period.")
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label=index_name, color='blue')
    plt.title(f'{index_name} Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{index_name.lower()}_trend.png')
    plt.close()
    print(f"Trend chart saved as '{index_name.lower()}_trend.png'")
    print("\n")

def volatility_analysis(data, index_name):
    print(f"4. Volatility Analysis - {index_name}")
    print("-" * 50)
    daily_returns = data['Close'].pct_change().dropna()
    rolling_vol = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
    print("Calculating 30-day rolling volatility (annualized).")
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_vol.index, rolling_vol, label='30-Day Rolling Volatility', color='red')
    plt.title(f'{index_name} Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{index_name.lower()}_volatility.png')
    plt.close()
    print(f"Volatility chart saved as '{index_name.lower()}_volatility.png'")
    print("\n")

def moving_average_analysis(data, index_name):
    print(f"5. Moving Average Analysis - {index_name}")
    print("-" * 50)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    print("Computed 20-day, 50-day, and 200-day moving averages.")
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label=index_name, color='black')
    plt.plot(data.index, data['MA20'], label='20-Day MA', color='blue')
    plt.plot(data.index, data['MA50'], label='50-Day MA', color='orange')
    plt.plot(data.index, data['MA200'], label='200-Day MA', color='red')
    plt.title(f'{index_name} with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{index_name.lower()}_moving_averages.png')
    plt.close()
    print(f"Moving averages chart saved as '{index_name.lower()}_moving_averages.png'")
    print("\n")

def comparative_analysis(data, index_name):
    print(f"6. Comparative Analysis - {index_name}")
    print("-" * 50)
    yearly_data = data.groupby('Year')['Close'].agg(['first', 'last', 'max', 'min'])
    yearly_data['Return'] = (yearly_data['last'] - yearly_data['first']) / yearly_data['first'] * 100
    print("Yearly Performance:")
    print(yearly_data[['first', 'last', 'Return', 'max', 'min']].round(2))
    plt.figure(figsize=(10, 6))
    plt.bar(yearly_data.index, yearly_data['Return'], color='blue')
    plt.title(f'{index_name} Yearly Returns')
    plt.xlabel('Year')
    plt.ylabel('Return (%)')
    plt.grid(True)
    plt.savefig(f'{index_name.lower()}_yearly_returns.png')
    plt.close()
    print(f"Yearly returns chart saved as '{index_name.lower()}_yearly_returns.png'")
    monthly_returns = data.groupby(['Year', 'Month'])['Close'].agg(['first', 'last'])
    monthly_returns['Return'] = (monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first'] * 100
    heatmap_data = monthly_returns['Return'].unstack()
    heatmap_data = heatmap_data.reindex(columns=range(1, 13))
    heatmap_data.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='RdYlGn', annot=True, fmt='.1f', center=0)
    plt.title(f'{index_name} Monthly Returns (%)')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.savefig(f'{index_name.lower()}_monthly_heatmap.png')
    plt.close()
    print(f"Monthly returns heatmap saved as '{index_name.lower()}_monthly_heatmap.png'")
    print("\n")

def seasonal_patterns(data, index_name):
    print(f"7. Seasonal Patterns - {index_name}")
    print("-" * 50)
    monthly_returns = data['Close'].resample('M').last().pct_change() * 100
    monthly_returns = monthly_returns.to_frame('Return')
    monthly_returns['Month'] = monthly_returns.index.month
    monthly_avg = monthly_returns.groupby('Month')['Return'].mean()
    win_rate = (monthly_returns.groupby('Month')['Return'] > 0).mean() * 100
    seasonal_df = pd.DataFrame({'Avg Return (%)': monthly_avg, 'Win Rate (%)': win_rate})
    seasonal_df.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print("Average Monthly Returns and Win Rates:")
    print(seasonal_df.round(2))
    print("Note: Patterns may vary by market. Common trends include year-end rallies and January effects.")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(seasonal_df.index, seasonal_df['Avg Return (%)'], color='blue', alpha=0.6, label='Avg Return (%)')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Avg Return (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(seasonal_df.index, seasonal_df['Win Rate (%)'], color='green', marker='o', label='Win Rate (%)')
    ax2.set_ylabel('Win Rate (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    fig.suptitle(f'{index_name} Seasonal Patterns')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.grid(True)
    plt.savefig(f'{index_name.lower()}_seasonal_patterns.png')
    plt.close()
    print(f"Seasonal patterns chart saved as '{index_name.lower()}_seasonal_patterns.png'")
    print("\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze stock index data and download historical data')
    parser.add_argument('--index', type=str, required=True, choices=index_tickers.keys(), help='Select the index to analyze (NASDAQ, NYSE, NIFTY, BSE)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD), default is 2018-06-01')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD), default is today')
    parser.add_argument('--years', type=int, help='Number of years back from end date')
    args = parser.parse_args()
    
    if args.years:
        if args.end:
            try:
                end_date = datetime.strptime(args.end, '%Y-%m-%d')
            except ValueError:
                print("Invalid end date format. Use YYYY-MM-DD.")
                return
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * args.years)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    else:
        start_date = args.start if args.start else '2018-06-01'
        end_date = args.end if args.end else datetime.now().strftime('%Y-%m-%d')
    
    ticker = index_tickers[args.index]
    print(f"Fetching data for {args.index} ({ticker}) from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    if data.empty:
        print(f"No data found for {args.index} ({ticker}) in the specified date range.")
        return
    data.to_csv(f'{args.index}_historical_data.csv')
    print(f"Historical data saved to {args.index}_historical_data.csv")
    
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['MonthName'] = data.index.strftime('%b')
    
    dataset_introduction(data, args.index)
    summary_statistics(data, args.index)
    trend_analysis(data, args.index)
    volatility_analysis(data, args.index)
    moving_average_analysis(data, args.index)
    comparative_analysis(data, args.index)
    seasonal_patterns(data, args.index)
    print("All analyses complete. Check saved PNG files for charts.")

if __name__ == "__main__":
    main()