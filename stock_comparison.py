import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import argparse

# CSS styling for the report
css = """
body {
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.6;
}
h1, h2, h3, h4 {
    color: #003366;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #f2f2f2;
}
"""

def fetch_general_info(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    info = stock.info
    name = info.get('longName', stock_symbol)
    sector = info.get('sector', 'Unknown')
    business_summary = info.get('longBusinessSummary', 'No summary available.')
    return name, sector, business_summary

def fetch_financial_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    dividends = stock.dividends
    return financials, balance_sheet, cashflow, dividends

def calculate_financial_metrics(financials, balance_sheet, cashflow, dividends):
    metrics = {}
    latest_year = financials.columns[0]
    try:
        revenue = financials.loc['Total Revenue', latest_year]
        cogs = financials.loc['Cost Of Revenue', latest_year] if 'Cost Of Revenue' in financials.index else 0
        gross_profit = revenue - cogs
        metrics['Gross Margin (%)'] = (gross_profit / revenue * 100) if revenue else 0
        operating_income = financials.loc['Operating Income', latest_year]
        metrics['Operating Margin (%)'] = (operating_income / revenue * 100) if revenue else 0
        net_income = financials.loc['Net Income', latest_year]
        metrics['Net Margin (%)'] = (net_income / revenue * 100) if revenue else 0
        equity = balance_sheet.loc['Total Stockholder Equity', latest_year]
        metrics['Return on Equity (ROE) (%)'] = (net_income / equity * 100) if equity else 0
        total_assets = balance_sheet.loc['Total Assets', latest_year]
        metrics['Return on Assets (ROA) (%)'] = (net_income / total_assets * 100) if total_assets else 0
        current_assets = balance_sheet.loc['Total Current Assets', latest_year]
        current_liabilities = balance_sheet.loc['Total Current Liabilities', latest_year]
        metrics['Current Ratio'] = (current_assets / current_liabilities) if current_liabilities else 0
        total_debt = balance_sheet.loc['Total Debt', latest_year] if 'Total Debt' in balance_sheet.index else (balance_sheet.get('Long Term Debt', 0) + balance_sheet.get('Short Long Term Debt', 0))
        metrics['Debt-to-Equity Ratio'] = (total_debt / equity) if equity else 0
        
        if len(financials.columns) >= 5:
            revenue_5y_ago = financials.loc['Total Revenue', financials.columns[-5]]
            metrics['Revenue Growth (5-year, %)'] = ((revenue / revenue_5y_ago) ** (1/5) - 1) * 100 if revenue_5y_ago else 0
            eps = financials.loc['Net Income', latest_year] / balance_sheet.loc['Common Stock', latest_year]
            eps_5y_ago = financials.loc['Net Income', financials.columns[-5]] / balance_sheet.loc['Common Stock', financials.columns[-5]]
            metrics['EPS Growth (5-year, %)'] = ((eps / eps_5y_ago) ** (1/5) - 1) * 100 if eps_5y_ago else 0
        else:
            metrics['Revenue Growth (5-year, %)'] = 0
            metrics['EPS Growth (5-year, %)'] = 0
        
        if not dividends.empty and len(dividends) >= 5*252:  # Approx 252 trading days per year
            div_5y = dividends.resample('Y').sum()
            if len(div_5y) >= 5:
                div_latest = div_5y[-1]
                div_5y_ago = div_5y[-5]
                metrics['Dividend Growth (5-year, %)'] = ((div_latest / div_5y_ago) ** (1/5) - 1) * 100 if div_5y_ago else 0
            else:
                metrics['Dividend Growth (5-year, %)'] = 0
        else:
            metrics['Dividend Growth (5-year, %)'] = 0
    except KeyError as e:
        print(f"Warning: Missing data for {e}. Setting affected metric to 0.")
        for metric in metrics:
            if metric not in metrics:
                metrics[metric] = 0
    return metrics

def fetch_historical_prices(stock_symbol, period='5y'):
    stock = yf.Ticker(stock_symbol)
    prices = stock.history(period=period)
    return prices

def calculate_performance(prices):
    one_year_ago = datetime.now() - timedelta(days=365)
    five_years_ago = datetime.now() - timedelta(days=5*365)
    
    latest_price = prices['Close'].iloc[-1]
    price_1y = prices['Close'][prices.index >= one_year_ago].iloc[0] if not prices[prices.index >= one_year_ago].empty else latest_price
    price_5y = prices['Close'][prices.index >= five_years_ago].iloc[0] if not prices[prices.index >= five_years_ago].empty else latest_price
    
    return_1y = ((latest_price - price_1y) / price_1y) * 100
    return_5y = ((latest_price - price_5y) / price_5y) * 100
    return return_1y, return_5y

def perform_technical_analysis(prices):
    prices = prices.tail(252)  # Last year for technicals
    sma50 = prices['Close'].rolling(window=50).mean().iloc[-1]
    sma200 = prices['Close'].rolling(window=200).mean().iloc[-1]
    rsi = ta.rsi(prices['Close'], length=14).iloc[-1]
    macd = ta.macd(prices['Close'], fast=12, slow=26, signal=9)
    macd_line = macd['MACD_12_26_9'].iloc[-1]
    signal_line = macd['MACDS_12_26_9'].iloc[-1]
    volume_avg = prices['Volume'].mean()
    
    score = 0
    if prices['Close'].iloc[-1] > sma50:
        score += 1
    if sma50 > sma200:
        score += 1
    if 30 < rsi < 70:
        score += 1
    if macd_line > signal_line:
        score += 1
    
    return score, prices['Close'].iloc[-1], sma50, sma200, rsi, macd_line, signal_line, volume_avg

def fetch_valuation_metrics(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    info = stock.info
    pe_ratio = info.get('trailingPE', 'N/A')
    pb_ratio = info.get('priceToBook', 'N/A')
    ev_ebitda = info.get('enterpriseToEbitda', 'N/A')
    return pe_ratio, pb_ratio, ev_ebitda

def assign_scores(stocks_data):
    scores = {stock: {} for stock in stocks_data}
    
    # Fundamental: Revenue Growth
    rev_growth = {s: d['metrics']['Revenue Growth (5-year, %)'] for s, d in stocks_data.items()}
    sorted_stocks = sorted(rev_growth, key=rev_growth.get, reverse=True)
    for i, s in enumerate(sorted_stocks):
        scores[s]['Fundamental Analysis'] = 5 - i if i < 2 else 3
    
    # Financial: ROE
    roe = {s: d['metrics']['Return on Equity (ROE) (%)'] for s, d in stocks_data.items()}
    sorted_stocks = sorted(roe, key=roe.get, reverse=True)
    for i, s in enumerate(sorted_stocks):
        scores[s]['Financial Analysis'] = 5 - i if i < 2 else 3
    
    # Performance: 5-year return
    returns_5y = {s: d['performance'][1] for s, d in stocks_data.items()}
    sorted_stocks = sorted(returns_5y, key=returns_5y.get, reverse=True)
    for i, s in enumerate(sorted_stocks):
        scores[s]['Company Performance'] = 5 - i if i < 2 else 3
    
    # Outlook: EPS Growth
    eps_growth = {s: d['metrics']['EPS Growth (5-year, %)'] for s, d in stocks_data.items()}
    sorted_stocks = sorted(eps_growth, key=eps_growth.get, reverse=True)
    for i, s in enumerate(sorted_stocks):
        scores[s]['Outlook'] = 5 - i if i < 2 else 3
    
    # Technical
    tech_scores = {s: d['technical'][0] for s, d in stocks_data.items()}
    sorted_stocks = sorted(tech_scores, key=tech_scores.get, reverse=True)
    for i, s in enumerate(sorted_stocks):
        scores[s]['Technical Analysis'] = 5 - i if i < 2 else 2 if i == 2 else 3
    
    # Valuation: P/E (lower better)
    pe_ratios = {s: d['valuation'][0] if d['valuation'][0] != 'N/A' else float('inf') for s, d in stocks_data.items()}
    sorted_stocks = sorted(pe_ratios, key=pe_ratios.get)
    for i, s in enumerate(sorted_stocks):
        scores[s]['Valuation'] = 5 - i if i < 2 else 3
    
    return scores

def generate_report(stocks, stocks_data, scores, author):
    html = f"<html><head><style>{css}</style></head><body>"
    
    # Title
    stock_names = [stocks_data[s]['name'] for s in stocks]
    sector = stocks_data[stocks[0]]['sector']
    html += f"<h1>Comparison Among {', '.join(stock_names[:-1])}, & {stock_names[-1]} in the {sector} by {author}</h1>"
    
    # Best Investment Opportunity
    total_scores = {s: sum(scores[s].values()) for s in stocks}
    best_stock = max(total_scores, key=total_scores.get)
    html += "<h2>Best Investment Opportunity</h2>"
    score_table = "<table><tr><th>Category</th>" + "".join(f"<th>{stocks_data[s]['name']}</th>" for s in stocks) + "</tr>"
    for category in scores[stocks[0]].keys():
        score_table += f"<tr><td>{category}</td>" + "".join(f"<td>{scores[s][category]}</td>" for s in stocks) + "</tr>"
    score_table += "<tr><td><b>Total Score</b></td>" + "".join(f"<td><b>{total_scores[s]}</b></td>" for s in stocks) + "</tr></table>"
    html += score_table
    html += f"<p><b>{stocks_data[best_stock]['name']}</b> is the best opportunity with a total score of {total_scores[best_stock]}, excelling in financial strength and market performance.</p>"
    
    # Fundamental Analysis
    html += "<h2>Fundamental Analysis</h2>"
    for s in stocks:
        data = stocks_data[s]
        html += f"<h3>{data['name']}:</h3>"
        html += f"<h4>Core Business and Competitive Positioning</h4><p>{data['business_summary']}</p>"
        html += "<h4>Recent Developments</h4><p>[Placeholder: Insert recent developments from news sources]</p>"
        html += "<h4>Company Sentiments</h4><p>[Placeholder: Insert sentiment analysis from analyst reports and social media]</p>"
    
    # Financial Analysis
    html += "<h2>Financial Analysis</h2>"
    metrics_table = "<table><tr><th>Metric</th>" + "".join(f"<th>{stocks_data[s]['name']}</th>" for s in stocks) + "</tr>"
    metric_names = list(stocks_data[stocks[0]]['metrics'].keys())
    for m in metric_names:
        metrics_table += f"<tr><td>{m}</td>" + "".join(f"<td>{stocks_data[s]['metrics'][m]:.2f}</td>" for s in stocks) + "</tr>"
    metrics_table += "</table>"
    html += metrics_table
    html += "<p><b>Commentary:</b> [Placeholder: Insert analysis of strengths and weaknesses based on metrics]</p>"
    
    # Company Performance
    html += "<h2>Company Performance</h2>"
    html += "<h3>Stock Price Performance:</h3>"
    perf_table = "<table><tr><th>Period</th>" + "".join(f"<th>{stocks_data[s]['name']}</th>" for s in stocks) + "</tr>"
    perf_table += "<tr><td>Past Year (%)</td>" + "".join(f"<td>{stocks_data[s]['performance'][0]:.1f}</td>" for s in stocks) + "</tr>"
    perf_table += "<tr><td>Five Years (%)</td>" + "".join(f"<td>{stocks_data[s]['performance'][1]:.1f}</td>" for s in stocks) + "</tr>"
    perf_table += "</table>"
    html += perf_table
    html += "<h3>Relative to Historical Performance</h3><p>[Placeholder: Compare to historical averages]</p>"
    html += "<h3>Relative to Industry Benchmark</h3><p>[Placeholder: Compare to industry benchmarks]</p>"
    html += "<h3>Relative to Management's Forecasts</h3><p>[Placeholder: Evaluate alignment with forecasts]</p>"
    html += "<h3>Performance for Investors</h3><p>[Placeholder: Discuss total returns]</p>"
    
    # Outlook
    html += "<h2>Outlook</h2>"
    for s in stocks:
        html += f"<h3>{stocks_data[s]['name']}:</h3>"
        html += "<h4>Growth Projections</h4><p>[Placeholder: Insert analyst growth projections]</p>"
        html += "<h4>Opportunities</h4><p>[Placeholder: Identify business opportunities]</p>"
        html += "<h4>Management's Forecasts</h4><p>[Placeholder: Discuss strategic focus]</p>"
        html += "<h4>Macroeconomic & Regulatory Condition</h4><p>[Placeholder: Discuss sector conditions]</p>"
        html += "<h4>Potential Risks</h4><p>[Placeholder: Outline risks]</p>"
    
    # Technical Analysis
    html += "<h2>Technical Analysis</h2>"
    for s in stocks:
        t = stocks_data[s]['technical']
        html += f"<h3>{stocks_data[s]['name']}:</h3>"
        trend = "uptrend" if t[1] > t[2] > t[3] else "downtrend" if t[1] < t[2] < t[3] else "sideways"
        rsi_status = "overbought" if t[4] > 70 else "oversold" if t[4] < 30 else "neutral"
        macd_status = "bullish" if t[5] > t[6] else "bearish"
        html += f"<p>Stock price at ${t[1]:.2f}, {trend} relative to 50-day (${t[2]:.2f}) and 200-day (${t[3]:.2f}) moving averages. RSI at {t[4]:.1f} ({rsi_status}). MACD {macd_status} (MACD: {t[5]:.2f}, Signal: {t[6]:.2f}). Volume avg: {t[7]:,.0f}.</p>"
    
    # Valuation
    html += "<h2>Valuation</h2>"
    val_table = "<table><tr><th>Metric</th>" + "".join(f"<th>{stocks_data[s]['name']}</th>" for s in stocks) + "</tr>"
    val_table += "<tr><td>P/E Ratio</td>" + "".join(f"<td>{stocks_data[s]['valuation'][0]}</td>" for s in stocks) + "</tr>"
    val_table += "<tr><td>P/B Ratio</td>" + "".join(f"<td>{stocks_data[s]['valuation'][1]}</td>" for s in stocks) + "</tr>"
    val_table += "<tr><td>EV/EBITDA</td>" + "".join(f"<td>{stocks_data[s]['valuation'][2]}</td>" for s in stocks) + "</tr>"
    val_table += "</table>"
    html += val_table
    html += "<p>[Placeholder: Comment on valuation relative to growth and industry]</p>"
    
    # Selecting the Best Opportunity
    html += "<h2>Selecting the Best Opportunity</h2>"
    html += score_table
    html += f"<p><b>{stocks_data[best_stock]['name']}</b> is the best opportunity with a total score of {total_scores[best_stock]}, excelling in financial strength and market performance.</p>"
    html += "<h3>Selection Rationale</h3><p>[Placeholder: Detailed rationale for selection]</p>"
    html += "<h3>Key Factors Making It Better Than Peers</h3><p>[Placeholder: List key factors]</p>"
    html += "<h3>Shortcomings of Other Stocks</h3>"
    for s in stocks:
        if s != best_stock:
            html += f"<p><b>{stocks_data[s]['name']}:</b> [Placeholder: Summarize shortcomings]</p>"
    
    # Citations
    html += "<h2>Key Citations</h2><p>[Placeholder: List data sources]</p>"
    
    html += "</body></html>"
    return html

def main():
    parser = argparse.ArgumentParser(description='Stock Comparison Analysis')
    parser.add_argument('--stocks', nargs=3, required=True, help='Three stock symbols')
    parser.add_argument('--author', required=True, help='Author name')
    args = parser.parse_args()
    
    stocks = args.stocks
    author = args.author
    stocks_data = {}
    
    for s in stocks:
        name, sector, business_summary = fetch_general_info(s)
        financials, balance_sheet, cashflow, dividends = fetch_financial_data(s)
        metrics = calculate_financial_metrics(financials, balance_sheet, cashflow, dividends)
        prices = fetch_historical_prices(s)
        performance = calculate_performance(prices)
        technical = perform_technical_analysis(prices)
        valuation = fetch_valuation_metrics(s)
        stocks_data[s] = {
            'name': name, 'sector': sector, 'business_summary': business_summary,
            'metrics': metrics, 'performance': performance, 'technical': technical,
            'valuation': valuation
        }
    
    scores = assign_scores(stocks_data)
    report = generate_report(stocks, stocks_data, scores, author)
    
    with open("stock_comparison_report.html", "w") as f:
        f.write(report)
    print("Report generated as 'stock_comparison_report.html'.")

if __name__ == "__main__":
    main()