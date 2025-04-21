import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import datetime
import os
import requests_cache

# Set random seed for reproducibility
np.random.seed(42)

# Define assets (Indian stocks from different sectors)
assets = {
    'RELIANCE.NS': 'Reliance Industries (Energy)',
    'TCS.NS': 'Tata Consultancy Services (Technology)',
    'HDFCBANK.NS': 'HDFC Bank (Financials)',
    'HINDUNILVR.NS': 'Hindustan Unilever (Consumer Goods)',
    'INFY.NS': 'Infosys (Technology)'
}

# Define time periods
training_start = '2019-04-01'
training_end = '2022-03-31'
testing_start = '2022-04-01'
testing_end = '2025-03-31'
initial_investment = 100000  # INR

# Download historical data with error handling
def download_data(tickers, start, end):
    try:
        data = yf.download(list(tickers.keys()), start=start, end=end, progress=False)
        # Ensure all tickers have data
        if data.empty or data.isna().all().any():
            raise ValueError("No valid data downloaded for one or more tickers.")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Calculate returns and covariance matrix
def calculate_metrics(data):
    try:
        returns = data.pct_change(fill_method=None).dropna()
        if returns.empty:
            raise ValueError("No valid returns data after processing.")
        mean_returns = returns.mean() * 252  # Annualized returns
        cov_matrix = returns.cov() * 252  # Annualized covariance
        return mean_returns, cov_matrix
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, None

# Portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 100  # In percentage
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights))) * 100
    return portfolio_return, portfolio_volatility

# Objective function (minimize negative Sharpe ratio)
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate * 100) / portfolio_volatility
    return -sharpe_ratio

# Constraints
def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
    )
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling
    initial_guess = num_assets * [1. / num_assets]
    result = minimize(
        neg_sharpe_ratio,
        initial_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x

# Efficient frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - 0.04 * 100) / portfolio_volatility
        weights_record.append(weights)
    return results, weights_record

# Backtesting
def backtest_portfolio(weights, data, initial_investment):
    try:
        returns = data.pct_change(fill_method=None).dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = initial_investment * cumulative_returns
        return portfolio_value, portfolio_returns
    except Exception as e:
        print(f"Error in backtesting: {e}")
        return None, None

# Plot efficient frontier
def plot_efficient_frontier(results, optimal_weights, mean_returns, cov_matrix):
    plt.figure(figsize=(10, 6))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    opt_return, opt_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    plt.scatter(opt_volatility, opt_return, c='red', marker='*', s=200, label='Optimal Portfolio')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Return (%)')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.savefig('efficient_frontier.png')
    plt.close()

# Plot portfolio performance
def plot_portfolio_performance(portfolio_value):
    plt.figure(figsize=(10, 6))
    portfolio_value.plot()
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (INR)')
    plt.title('Portfolio Performance (Backtest)')
    plt.grid(True)
    plt.savefig('portfolio_performance.png')
    plt.close()

# Main execution
def main():
    # Download training data
    training_data = download_data(assets, training_start, training_end)
    if training_data is None:
        print("Failed to download training data. Exiting.")
        return

    # Calculate metrics
    mean_returns, cov_matrix = calculate_metrics(training_data)
    if mean_returns is None or cov_matrix is None:
        print("Failed to calculate metrics. Exiting.")
        return

    # Optimize portfolio
    optimal_weights = optimize_portfolio(mean_returns, cov_matrix)
    opt_return, opt_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    sharpe_ratio = (opt_return - 0.04 * 100) / opt_volatility

    # Efficient frontier
    results, weights_record = efficient_frontier(mean_returns, cov_matrix)

    # Download testing data
    testing_data = download_data(assets, testing_start, testing_end)
    if testing_data is None:
        print("Failed to download testing data. Exiting.")
        return

    # Backtesting
    portfolio_value, portfolio_returns = backtest_portfolio(optimal_weights, testing_data, initial_investment)
    if portfolio_value is None or portfolio_returns is None:
        print("Failed to backtest portfolio. Exiting.")
        return

    # Performance metrics
    annualized_return = portfolio_returns.mean() * 252 * 100
    annualized_volatility = portfolio_returns.std() * np.sqrt(252) * 100
    final_value = portfolio_value.iloc[-1]
    total_pnl = final_value - initial_investment
    backtest_sharpe = (annualized_return - 0.04 * 100) / annualized_volatility

    # Visualizations
    plot_efficient_frontier(results, optimal_weights, mean_returns, cov_matrix)
    plot_portfolio_performance(portfolio_value)

    # Documentation
    documentation = f"""
Portfolio Optimization Report
============================

Asset Selection
---------------
Selected Assets and Justification:
{chr(10).join([f"- {ticker}: {desc}" for ticker, desc in assets.items()])}

Criteria:
- **Diversification**: Assets span Energy, Technology, Financials, and Consumer Goods to reduce sector-specific risk.
- **Historical Performance**: Chosen based on stable returns and market leadership from April 2019 to March 2022.
- **Liquidity**: All are large-cap stocks with high trading volumes, ensuring reliable data.

Optimized Portfolio
------------------
Optimal Weights (based on training data, April 2019–March 2022):
{chr(10).join([f"- {ticker}: {weight*100:.2f}%" for ticker, weight in zip(assets.keys(), optimal_weights)])}

Expected Annual Return: {opt_return:.2f}%
Expected Annual Volatility: {opt_volatility:.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}

Portfolio PnL
-------------
Initial Investment: INR {initial_investment:,.2f}
Final Portfolio Value (March 2025): INR {final_value:,.2f}
Total Profit/Loss: INR {total_pnl:,.2f}

Performance Metrics
------------------
- Annualized Return (Backtest, April 2022–March 2025): {annualized_return:.2f}%
- Annualized Volatility (Backtest): {annualized_volatility:.2f}%
- Sharpe Ratio (Backtest): {backtest_sharpe:.2f}

Data Sources
------------
- Historical price data: Yahoo Finance API (yfinance library)
- Training Period: April 2019–March 2022
- Testing Period: April 2022–March 2025

Methodology
-----------
- **Returns**: Calculated as daily percentage changes, annualized by multiplying by 252.
- **Covariance Matrix**: Annualized daily covariance matrix (252 trading days).
- **Optimization**: Maximized Sharpe ratio using scipy.optimize.minimize with SLSQP method.
- **Constraints**: No short selling (weights ≥ 0), full investment (weights sum to 1).
- **Backtesting**: Applied optimal weights to testing period data to compute portfolio value and returns.

Visualizations
--------------
- Efficient Frontier: Saved as 'efficient_frontier.png'
- Portfolio Performance: Saved as 'portfolio_performance.png'
"""
    with open('portfolio_documentation.txt', 'w', encoding='utf-8') as f:
        f.write(documentation)

    print("Portfolio optimization complete. Check 'efficient_frontier.png', 'portfolio_performance.png', and 'portfolio_documentation.txt' for results.")

if __name__ == "__main__":
    main()