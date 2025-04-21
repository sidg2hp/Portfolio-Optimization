
Portfolio Optimization Report
============================

Asset Selection
---------------
Selected Assets and Justification:
- RELIANCE.NS: Reliance Industries (Energy)
- TCS.NS: Tata Consultancy Services (Technology)
- HDFCBANK.NS: HDFC Bank (Financials)
- HINDUNILVR.NS: Hindustan Unilever (Consumer Goods)
- INFY.NS: Infosys (Technology)

Criteria:
- **Diversification**: Assets span Energy, Technology, Financials, and Consumer Goods to reduce sector-specific risk.
- **Historical Performance**: Chosen based on stable returns and market leadership from April 2019 to March 2022.
- **Liquidity**: All are large-cap stocks with high trading volumes, ensuring reliable data.

Optimized Portfolio
------------------
Optimal Weights (based on training data, April 2019–March 2022):
- RELIANCE.NS: 0.00%
- TCS.NS: 0.00%
- HDFCBANK.NS: 71.18%
- HINDUNILVR.NS: 23.21%
- INFY.NS: 5.61%

Expected Annual Return: 35.93%
Expected Annual Volatility: 426.89%
Sharpe Ratio: 0.07

Portfolio PnL
-------------
Initial Investment: INR 100,000.00
Final Portfolio Value (March 2025): INR 94,622.40
Total Profit/Loss: INR -5,377.60

Performance Metrics
------------------
- Annualized Return (Backtest, April 2022–March 2025): 0.18%
- Annualized Volatility (Backtest): 20.30%
- Sharpe Ratio (Backtest): -0.19

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
