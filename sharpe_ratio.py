import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestClassifier
import os

# 1. Setup
tickers = ["EWA", "EWC", "^VIX"]
print("Downloading data...")
df = yf.download(tickers, start="2020-01-01", end="2025-12-31", progress=False)
prices = df['Close'].dropna()

# 2. Process Data
prices_norm = (prices / prices.iloc[0]) * 100

# 3. Kalman Filter Math
print("Calculating Adaptive Relationship (Kalman)...")
obs_mat = np.vstack([prices_norm['EWC'], np.ones(len(prices_norm))]).T[:, np.newaxis, :]
kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=1.0,
                  transition_covariance=np.eye(2) * 1e-4)

state_means, _ = kf.filter(prices_norm['EWA'].values)
prices_norm['Smart_Spread'] = prices_norm['EWA'] - (prices_norm['EWC'] * state_means[:, 0] + state_means[:, 1])

# 4. Machine Learning Logic
print("Running ML Gatekeeper...")
prices_norm['Spread_Vol'] = prices_norm['Smart_Spread'].rolling(10).std()
prices_norm['VIX_Level'] = prices_norm['^VIX']
prices_norm['Target'] = (prices_norm['Smart_Spread'].shift(-5).abs() < prices_norm['Smart_Spread'].abs()).astype(int)

ml_data = prices_norm[['Spread_Vol', 'VIX_Level', 'Smart_Spread', 'Target']].dropna()
X = ml_data[['Spread_Vol', 'VIX_Level', 'Smart_Spread']]
y = ml_data['Target']

split = int(len(X) * 0.8)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X[:split], y[:split])

# 5. Backtest Calculations
prices_norm['ML_Prediction'] = model.predict(X.reindex(prices_norm.index).fillna(0))
# Entry logic: Spread is wide AND ML predicts a return to normal
prices_norm['Signal'] = np.where((prices_norm['Smart_Spread'].abs() > prices_norm['Spread_Vol'] * 2) & 
                                 (prices_norm['ML_Prediction'] == 1), 1, 0)

# Strategy Returns (Change in spread * our signal)
prices_norm['Strategy_Ret'] = prices_norm['Smart_Spread'].diff() * prices_norm['Signal'].shift(1)
prices_norm['Cumulative_PnL'] = prices_norm['Strategy_Ret'].cumsum()

# 6. SHARPE & RISK METRICS
print("Calculating Final Performance Stats...")
daily_returns = prices_norm['Strategy_Ret'].dropna()

# Annualized Sharpe (Assumes 252 trading days)
if daily_returns.std() != 0:
    sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
else:
    sharpe = 0.0

# Max Drawdown
cum_pnl = prices_norm['Cumulative_PnL']
running_max = cum_pnl.cummax()
drawdown = cum_pnl - running_max
max_dd = drawdown.min()

# 7. Print the "Monday Morning" Report
print("\n" + "="*40)
print("  MULTI-STRAT QUANT PROJECT: FINAL STATS")
print("="*40)
print(f"Asset Pair:           Australia (EWA) / Canada (EWC)")
print(f"Annualized Sharpe:    {sharpe:.2f}")
print(f"Max Strategy Drop:    {max_dd:.2f} units")
print(f"Winning Signals:      {prices_norm['Signal'].sum()} days")
print(f"End-of-Period PnL:    {prices_norm['Cumulative_PnL'].iloc[-1]:.2f}")
print("="*40)

# Save the final masterpiece
plt.figure(figsize=(10, 6))
plt.plot(prices_norm.index, prices_norm['Cumulative_PnL'], color='gold', lw=2)
plt.title(f"Equity Curve | Sharpe: {sharpe:.2f}")
plt.grid(True, alpha=0.2)
plt.show()