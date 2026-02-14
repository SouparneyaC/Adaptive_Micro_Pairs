import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestClassifier
import os

# 1. Setup Folders
folder_path = "disjoint_pairs_stock/plots"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 2. Data Acquisition
tickers = ["EWA", "EWC", "^VIX"]
print("Step 1: Downloading market data...")
df = yf.download(tickers, start="2020-01-01", end="2025-12-31", progress=False)
prices = df['Close'].dropna()

# 3. Normalization
prices_norm = (prices / prices.iloc[0]) * 100

# 4. The Kalman Filter (Our "Smart" Math)
print("Step 2: Running Kalman Filter to find the hidden relationship...")
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

# 5. Machine Learning (Our "Gatekeeper")
print("Step 3: Training the ML Gatekeeper...")

# Feature Engineering: VIX and Spread Volatility
prices_norm['Spread_Vol'] = prices_norm['Smart_Spread'].rolling(10).std()
prices_norm['VIX_Level'] = prices_norm['^VIX']

# Target: 1 if the spread returns toward zero in 5 days, 0 if it doesn't
prices_norm['Target'] = (prices_norm['Smart_Spread'].shift(-5).abs() < prices_norm['Smart_Spread'].abs()).astype(int)

# Clean up data for ML
ml_data = prices_norm[['Spread_Vol', 'VIX_Level', 'Smart_Spread', 'Target']].dropna()
X = ml_data[['Spread_Vol', 'VIX_Level', 'Smart_Spread']]
y = ml_data['Target']

# Split and Train
split = int(len(X) * 0.8)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X[:split], y[:split])

# 6. Final Strategy Backtest (The "Money" Part)
# Signal: If Spread > 2*Vol AND ML says it will revert, go Long/Short
prices_norm['ML_Prediction'] = model.predict(X.reindex(prices_norm.index).fillna(0))
prices_norm['Signal'] = np.where((prices_norm['Smart_Spread'].abs() > prices_norm['Spread_Vol'] * 2) & 
                                 (prices_norm['ML_Prediction'] == 1), 1, 0)

# Calculate Daily Returns
prices_norm['Strategy_Ret'] = prices_norm['Smart_Spread'].diff() * prices_norm['Signal'].shift(1)
prices_norm['Cumulative_PnL'] = prices_norm['Strategy_Ret'].cumsum()

# 7. Final Output
plt.figure(figsize=(12, 6))
plt.plot(prices_norm.index, prices_norm['Cumulative_PnL'], color='gold', linewidth=2)
plt.title("Strategy Performance: Cumulative P&L (Equity Curve)", fontsize=14)
plt.ylabel("Profit/Loss Units")
plt.grid(True, alpha=0.3)

save_path = os.path.join(folder_path, "final_strategy_pnl.png")
plt.savefig(save_path)
print(f"DONE! Final P&L chart saved to {save_path}")
plt.show()