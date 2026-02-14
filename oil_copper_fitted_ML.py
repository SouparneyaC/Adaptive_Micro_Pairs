import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestClassifier

# 1. Download Data
tickers = ["EWA", "EWC", "^VIX", "HG=F", "CL=F"]
print("Downloading Macro-Data...")
df = yf.download(tickers, start="2020-01-01", end="2025-12-31", progress=False)
prices = df['Close'].dropna()

# 2. Normalize and Kalman Filter
prices_norm = (prices / prices.iloc[0]) * 100
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

# 3. FEATURE ENGINEERING (Must happen BEFORE the ML step)
prices_norm['Oil_5D_Trend'] = prices['CL=F'].pct_change(5).shift(1)
prices_norm['Copper_5D_Trend'] = prices['HG=F'].pct_change(5).shift(1)
prices_norm['Spread_Vol'] = prices_norm['Smart_Spread'].rolling(10).std()
prices_norm['VIX_Level'] = prices_norm['^VIX']

# 4. ML Gatekeeper
prices_norm['Target'] = (prices_norm['Smart_Spread'].shift(-5).abs() < prices_norm['Smart_Spread'].abs()).astype(int)

# Use the features we just created
ml_cols = ['Spread_Vol', 'VIX_Level', 'Oil_5D_Trend', 'Copper_5D_Trend', 'Smart_Spread']
ml_data = prices_norm[ml_cols + ['Target']].dropna()

model = RandomForestClassifier(n_estimators=100, random_state=42)
split = int(len(ml_data) * 0.8)
model.fit(ml_data[ml_cols][:split], ml_data['Target'][:split])

# 5. STRATEGY SETTINGS
THRESHOLD = 2.0  # High conviction only

prices_norm['ML_Prediction'] = model.predict(prices_norm[ml_cols].fillna(0))
prices_norm['Signal'] = np.where((prices_norm['Smart_Spread'].abs() > prices_norm['Spread_Vol'] * THRESHOLD) & 
                                 (prices_norm['ML_Prediction'] == 1), 1, 0)

# 6. PERFORMANCE STATS
prices_norm['Strategy_Ret'] = prices_norm['Smart_Spread'].diff() * prices_norm['Signal'].shift(1)
prices_norm['Cumulative_PnL'] = prices_norm['Strategy_Ret'].cumsum()

daily_returns = prices_norm['Strategy_Ret'].dropna()
if daily_returns.std() != 0:
    final_sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
else:
    final_sharpe = 0

print("\n" + "="*30)
print("   MONDAY MORNING STATS")
print("="*30)
print(f"Final Sharpe Ratio: {final_sharpe:.2f}")
print(f"Total Trade Days:   {prices_norm['Signal'].sum()}")
print(f"Max PnL Achieved:   {prices_norm['Cumulative_PnL'].max():.2f}")
print("="*30)

plt.figure(figsize=(10,6))
plt.plot(prices_norm['Cumulative_PnL'], color='gold')
plt.title(f"Macro-Enhanced Pair Strategy (Sharpe: {final_sharpe:.2f})")
plt.show()