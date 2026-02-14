import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import os

# 1. Setup Folders
folder_path = "disjoint_pairs_stock/plots"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 2. Get Data
tickers = ["EWA", "EWC", "^VIX"]
print("Downloading data...")
df = yf.download(tickers, start="2020-01-01", end="2025-12-31")
prices = df['Close'].dropna()

# 3. Normalize (Start both at 100)
prices_norm = (prices / prices.iloc[0]) * 100

# 4. Prepare Kalman Filter Inputs
# We want to find: EWA = (Beta * EWC) + Intercept
obs_mat = np.vstack([prices_norm['EWC'], np.ones(len(prices_norm))]).T[:, np.newaxis, :]

kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=1.0,
                  transition_covariance=np.eye(2) * 1e-4)

# 5. Run the Math
print("Running Kalman Filter...")
state_means, _ = kf.filter(prices_norm['EWA'].values)

# Extract the 'Smart' Beta and Intercept
prices_norm['Dynamic_Beta'] = state_means[:, 0]
prices_norm['Dynamic_Intercept'] = state_means[:, 1]

# Calculate the 'Smart Spread'
prices_norm['Smart_Spread'] = prices_norm['EWA'] - (prices_norm['EWC'] * prices_norm['Dynamic_Beta'] + prices_norm['Dynamic_Intercept'])

# 6. Compare 'Dumb' vs 'Smart' Spread
# 'Dumb' spread is just simple subtraction
prices_norm['Dumb_Spread'] = prices_norm['EWA'] - prices_norm['EWC']

# 7. Plotting the Comparison
plt.figure(figsize=(12, 6))
plt.plot(prices_norm.index, prices_norm['Dumb_Spread'], label='Dumb Spread (Simple)', color='red', alpha=0.3)
plt.plot(prices_norm.index, prices_norm['Smart_Spread'], label='Smart Spread (Kalman)', color='blue')
plt.axhline(0, color='black', linestyle='--')
plt.title("Dumb vs. Smart (Kalman) Spread", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
save_path = os.path.join(folder_path, "kalman_comparison.png")
plt.savefig(save_path)
print(f"Success! Plot saved to {save_path}")
plt.show()