import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Create the folder structure
# 'os.makedirs' creates the folders if they don't exist already
folder_path = "disjoint_pairs_stock/plots"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created directory: {folder_path}")

# 2. Setup Tickers (Australia, Canada, and Fear Index)
tickers = ["EWA", "EWC", "^VIX"]
print("Downloading data...")
df = yf.download(tickers, start="2020-01-01", end="2025-12-31")

# 3. Data Cleaning
prices = df['Close'].dropna()

# 4. Normalize and Calculate Spread
# We divide by the first row so both stocks start at 100
prices_norm = (prices / prices.iloc[0]) * 100
prices_norm['Spread'] = prices_norm['EWA'] - prices_norm['EWC']

# 5. Create the Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: The Stocks
ax1.plot(prices_norm.index, prices_norm['EWA'], label='Australia (EWA)', color='#1f77b4')
ax1.plot(prices_norm.index, prices_norm['EWC'], label='Canada (EWC)', color='#d62728')
ax1.set_title("Price Comparison (Normalized to 100)", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: The Spread
ax2.plot(prices_norm.index, prices_norm['Spread'], label='Spread (EWA - EWC)', color='#2ca02c')
ax2.axhline(0, color='black', linestyle='--')
ax2.set_title("The Spread (The 'Distance' between the two)", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# 6. Save the plot to the specific folder
save_path = os.path.join(folder_path, "ewa_ewc_spread.png")
plt.savefig(save_path)
print(f"Plot successfully saved to: {save_path}")

# Optional: Show it on screen too
plt.show()