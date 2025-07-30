import pandas as pd
import math

# Load data
data = pd.read_csv('data.csv')
km = data['km']
actual_price = data['price']

# Load trained parameters
params = pd.read_csv('params.csv')
theta0 = params['theta0'][0]
theta1 = params['theta1'][0]

# Predict prices
predicted_price = theta0 + theta1 * km

# Calculate Mean Squared Error (MSE)
squared_errors = (actual_price - predicted_price) ** 2
mse = squared_errors.mean()

# Calculate Root Mean Squared Error (RMSE)
rmse = math.sqrt(mse)

min_price = min(data['price'])
max_price = max(data['price'])
error_off = rmse / (max_price - min_price) * 100
print(f"Minimum Price: {min_price}")
print(f"Maximum Price: {max_price}")
print(f"Root Mean Squared Error (RMSE): €{rmse:.2f}")
print(f"Current Precision of Model: {error_off:.2f}%")