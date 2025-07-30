import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

try:
    mileage = float(input("Enter mileage (in km): "))
except ValueError:
    print("Invalid input. Please enter a numeric mileage.")
    sys.exit(1)
    
try:
    params = pd.read_csv('params.csv')
    theta0 = params['theta0'][0]
    theta1 = params['theta1'][0]
except FileNotFoundError:
    theta0 = 0
    theta1 = 0

# Calculate price estimate
estimated_price = theta0 + theta1 * mileage
print(f"Estimated price: €{estimated_price:.2f}")

# Load dataset for plotting
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Warning: data.csv not found, skipping plot.")
    sys.exit(0)

km = data['km']
price = data['price']

plt.figure(figsize=(10,6))
plt.scatter(km, price, color='blue', alpha=0.6, label='Data points', s=60, edgecolors='k')

# Smooth regression line over km range
km_range = np.linspace(km.min(), km.max(), 200)
regression_line = theta0 + theta1 * km_range
plt.plot(km_range, regression_line, color='red', linewidth=3, label='Regression Line')

# Vertical line at input mileage
plt.axvline(x=mileage, color='green', linestyle='--', linewidth=2, label=f'Mileage Input: {mileage:.1f} km')

# Annotate estimated price near vertical line
plt.annotate(f"Estimated price:\n€{estimated_price:.2f}",
             xy=(mileage, estimated_price),
             xytext=(mileage + (km.max()-km.min())*0.05, estimated_price),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

plt.xlabel('Mileage (km)', fontsize=14)
plt.ylabel('Price (€)', fontsize=14)
plt.title(f'Car Price Prediction\nθ0 = {theta0:.2f}, θ1 = {theta1:.5f}', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()