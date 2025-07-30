import pandas as pd
import matplotlib.pyplot as plt

# Load model parameters
try:
    params = pd.read_csv('params.csv')
except FileNotFoundError:
    print("Error: params.csv not found. Please run train.py first.")
    exit()

theta0 = params['theta0'][0]
theta1 = params['theta1'][0]

try:
    mileage = float(input("Enter mileage (in km): "))
except ValueError:
    print("Invalid input. Please enter a numeric mileage.")
    exit()

predicted_price = theta0 + theta1 * mileage
print(f"Estimated price: €{predicted_price:.2f}")

try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Warning: data.csv not found. Skipping plot.")
    exit()

plt.scatter(data['km'], data['price'], color="black", label="Original Data")
x_vals = [data['km'].min(), data['km'].max()]
y_vals = [theta0 + theta1 * x for x in x_vals]
plt.plot(x_vals, y_vals, color="blue", label="Regression Line")

plt.scatter(mileage, predicted_price, color="red", label="Estimate", zorder=5)

plt.title("Car Price Prediction")
plt.xlabel("Mileage (km)")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.show()