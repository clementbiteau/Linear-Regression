import pandas as pd
import matplotlib.pyplot as plt

# load the data
data = pd.read_csv('data.csv')
og_km = data['km']
og_price = data['price']

# normalise data
km_mean = data['km'].mean()
km_std = data['km'].std()

price_mean = data['price'].mean() 
price_std = data['price'].std()
data['km'] = (og_km - km_mean) / km_std
data['price'] = (og_price - price_mean) / price_std

# gradient descent function
def gradient_descent(m_current, b_current, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].km
        y = points.iloc[i].price
        error = y - (m_current * x + b_current)
        m_gradient += -(2/n) * x * error
        b_gradient += -(2/n) * error
    
    m = m_current - m_gradient * L
    b = b_current - b_gradient * L
    
    return m, b

# learning routine
m = 0
b = 0
L = 0.01
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)

# de normalise data
normalised_km = (og_km - km_mean) / km_std
norm_preds = m * normalised_km + b
real_prediction = norm_preds * price_std + price_mean

# use plt to show the results
plt.scatter(og_km, og_price, color="black", label="Original Prices")
plt.plot(og_km, real_prediction, color="green", label="Prediction Line")
plt.title("Linear Regression - Price vs Km - Cars")
plt.xlabel("Mileage in Kilometers")
plt.ylabel("Prices in Euros")
plt.legend()
plt.show()