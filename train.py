import pandas as pd

# Load dataset
data = pd.read_csv('data.csv')
km = data['km']
price = data['price']

# Normalize data
km_mean = km.mean()
km_std = km.std()
price_mean = price.mean()
price_std = price.std()

data['km'] = (km - km_mean) / km_std
data['price'] = (price - price_mean) / price_std

# Gradient descent function
def gradient_descent(m_current, b_current, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].km
        y = points.iloc[i].price
        prediction = m_current * x + b_current
        error = y - prediction
        m_gradient += -(2/n) * x * error
        b_gradient += -(2/n) * error
    new_m = m_current - learning_rate * m_gradient
    new_b = b_current - learning_rate * b_gradient
    return new_m, new_b

# Training loop
m = 0
b = 0
learning_rate = 0.01
epochs = 1000

for _ in range(epochs):
    m, b = gradient_descent(m, b, data, learning_rate)

# Denormalize parameters to original scale
theta1 = m * (price_std / km_std)
theta0 = price_mean + price_std * b - theta1 * km_mean

params = pd.DataFrame({'theta0': [theta0], 'theta1': [theta1]})
params.to_csv('params.csv', index=False)
print("Training complete.")
print(f"Theta0: {theta0}, Theta1: {theta1}")
