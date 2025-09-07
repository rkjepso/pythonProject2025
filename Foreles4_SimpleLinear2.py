
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
X = np.linspace(0, 20, 50).reshape(-1, 1)
noise = np.random.normal(0, 1, size=X.shape) * 2 + 20
Y = 200 - 2 * X + noise
Y = [y if y < 30 else 60 - y  for y in Y]

# Split into training and testing sets
X_Train, X_Test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_Train, Y_train)

# Predict
Y_pred_test = model.predict(X_Test)

# Evaluate
mse_test = mean_squared_error(Y_test, Y_pred_test)
r2_test = r2_score(Y_test, Y_pred_test)

print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Slope: {model.coef_[0][0]:.2f}")
print(f"Test MSE: {mse_test:.2f}, R²: {r2_test:.2f}")

# Plot
plt.scatter(X_Train, Y_train, color='blue', label='Train data')
plt.scatter(X_Test, Y_test, color='green', label='Test data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Train-Test Split')
plt.legend()
plt.show()
