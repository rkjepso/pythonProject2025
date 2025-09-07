
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
x = np.linspace(0, 20, 20).reshape(-1, 1)
y = 2500 + pow(x - 14, 2) * 4 + np.random.randint(0,200, size=x.shape)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x) # make tables for all x,x2,x3....

# Train the model
model = LinearRegression()
model.fit(X_poly, y)  # y = Ax + Bx2..Cx3 -> calulate A, B, C...

# Predict, find the curve
y_pred = model.predict(X_poly) # calulate y = Ax + Bx2..
# Evaluate model quality
std = np.sqrt(mean_squared_error(y, y_pred)) # standard deviation/avvk
r_score = r2_score(y, y_pred)

# Plot
plt.scatter(x, y, color='blue', label='Train data')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('km from Bergen')
plt.ylabel('Rain (mm)')
plt.title(f'Linear Regression STD: {std:.2f}, R: {r_score:.2f}')
plt.legend()
plt.show()





