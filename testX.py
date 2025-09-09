import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Sample dataset
data = {
    'm2': [150, 180, 20, 30, 50, 90, 150, 110, 80, 20],
    'std': [3, 1, 1, 2, 2, 3, 1, 1, 2, 3],
    'price': [7, 16, 3, 2.5, 6, 12, 10, 8, 5,1.8]
}

df = pd.DataFrame(data)

# Features and target
X_train = df[['m2', 'std']]
y_train = df['price']

# Split into training and test sets
#  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
poly = PolynomialFeatures(degree=2)
# Create and train the model
model = LinearRegression()
X_poly = poly.fit_transform(X_train)
model.fit(X_poly, y_train)

# Predict and evaluate
dataTest = {
    'm2': [170, 180, 120, 30, 50, 90],
    'std': [1, 3, 2, 2, 2, 1],
    'price': [13, 12, 10, 2.5, 6, 8]
}
dfTest = pd.DataFrame(dataTest)
X_test = dfTest[['m2', 'std']]
y_test = dfTest['price']
X_polytest = poly.fit_transform(X_test)
y_pred = model.predict(X_polytest)
mse = mean_squared_error(y_test, y_pred)
dfTest["priceEst"] = y_pred
print(dfTest)
# Output results
print("Predicted prices:", y_pred)
print("Mean Squared Error:", mse)
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)
