import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'm2': [150, 180, 20, 30, 50, 90],
    'bedrooms': [3, 4, 1, 1, 1, 2],
    'age': [10, 15, 20, 5, 8, 1],
    'price': [8, 12, 2, 2.5, 6, 8]
}

df = pd.DataFrame(data)

# Features and target
X = df[['m2', 'bedrooms', 'age']]
y = df['price']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(X_test)
# Output results
print("Predicted prices:", y_pred)
print("Mean Squared Error:", mse)
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)
