
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def to_rgb(value):
    value = max(0, min(100, value))  # Clamp to [0, 100]
    if value < 50:   # Green to Yellow
        r = value / 50
        g = 1.0
    else:  # Green to Yellow
        r = 1.0
        g = 1 - (value - 50) / 50
    return r, g, 0.0


df = pd.read_csv('house_data.csv')
# Prepare plot
fig = plt.figure(figsize=(10,6))
ax = fig.add_axes((.1, .2, .85, .75), projection="3d")
ax.set_title("Leiligheter i Bergen (Est/Real)")


X = df[['m2', 'Standard']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Standard avvik:", int(np.sqrt(mse)))


x3 = np.linspace(25,250, 10)
y3 = [1,1,2,2,3,3,4,4,1,2]
z3 = np.zeros(10)

dx = np.ones(10) * 10
dy = np.ones(10) * 0.3
dz = [1,2,3,4,5,6,7,8,9,10]

zmax = max(dz)
colors =[to_rgb(z/zmax*100) for z in dz]
ax.bar3d(x3, y3, z3, dx, dy, dz, color=colors)

ax.set_ylim(1,4)
ax.set_xlabel('m2')
ax.set_ylabel('standard')
ax.set_zlabel('pris')

plt.show()