import random
from random import randint
import matplotlib.pyplot as py
import numpy as np
from matplotlib.pyplot import legend
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_kmNS = np.linspace(0,50, 11)
X_kmEW = np.linspace(0,50, 11)

for x,y in zip(X_kmNS, X_kmEW):
    y_nedbor = (20-x)*(20-y)

fig = py.figure(figsize=(10,4))
ax = fig.add_axes((0.1,0.1, 0.8,0.8))
ax.scatter(X_km, y_nedbor)

poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X_km.reshape(-1,1))
# Train the model
model = LinearRegression()
model.fit(X_poly, y_nedbor)
y_nedborPredict = model.predict(X_poly)

ax.plot(X_km, y_nedborPredict, color="red")
ax.legend(["real", "regression line"])
ax.set_title("Nedbør fra Bergen Sentrum")
py.show()