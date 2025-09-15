
import matplotlib.pyplot as py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_km = np.linspace(0,50, 11).reshape(-1,1) # konverterer [0,5..] til en kolonne
y_nedbor = np.array([2700,2500,2300,2700,2400,2300,2200,1800,2100,1600,1800])

fig = py.figure(figsize=(10,4))
ax = fig.add_axes((0.1,0.14, 0.8,0.8))
ax.scatter(X_km, y_nedbor)

#poly = PolynomialFeatures(degree=1)
#X_poly = poly.fit_transform(X_km.reshape(-1,1)) #from row to column reshape
# Train the model
model = LinearRegression()
model.fit(X_km, y_nedbor)
y_nedborPredict = model.predict(X_km)

ax.plot(X_km, y_nedborPredict, color="red")
ax.legend(["real", "regression line"])
ax.set_title("Nedbør fra Bergen Sentrum")
ax.set_xlabel("km fra sentrum")
ax.set_ylabel("mm nebor")
py.show()