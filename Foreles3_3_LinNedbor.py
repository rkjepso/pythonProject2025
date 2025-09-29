
import matplotlib.pyplot as py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_km = np.linspace(0,50, 11) # konverterer [0,5..] til en kolonne
y_nedbor = np.array([2700,2500,2300,2700,2400,2300,2200,1800,2100,1600,1800])

fig = py.figure(figsize=(10,4))
ax = fig.add_axes((0.1,0.14, 0.8,0.8))
ax.scatter(X_km, y_nedbor, color="blue")

model = LinearRegression()
X_km = X_km.reshape(-1,1) # må konvertere til kolonne !!
model.fit(X_km, y_nedbor)
y_nedborPredict = model.predict(X_km)
# Plot the results
ax.scatter(X_km, y_nedborPredict, color="green")
ax.plot(X_km, y_nedborPredict, color="red")
ax.legend(["real", "predicted", "regression line"])
ax.set_xlabel("km fra sentrum")
ax.set_ylabel("mm nebor")
# Quality of the model
std = mean_squared_error(y_nedbor, y_nedborPredict)**0.5 # standard deviation/avvk
r_score = r2_score(y_nedbor, y_nedborPredict)
ax.set_title(f"Nedbør fra Bergen Sentrum STD:{std:.0f} R2:{r_score:.1f}")
py.show()