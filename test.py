import pandas as pd
from sklearn.linear_model import LinearRegression

dataHus = [[50, 1, 4.5],[70, 4, 4.0],[90, 3, 6.0],[110, 2, 12.0]]
df = pd.DataFrame(dataHus, columns =["m2","Std", "Price"])
X = df[['m2','Std']] # X = Input, Y = Output
y = df['Price']  # Create and train the model
model = LinearRegression()
model.fit(X, y)
leil = [[150,4], [140,1]]
y_pred = model.predict(leil)
print("Leiligheter i sentrum")
print(f"{leil[0]} est pris {y_pred[0]:.1f}")
print(f"{leil[1]} est pris {y_pred[1]:.1f}")

