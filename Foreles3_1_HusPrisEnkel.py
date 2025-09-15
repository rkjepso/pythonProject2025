import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
dataLeil = [[50, 1, 4.5],
            [70, 4, 5.0],
            [90, 3, 6.0],
            [110, 2, 9.0]]
df = pd.DataFrame(columns=["m2","Std", "Price"], data=dataLeil)
# X = Input/Independent y = Output
X = df[['m2','Std']]
y = df['Price']  # Create and train the model
model = LinearRegression()
model.fit(X, y)  # Build a model based on X (table) and y (1d table/vector)

leil1 = [150,4]  # Predict some data
leil2 = [140,1]
dfPred = pd.DataFrame(columns=["m2","Std"], data=[leil1, leil2])
y_pred = model.predict(dfPred)
print("Leiligheter i Bergen sentrum")
print(f"{leil1} estimert pris {y_pred[0]:.1f}")
print(f"{leil2} estimert pris {y_pred[1]:.1f}")
plt.show()
