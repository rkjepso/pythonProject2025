import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
dataLeil = [[50, 1, 2.5],   [20,1,1.0],         [70, 4, 4.0],
            [90, 3, 6.0],            [110, 2, 12.0]]
df = pd.DataFrame(columns=["m2","Std", "Price"], data=dataLeil)
X = df[['m2','Std']]
y = df['Price']  # Create and train the model
model = LinearRegression()
model.fit(X, y)
leiligheterEst = [[100,2],[200,4], [50,4]]

dfPred = pd.DataFrame(columns=["m2","Std"], data=leiligheterEst)
y_pred = model.predict(dfPred)
print("Leiligheter i Bergen sentrum")
for i, pris in enumerate(y_pred):
    print(f"{leiligheterEst[i][0]} estimert pris {pris:.1f}")

plt.show()
