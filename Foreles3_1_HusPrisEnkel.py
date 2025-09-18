import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
solgtLeil = [[50, 1, 2.5],   [60, 2, 4.5], [30,1,1.5],         [70, 4, 4.0],
            [90, 3, 6.0],   [110, 2, 12.0], [50, 4, 3.5]]
df = pd.DataFrame(columns=["m2","Std", "Price"], data=solgtLeil)
X = df[['m2','Std']]
y = df['Price']
model = LinearRegression()  # Create model
model.fit(X, y)             # Train
estLeil = [[100,2],[100,4], [220,1], [200,4], [50,4]]

dfPred = pd.DataFrame(columns=["m2","Std"], data=estLeil)
y_pred = model.predict(dfPred)
print("Leiligheter i Bergen sentrum Standard(4=best) ")
for i, pris in enumerate(y_pred):
    print(f"m2 {estLeil[i][0]:>3} Standard {estLeil[i][1]} Estimert pris {pris:.1f}")


