import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
# list of lists of house data
dataLeil = [[50, 1, 2.5],            [70, 4, 4.0],
            [90, 3, 6.0],            [110, 2, 12.0]]
arr = np.array(dataLeil)
df = pd.DataFrame(columns=["m2","Std", "Price"], data=arr)
X = df[['m2','Std']] # 2D tabell
y = df['Price']  # Create and train the model
model = LinearRegression()
model.fit(X, y)
leil1 = [150,4]
leil2 = [140,1]
dfPred = pd.DataFrame(columns=["m2","Std"], data=[leil1, leil2])
y_pred = model.predict(dfPred)
print("Leiligheter i Bergen sentrum")
print(f"{leil1} estimert pris {y_pred[0]:.1f}")
print(f"{leil2} estimert pris {y_pred[1]:.1f}")
plt.show()
