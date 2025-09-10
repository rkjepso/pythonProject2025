
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
df = pd.read_csv('house_data.csv')

# X = Input/Independent y = Output
X = df[['m2', 'Standard']]
y = df['Price']
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

dfTest = pd.DataFrame(X_test)
dfTest["Estimated"] = [int(e) for e in y_pred]
print(dfTest)

# Output results
print("Standard avvik:", int(np.sqrt(mse)))

# Create 3D plot
fig = plt.figure()
ax = fig.add_axes((.1, .1, .85, .85))
# Label axes
ax.set_title("Huspriser i Bergen")
ax.set_xlabel('Size (m2)')
ax.set_ylabel('Standard')

#ax.set_zlabel('Price')
estM = dfTest["Estimated"] / 1000_000
colors = estM
#labels = [10 for p in dfTest["Estimated"]]
scatter = ax.scatter(dfTest['m2'], dfTest['Standard'],
                     c=colors, cmap='plasma', s=estM*60)
xT = dfTest['m2'].tolist()
yT = dfTest['Standard'].tolist()
eT = estM.tolist()
for i, y in enumerate(estM):
    ax.text(xT[i], yT[i], s=f"{eT[i]:.1f}", color='lightgrey' if eT[i] < 10 else 'black', fontsize=12, ha='center', va='center')
# Add color bar
cbar = plt.colorbar(scatter, ax=ax, pad=0.05)
cbar.set_label('Beløp (millioner)')

plt.show()
