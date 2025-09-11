
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
df = pd.read_csv('house_data.csv')

# Prepare plot
fig = plt.figure()
ax = fig.add_axes((.1, .2, .85, .75))
ax.set_title("Huspriser i Bergen")
ax.set_xlabel('Size (m2)')
ax.set_ylabel('Standard')

# X = Input/Independent y = Output
X = df[['m2', 'Standard']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Standard avvik:", int(np.sqrt(mse)))

dfTest = pd.DataFrame(X_test)
dfTest["Estimated"] = [int(e) for e in y_pred]
estM = dfTest["Estimated"] / 1000_000

def plot_bubbles():
    colors = estM
    scatter = ax.scatter(dfTest['m2'], dfTest['Standard'],
                         c=colors, cmap='plasma', s=estM*60)
    xT = dfTest['m2'].tolist()
    yT = dfTest['Standard'].tolist()
    eT = estM.tolist()
    for i in range(1, len(xT)):
        ax.text(xT[i], yT[i], s=f"{eT[i]:.1f}", color='lightgrey' if eT[i] < 10 else 'black', fontsize=12, ha='center', va='center')
    cbar = plt.colorbar(scatter, ax=ax, pad=0.05) # Add color bar
    cbar.set_label('Beløp (millioner)')

plot_bubbles()

def submit(expression):
    ax.set_title(expression)
    plt.draw()

axbox = fig.add_axes((0.3, 0.05, 0.4, 0.075))
text_box = TextBox(axbox, "Angi m2, standard(1-4) : ", textalignment="center")
text_box.on_submit(submit)
text_box.set_val("100,2")

plt.show()