
from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from matplotlib.widgets import TextBox

df = pd.read_csv('house_data.csv')
# Prepare plot
fig = plt.figure()
ax = fig.add_axes((.1, .2, .85, .75))
ax.set_title("Leiligheter i Bergen")
ax.set_xlabel('(m2)')
ax.set_ylabel('Standard (4=top,1=dårlig)')

# X = Input/Independent y = Output
X = df[['m2', 'Standard']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Standard avvik:", int(np.sqrt(mse)))
dfTest = pd.DataFrame(X_test)
dfTest["Estimated"] = [int(e) for e in y_pred]

def plot_bubble(m2,standard, priceM):
    cmap = colormaps['viridis']
    colorBack = cmap(priceM/20)
    colorPrice = 'white'
    if priceM > 12:
        colorPrice = 'black'
    ax.scatter(m2, standard, color=colorBack, s=priceM*100)
    ax.text(m2, standard, s=f"{priceM:.1f}", color=colorPrice, fontsize=12, ha='center', va='center')
    plt.draw()

def submit(exp):
    arr = np.fromstring(exp, sep=",")
    XS = np.array([arr])
    priceEst = model.predict(XS) / 1000_000
    plot_bubble(arr[0], arr[1], priceEst[0])

axbox = fig.add_axes((0.4, 0.03, 0.5, 0.06))
text_box = TextBox(axbox, "Angi m2, standard(1-4) : ", textalignment="center")
text_box.text_disp.set_fontsize(12)
text_box.label.set_fontsize(12)
text_box.on_submit(submit)
text_box.set_val("100,2")

estM = [e / 1e6 for e in dfTest["Estimated"]] # converter til mill
xT = dfTest['m2'].tolist()
yT = dfTest['Standard'].tolist()
for i in range(1, len(xT)):
    plot_bubble(xT[i], yT[i], estM[i])

plt.show()