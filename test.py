import pandas
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

dataHus = [[50, 1, 4.5],[70, 4, 4.0],[90, 3, 6.0],[110, 2, 12.0]]
df = pandas.DataFrame(dataHus, columns =["m2","Std", "Price"])
# X = Input/Independent y = Output
X = df[['m2','Std']]
y = df['Price']  # Create and train the model
model = LinearRegression()
model.fit(X, y)
lei1 = [150,4]
lei2 = [140,1]
y_pred = model.predict([lei1,lei2])
print("Leiligheter i sentrum")
print(f"{lei1} est pris {y_pred[0]:.1f}")
print(f"{lei2} est pris {y_pred[1]:.1f}")
plt.show()
