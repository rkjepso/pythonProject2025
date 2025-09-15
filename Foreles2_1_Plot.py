
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,31)
y = np.sin(x)

fig = plt.figure(figsize=(13, 7))

axis = fig.add_axes((.1, .1, .8, .8))
axis.set_title("Sin")
axis.plot(x, y)
plt.show()

