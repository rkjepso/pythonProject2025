
import matplotlib.pyplot as plt
import numpy as np

x:np.array = np.linspace(0,10,51) # array/tabell. Mye brukt
y:np.array = np.sin(x)
fig = plt.figure(figsize=(13, 7))

axis = fig.add_axes(rect=(.1, .1, .8, .8))
axis.set_title("y = sin(x)")
axis.set_xlabel("x verdier")
axis.set_ylabel("y verdier")
axis.plot(x, y)
plt.show()

