from random import randint
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

fig = plt.figure(figsize=(4,4))
axButn = plt.axes((0.3, 0.3, 0.4, 0.4))
btn = Button(axButn, label="Random (1..1024)", color='lightgray', hovercolor='yellow')
n = 0
def on_plot(event):
    global n
    n = n + 1
    nrand = randint(0, 1024)
    axButn.set_title(f'Tallet Er : {nrand}')
    fig.suptitle(f"Clicket {n} ganger")
    plt.show()

btn.on_clicked(on_plot)
plt.show()










