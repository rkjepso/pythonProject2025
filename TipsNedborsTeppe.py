import random

import numpy as np
import matplotlib.pyplot as plt
#Eksempel som viser grafisk to input (uavhengige) og en output (avhengig)
# Create meshgrid data
x = np.linspace(-1, 1, 15)
y = np.linspace(-1, 1, 15)
X, Y = np.meshgrid(x, y)

# Define Z as a function of X and Y
Z = np.cos(X)*np.cos(Y*0.7) * random.randrange(2000, 2500)

# Parameters for Gaussian noise
mean = 0
std_dev = 100  # Small standard deviation

# Generate noise of the same shape
noise = np.random.normal(mean, std_dev, Z.shape)

# Add noise to the existing array
Z = Z + noise

# Plot 3D surface with color gradient based on Z value
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Use a colormap to show the gradient
ax.grid(True)
surf = ax.plot_surface(X, Y, Z, cmap='viridis_r', edgecolor='none', alpha=0.5)
# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
# Axis labels and title

ax.set_xlabel('X - Vest/Øst')
ax.set_ylabel('Y - Nord/Sør')
ax.set_zlabel('Z - Nedbør')
ax.set_title('3D "Nedbørsteppe"')
ax.set_xticks([-1,0,1])
ax.set_yticks([-1,0,1])
# Add a color bar to show the mapping of colors to Z values
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Z value')

plt.tight_layout()
plt.show()