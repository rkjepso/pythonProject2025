import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'Size (m2)': [100, 150, 200, 120, 180],
    'House Price': [150000, 250000, 350000, 180000, 300000],
    'Price per m2': [1500, 1666.67, 1750, 1500, 1666.67]
})

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Set black background for axes
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Scatter plot with plasma colormap
scatter = ax.scatter(df['Size (m2)'], df['House Price'], df['Price per m2'],
                     c=df['Price per m2'], cmap='plasma', s=100)

# Add color bar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Price per m2')

# Label axes
ax.set_xlabel('Size (m2)', color='white')
ax.set_ylabel('House Price', color='white')
ax.set_zlabel('Price per m2', color='white')
ax.set_title('3D Scatter Plot with Black Axis Background', color='white')

# Set tick label colors
ax.tick_params(colors='white')

plt.show()
