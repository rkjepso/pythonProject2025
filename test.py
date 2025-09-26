import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data points
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
z = [5, 6, 7, 8, 9]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x, y, z, c='r', marker='o')  # 'c' sets color, 'marker' sets point style

# Add labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Show the plot
plt.show()