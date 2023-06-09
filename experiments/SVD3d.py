# load json file
import json

import numpy as np

with open('/Users/gdk/Downloads/pour_milk.json') as f:
    data = json.load(f)

carton_positions = [data[str(frame)]["nodes"]["milk carton"]["pose"] for frame in range(len(data))]
cup_positions = [data[str(frame)]["nodes"]["cup"]["pose"] for frame in range(len(data))]

carton_positions = np.stack(carton_positions)
cup_positions = np.stack(cup_positions)

diff_positions = carton_positions - cup_positions

# draw 3d points diff_positions
import matplotlib.pyplot as plt

# draw points in 3d


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(diff_positions[:50, 0], diff_positions[:50, 1], diff_positions[:50, 2], c='g')
ax.scatter(diff_positions[50:100, 0], diff_positions[50:100, 1], diff_positions[50:100, 2], c='b')
ax.scatter(diff_positions[100:150, 0], diff_positions[100:150, 1], diff_positions[100:150, 2], c='k')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# draw a red point at origin
ax.scatter(0, 0, 0, c='r', marker='o')

plt.show()

# def fit_plane_through_origin(points):
#     # Step 1: Calculate the centroid
#     centroid = np.mean(points, axis=0)
#
#     # Step 2: Translate points and centroid to the origin
#     translated_points = points - centroid
#
#     # Step 3: Solve for the normal vector
#     p1, p2 = translated_points[:2]
#     normal = np.cross(p1, p2)
#
#     # Step 4: Normalize the normal vector
#     normal /= np.linalg.norm(normal)
#
#     return normal
#
#
# def fit_least_squares_plane(points):
#     # Construct the design matrix
#     X = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
#
#     # Solve for the plane coefficients using least squares
#     coeffs, _, _, _ = np.linalg.lstsq(X, points[:, 2], rcond=None)
#
#     # Extract the plane coefficients
#     A, B, C = coeffs
#
#     # Normalize the coefficients
#     norm = np.sqrt(A**2 + B**2 + 1)
#     A /= norm
#     B /= norm
#     C /= norm
#
#     return A, B, C
#
#
# # Fit plane through origin
# plane_normal = fit_plane_through_origin(diff_positions)
#
# # Create a meshgrid to represent the plane
# x_range = np.linspace(-0.2, 0., 10)
# y_range = np.linspace(-0.3, 0, 10)
# X, Y = np.meshgrid(x_range, y_range)
#
# # Calculate the corresponding z-values for the plane
# Z = (-plane_normal[0] * X - plane_normal[1] * Y) / plane_normal[2]
#
# # Plot the points and the plane
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# # Plot the points
# # ax.scatter(diff_positions[:, 0], diff_positions[:, 1], diff_positions[:, 2], color='red')
#
# # Plot the plane
# # ax.plot_surface(X, Y, Z, alpha=0.5)
#
# ax.scatter(diff_positions[:50, 0], diff_positions[:50, 1], diff_positions[:50, 2], c='g')
# ax.scatter(diff_positions[50:100, 0], diff_positions[50:100, 1], diff_positions[50:100, 2], c='b')
# ax.scatter(diff_positions[100:150, 0], diff_positions[100:150, 1], diff_positions[100:150, 2], c='k')
#
# # draw a red point at origin
# ax.scatter(0, 0, 0, c='r', marker='o')
#
# A, B, C = fit_least_squares_plane(diff_positions)
#
# # Create a meshgrid to represent the plane
# x_range = np.linspace(-0.2, 0., 10)
# y_range = np.linspace(-0.3, 0, 10)
# X, Y = np.meshgrid(x_range, y_range)
#
# # Calculate the corresponding z-values for the plane
# Z = (-A * X - B * Y) / C
#
# # ax.plot_surface(X, Y, Z, alpha=0.5)
#
#
# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Fitted Plane')
#
# # Show the plot
# plt.show()

points = diff_positions
U, S, V = np.linalg.svd(points)
normal = V[2]

projected_points = points - ((np.dot(points, normal)) / np.linalg.norm(normal)**2)[:, np.newaxis] * normal


A, B, C = normal
xx, yy = np.meshgrid(np.linspace(-0.2, 0., 10), np.linspace(-0.2, 0., 10))
zz = (-A * xx - B * yy) / C
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue')

# Plot the plane
ax.plot_surface(xx, yy, zz, alpha=0.5)

ax.scatter(0, 0, 0, c='r', marker='o')

ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], color='purple')



# Set labels and display the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


x_coordinates = np.dot(projected_points, V[0])
y_coordinates = np.dot(projected_points, V[1])

plt.scatter(x_coordinates[:50], y_coordinates[:50], color='blue')
plt.scatter(x_coordinates[50:100], y_coordinates[50:100], color='green')
plt.scatter(x_coordinates[100:150], y_coordinates[100:150], color='red')

plt.xlim(-0.3, 0)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Coordinates on the Fitted Plane')
plt.show()
