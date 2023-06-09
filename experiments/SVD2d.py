import json

import numpy as np
import matplotlib.pyplot as plt

with open('/Users/gdk/Downloads/pour_milk.json') as f:
    data = json.load(f)

carton_positions = [data[str(frame)]["nodes"]["milk carton"]["pose"] for frame in range(len(data))]
cup_positions = [data[str(frame)]["nodes"]["cup"]["pose"] for frame in range(len(data))]

carton_positions = np.stack(carton_positions)
cup_positions = np.stack(cup_positions)

diff_positions = carton_positions - cup_positions

x = diff_positions[:, 0]
y = diff_positions[:, 1]

A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(m, c)

# Generate points for the line
line_x = np.linspace(0, max(x), 100)
line_y = m*line_x + c

# Plot the points and the line
plt.scatter(x, y, label='Data Points')
plt.plot(line_x, line_y, 'r', label='Least Squares Line')



points = diff_positions[:, :2]
# centered_points = points - np.mean(points, axis=0)

# Perform SVD
U, s, Vt = np.linalg.svd(points)

# Calculate the direction vector of the best-fitting line
direction = Vt[-1]
print(Vt)

# Calculate the intercept of the line
intercept = np.mean(points, axis=0)

# Define the line equation
x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
y = direction[1] / direction[0] * (x - intercept[0]) + intercept[1]

plt.plot(x, y, 'g', label='SVD Line')

# A = np.vstack((x, np.ones(len(x)))).T
# U, S, VT = np.linalg.svd(A)
# S_inv = np.zeros((A.shape[1], A.shape[0]))
# S_inv[:A.shape[1], :A.shape[1]] = np.linalg.inv(np.diag(S))
# print(np.linalg.inv(np.diag(S)).shape, S_inv.shape, U.shape, VT.shape)
#
# A_plus = VT.T.dot(S_inv).dot(U.T)
#
# m, c = A_plus.dot(y)

# coefficients = V.T @ np.linalg.pinv(S) @ U.T @ y
# print(coefficients)
# m, b = coefficients[0], coefficients[1]
# line_x = np.linspace(0, max(x), 100)
# line_y = m*line_x + c
# print(m, c)

direction = Vt[0]
print(Vt)

# Calculate the intercept of the line
# intercept = np.mean(points, axis=0)
#
# # Define the line equation
x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
y = direction[1] / direction[0] * (x - intercept[0]) + intercept[1]

plt.plot(x, y, 'purple', label='SVD Line perpendicular')
# plt.plot([0, -direction[1]], [0, direction[0]], 'g', label='dominant direction')


plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
plt.show()