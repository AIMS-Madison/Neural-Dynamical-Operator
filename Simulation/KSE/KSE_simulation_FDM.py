import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 20.0                 # Length of the domain
N = 128                  # Number of grid points
dx = L / N               # Grid spacing
dt = 0.001               # Reduced time step
T = 10.0                 # Maximum time
Nt = int(T / dt)         # Number of time steps
max_u_value = 1e5        # Arbitrary large value for stability check

x = np.linspace(0, L, N, endpoint=False)  # Spatial grid
u = np.sin(x)            # Initial condition

# Helper function for periodic boundary conditions
def periodic_diff(array, order):
    if order == 1:
        return (np.roll(array, -1) - np.roll(array, 1)) / (2 * dx)
    elif order == 2:
        return (np.roll(array, -1) - 2 * array + np.roll(array, 1)) / dx**2
    elif order == 4:
        return (np.roll(array, -2) - 4 * np.roll(array, -1) + 6 * array - 4 * np.roll(array, 1) + np.roll(array, 2)) / dx**4
    else:
        raise ValueError("Order not supported")

# Time integration using finite difference
for t in range(Nt):
    u_xx = periodic_diff(u, 2)
    u_xxxx = periodic_diff(u, 4)
    u_x = periodic_diff(u, 1)

    u_new = u - dt * (u * u_x + u_xx + u_xxxx)

    # Stability check
    if np.max(np.abs(u_new)) > max_u_value:
        print(f"Solution became unstable at t = {t*dt}")
        break

    u = u_new

# Plotting the result
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of KSE')
plt.grid(True)
plt.show()
