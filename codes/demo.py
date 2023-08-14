import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Given function
def gravity_on_ellipse(latitude, a, b):
    g_e = 9.78033
    g_p = 9.8322
    k = a**2 * g_e - b**2 * g_p
    phi = np.radians(latitude)
    g_phi = g_e * (1 + k * np.cos(phi)**2 / a**2)
    return g_phi

# Define the equatorial and polar radii for Earth (in meters)
a = 6378137.0
b = 6356752.3

# Create an array of latitudes from -90 to 90
latitudes = np.linspace(-90, 90, 500)
test_y = 9.9
test_x = 45
gravities = gravity_on_ellipse(latitudes, a, b)

# Normalize the gravities for colormap
norm = plt.Normalize(gravities.min(), gravities.max())

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(latitudes, gravities, c=gravities, cmap=cm.jet, s=15, norm=norm)
ax.scatter(test_x, test_y, s=15)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Gravitational acceleration (m/s^2)', rotation=270, labelpad=20)
ax.set_title('Gravitational Acceleration vs. Latitude')
ax.set_xlabel('Latitude (degrees)')
ax.set_ylabel('Gravitational Acceleration (m/s^2)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
