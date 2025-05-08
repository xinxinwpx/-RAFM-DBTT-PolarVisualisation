#A Data-Driven Machine Learning Model for Radiation-Induced DBTT Shifts in RAFM Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import BoundaryNorm

# Set font
plt.rcParams['font.family'] = 'Times New Roman'

# === Read data ===
df = pd.read_excel('CrPython.xlsx')  # Replace with your file path
angle_step = 15
radius_step = 0.1  # Dose precision

# Discretize angle and radius
df['theta_deg'] = np.round(df['Tirr (℃)'] / angle_step) * angle_step
df['radius_bin'] = np.round(df['Dose (dap)'] / radius_step) * radius_step
df['theta_rad'] = np.deg2rad(df['theta_deg'])

# Raw data
theta = df['theta_rad'].values
radius = df['radius_bin'].values
values = df['DBTT (℃)'].values

# Create interpolation grid (radius from 2 to 10)
theta_grid = np.linspace(0, 2 * np.pi, 360)
radius_grid = np.linspace(2, 10, 300)
T, R = np.meshgrid(theta_grid, radius_grid)

# Interpolation (nearest to avoid blank spots)
points = np.stack((theta, radius), axis=-1)
grid_points = np.stack((T.ravel(), R.ravel()), axis=-1)
Z = griddata(points, values, grid_points, method='nearest').reshape(T.shape)

# Color map settings
levels = np.arange(-50, 310, 25)
cmap = plt.get_cmap('RdYlGn_r')
norm = BoundaryNorm(levels, cmap.N)

# === Plot ===
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 8))
c = ax.contourf(T, R, Z, levels=levels, cmap=cmap, norm=norm, extend='both')

# Colorbar (simplified ticks)
cb = fig.colorbar(c, ax=ax, pad=0.1, ticks=[-50, 0, 50, 100, 150, 200, 250, 300])
cb.set_label('DBTT (℃)', fontsize=22)
cb.ax.tick_params(labelsize=20)
for label in cb.ax.get_yticklabels():
    label.set_weight('bold')

# Polar angle settings
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
custom_degrees = [0, 60, 120, 180, 240, 300]
custom_labels = ['0°C', '75°C', '150°C', '225°C', '300°C', '375°C']
ax.set_xticks(np.deg2rad(custom_degrees))
ax.set_xticklabels(custom_labels, fontsize=22, fontweight='bold')
ax.tick_params(axis='x', pad=19)

# Radius settings (from 2 to 10, hide labels)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels([])  # Hide labels, show gridlines only

# Gridline style
ax.yaxis.grid(True, linestyle='--', color='gray', linewidth=1.2)
ax.xaxis.grid(True, linestyle='--', color='gray', linewidth=1.2)

# Layout adjustment
plt.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.1)

# Save figure
plt.savefig('DBTT_Polar_Radius2to10_HideYLabel.png', dpi=600, bbox_inches='tight')
plt.show()
