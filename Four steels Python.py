#A Data-Driven Machine Learning Model for Radiation-Induced DBTT Shifts in RAFM Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import BoundaryNorm, ListedColormap

# Set font
plt.rcParams['font.family'] = 'Times New Roman'

# === Custom colormap (your provided segmented red-blue color bands) ===
levels = [-100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 160, 205]
colors = ['#313695', '#4575b4', '#74add1', '#abd9e9',
          '#e0f3f8', '#ffffbf', '#fee090', '#fdae61',
          '#f46d43', '#d73027', '#a50026', '#67001f']
cmap = ListedColormap(colors)
norm = BoundaryNorm(levels, ncolors=cmap.N)

# === Read data ===
df = pd.read_excel('steeldata.xlsx')  # Replace with your file path
angle_step = 15
radius_step = 1  # Adapt for radius 0–50

# Discretize angle and radius
df['theta_deg'] = np.round(df['Tirr (℃)'] / angle_step) * angle_step
df['radius_bin'] = np.round(df['Dose (dap)'] / radius_step) * radius_step
df['theta_rad'] = np.deg2rad(df['theta_deg'])

# Raw data
theta = df['theta_rad'].values
radius = df['radius_bin'].values
values = df['DBTT (℃)'].values

# Create interpolation grid
theta_grid = np.linspace(0, 2 * np.pi, 360)
radius_grid = np.linspace(0, 50, 500)
T, R = np.meshgrid(theta_grid, radius_grid)

# Interpolation
points = np.stack((theta, radius), axis=-1)
grid_points = np.stack((T.ravel(), R.ravel()), axis=-1)
Z = griddata(points, values, grid_points, method='nearest').reshape(T.shape)

# === Plot ===
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 8))
c = ax.contourf(T, R, Z, levels=levels, cmap=cmap, norm=norm, extend='both')

# Colorbar (optional ticks)
cb = fig.colorbar(c, ax=ax, pad=0.1, ticks=[-100, -50, 0, 50, 100, 150, 205])
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

# Radius settings (0–50), hide labels
ax.set_yticks([10, 20, 30, 40, 50])
ax.set_yticklabels([])

# Grid style
ax.yaxis.grid(True, linestyle='--', color='gray', linewidth=1.2)
ax.xaxis.grid(True, linestyle='--', color='gray', linewidth=1.2)

# Layout optimization
plt.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.1)

# Save figure
plt.savefig('DBTT_Polar_CustomLevelsColormapornlPythonPlot.png', dpi=600, bbox_inches='tight')
plt.show()
