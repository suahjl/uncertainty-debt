# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
from dotenv import load_dotenv
import ast
import time
from tqdm import tqdm

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
path_ceic = "./ceic/"

# %%
# I --- Data
df = pd.read_csv(
    path_output
    + "reg_quadrant_thresholdselection_fe_modwith_maxminepu_maxminstir_aiccsearch.csv"
)

# %%
# II --- Plot
# Create a grid of x1 and x2 values
x_values = np.linspace(df["hhdebt_ngdp_ref_threshold"].min(), df["hhdebt_ngdp_ref_threshold"].max(), len(df["hhdebt_ngdp_ref_threshold"]))
y_values = np.linspace(df["govdebt_ngdp_ref_threshold"].min(), df["govdebt_ngdp_ref_threshold"].max(), len(df["govdebt_ngdp_ref_threshold"]))
x_grid, y_grid = np.meshgrid(x_values, y_values)


# Interpolate the y values on the grid
z_grid = griddata(
    (df["hhdebt_ngdp_ref_threshold"], df["govdebt_ngdp_ref_threshold"]),
    df["aicc"],
    (x_grid, y_grid),
    method="linear",
)

# Plot the contour map and save to PNG
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.contour3D(x_grid, y_grid, z_grid, 50, cmap="viridis")

ax.set_xlabel("HH debt")
ax.set_ylabel("Government debt")
ax.set_zlabel("AICc")

# Save the plot as a PNG file
plt.savefig(path_output + "thresholdselection_aicc_contour.png", dpi=300)


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
