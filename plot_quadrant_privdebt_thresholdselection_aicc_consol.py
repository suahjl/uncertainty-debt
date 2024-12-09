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

# %%
# TEMPORARY WORKAROUND FOR TCL AND TK FOR PY 3.13.0
os.environ["TCL_LIBRARY"] = (
    r"C:\Users\suahj\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
)
os.environ["TK_LIBRARY"] = (
    r"C:\Users\suahj\AppData\Local\Programs\Python\Python313\tcl\tk8.6"
)


# %%
# I --- Do everything function
def do_everything(file_suffixes, uncertainty_variable, mp_variable):
    # Load data
    df = pd.read_csv(
        path_output
        + "reg_quadrant_privdebt_thresholdselection_"
        + file_suffixes
        + "fe_"
        + "modwith_"
        + uncertainty_variable
        + "_"
        + mp_variable
        + "_aiccsearch"
        + ".csv"
    )

    # Save optimal threshold values
    privdebt_optimal = df.loc[
        df["aicc"] == df["aicc"].min(),
        "privdebt_ngdp_ref_threshold",
    ].reset_index(drop=True)[0]
    govdebt_optimal = df.loc[
        df["aicc"] == df["aicc"].min(),
        "govdebt_ngdp_ref_threshold",
    ].reset_index(drop=True)[0]
    aicc_optimal = df["aicc"].min()

    # Create a grid of x1 and x2 values
    x_values = np.linspace(
        df["privdebt_ngdp_ref_threshold"].min(),
        df["privdebt_ngdp_ref_threshold"].max(),
        len(df["privdebt_ngdp_ref_threshold"]),
    )
    y_values = np.linspace(
        df["govdebt_ngdp_ref_threshold"].min(),
        df["govdebt_ngdp_ref_threshold"].max(),
        len(df["govdebt_ngdp_ref_threshold"]),
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Interpolate the y values on the grid
    z_grid = griddata(
        (df["privdebt_ngdp_ref_threshold"], df["govdebt_ngdp_ref_threshold"]),
        df["aicc"],
        (x_grid, y_grid),
        method="linear",
    )

    # Gennerate matplotlib 3D object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Add the contour plot
    ax.contour3D(x_grid, y_grid, z_grid, 50, cmap="viridis")

    # Add a diamond marker at the intersection
    # ax.scatter(
    #     privdebt_optimal,
    #     govdebt_optimal,
    #     aicc_optimal,
    #     color="red",
    #     s=50,
    #     marker="D",
    #     label="Private debt: "
    #     + str(privdebt_optimal)
    #     + "% of GDP"
    #     + "\nGovernment debt: "
    #     + str(govdebt_optimal)
    #     + "% of GDP",
    #     zorder=100,
    # )

    # Labels
    ax.set_xlabel("Private debt")
    ax.set_ylabel("Government debt")
    ax.set_zlabel("AICc")

    ax.set_title(
        "Grid search of optimal debt thresholds by AICc-minimisation\n"
        + "Optimal thresholds at\n"
        + "Private debt: "
        + str(privdebt_optimal)
        + "% of GDP"
        + "; Government debt: "
        + str(govdebt_optimal)
        + "% of GDP"
    )

    # Save the plot as a PNG file
    plt.savefig(
        path_output
        + "thresholdselection_"
        + file_suffixes
        + "fe_"
        + "modwith_"
        + uncertainty_variable
        + "_"
        + mp_variable
        + "_aicc_contour.png",
        dpi=300,
    )


# %%
# II --- Do everything
# EPU
do_everything(
    file_suffixes="", uncertainty_variable="maxminepu", mp_variable="maxminstir"
)

# WUI
do_everything(
    file_suffixes="", uncertainty_variable="maxminwui", mp_variable="maxminstir"
)

# UCT
do_everything(
    file_suffixes="", uncertainty_variable="maxminuct", mp_variable="maxminstir"
)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
