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
# I --- Functions
def do_everything(
    list_mp_variables: list[str],
    list_uncertainty_variables: list[str],
    responses_to_plot: list[str],
    xyz_labels: list[str],
    z_quantiles_to_fix: list[float],
    file_suffixes: str,  # format: "abc_" or ""
):
    for mp_variable in tqdm(list_mp_variables):
        for uncertainty_variable in tqdm(list_uncertainty_variables):
            for shock in [uncertainty_variable, mp_variable]:
                for response in responses_to_plot:
                    grid_full = pd.read_parquet(
                        path_output
                        + "octant_interaction_panellp_grid_impactsize_"
                        + file_suffixes
                        + "irf_"
                        + "modwith_"
                        + uncertainty_variable
                        + "_"
                        + mp_variable
                        + "_"
                        + "shock"
                        + shock
                        + "_"
                        + "response"
                        + response
                        + ".parquet"
                    )

                    plane_count = 0
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")
                    for z_quantile in z_quantiles_to_fix:
                        grid = grid_full[
                            grid_full[xyz_labels[2]]
                            == grid_full[xyz_labels[2]].quantile(
                                z_quantile, interpolation="nearest"
                            )
                        ].copy()
                        xi = np.linspace(
                            grid[xyz_labels[0]].min(), grid[xyz_labels[0]].max(), 150
                        )
                        yi = np.linspace(
                            grid[xyz_labels[1]].min(), grid[xyz_labels[1]].max(), 150
                        )
                        xi, yi = np.meshgrid(xi, yi)

                        # Interpolate the z values onto the grid
                        zi = griddata(
                            (grid[xyz_labels[0]], grid[xyz_labels[1]]),
                            grid["impact"],
                            (xi, yi),
                            method="linear",
                        )
                        # Add contour map to figure
                        ax.contour3D(
                            xi,
                            yi,
                            zi,
                            levels=150,
                            cmap="viridis",
                            alpha=0.2 + plane_count * 0.1,
                        )
                        # Next
                        plane_count += 1
                    # Configure plot
                    ax.set_xlabel(xyz_labels[0])
                    ax.set_ylabel(xyz_labels[1])
                    ax.set_zlabel(xyz_labels[2])
                    plt.title(
                        "Impact of "
                        + shock
                        + " on "
                        + response
                        + "\nFor "
                        + xyz_labels[2]
                        + ": ["
                        + ",".join([str(i) for i in z_quantiles_to_fix])
                        + "]"
                    )
                    # Save the plot as a PNG file
                    plt.savefig(
                        path_output
                        + "octant_interaction_panellp_grid_impactsize_contour_"
                        + file_suffixes
                        + "irf_"
                        + "modwith_"
                        + uncertainty_variable
                        + "_"
                        + mp_variable
                        + "_"
                        + "shock"
                        + shock
                        + "_"
                        + "response"
                        + response
                        + ".png",
                        dpi=300,
                    )
                    # Close to save memory
                    plt.close()


# %%
# II --- Some objects for quick ref later
cols_endog_long = [
    "hhdebt",  # _ngdp
    "corpdebt",  # _ngdp
    "govdebt",  # _ngdp
    "gdp",  # urate gdp
    "corecpi",  # corecpi cpi
    "reer",
]
cols_endog_short = [
    # "hhdebt",  # _ngdp
    # "corpdebt",  # _ngdp
    # "govdebt",  # _ngdp
    "gdp",  # urate gdp
    "corecpi",  # corecpi cpi
    "reer",
]
cols_endog_essential = ["gdp", "corecpi"]
cols_threshold_hh_gov_epu = ["hhdebt_ngdp_ref", "govdebt_ngdp_ref", "epu_ref"]
cols_threshold_hh_gov_wui = ["hhdebt_ngdp_ref", "govdebt_ngdp_ref", "wui_ref"]
epu_quantiles_to_fix = [0, 0.2, 0.4, 0.6, 0.8, 1]  # 0, 100, 200, 300, 400, 500
wui_quantiles_to_fix = [0, 0.2, 0.4, 0.6, 0.8, 1]  # normalised as 0 to 1

# %%
# III --- Do everything with EPU as uncertainty shock
# STIR
do_everything(
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    responses_to_plot=cols_endog_essential,
    xyz_labels=cols_threshold_hh_gov_epu,
    z_quantiles_to_fix=epu_quantiles_to_fix,
    file_suffixes="",
)
# STIR (reduced)
do_everything(
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    responses_to_plot=cols_endog_essential,
    xyz_labels=cols_threshold_hh_gov_epu,
    z_quantiles_to_fix=epu_quantiles_to_fix,
    file_suffixes="reduced_",
)
# M2
do_everything(
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminepu"],
    responses_to_plot=cols_endog_essential,
    xyz_labels=cols_threshold_hh_gov_epu,
    z_quantiles_to_fix=epu_quantiles_to_fix,
    file_suffixes="m2_",
)
# M2 (reduced)
do_everything(
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminepu"],
    responses_to_plot=cols_endog_essential,
    xyz_labels=cols_threshold_hh_gov_epu,
    z_quantiles_to_fix=epu_quantiles_to_fix,
    file_suffixes="m2_reduced_",
)

# %%
# III.B --- Do everything with WUI as uncertainty shock
# STIR
do_everything(
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminwui"],
    responses_to_plot=cols_endog_essential,
    xyz_labels=cols_threshold_hh_gov_wui,
    z_quantiles_to_fix=wui_quantiles_to_fix,
    file_suffixes="",
)

# STIR (reduced)
do_everything(
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminwui"],
    responses_to_plot=cols_endog_essential,
    xyz_labels=cols_threshold_hh_gov_wui,
    z_quantiles_to_fix=wui_quantiles_to_fix,
    file_suffixes="reduced_",
)

# M2
do_everything(
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminwui"],
    responses_to_plot=cols_endog_essential,
    xyz_labels=cols_threshold_hh_gov_wui,
    z_quantiles_to_fix=wui_quantiles_to_fix,
    file_suffixes="m2_",
)

# M2 (reduced)
do_everything(
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminwui"],
    responses_to_plot=cols_endog_essential,
    xyz_labels=cols_threshold_hh_gov_wui,
    z_quantiles_to_fix=wui_quantiles_to_fix,
    file_suffixes="m2_reduced_",
)


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
