# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles
import localprojections as lp
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast
from tabulate import tabulate
import ruptures as rpt
from chow_test import chow_test
import warnings
import plotly.graph_objects as go
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import itertools

time_start = time.time()


# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
path_ceic = "./ceic/"
tel_config = os.getenv("TEL_CONFIG")
pd.options.mode.chained_assignment = None
warnings.filterwarnings(
    "ignore"
)  # MissingValueWarning when localprojections implements shift operations


# %%
# I --- Do everything function
def do_everything_octant_interaction_panellp(
    cols_endog_after_shocks: list[str],
    cols_all_exog: list[str],
    list_mp_variables: list[str],
    list_uncertainty_variables: list[str],
    cols_state_dependency: list[str],
    grid_state_dependency_ranges: list[list[float]],
    state_dependency_nice_for_title: str,  # HH debt, Gov debt
    countries_drop: list[str],
    file_suffixes: str,  # format: "abc_" or ""
    beta_values_to_simulate: list[list[float]],
    irf_colours_for_each_beta: list[str],
    t_start: date = date(1990, 1, 1),
    t_end: date = None,
    input_df_suffix="yoy",
):
    # Nested functions
    def check_balance_timing(input):
        min_quarter_by_country = input.copy()
        min_quarter_by_country = min_quarter_by_country.dropna(axis=0)
        min_quarter_by_country = (
            min_quarter_by_country.groupby("country")["quarter"].min().reset_index()
        )
        print(tabulate(min_quarter_by_country, headers="keys", tablefmt="pretty"))

    def check_balance_endtiming(input):
        max_quarter_by_country = input.copy()
        max_quarter_by_country = max_quarter_by_country.dropna(axis=0)
        max_quarter_by_country = (
            max_quarter_by_country.groupby("country")["quarter"].max().reset_index()
        )
        print(tabulate(max_quarter_by_country, headers="keys", tablefmt="pretty"))

    def generate_n_interactions(df: pd.DataFrame, labels_to_be_interacted: list[str]):
        # A --- Create labels
        result = []
        # Add individual elements
        result.extend(labels_to_be_interacted)
        # Add pairwise combinations
        result.extend(
            "_".join(pair) for pair in combinations(labels_to_be_interacted, 2)
        )
        # Add triplet combinations
        result.extend(
            "_".join(triplet) for triplet in combinations(labels_to_be_interacted, 3)
        )

        # Add quartet combinations
        result.extend(
            "_".join(quartet) for quartet in combinations(labels_to_be_interacted, 4)
        )

        # B --- Create new columns in df
        # Loop over pairs
        for pair in combinations(labels_to_be_interacted, 2):
            col_name = "_".join(pair)
            df[col_name] = df[pair[0]] * df[pair[1]]

        # Loop over trios
        for triple in combinations(labels_to_be_interacted, 3):
            col_name = "_".join(triple)
            df[col_name] = df[triple[0]] * df[triple[1]] * df[triple[2]]

        # Loop over quartets
        for quartet in combinations(labels_to_be_interacted, 4):
            col_name = "_".join(quartet)
            df[col_name] = (
                df[quartet[0]] * df[quartet[1]] * df[quartet[2]] * df[quartet[3]]
            )

        # C --- Output
        return df, result

    def compute_grid_octant_average_effects(
        input: pd.DataFrame,
        response_variable: str,  # which IRF?
        xyz_labels: list[str],  # [x, y, z]
        xyz_ranges: list[list[float]],  # for v2, v3, v4
        shock_size: float,  # size of shock of interest (v1)
        grid_size: int,
        int1_label: str,
        int12_label: str,
        int13_label: str,
        int14_label: str,
        int123_label: str,
        int124_label: str,
        int134_label: str,
        int1234_label: str,
        n_periods_to_average_over: int,
    ):
        # prelims
        irf = input.copy()  # must be output from LP modules
        # generate average irf
        irf = irf[irf["Horizon"] < n_periods_to_average_over]  # e.g., 4 ==> 0,1,2,3
        irf = (
            irf.groupby(["Shock", "Response"])[["Mean"]].mean().reset_index(drop=False)
        )
        # create grid of values of x, y and z
        x_values = np.linspace(xyz_ranges[0][0], xyz_ranges[0][1], grid_size)
        y_values = np.linspace(xyz_ranges[1][0], xyz_ranges[1][1], grid_size)
        z_values = np.linspace(xyz_ranges[2][0], xyz_ranges[2][1], grid_size)
        grid = itertools.product(x_values, y_values, z_values)
        grid = pd.DataFrame(grid, columns=[xyz_labels[0], xyz_labels[1], xyz_labels[2]])

        # Extract betas
        b1 = irf.loc[
            ((irf["Shock"] == int1_label) & (irf["Response"] == response_variable)),
            "Mean",
        ].reset_index(drop=True)[0]
        b12 = irf.loc[
            ((irf["Shock"] == int12_label) & (irf["Response"] == response_variable)),
            "Mean",
        ].reset_index(drop=True)[0]
        b13 = irf.loc[
            ((irf["Shock"] == int13_label) & (irf["Response"] == response_variable)),
            "Mean",
        ].reset_index(drop=True)[0]
        b14 = irf.loc[
            ((irf["Shock"] == int14_label) & (irf["Response"] == response_variable)),
            "Mean",
        ].reset_index(drop=True)[0]
        b123 = irf.loc[
            ((irf["Shock"] == int123_label) & (irf["Response"] == response_variable)),
            "Mean",
        ].reset_index(drop=True)[0]
        b124 = irf.loc[
            ((irf["Shock"] == int124_label) & (irf["Response"] == response_variable)),
            "Mean",
        ].reset_index(drop=True)[0]
        b134 = irf.loc[
            ((irf["Shock"] == int134_label) & (irf["Response"] == response_variable)),
            "Mean",
        ].reset_index(drop=True)[0]
        b1234 = irf.loc[
            ((irf["Shock"] == int1234_label) & (irf["Response"] == response_variable)),
            "Mean",
        ].reset_index(drop=True)[0]
        # Fill grid
        grid["impact"] = (
            (b1 * shock_size)
            + (b12 * grid[xyz_labels[0]])
            + (b13 * grid[xyz_labels[1]])
            + (b14 * grid[xyz_labels[2]])
            + (b123 * grid[xyz_labels[0]] * grid[xyz_labels[1]])
            + (b124 * grid[xyz_labels[0]] * grid[xyz_labels[2]])
            + (b134 * grid[xyz_labels[1]] * grid[xyz_labels[2]])
            + (b1234 * grid[xyz_labels[0]] * grid[xyz_labels[1]] * grid[xyz_labels[2]])
        )
        # Output
        return grid

    def irf_interaction_only_wide(
        irf: pd.DataFrame,
        int1_label: str,
        int12_label: str,
        int13_label: str,
        int14_label: str,
        int123_label: str,
        int124_label: str,
        int134_label: str,
        int1234_label: str,
        beta_ints: list[list[float]],
    ):
        # Triple interacted model: y = a + b1A + b2B + b3C + b4AB + b5AC + b6BC + b7ABC
        # subset b1 and b3
        irf_b1 = irf[irf["Shock"] == int1_label].copy()
        irf_b12 = irf[irf["Shock"] == int12_label].copy()
        irf_b13 = irf[irf["Shock"] == int13_label].copy()
        irf_b14 = irf[irf["Shock"] == int14_label].copy()
        irf_b123 = irf[irf["Shock"] == int123_label].copy()
        irf_b124 = irf[irf["Shock"] == int124_label].copy()
        irf_b134 = irf[irf["Shock"] == int134_label].copy()
        irf_b1234 = irf[irf["Shock"] == int1234_label].copy()
        # rename all column labels for Mean, LB and UB for easy merging later
        irf_b1 = irf_b1.rename(
            columns={
                "Mean": "Mean_" + int1_label,
                "LB": "LB_" + int1_label,
                "UB": "UB_" + int1_label,
            }
        )
        irf_b12 = irf_b12.rename(
            columns={
                "Mean": "Mean_" + int12_label,
                "LB": "LB_" + int12_label,
                "UB": "UB_" + int12_label,
            }
        )
        irf_b13 = irf_b13.rename(
            columns={
                "Mean": "Mean_" + int13_label,
                "LB": "LB_" + int13_label,
                "UB": "UB_" + int13_label,
            }
        )
        irf_b14 = irf_b14.rename(
            columns={
                "Mean": "Mean_" + int14_label,
                "LB": "LB_" + int14_label,
                "UB": "UB_" + int14_label,
            }
        )
        irf_b123 = irf_b123.rename(
            columns={
                "Mean": "Mean_" + int123_label,
                "LB": "LB_" + int123_label,
                "UB": "UB_" + int123_label,
            }
        )
        irf_b124 = irf_b124.rename(
            columns={
                "Mean": "Mean_" + int124_label,
                "LB": "LB_" + int124_label,
                "UB": "UB_" + int124_label,
            }
        )
        irf_b134 = irf_b134.rename(
            columns={
                "Mean": "Mean_" + int134_label,
                "LB": "LB_" + int134_label,
                "UB": "UB_" + int134_label,
            }
        )
        irf_b1234 = irf_b1234.rename(
            columns={
                "Mean": "Mean_" + int1234_label,
                "LB": "LB_" + int1234_label,
                "UB": "UB_" + int1234_label,
            }
        )
        del irf_b12["Shock"]
        del irf_b13["Shock"]
        del irf_b14["Shock"]
        del irf_b123["Shock"]
        del irf_b124["Shock"]
        del irf_b134["Shock"]
        del irf_b1234["Shock"]
        # merge left
        irf_interactions = irf_b1.merge(irf_b12, on=["Response", "Horizon"])
        irf_interactions = irf_interactions.merge(irf_b13, on=["Response", "Horizon"])
        irf_interactions = irf_interactions.merge(irf_b14, on=["Response", "Horizon"])
        irf_interactions = irf_interactions.merge(irf_b123, on=["Response", "Horizon"])
        irf_interactions = irf_interactions.merge(irf_b124, on=["Response", "Horizon"])
        irf_interactions = irf_interactions.merge(irf_b134, on=["Response", "Horizon"])
        irf_interactions = irf_interactions.merge(irf_b1234, on=["Response", "Horizon"])
        irf_interactions["Shock"] = (
            int1_label
            + "_mult_"
            + int12_label
            + "_mult_"
            + int13_label
            + "_mult_"
            + int14_label
            + "_mult_"
            + int123_label
            + "_mult_"
            + int124_label
            + "_mult_"
            + int134_label
            + "_mult_"
            + int1234_label
        )
        # generate new irfs
        for beta_int in beta_ints:
            for irf_moment in ["Mean", "LB", "UB"]:
                irf_interactions[
                    irf_moment
                    + "_b12"
                    + str(beta_int[0])
                    + "_b13"
                    + str(beta_int[1])
                    + "_b14"
                    + str(beta_int[2])
                    + "_b123"
                    + str(beta_int[0] * beta_int[1])
                    + "_b124"
                    + str(beta_int[0] * beta_int[2])
                    + "_b134"
                    + str(beta_int[1] * beta_int[2])
                    + "_b1234"
                    + str(beta_int[0] * beta_int[1] * beta_int[2])
                ] = (
                    irf_interactions[irf_moment + "_" + int1_label]
                    + (beta_int[0] * irf_interactions[irf_moment + "_" + int12_label])
                    + (beta_int[1] * irf_interactions[irf_moment + "_" + int13_label])
                    + (beta_int[2] * irf_interactions[irf_moment + "_" + int14_label])
                    + (
                        beta_int[0]
                        * beta_int[1]
                        * irf_interactions[irf_moment + "_" + int123_label]
                    )
                    + (
                        beta_int[0]
                        * beta_int[2]
                        * irf_interactions[irf_moment + "_" + int124_label]
                    )
                    + (
                        beta_int[1]
                        * beta_int[2]
                        * irf_interactions[irf_moment + "_" + int134_label]
                    )
                    + (
                        beta_int[0]
                        * beta_int[1]
                        * beta_int[2]
                        * irf_interactions[irf_moment + "_" + int1234_label]
                    )
                )
        # clean house
        for irf_moment in ["Mean", "LB", "UB"]:
            del irf_interactions[irf_moment + "_" + int1_label]
            del irf_interactions[irf_moment + "_" + int12_label]
            del irf_interactions[irf_moment + "_" + int13_label]
            del irf_interactions[irf_moment + "_" + int14_label]
            del irf_interactions[irf_moment + "_" + int123_label]
            del irf_interactions[irf_moment + "_" + int124_label]
            del irf_interactions[irf_moment + "_" + int134_label]
            del irf_interactions[irf_moment + "_" + int1234_label]
        irf_interactions = irf_interactions[
            ~(
                (irf_interactions["Response"] == int12_label)
                | (irf_interactions["Response"] == int13_label)
                | (irf_interactions["Response"] == int14_label)
                | (irf_interactions["Response"] == int123_label)
                | (irf_interactions["Response"] == int124_label)
                | (irf_interactions["Response"] == int134_label)
                | (irf_interactions["Response"] == int1234_label)
            )
        ]
        # export
        return irf_interactions

    def plot_quartet_interaction_wide_irf(
        irf: pd.DataFrame,
        shock_variable: str,  # only used in title and file suffix
        response_variable: str,
        show_ci: bool,
        beta_ints: list[list[float]],  # [[60.5, 91, 30], [30, 30, 90]]
        beta_int_colours: list[str],
        interacted_variable_label_nice: str,  # taken from b1_label + "_mult_" + b4_label + "_mult_" + b5_label + "_mult_" + b7_label
    ):
        fig = go.Figure()  # 4 lines per chart
        irf_count = 0
        for beta_int in tqdm(beta_ints):
            # subset
            irf_sub = irf[irf["Response"] == response_variable].copy()
            # mean irf
            fig.add_trace(
                go.Scatter(
                    x=irf_sub["Horizon"],
                    y=irf_sub[
                        "Mean"
                        + "_b12"
                        + str(beta_int[0])
                        + "_b13"
                        + str(beta_int[1])
                        + "_b14"
                        + str(beta_int[2])
                        + "_b123"
                        + str(beta_int[0] * beta_int[1])
                        + "_b124"
                        + str(beta_int[0] * beta_int[2])
                        + "_b134"
                        + str(beta_int[1] * beta_int[2])
                        + "_b1234"
                        + str(beta_int[0] * beta_int[1] * beta_int[2])
                    ],
                    name=interacted_variable_label_nice
                    + " = "
                    + ", ".join([str(i) for i in beta_int]),
                    mode="lines",
                    line=dict(
                        color=beta_int_colours[irf_count],
                        dash="solid",
                    ),
                )
            )
            if show_ci:
                # lower bound
                fig.add_trace(
                    go.Scatter(
                        x=irf_sub["Horizon"],
                        y=irf_sub[
                            "LB"
                            + "_b12"
                            + str(beta_int[0])
                            + "_b13"
                            + str(beta_int[1])
                            + "_b14"
                            + str(beta_int[2])
                            + "_b123"
                            + str(beta_int[0] * beta_int[1])
                            + "_b124"
                            + str(beta_int[0] * beta_int[2])
                            + "_b134"
                            + str(beta_int[1] * beta_int[2])
                            + "_b1234"
                            + str(beta_int[0] * beta_int[1] * beta_int[2])
                        ],
                        name="",
                        mode="lines",
                        line=dict(
                            color=beta_int_colours[irf_count],
                            width=1,
                            dash="dash",
                        ),
                    )
                )
                # upper bound
                fig.add_trace(
                    go.Scatter(
                        x=irf_sub["Horizon"],
                        y=irf_sub[
                            "UB"
                            + "_b12"
                            + str(beta_int[0])
                            + "_b13"
                            + str(beta_int[1])
                            + "_b14"
                            + str(beta_int[2])
                            + "_b123"
                            + str(beta_int[0] * beta_int[1])
                            + "_b124"
                            + str(beta_int[0] * beta_int[2])
                            + "_b134"
                            + str(beta_int[1] * beta_int[2])
                            + "_b1234"
                            + str(beta_int[0] * beta_int[1] * beta_int[2])
                        ],
                        name="",
                        mode="lines",
                        line=dict(
                            color=beta_int_colours[irf_count],
                            width=1,
                            dash="dash",
                        ),
                    )
                )
            # next
            irf_count += 1
        # format
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="darkgrey",
            line_width=1,
        )
        fig.update_layout(
            title="Panel LP IRF: Response of "
            + response_variable
            + " to "
            + shock_variable
            + " shocks "
            + "conditional on "
            + interacted_variable_label_nice,
            plot_bgcolor="white",
            hovermode="x unified",
            showlegend=True,
            font=dict(color="black", size=12),
        )
        # save image
        if show_ci:
            file_ci_suffix = "_withci"
        elif not show_ci:
            file_ci_suffix = ""
        fig.write_image(
            path_output
            + "octant_interaction_panellp_"
            + file_suffixes
            + "irf_"
            + "modwith_"
            + uncertainty_variable
            + "_"
            + mp_variable
            + "_"
            + "shock"
            + shock_variable
            + "_"
            + "response"
            + response_variable
            + file_ci_suffix
            + ".png",
            height=768,
            width=1366,
        )

    # Loop to estimate
    for mp_variable in tqdm(list_mp_variables):
        for uncertainty_variable in tqdm(list_uncertainty_variables):
            print("\nMP variable is " + mp_variable)
            print("Uncertainty variable is " + uncertainty_variable)
            # II --- Load data
            df = pd.read_parquet(
                path_data + "data_macro_" + input_df_suffix + ".parquet"
            )
            # III --- Additional wrangling
            # Groupby ref
            cols_groups = ["country", "quarter"]
            # Retrieve full list of endogenous variables
            cols_all_endog = [
                uncertainty_variable,
                mp_variable,
            ] + cols_endog_after_shocks
            # Trim columns
            df = df[
                cols_groups + cols_all_endog + cols_all_exog + cols_state_dependency
            ].copy()
            # Check when the panel becomes balanced
            check_balance_timing(input=df)
            check_balance_endtiming(input=df)
            # Create interaction terms

            # Trim more countries
            df = df[~df["country"].isin(countries_drop)].copy()
            # Check again when panel becomes balanced
            check_balance_timing(input=df)
            check_balance_endtiming(input=df)
            # Timebound
            df["date"] = pd.to_datetime(df["quarter"]).dt.date
            df = df[(df["date"] >= t_start)]
            if t_end is None:
                pass
            else:
                df = df[(df["date"] <= t_end)]
            del df["date"]
            # Drop NA
            df = df.dropna(axis=0)

            # Reset index
            df = df.reset_index(drop=True)
            # Numeric time
            df["time"] = df.groupby("country").cumcount()
            del df["quarter"]
            # Set multiindex
            df = df.set_index(["country", "time"])

            # IV --- Analysis
            df, cols_interaction = generate_n_interactions(
                df=df,
                labels_to_be_interacted=[uncertainty_variable]
                + [mp_variable]
                + cols_state_dependency,
            )
            irf = lp.PanelLPX(
                data=df,
                Y=cols_all_endog + cols_interaction,
                X=cols_all_exog,
                response=cols_all_endog + cols_interaction,
                horizon=12,
                lags=1,
                varcov="kernel",
                ci_width=0.8,
            )  # actual model is estimated only once
            for shock in [uncertainty_variable, mp_variable]:
                for response in cols_all_endog:
                    # Compile 1Y and 2Y average marginal effects
                    grid = compute_grid_octant_average_effects(
                        input=irf,
                        response_variable=shock,
                        xyz_labels=cols_state_dependency,
                        xyz_ranges=grid_state_dependency_ranges,
                        grid_size=100,
                        shock_size=1,
                        int1_label=shock,
                        int12_label=shock + "_" + cols_state_dependency[0],
                        int13_label=shock + "_" + cols_state_dependency[1],
                        int14_label=shock + "_" + cols_state_dependency[2],
                        int123_label=shock
                        + "_"
                        + cols_state_dependency[0]
                        + "_"
                        + cols_state_dependency[1],
                        int124_label=shock
                        + "_"
                        + cols_state_dependency[0]
                        + "_"
                        + cols_state_dependency[2],
                        int134_label=shock
                        + "_"
                        + cols_state_dependency[1]
                        + "_"
                        + cols_state_dependency[2],
                        int1234_label=shock
                        + "_"
                        + cols_state_dependency[0]
                        + "_"
                        + cols_state_dependency[1]
                        + "_"
                        + cols_state_dependency[2],
                        n_periods_to_average_over=4,
                    )
                    grid.to_parquet(
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
                    )  # save grids for possible separate processing
                irf_interaction = irf_interaction_only_wide(
                    irf=irf,
                    int1_label=shock,
                    int12_label=shock + "_" + cols_state_dependency[0],
                    int13_label=shock + "_" + cols_state_dependency[1],
                    int14_label=shock + "_" + cols_state_dependency[2],
                    int123_label=shock
                    + "_"
                    + cols_state_dependency[0]
                    + "_"
                    + cols_state_dependency[1],
                    int124_label=shock
                    + "_"
                    + cols_state_dependency[0]
                    + "_"
                    + cols_state_dependency[2],
                    int134_label=shock
                    + "_"
                    + cols_state_dependency[1]
                    + "_"
                    + cols_state_dependency[2],
                    int1234_label=shock
                    + "_"
                    + cols_state_dependency[0]
                    + "_"
                    + cols_state_dependency[1]
                    + "_"
                    + cols_state_dependency[2],
                    beta_ints=beta_values_to_simulate,
                )  # interaction terms are calculated twice
                for endog in cols_all_endog:
                    plot_quartet_interaction_wide_irf(
                        irf=irf_interaction,
                        response_variable=endog,
                        shock_variable=shock,  # this way, the file name will only say if mp or unc shocks were used
                        show_ci=False,
                        beta_ints=beta_values_to_simulate,  # [[60.5, 91, 160], [30, 30, 160]]
                        beta_int_colours=irf_colours_for_each_beta,  # ["blue", "red"]
                        interacted_variable_label_nice=state_dependency_nice_for_title,
                    )  # IRF plots are generated for all variables per shock


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
cols_threshold_hh_gov_epu = ["hhdebt_ngdp_ref", "govdebt_ngdp_ref", "epu_ref"]
cols_threshold_hh_gov_wui = ["hhdebt_ngdp_ref", "govdebt_ngdp_ref", "wui_ref"]
grid_range_hh_gov_epu = [[0, 150], [0, 250], [0, 500]]
grid_range_hh_gov_wui = [[0, 150], [0, 250], [0, 1]]
hh_gov_epu_values_combos = [
    [30, 60, 80],  # HH low, gov low, uncertainty low
    # [60, 60, 80],  # HH med, gov low, uncertainty low
    [90, 60, 80],  # HH high, gov low, uncertainty low
    # [30, 90, 80],  # HH low, gov med, uncertainty low
    # [60, 90, 80],  # HH med, gov med, uncertainty low
    # [90, 90, 80],  # HH high, gov med, uncertainty low
    [30, 120, 80],  # HH low, gov high, uncertainty low
    # [60, 120, 80],  # HH med, gov high, uncertainty low
    [90, 120, 80],  # HH high, gov high, uncertainty low
    [30, 60, 160],  # HH low, gov low, uncertainty high
    # [60, 60, 160],  # HH med, gov low, uncertainty high
    [90, 60, 160],  # HH high, gov low, uncertainty high
    # [30, 90, 160],  # HH low, gov med, uncertainty high
    # [60, 90, 160],  # HH med, gov med, uncertainty high
    # [90, 90, 160],  # HH high, gov med, uncertainty high
    [30, 120, 160],  # HH low, gov high, uncertainty high
    # [60, 120, 160],  # HH med, gov high, uncertainty high
    [90, 120, 160],  # HH high, gov high, uncertainty high
]
hh_gov_wui_values_combos = [
    [30, 60, 0.2],  # HH low, gov low, uncertainty low
    # [60, 60, 0.2],  # HH med, gov low, uncertainty low
    [90, 60, 0.2],  # HH high, gov low, uncertainty low
    # [30, 90, 0.2],  # HH low, gov med, uncertainty low
    # [60, 90, 0.2],  # HH med, gov med, uncertainty low
    # [90, 90, 0.2],  # HH high, gov med, uncertainty low
    [30, 120, 0.2],  # HH low, gov high, uncertainty low
    # [60, 120, 0.2],  # HH med, gov high, uncertainty low
    [90, 120, 0.2],  # HH high, gov high, uncertainty low
    [30, 60, 0.4],  # HH low, gov low, uncertainty high
    # [60, 60, 0.4],  # HH med, gov low, uncertainty high
    [90, 60, 0.4],  # HH high, gov low, uncertainty high
    # [30, 90, 0.4],  # HH low, gov med, uncertainty high
    # [60, 90, 0.4],  # HH med, gov med, uncertainty high
    # [90, 90, 0.4],  # HH high, gov med, uncertainty high
    [30, 120, 0.4],  # HH low, gov high, uncertainty high
    # [60, 120, 0.4],  # HH med, gov high, uncertainty high
    [90, 120, 0.4],  # HH high, gov high, uncertainty high
]
# debt_values_combos_irf_line_colours=[
#     "lightblue",
#     "blue",
#     "darkblue",
#     "lightgrey",
#     "grey",
#     "black",
#     "pink",
#     "red",
#     "crimson",
# ]
debt_values_combos_irf_line_colours = [
    "grey",
    "pink",
    "blue",
    "crimson",
    "black",
    "orangered",
    "midnightblue",
    "maroon",
]

# %%
# III --- Do everything
# With STIR
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_hh_gov_epu,
    grid_state_dependency_ranges=grid_range_hh_gov_epu,
    state_dependency_nice_for_title="HH debt, Gov debt, EPU",  # HH debt, Gov debt, EPU
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "colombia",  # 2006 Q4
        "germany",  # 2006 Q1
        "sweden",  # ends 2020 Q3 --- epu
    ],
    file_suffixes="",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_epu_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
)

# With STIR (reduced)
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_hh_gov_epu,
    grid_state_dependency_ranges=grid_range_hh_gov_epu,
    state_dependency_nice_for_title="HH debt, Gov debt, EPU",  # HH debt, Gov debt, EPU
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "colombia",  # 2006 Q4
        "germany",  # 2006 Q1
        "sweden",  # ends 2020 Q3 --- epu
    ],
    file_suffixes="reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_epu_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
)

# With M2
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["m2"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_hh_gov_epu,
    grid_state_dependency_ranges=grid_range_hh_gov_epu,
    state_dependency_nice_for_title="HH debt, Gov debt, EPU",  # HH debt, Gov debt, EPU
    countries_drop=[
        "india",  # 2012 Q1
        "china",  # 2007 Q1 and potentially exclusive case
        "chile",  # 2010 Q1 and potentially exclusive case
        "colombia",  # 2005 Q4
        "singapore",  # 2005 Q1
    ],
    file_suffixes="m2_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_epu_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
)

# With M2 (reduced)
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["m2"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_hh_gov_epu,
    grid_state_dependency_ranges=grid_range_hh_gov_epu,
    state_dependency_nice_for_title="HH debt, Gov debt, EPU",  # HH debt, Gov debt, EPU
    countries_drop=[
        "india",  # 2012 Q1
        "china",  # 2007 Q1 and potentially exclusive case
        "chile",  # 2010 Q1 and potentially exclusive case
        "colombia",  # 2005 Q4
        "singapore",  # 2005 Q1
    ],
    file_suffixes="m2_reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_epu_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
)

# %%
# III.B --- Do everything but with WUI as uncertainty shocks
# With STIR
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminwui"],
    cols_state_dependency=cols_threshold_hh_gov_wui,
    grid_state_dependency_ranges=grid_range_hh_gov_wui,
    state_dependency_nice_for_title="HH debt, Gov debt, WUI",  # HH debt, Gov debt, WUI
    countries_drop=[
        "argentina",
        "china",
        "germany",
        "india",
        "indonesia",
        "israel",
        "malaysia",
        "turkey",
        "thailand",
        "denmark",
        "norway",
        "sweden",
    ],
    file_suffixes="",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_wui_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
    input_df_suffix="large_yoy",
)

# With STIR (reduced)
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminwui"],
    cols_state_dependency=cols_threshold_hh_gov_wui,
    grid_state_dependency_ranges=grid_range_hh_gov_wui,
    state_dependency_nice_for_title="HH debt, Gov debt, WUI",  # HH debt, Gov debt, WUI
    countries_drop=[
        "argentina",
        "china",
        "germany",
        "india",
        "indonesia",
        "israel",
        "malaysia",
        "turkey",
        "thailand",
        "denmark",
        "norway",
        "sweden",
    ],
    file_suffixes="reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_wui_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
    input_df_suffix="large_yoy",
)

# With M2
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["m2"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminwui"],
    cols_state_dependency=cols_threshold_hh_gov_wui,
    grid_state_dependency_ranges=grid_range_hh_gov_wui,
    state_dependency_nice_for_title="HH debt, Gov debt, WUI",  # HH debt, Gov debt, WUI
    countries_drop=[
        "argentina",
        "chile",
        "china",
        "india",
        "indonesia",
        "malaysia",
        "singapore",
        "turkey",
    ],
    file_suffixes="m2_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_wui_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
    input_df_suffix="large_yoy",
)

# With M2 (reduced)
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["m2"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminwui"],
    cols_state_dependency=cols_threshold_hh_gov_wui,
    grid_state_dependency_ranges=grid_range_hh_gov_wui,
    state_dependency_nice_for_title="HH debt, Gov debt, WUI",  # HH debt, Gov debt, WUI
    countries_drop=[
        "argentina",
        "chile",
        "china",
        "india",
        "indonesia",
        "malaysia",
        "singapore",
        "turkey",
    ],
    file_suffixes="m2_reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_wui_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
    input_df_suffix="large_yoy",
)

# %%
# III.C --- Do everything but with +8Q as reference for maxminshocks
# EPU and STIR
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_hh_gov_epu,
    grid_state_dependency_ranges=grid_range_hh_gov_epu,
    state_dependency_nice_for_title="HH debt, Gov debt, EPU",  # HH debt, Gov debt, EPU
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "colombia",  # 2006 Q4
        "germany",  # 2006 Q1
        "sweden",  # ends 2020 Q3 --- epu
    ],
    file_suffixes="maxminref8_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_epu_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
    input_df_suffix="yoy_maxminref8",
)

# WUI and STIR
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminwui"],
    cols_state_dependency=cols_threshold_hh_gov_wui,
    grid_state_dependency_ranges=grid_range_hh_gov_wui,
    state_dependency_nice_for_title="HH debt, Gov debt, WUI",  # HH debt, Gov debt, WUI
    countries_drop=[
        "argentina",
        "china",
        "germany",
        "india",
        "indonesia",
        "israel",
        "malaysia",
        "turkey",
        "thailand",
        "denmark",
        "norway",
        "sweden",
    ],
    file_suffixes="maxminref8_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_wui_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
    input_df_suffix="large_yoy_maxminref8",
)


# %%
# III.D --- Do everything but with +6Q as reference for maxminshocks
# EPU and STIR
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_hh_gov_epu,
    grid_state_dependency_ranges=grid_range_hh_gov_epu,
    state_dependency_nice_for_title="HH debt, Gov debt, EPU",  # HH debt, Gov debt, EPU
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "colombia",  # 2006 Q4
        "germany",  # 2006 Q1
        "sweden",  # ends 2020 Q3 --- epu
    ],
    file_suffixes="maxminref6_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_epu_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
    input_df_suffix="yoy_maxminref6",
)

# WUI and STIR
do_everything_octant_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminwui"],
    cols_state_dependency=cols_threshold_hh_gov_wui,
    grid_state_dependency_ranges=grid_range_hh_gov_wui,
    state_dependency_nice_for_title="HH debt, Gov debt, WUI",  # HH debt, Gov debt, WUI
    countries_drop=[
        "argentina",
        "china",
        "germany",
        "india",
        "indonesia",
        "israel",
        "malaysia",
        "turkey",
        "thailand",
        "denmark",
        "norway",
        "sweden",
    ],
    file_suffixes="maxminref6_",  # format: "abc_" or ""
    beta_values_to_simulate=hh_gov_wui_values_combos,
    irf_colours_for_each_beta=debt_values_combos_irf_line_colours,
    input_df_suffix="large_yoy_maxminref6",
)


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%


# # %%
# # TESTING
# df = pd.read_parquet(path_data + "data_macro_large_yoy.parquet")
# df = df[
#     ["country", "quarter", "m2", "maxminbrent", "wui_ref", "maxminwui", "maxminm2"]
#     + cols_endog_long
# ]


# def check_balance_timing(input):
#     min_quarter_by_country = input.copy()
#     min_quarter_by_country = min_quarter_by_country.dropna(axis=0)
#     min_quarter_by_country = (
#         min_quarter_by_country.groupby("country")["quarter"].min().reset_index()
#     )
#     print(tabulate(min_quarter_by_country, headers="keys", tablefmt="pretty"))


# def check_balance_endtiming(input):
#     max_quarter_by_country = input.copy()
#     max_quarter_by_country = max_quarter_by_country.dropna(axis=0)
#     max_quarter_by_country = (
#         max_quarter_by_country.groupby("country")["quarter"].max().reset_index()
#     )
#     print(tabulate(max_quarter_by_country, headers="keys", tablefmt="pretty"))


# check_balance_timing(df)
# check_balance_endtiming(df)
# # %%
# countries_drop = [
#     "argentina",
#     "chile",
#     "china",
#     "india",
#     "indonesia",
#     "malaysia",
#     "singapore",
#     "turkey",
# ]

# df = df[~df["country"].isin(countries_drop)].copy()
# check_balance_timing(df)
# check_balance_endtiming(df)

# # %%
