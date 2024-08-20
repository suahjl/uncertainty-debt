import pandas as pd
import numpy as np
from helper import (
    telsendfiles,
    telsendimg,
    telsendmsg,
    subplots_scatterplots,
    scatterplot,
    scatterplot_layered,
    subplots_linecharts,
    stacked_barchart,
    stacked_barchart_overlaycallouts,
    lineplot,
    pil_img2pdf,
    heatmap,
)
from datetime import date, timedelta
import statsmodels.formula.api as smf
import statsmodels.tsa.api as sm
import plotly.graph_objects as go
import plotly.express as px
from tabulate import tabulate
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
path_ceic = "./ceic/"
tel_config = os.getenv("TEL_CONFIG")
t_start = date(1990, 1, 1)

# For loading thresholds
mp_variable = "maxminstir"  # maxminstir
uncertainty_variable = "maxminepu"  # maxepu

# %%
# I --- Load data
df = pd.read_parquet(path_data + "data_macro_yoy_ratesinlevels.parquet")

# %%
# II --- Additional wrangling
list_shock_prefixes = ["max", "min", "maxmin"]
list_mp_variables = [i + "stir" for i in list_shock_prefixes] + ["stir"]
list_uncertainty_variables = [i + "epu" for i in list_shock_prefixes] + ["epu"]

# Groupby ref
cols_groups = ["country", "quarter"]
# Trim columns
cols_all = [
    "hhdebt",
    "corpdebt",
    "govdebt",
    "hhdebt_ngdp_ref",
    "corpdebt_ngdp_ref",
    "govdebt_ngdp_ref",
    "gdp",
    "urate",
    "corecpi",
    "cpi",
    "reer",
    "brent",
    "maxminbrent",
] + list_mp_variables
# colours_all = [
#     "red",
#     "green",
#     "blue",
#     "darkred",
#     "darkgreen",
#     "darkblue",
#     "black",
#     "lightgrey",
#     "magenta",
#     "pink",
#     "purple",
#     "mediumpurple",
#     "plum",
#     "yellowgreen",
#     "orange",
#     "mediumslateblue",
#     "darkslateblue",
# ]
df = df[cols_groups + cols_all + list_uncertainty_variables].copy()
# Trim more countries
# if "stir" in mp_variable:
countries_drop = [
    "india",  # 2016 Q3
    "belgium",  # ends 2022 Q3
    "denmark",  # ends 2019 Q3
    "china",  # 2007 Q4 and potentially exclusive case
    "colombia",  # 2006 Q4
    "germany",  # 2006 Q1
    "sweden",  # ends 2020 Q3 --- epu
    # "mexico",  # ends 2023 Q1 --- ngdp (keep if %yoy for debt and not %diff_ngdp)
    # "russia",  # basket case
]  # 12-13 countries
# Timebound
df["date"] = pd.to_datetime(df["quarter"]).dt.date
df = df[(df["date"] >= t_start)]
del df["date"]
# Drop NA
df = df.dropna(axis=0)
# Threshold


def find_quadrant_thresholds(
    df: pd.DataFrame,
    threshold_variables: list[str],
    option: str,
    param_choices: list[float],
):
    if option == "dumb":
        df.loc[
            df[threshold_variables[0] + "_ref"] >= param_choices[0],
            threshold_variables[0] + "_above_threshold",
        ] = 1
        df.loc[
            df[threshold_variables[0] + "_ref"] < param_choices[0],
            threshold_variables[0] + "_above_threshold",
        ] = 0
        df.loc[
            df[threshold_variables[1] + "_ref"] >= param_choices[1],
            threshold_variables[0] + "_above_threshold",
        ] = 1
        df.loc[
            df[threshold_variables[1] + "_ref"] < param_choices[1],
            threshold_variables[1] + "_above_threshold",
        ] = 0
        # quadrant 1 (0,0)
        df.loc[
            (
                (df[threshold_variables[0] + "_above_threshold"] == 0)
                & (df[threshold_variables[1] + "_above_threshold"] == 0)
            ),
            threshold_variables[0] + "0" + "_" + threshold_variables[1] + "0",
        ] = 1
        df.loc[
            ~(
                (df[threshold_variables[0] + "_above_threshold"] == 0)
                & (df[threshold_variables[1] + "_above_threshold"] == 0)
            ),
            threshold_variables[0] + "0" + "_" + threshold_variables[1] + "0",
        ] = 0
        # quadrant 2 (1,0)
        df.loc[
            (
                (df[threshold_variables[0] + "_above_threshold"] == 1)
                & (df[threshold_variables[1] + "_above_threshold"] == 0)
            ),
            threshold_variables[0] + "1" + "_" + threshold_variables[1] + "0",
        ] = 1
        df.loc[
            ~(
                (df[threshold_variables[0] + "_above_threshold"] == 1)
                & (df[threshold_variables[1] + "_above_threshold"] == 0)
            ),
            threshold_variables[0] + "1" + "_" + threshold_variables[1] + "0",
        ] = 0
        # quadrant 3 (0,1)
        df.loc[
            (
                (df[threshold_variables[0] + "_above_threshold"] == 0)
                & (df[threshold_variables[1] + "_above_threshold"] == 1)
            ),
            threshold_variables[0] + "0" + "_" + threshold_variables[1] + "1",
        ] = 1
        df.loc[
            ~(
                (df[threshold_variables[0] + "_above_threshold"] == 0)
                & (df[threshold_variables[1] + "_above_threshold"] == 1)
            ),
            threshold_variables[0] + "0" + "_" + threshold_variables[1] + "1",
        ] = 0
        # quadrant 4 (1,1)
        df.loc[
            (
                (df[threshold_variables[0] + "_above_threshold"] == 1)
                & (df[threshold_variables[1] + "_above_threshold"] == 1)
            ),
            threshold_variables[0] + "1" + "_" + threshold_variables[1] + "1",
        ] = 1
        df.loc[
            ~(
                (df[threshold_variables[0] + "_above_threshold"] == 1)
                & (df[threshold_variables[1] + "_above_threshold"] == 1)
            ),
            threshold_variables[0] + "1" + "_" + threshold_variables[1] + "1",
        ] = 0
        print("Threshold of " + threshold_variables[0] + " is " + str(param_choices[0]))
        print("Threshold of " + threshold_variables[1] + " is " + str(param_choices[1]))
    elif option == "reg_thresholdselection":
        df_opt_threshold = pd.read_csv(
            path_output
            + "reg_quadrant_thresholdselection_fe_"
            + "modwith_"
            + uncertainty_variable
            + "_"
            + mp_variable
            + "_opt_threshold"
            + ".csv"
        )
        opt_threshold0 = df_opt_threshold.iloc[0, 0]
        opt_threshold1 = df_opt_threshold.iloc[0, 1]
        # first
        df.loc[
            df[threshold_variables[0] + "_ref"] >= opt_threshold0,
            threshold_variables[0] + "_above_threshold",
        ] = 1
        df.loc[
            df[threshold_variables[0] + "_ref"] < opt_threshold0,
            threshold_variables[0] + "_above_threshold",
        ] = 0
        # second
        df.loc[
            df[threshold_variables[1] + "_ref"] >= opt_threshold1,
            threshold_variables[1] + "_above_threshold",
        ] = 1
        df.loc[
            df[threshold_variables[1] + "_ref"] < opt_threshold1,
            threshold_variables[1] + "_above_threshold",
        ] = 0
        # quadrant 1 (0,0)
        df.loc[
            (
                (df[threshold_variables[0] + "_above_threshold"] == 0)
                & (df[threshold_variables[1] + "_above_threshold"] == 0)
            ),
            threshold_variables[0] + "0" + "_" + threshold_variables[1] + "0",
        ] = 1
        df.loc[
            ~(
                (df[threshold_variables[0] + "_above_threshold"] == 0)
                & (df[threshold_variables[1] + "_above_threshold"] == 0)
            ),
            threshold_variables[0] + "0" + "_" + threshold_variables[1] + "0",
        ] = 0
        # quadrant 2 (1,0)
        df.loc[
            (
                (df[threshold_variables[0] + "_above_threshold"] == 1)
                & (df[threshold_variables[1] + "_above_threshold"] == 0)
            ),
            threshold_variables[0] + "1" + "_" + threshold_variables[1] + "0",
        ] = 1
        df.loc[
            ~(
                (df[threshold_variables[0] + "_above_threshold"] == 1)
                & (df[threshold_variables[1] + "_above_threshold"] == 0)
            ),
            threshold_variables[0] + "1" + "_" + threshold_variables[1] + "0",
        ] = 0
        # quadrant 3 (0,1)
        df.loc[
            (
                (df[threshold_variables[0] + "_above_threshold"] == 0)
                & (df[threshold_variables[1] + "_above_threshold"] == 1)
            ),
            threshold_variables[0] + "0" + "_" + threshold_variables[1] + "1",
        ] = 1
        df.loc[
            ~(
                (df[threshold_variables[0] + "_above_threshold"] == 0)
                & (df[threshold_variables[1] + "_above_threshold"] == 1)
            ),
            threshold_variables[0] + "0" + "_" + threshold_variables[1] + "1",
        ] = 0
        # quadrant 4 (1,1)
        df.loc[
            (
                (df[threshold_variables[0] + "_above_threshold"] == 1)
                & (df[threshold_variables[1] + "_above_threshold"] == 1)
            ),
            threshold_variables[0] + "1" + "_" + threshold_variables[1] + "1",
        ] = 1
        df.loc[
            ~(
                (df[threshold_variables[0] + "_above_threshold"] == 1)
                & (df[threshold_variables[1] + "_above_threshold"] == 1)
            ),
            threshold_variables[0] + "1" + "_" + threshold_variables[1] + "1",
        ] = 0
        print(
            "optimal thresholds: "
            + threshold_variables[0]
            + " = "
            + str(opt_threshold0)
            + " and "
            + threshold_variables[1]
            + " = "
            + str(opt_threshold1)
        )
    return df


# Threshold
threshold_variables = ["hhdebt_ngdp", "govdebt_ngdp"]
df = find_quadrant_thresholds(
    df=df,
    threshold_variables=threshold_variables,
    option="reg_thresholdselection",
    param_choices=[0, 0],
)

# Reset index
df = df.reset_index(drop=True)

# %%
# III --- Plot
plot_cbyc = False
if plot_cbyc:
    pic_names = []
    for threshold_variable in threshold_variables:
        for x in ["epu", "maxminepu", "maxminstir"]:
            for y in tqdm(cols_all):
                # split x-axis columns into when H = 0 and H = 1
                df.loc[
                    df[threshold_variable + "_above_threshold"] == 1,
                    x + "_when_" + threshold_variable + "_is_above_threshold",
                ] = df[x].copy()
                df.loc[
                    df[threshold_variable + "_above_threshold"] == 0,
                    x + "_when_" + threshold_variable + "_is_below_threshold",
                ] = df[x].copy()
                fig = subplots_scatterplots(
                    data=df,
                    col_group="country",
                    cols_x=[
                        x + "_when_" + threshold_variable + "_is_above_threshold",
                        x + "_when_" + threshold_variable + "_is_below_threshold",
                    ],
                    cols_y=[y, y],
                    annot_size=9,
                    font_size=9,
                    marker_colours=["red", "black"],  # black and red for easy reference
                    marker_sizes=[3, 3],
                    include_best_fit=True,
                    best_fit_colours=["red", "black"],
                    best_fit_widths=[2, 2],
                    main_title=y
                    + " against "
                    + x
                    + " when "
                    + threshold_variable
                    + " is above and below threshold",
                    maxrows=5,
                    maxcols=4,
                    add_horizontal_at_yzero=True,
                    add_vertical_at_xzero=True,
                )
                pic_name = (
                    path_output
                    + "scatter_regime_ratesinlevels"
                    + "_"
                    + threshold_variable
                    + "_"
                    + y
                    + "_against_"
                    + x
                )
                pic_names += [pic_name]
                fig.write_image(
                    pic_name + ".png",
                    height=768,
                    width=1366,
                )
    pdf_name = path_output + "scatter_regime_ratesinlevels"
    pil_img2pdf(list_images=pic_names, extension="png", pdf_name=pdf_name)

plot_pooled = True
if plot_pooled:
    pic_names = []
    for threshold_variable in threshold_variables:
        for x in ["epu", "maxminepu", "maxminstir"]:
            for y in tqdm(cols_all):
                # split x-axis columns into when H = 0 and H = 1
                df.loc[
                    df[threshold_variable + "_above_threshold"] == 1,
                    x + "_when_" + threshold_variable + "_is_above_threshold",
                ] = df[x].copy()
                df.loc[
                    df[threshold_variable + "_above_threshold"] == 0,
                    x + "_when_" + threshold_variable + "_is_below_threshold",
                ] = df[x].copy()
                fig = scatterplot_layered(
                    data=df,
                    x_cols=[
                        x + "_when_" + threshold_variable + "_is_above_threshold",
                        x + "_when_" + threshold_variable + "_is_below_threshold",
                    ],
                    x_cols_nice=[
                        x + " when " + threshold_variable + " is above threshold",
                        x + " when " + threshold_variable + " is below threshold",
                    ],
                    y_cols=[y, y],
                    y_cols_nice=[y, y],
                    font_size=22,
                    marker_colours=["red", "black"],  # black and red for easy reference
                    marker_sizes=[4, 4],
                    best_fit_colours=["red", "black"],
                    best_fit_widths=[5, 5],
                    main_title=(
                        "Pooled: "
                        + y
                        + " against "
                        + x
                        + " when "
                        + threshold_variable
                        + " is above and below threshold"
                    ),
                    add_horizontal_at_yzero=True,
                    add_vertical_at_xzero=True,
                )
                pic_name = (
                    path_output
                    + "scatter_regime_ratesinlevels"
                    + "_"
                    + threshold_variable
                    + "_"
                    + y
                    + "_against_"
                    + x
                )
                pic_names += [pic_name]
                fig.write_image(
                    pic_name + ".png",
                    height=768,
                    width=1366,
                    # height=1366,
                    # width=1100,
                )
    pdf_name = path_output + "scatter_regime_ratesinlevels_pooled"
    pil_img2pdf(list_images=pic_names, extension="png", pdf_name=pdf_name)


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
