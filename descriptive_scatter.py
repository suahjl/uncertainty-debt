import pandas as pd
import numpy as np
from helper import (
    telsendfiles,
    telsendimg,
    telsendmsg,
    subplots_scatterplots,
    scatterplot,
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
    "hhdebt_ngdp",
    "corpdebt_ngdp",
    "govdebt_ngdp",
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
colours_all = [
    "red",
    "green",
    "blue",
    "darkred",
    "darkgreen",
    "darkblue",
    "black",
    "lightgrey",
    "magenta",
    "pink",
    "purple",
    "mediumpurple",
    "plum",
    "yellowgreen",
    "orange",
    "mediumslateblue",
    "darkslateblue",
]
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
    "mexico",  # ends 2023 Q1 --- epu
]  # 12 countries
# Timebound
df["date"] = pd.to_datetime(df["quarter"]).dt.date
df = df[(df["date"] >= t_start)]
del df["date"]
# Drop NA
df = df.dropna(axis=0)
# Threshold
threshold_variable = "hhdebt_ngdp"


def find_threshold(
    df: pd.DataFrame, threshold_variable: str, option: str, param_choice: float
):
    if option == "dumb":
        df.loc[
            df[threshold_variable + "_ref"] >= param_choice,
            threshold_variable + "_above_threshold",
        ] = 1
        df.loc[
            df[threshold_variable + "_ref"] < param_choice,
            threshold_variable + "_above_threshold",
        ] = 0
        print("Threshold is " + str(param_choice))
    elif option == "global_quantile":
        df.loc[
            df[threshold_variable + "_ref"]
            >= df[threshold_variable + "_ref"].quantile(param_choice),
            threshold_variable + "_above_threshold",
        ] = 1
        df.loc[
            df[threshold_variable + "_ref"]
            < df[threshold_variable + "_ref"].quantile(param_choice),
            threshold_variable + "_above_threshold",
        ] = 0
        print(
            "Threshold is "
            + str(df[threshold_variable + "_ref"].quantile(param_choice))
        )
    elif option == "country_quantile":
        ref = pd.DataFrame(
            df.groupby("country")[threshold_variable + "_ref"].quantile(0.8)
        ).reset_index()
        ref = ref.rename(
            columns={threshold_variable + "_ref": threshold_variable + "_threshold"}
        )
        df = df.merge(ref, how="left", on="country")
        df.loc[
            df[threshold_variable + "_ref"] >= df[threshold_variable + "_threshold"],
            threshold_variable + "_above_threshold",
        ] = 1
        df.loc[
            df[threshold_variable + "_ref"] < df[threshold_variable + "_threshold"],
            threshold_variable + "_above_threshold",
        ] = 0
    return df


df = find_threshold(
    df=df,
    threshold_variable="hhdebt_ngdp",
    option="dumb",
    param_choice=80,
)

# Reset index
df = df.reset_index(drop=True)

# %%
# III --- Plot
for x in ["epu", "maxepu", "maxminstir"]:
    for y, ycolour in tqdm(zip(cols_all, colours_all)):
        fig = subplots_scatterplots(
            data=df,
            col_group="country",
            cols_x=[x],
            cols_y=[y],
            annot_size=9,
            font_size=9,
            marker_colours=[ycolour],
            marker_sizes=[3],
            include_best_fit=True,
            best_fit_colours=[ycolour],
            best_fit_widths=[2],
            main_title=y + " against " + x,
            maxrows=5,
            maxcols=4,
            add_horizontal_at_yzero=True,
            add_vertical_at_xzero=True,
        )
        fig.write_image(
            path_output + "scatter" + "_" + y + "_against_" + x + ".png",
            height=768,
            width=1366,
        )

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
