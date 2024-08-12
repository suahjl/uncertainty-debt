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
cols_all = (
    [
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
    ]
    + list_mp_variables
    + list_uncertainty_variables
)
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
    "darkcyan",
    "darkseagreen",
    "darkkhaki",
    "mediumaquamarine",
]
dashes_all = ["solid"] * len(colours_all)
df = df[cols_groups + cols_all].copy()
# Add y=0
df["y_is_zero"] = 0
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
# Reset index
df = df.reset_index(drop=True)

# %%
# III --- Plot
pic_names = []
for y, ycolour, dash in tqdm(zip(cols_all, colours_all, dashes_all)):
    fig = subplots_linecharts(
        data=df,
        col_group="country",
        cols_values=[y, "y_is_zero"],
        cols_values_nice=[y, "Y=0"],
        col_time="quarter",
        annot_size=9,
        font_size=9,
        line_colours=[ycolour, "grey"],
        line_dashes=[dash, "dot"],
        main_title=y,
        maxrows=5,
        maxcols=4,
        title_size=24,
    )
    pic_name = path_output + "lineplot_ratesinlevels" + "_" + y
    pic_names += [pic_name]
    fig.write_image(
        pic_name + ".png",
        height=768,
        width=1366,
    )
pdf_name = path_output + "lineplot_ratesinlevels"
pil_img2pdf(list_images=pic_names, extension="png", pdf_name=pdf_name)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
