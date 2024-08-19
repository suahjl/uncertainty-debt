# %%
# Special one-off charts
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
    stacked_area_lineplot,
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
path_data = "./data/ceic_for_special_charts/"
path_output = "./output/"

# %%
# I --- Global debt stacked area chart
df = pd.read_csv(path_data + "ceic_global_debt_for_charts.csv")
df["quarter"] = pd.to_datetime(df["quarter"].astype("str"), format='%b-%y').astype("str")
df["quarter"] = pd.to_datetime(df["quarter"].astype("str")).dt.to_period("Q")
df["quarter"] = df["quarter"].astype("str")
fig = stacked_area_lineplot(
    data=df,
    x_col="quarter",
    y_cols_area=["hhdebt", "corpdebt", "govdebt"],
    y_cols_area_nice=["Household", "Non-financial corporations", "Government"],
    y_cols_line=["debt"],
    y_cols_line_nice=["Total"],
    colours_area=["red", "blue", "green"],
    colours_line=["black"],
    main_title="Global PPP-weighted credit to the non-financial sector",
    font_size=24,
)
fig.write_image(path_output + "stacked_area_lineplot_globaldebt.png")

# %%
# II --- Global EPU line plot
df = pd.read_csv(path_data + "ceic_gepu_for_charts.csv")
df["month"] = pd.to_datetime(df["month"].astype("str"), format='%b-%y').astype("str")
df["month"] = pd.to_datetime(df["month"].astype("str")).dt.to_period("M")
df["month"] = df["month"].astype("str")
df["gepu_12mma"] = df["gepu"].rolling(12).mean()
fig = lineplot(
    data=df,
    y_cols=["gepu", "gepu_12mma"],
    y_cols_nice=["GEPU", "GEPU (12MMA)"],
    x_col="month",
    x_col_nice="Month",
    line_colours=["black", "black"],
    line_widths=[3, 3],
    line_dashes=["solid", "dash"],
    main_title="Global PPP-weighted economic policy uncertainty (dotted: 12-month moving average)",
    font_size=20,
    show_legend=False
)
fig.write_image(path_output + "lineplot_gepu.png")

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
