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
path_data_special = "./data/ceic_for_special_charts/"
path_data = "./data/"
path_output = "./output/"

# %%
# --- Global debt stacked area chart
df = pd.read_csv(path_data_special + "ceic_global_debt_for_charts.csv")
df["quarter"] = pd.to_datetime(df["quarter"].astype("str"), format="%b-%y").astype(
    "str"
)
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
# --- Global GDP breakdown stacked area chart
df = pd.read_csv(path_data_special + "ceic_wb_globalgdp_breakdown.csv")
df["inventories_gdp_share"] = df["gcf_gdp_share"] - df["gfcf_gdp_share"]
df["errors_gdp_share"] = 100 - df[
    [
        "pce_gdp_share",
        "gce_gdp_share",
        "gfcf_gdp_share",
        "nx_gdp_share",
        "inventories_gdp_share",
    ]
].sum(axis=1)
df["total"] = 100
fig = stacked_area_lineplot(
    data=df,
    x_col="year",
    y_cols_area=[
        "errors_gdp_share",
        "nx_gdp_share",
        "inventories_gdp_share",
        "pce_gdp_share",
        "gce_gdp_share",
        "gfcf_gdp_share",
    ],
    y_cols_area_nice=[
        "Errors and omissions",
        "Net exports",
        "Change in inventories",
        "Household consumption",
        "Government consumption",
        "Investment",
    ],
    y_cols_line=["total"],
    y_cols_line_nice=["Total"],
    colours_area=[
        "lightgrey",
        "orange",
        "grey",
        "crimson",
        "darkgreen",
        "darkblue",
    ],
    colours_line=["black"],
    main_title="Expenditure share breakdown of global GDP",
    font_size=22,
)
fig.write_image(path_output + "stacked_area_lineplot_globalgdp_breakdown.png")

# %%
# --- Global debt line plots
df = pd.read_csv(path_data_special + "ceic_global_debt_for_charts.csv")
df["quarter"] = pd.to_datetime(df["quarter"].astype("str"), format="%b-%y").astype(
    "str"
)
df["quarter"] = pd.to_datetime(df["quarter"].astype("str")).dt.to_period("Q")
df["quarter"] = df["quarter"].astype("str")
for col in tqdm([i for i in list(df.columns) if "quarter" not in i]):
    df_sub = df[["quarter", col]].copy()
    df_sub = df_sub.dropna()
    fig = lineplot(
        data=df_sub,
        y_cols=[col],
        y_cols_nice=[col],
        x_col="quarter",
        x_col_nice="Quarter",
        line_colours=["black"],
        line_widths=[3],
        line_dashes=["solid"],
        main_title="",
        font_size=20,
        show_legend=False,
    )
    fig.write_image(path_output + "lineplot_global_" + col + ".png")

# %%
df = pd.read_csv(path_data + "ceic_ae_eme_debt_for_charts.csv")
df["quarter"] = pd.to_datetime(df["quarter"].astype("str"), format="%b-%y").astype(
    "str"
)
df["quarter"] = pd.to_datetime(df["quarter"].astype("str")).dt.to_period("Q")
df["quarter"] = df["quarter"].astype("str")
for col in tqdm([i for i in list(df.columns) if "quarter" not in i]):
    df_sub = df[["quarter", col]].copy()
    df_sub = df_sub.dropna()
    fig = lineplot(
        data=df_sub,
        y_cols=[col],
        y_cols_nice=[col],
        x_col="quarter",
        x_col_nice="Quarter",
        line_colours=["black"],
        line_widths=[3],
        line_dashes=["solid"],
        main_title="",
        font_size=20,
        show_legend=False,
    )
    fig.write_image(path_output + "lineplot_global_" + col + ".png")

# %%
# --- Global EPU line plot
df = pd.read_csv(path_data_special + "ceic_gepu_for_charts.csv")
df["month"] = pd.to_datetime(df["month"].astype("str"), format="%b-%y").astype("str")
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
    show_legend=False,
)
fig.write_image(path_output + "lineplot_gepu.png")

# %%
# --- Distribution of HH and gov debt
df = pd.read_parquet(path_data + "data_macro_yoy.parquet")
df = df[
    ["country", "quarter", "hhdebt_ngdp_ref", "corpdebt_ngdp_ref", "govdebt_ngdp_ref"]
]
fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=df["hhdebt_ngdp_ref"], name="HH debt", opacity=0.75, marker=dict(color="red")
    )
)
fig.add_trace(
    go.Histogram(
        x=df["govdebt_ngdp_ref"],
        name="Government debt",
        opacity=0.75,
        marker=dict(color="darkblue"),
    )
)
fig.update_layout(
    title="Histogram of HH and government debt-to-GDP ratios",
    plot_bgcolor="white",
    font=dict(color="black", size=12),
    height=768,
    width=1366,
)
fig.write_image(path_output + "histogram_debt_ngdp.png")


def find_quantile(value, column):
    return (
        df[column]
        .dropna()
        .rank(pct=True)
        .loc[((df[column] <= (value + 0.05)) & (df[column] >= (value - 0.05)))]
        .mean()
    )


hhdebt_threshold = 60.5
govdebt_threshold = 95

quantile_hhdebt = find_quantile(hhdebt_threshold, "hhdebt_ngdp_ref")
quantile_govdebt = find_quantile(govdebt_threshold, "govdebt_ngdp_ref")

print(
    f"The value {hhdebt_threshold} in hhdebt_ngdp_ref is in the {quantile_hhdebt:.2%} quantile."
)
print(
    f"The value {govdebt_threshold} in govdebt_ngdp_ref is in the {quantile_govdebt:.2%} quantile."
)

df_quantile = pd.DataFrame(
    {
        "variable": ["hhdebt_ngdp", "govdebt_ngdp"],
        "threshold": [hhdebt_threshold, govdebt_threshold],
        "quantile": [100 * quantile_hhdebt, 100 * quantile_govdebt],
    }
)
df_quantile.to_csv(
    path_output + "debt_ngdp_threshold_quantile_positions.csv", index=False
)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
