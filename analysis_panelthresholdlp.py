# %%
import pandas as pd
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
df = pd.read_parquet(path_data + "data_macro_yoy.parquet")


# %%
# II --- Functions
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


# %%
# III --- Additional wrangling
# Groupby ref
cols_groups = ["country", "quarter"]
# Trim columns
cols_all_endog = [
    "epu",
    "hhdebt_ngdp",
    "corpdebt_ngdp",
    "govdebt_ngdp",
    "stir",
    "gdp",
    "capflows_ngdp",
    "corecpi",
    "reer",
]
cols_all_exog = ["brent"]
cols_threshold = ["hhdebt_ngdp_ref", "corpdebt_ngdp_ref"]
df = df[cols_groups + cols_all_endog + cols_all_exog + cols_threshold].copy()
# Check when the panel becomes balanced
check_balance_timing(input=df)
# Trim more countries
countries_drop = [
    "australia",  # inflation
    "india",
    "china",
    "colombia",
    "germany",  # inflation
    "south_korea",  # stir
    "denmark",  # ends before covid
    "belgium",  # ends 2022
    "sweden",  # ends 2020
    "mexico",  # ends 2023Q1
]
df = df[~df["country"].isin(countries_drop)].copy()
# Check again when panel becomes balanced
check_balance_timing(input=df)
check_balance_endtiming(input=df)
# Timebound
df["date"] = pd.to_datetime(df["quarter"]).dt.date
df = df[(df["date"] >= t_start)]
del df["date"]
# Drop NA
df = df.dropna(axis=0)
# Threshold
df.loc[df["hhdebt_ngdp_ref"] >= 80, "hhdebt_above_threshold"] = 1
df.loc[df["hhdebt_ngdp_ref"] < 80, "hhdebt_above_threshold"] = 0
# Reset index
df = df.reset_index(drop=True)
# Numeric time
df["time"] = df.groupby("country").cumcount()
del df["quarter"]
# Set multiindex
df = df.set_index(["country", "time"])

# %%
# IV --- Analysis
# estimate model
irf_on, irf_off = lp.ThresholdPanelLPX(
    data=df,
    Y=cols_all_endog,
    X=cols_all_exog,
    threshold_var="hhdebt_above_threshold",
    response=cols_all_endog,
    horizon=12,
    lags=1,
    varcov="kernel",
    ci_width=0.8,
)
# plot irf
fig = lp.ThresholdIRFPlot(
    irf_threshold_on=irf_on,
    irf_threshold_off=irf_off,
    response=cols_all_endog,
    shock=["epu"],
    n_columns=3,
    n_rows=3,
    maintitle="EPU shock when HH debt is above threshold",
    show_fig=False,
    save_pic=False,
    annot_size=14,
    font_size=14,
)
# save irf (need to use kaleido==0.1.0post1)
fig.write_image(path_output + "panelthresholdlp_irf.png", height=768, width=1366)  

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
