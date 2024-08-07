# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, get_data_from_api_ceic
import statsmodels.tsa.api as smt
from statsmodels.tsa.ar_model import ar_select_order
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
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
manual_data = ast.literal_eval(os.getenv("MANUAL_DOWNLOAD_DATA"))

# %%
# I --- Wrangle YoY
# Load data
df = pd.read_parquet(path_data + "data_macro_raw.parquet")
# Set groupby cols
cols_groups = ["country", "quarter"]
# Check countries with domestic EPU
# df.loc[~(df["epu"].isna()), :].copy()["country"].unique()
# Reduce countries (exclude aggregates)
list_countries_keep = [
    "australia",
    "belgium",
    "brazil",
    "canada",
    "chile",
    "china",
    "colombia",
    "croatia",
    "denmark",
    "france",
    "germany",
    "greece",
    "hong_kong_sar_china_",
    "india",
    "ireland",
    "italy",
    "japan",
    "mexico",
    "netherlands",
    "pakistan",
    "russian_federation",
    "singapore",
    "south_korea",
    "spain",
    "sweden",
    "united_kingdom",
    "united_states",
]
df = df[df["country"].isin(list_countries_keep)].copy()
df = df.reset_index(drop=True)
# Relative to nominal GDP in USD
# flows and reserves are in mil
for col in ["fdi", "pi_debt", "pi_equity", "fxr"]:
    df[col + "_ngdp"] = 100 * (df[col] / df["ngdp_usd_nsa"])
# debt is in bil (use 4q rolling sum for denominator)
df["ngdp_usd_nsa_4qrollsum"] = df.groupby("country")["ngdp_usd_nsa"].rolling(4).sum().reset_index(drop=True)
for col in ["privdebt", "govdebt", "hhdebt", "corpdebt"]:
    df[col + "_ngdp"] = 100 * ((1000 * df[col]) / df["ngdp_usd_nsa_4qrollsum"])
# Retain these columns as levels for reference later
for col in ["privdebt", "govdebt", "hhdebt", "corpdebt"]:
    df[col + "_ngdp" + "_ref"] = df[col + "_ngdp"].copy()
for col in ["epu"]:
    df[col + "_ref"] = df[col].copy()
# YoY growth
for col in [
    "reer",
    "ber",
    "equity",
    "privdebt",
    "govdebt",
    "hhdebt",
    "corpdebt",
    "fxr",
    "brent"
]:
    df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
# YoY diff
cols_flow_ngdp = [i + "_ngdp" for i in ["fdi", "pi_debt", "pi_equity"]]
cols_stock_ngdp = [
    i + "_ngdp" for i in ["fxr", "privdebt", "govdebt", "hhdebt", "corpdebt"]
]
cols_rates = ["urate", "policyrate", "stir", "ltir", "blr"] + ["epu"]
for col in cols_rates + cols_stock_ngdp + cols_flow_ngdp:
    df[col] = df[col] - df.groupby("country")[col].shift(4)
# Capital flows
df["pi_ngdp"] = df["pi_equity_ngdp"] + df["pi_debt_ngdp"]
df["capflows_ngdp"] = df["pi_equity_ngdp"] + df["pi_debt_ngdp"] + df["fdi_ngdp"]
# Save processed output
df.to_parquet(path_data + "data_macro_yoy" + ".parquet")

# %%
# II --- Wrangle levels
# Load data
df = pd.read_parquet(path_data + "data_macro_raw.parquet")
# Set groupby cols
cols_groups = ["country", "quarter"]
# Check countries with domestic EPU
# df.loc[~(df["epu"].isna()), :].copy()["country"].unique()
# Reduce countries (exclude aggregates)
list_countries_keep = [
    "australia",
    "belgium",
    "brazil",
    "canada",
    "chile",
    "china",
    "colombia",
    "croatia",
    "denmark",
    "france",
    "germany",
    "greece",
    "hong_kong_sar_china_",
    "india",
    "ireland",
    "italy",
    "japan",
    "mexico",
    "netherlands",
    "pakistan",
    "russian_federation",
    "singapore",
    "south_korea",
    "spain",
    "sweden",
    "united_kingdom",
    "united_states",
]
df = df[df["country"].isin(list_countries_keep)].copy()
df = df.reset_index(drop=True)
# Relative to nominal GDP in USD
# flows and reserves are in mil
for col in ["fdi", "pi_debt", "pi_equity", "fxr"]:
    df[col + "_ngdp"] = 100 * (df[col] / df["ngdp_usd_nsa"])
# debt is in bil (use 4q rolling sum for denominator)
df["ngdp_usd_nsa_4qrollsum"] = df.groupby("country")["ngdp_usd_nsa"].rolling(4).sum().reset_index(drop=True)
for col in ["privdebt", "govdebt", "hhdebt", "corpdebt"]:
    df[col + "_ngdp"] = 100 * ((1000 * df[col]) / df["ngdp_usd_nsa_4qrollsum"])
# Retain these columns as levels for reference later
for col in ["privdebt", "govdebt", "hhdebt", "corpdebt"]:
    df[col + "_ngdp" + "_ref"] = df[col + "_ngdp"].copy()
for col in ["epu"]:
    df[col + "_ref"] = df[col].copy()
# YoY growth
for col in [
    "reer",
    "ber",
    "equity",
    "privdebt",
    "govdebt",
    "hhdebt",
    "corpdebt",
    "fxr",
    "brent"
]:
    df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
# YoY diff
cols_flow_ngdp = [i + "_ngdp" for i in ["fdi", "pi_debt", "pi_equity"]]
cols_stock_ngdp = [
    i + "_ngdp" for i in ["fxr", "privdebt", "govdebt", "hhdebt", "corpdebt"]
]
# cols_rates = ["urate", "policyrate", "stir", "ltir", "blr"] + ["epu"]
for col in cols_stock_ngdp + cols_flow_ngdp:
    df[col] = df[col] - df.groupby("country")[col].shift(4)
# Capital flows
df["pi_ngdp"] = df["pi_equity_ngdp"] + df["pi_debt_ngdp"]
df["capflows_ngdp"] = df["pi_equity_ngdp"] + df["pi_debt_ngdp"] + df["fdi_ngdp"]
# Save processed output
df.to_parquet(path_data + "data_macro_yoy_ratesinlevels" + ".parquet")

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
