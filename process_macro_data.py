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
# I --- Function
def wrangle_data(option: str):
    # Load data
    df = pd.read_parquet(path_data + "data_macro_raw.parquet")
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
    for col in ["epu", "wui"]:  # uncertainty
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
    cols_rates = ["urate", "policyrate", "stir", "ltir", "blr", "stgby"] + ["epu", "wui"]  # uncertainty
    if option == "yoy":
        for col in cols_rates + cols_stock_ngdp + cols_flow_ngdp:
            df[col] = df[col] - df.groupby("country")[col].shift(4)
    elif option == "yoy_ratesinlevels":
        for col in cols_stock_ngdp + cols_flow_ngdp:
            df[col] = df[col] - df.groupby("country")[col].shift(4)
    # Capital flows
    df["pi_ngdp"] = df["pi_equity_ngdp"] + df["pi_debt_ngdp"]
    df["capflows_ngdp"] = df["pi_equity_ngdp"] + df["pi_debt_ngdp"] + df["fdi_ngdp"]
    # Nested functions to calculate shocks
    def hamilton_shock(col: str, option: str):
        df["_zero"] = 0
        col_x_cands = []
        for i in range(1, 5):
            df[col + str(i)] = df[col].shift(i)
            col_x_cands = col_x_cands + [col + str(i)]
        df["_x"] = df[col_x_cands].max(axis=1)
        if option == "growth":
            df["_z"] = 100 * ((df[col] / df["_x"]) - 1)
        elif option == "diff":
            df["_z"] = df[col] - df["_x"]
        df["max" + col] = df[["_zero", "_z"]].max(axis=1)
        for i in ["_zero", "_x", "_z"] + col_x_cands:
            del df[i]
    
    def reverse_hamilton_shock(col: str, option: str):
        df["_zero"] = 0
        col_x_cands = []
        for i in range(1, 5):
            df[col + str(i)] = df[col].shift(i)
            col_x_cands = col_x_cands + [col + str(i)]
        df["_x"] = df[col_x_cands].min(axis=1)
        if option == "growth":
            df["_z"] = 100 * ((df[col] / df["_x"]) - 1)
        elif option == "diff":
            df["_z"] = df[col] - df["_x"]
        df["min" + col] = df[["_zero", "_z"]].min(axis=1)
        for i in ["_zero", "_x", "_z"] + col_x_cands:
            del df[i]

    def twoway_hamilton_shock(col: str, option: str):
        df["_zero"] = 0
        col_x_cands = []
        for i in range(1, 5):
            df[col + str(i)] = df[col].shift(i)
            col_x_cands = col_x_cands + [col + str(i)]
        df["_xmax"] = df[col_x_cands].max(axis=1)
        df["_xmin"] = df[col_x_cands].min(axis=1)
        if option == "growth":
            df["_zmax"] = 100 * ((df[col] / df["_xmax"]) - 1)
            df["_zmin"] = 100 * ((df[col] / df["_xmin"]) - 1)
        elif option == "diff":
            df["_zmax"] = df[col] - df["_xmax"]
            df["_zmin"] = df[col] - df["_xmin"]
        df["max" + col] = df[["_zero", "_zmax"]].max(axis=1)
        df["min" + col] = df[["_zero", "_zmin"]].min(axis=1)
        df["maxmin" + col] = df["max" + col] + df["min" + col]
        for i in ["_zero", "_xmin", "_zmin", "_xmax", "_zmax"] + col_x_cands:
            del df[i]

    # Diff shocks
    for col in ["epu", "wui", "stir", "ltir", "m2", "us_jln"]:  # uncertainty
        hamilton_shock(col, option="diff")
        reverse_hamilton_shock(col, option="diff")
        twoway_hamilton_shock(col, option="diff")
    # Growth shocks
    for col in ["brent"]:
        hamilton_shock(col, option="growth")
        reverse_hamilton_shock(col, option="growth")
        twoway_hamilton_shock(col, option="growth")

    # Generate processed output
    return df

# %%
# II --- Wrangle YoY
# Wrangle
df_yoy = wrangle_data(option="yoy")
df_yoy_ratesinlevels = wrangle_data(option="yoy_ratesinlevels")
# Save processed output
df_yoy.to_parquet(path_data + "data_macro_yoy" + ".parquet")
df_yoy_ratesinlevels.to_parquet(path_data + "data_macro_yoy_ratesinlevels" + ".parquet")

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
