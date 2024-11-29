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
from tabulate import tabulate

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
path_ceic = "./ceic/"


# %%
# I --- Function
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


def wrangle_data(option: str, country_coverage="epu", maxminshocks_reflength=4):
    # Load data
    df = pd.read_parquet(path_data + "data_macro_raw.parquet")
    # Check countries with domestic EPU
    # df.loc[~(df["epu"].isna()), :].copy()["country"].unique()
    # Reduce countries (exclude aggregates)
    if country_coverage == "epu":
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
    elif country_coverage == "wui":
        list_countries_keep = [
            "albania",
            "algeria",
            "argentina",
            "armenia",
            "australia",
            "austria",
            "azerbaijan",
            "bangladesh",
            "belarus",
            "belgium",
            "bolivia",
            "bosnia_and_herzegovina",
            "botswana",
            "brazil",
            "bulgaria",
            "cambodia",
            "canada",
            "chile",
            "china",
            "colombia",
            "croatia",
            "czech_republic",
            "denmark",
            "ecuador",
            "egypt",
            "finland",
            "france",
            "georgia",
            "germany",
            "ghana",
            "greece",
            "hong_kong_sar_china_",
            "hungary",
            "india",
            "indonesia",
            "iran",
            "ireland",
            "israel",
            "italy",
            "ivory_coast",
            "japan",
            "jordan",
            "kazakhstan",
            "kenya",
            "kuwait",
            "kyrgyzstan",
            "laos",
            "latvia",
            "lebanon",
            "lithuania",
            "malawi",
            "malaysia",
            "mexico",
            "moldova",
            "mongolia",
            "morocco",
            "mozambique",
            "myanmar",
            "nepal",
            "netherlands",
            "new_zealand",
            "nigeria",
            "north_macedonia",
            "norway",
            "oman",
            "pakistan",
            "panama",
            "paraguay",
            "peru",
            "philippines",
            "poland",
            "portugal",
            "qatar",
            "romania",
            "saudi_arabia",
            "singapore",
            "slovakia",
            "slovenia",
            "south_africa",
            "south_korea",
            "spain",
            "sri_lanka",
            "sudan",
            "sweden",
            "switzerland",
            "taiwan",
            "tajikistan",
            "thailand",
            "tunisia",
            "turkey",
            "ukraine",
            "united_arab_emirates",
            "united_kingdom",
            "united_states",
            "uruguay",
            "uzbekistan",
            "venezuela",
            "vietnam",
            "yemen",
            "zambia",
        ]
    elif country_coverage == "uct":
        list_countries_keep = list(df["country"].unique())  # global variable
    df = df[df["country"].isin(list_countries_keep)].copy()
    df = df.reset_index(drop=True)
    # Relative to nominal GDP in USD
    # flows and reserves are in mil
    for col in ["fdi", "pi_debt", "pi_equity", "fxr"]:
        df[col + "_ngdp"] = 100 * (df[col] / df["ngdp_usd_nsa"])
    # debt is in bil (use 4q rolling sum for denominator)
    df["ngdp_usd_nsa_4qrollsum"] = (
        df.groupby("country")["ngdp_usd_nsa"].rolling(4).sum().reset_index(drop=True)
    )
    for col in ["privdebt", "govdebt", "hhdebt", "corpdebt"]:
        df[col + "_ngdp"] = 100 * ((1000 * df[col]) / df["ngdp_usd_nsa_4qrollsum"])
    # Retain these columns as levels for reference later
    for col in ["privdebt", "govdebt", "hhdebt", "corpdebt"]:
        df[col + "_ngdp" + "_ref"] = df[col + "_ngdp"].copy()
    for col in ["epu", "wui", "uct"]:  # uncertainty
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
        "brent",
    ]:
        df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
    # YoY diff
    cols_flow_ngdp = [i + "_ngdp" for i in ["fdi", "pi_debt", "pi_equity"]]
    cols_stock_ngdp = [
        i + "_ngdp" for i in ["fxr", "privdebt", "govdebt", "hhdebt", "corpdebt"]
    ]
    cols_rates = ["urate", "policyrate", "stir", "ltir", "blr", "stgby"] + [
        "epu",
        "wui",
        "uct",
    ]  # uncertainty
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
    def hamilton_shock(col: str, option: str, ref_period: int = 4):
        df["_zero"] = 0
        col_x_cands = []
        for i in range(1, ref_period + 1):
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

    def reverse_hamilton_shock(col: str, option: str, ref_period: int = 4):
        df["_zero"] = 0
        col_x_cands = []
        for i in range(1, ref_period + 1):
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

    def twoway_hamilton_shock(col: str, option: str, ref_period: int = 4):
        df["_zero"] = 0
        col_x_cands = []
        for i in range(1, ref_period + 1):
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
    for col in ["epu", "wui", "uct", "stir", "ltir", "m2", "us_jln"]:  # uncertainty
        hamilton_shock(col, option="diff", ref_period=maxminshocks_reflength)
        reverse_hamilton_shock(col, option="diff", ref_period=maxminshocks_reflength)
        twoway_hamilton_shock(col, option="diff", ref_period=maxminshocks_reflength)
    # Growth shocks
    for col in ["brent"]:
        hamilton_shock(col, option="growth", ref_period=maxminshocks_reflength)
        reverse_hamilton_shock(col, option="growth", ref_period=maxminshocks_reflength)
        twoway_hamilton_shock(col, option="growth", ref_period=maxminshocks_reflength)

    # Generate processed output
    return df


# %%
# II --- Wrangle YoY
# Wrangle using EPU data coverage
df_yoy = wrangle_data(option="yoy")
df_yoy_maxminref8 = wrangle_data(option="yoy", maxminshocks_reflength=8)
df_yoy_maxminref6 = wrangle_data(option="yoy", maxminshocks_reflength=6)
df_yoy_ratesinlevels = wrangle_data(option="yoy_ratesinlevels")
# Wrangle using WUI data coverage
df_large_yoy = wrangle_data(option="yoy", country_coverage="wui")
df_large_yoy_maxminref8 = wrangle_data(
    option="yoy", country_coverage="wui", maxminshocks_reflength=8
)
df_large_yoy_maxminref6 = wrangle_data(
    option="yoy", country_coverage="wui", maxminshocks_reflength=6
)
df_large_yoy_ratesinlevels = wrangle_data(
    option="yoy_ratesinlevels", country_coverage="wui"
)
# Wrangle using UCT / global data coverage (no countries will be dropped here)
df_full_yoy = wrangle_data(option="yoy", country_coverage="uct")
df_full_yoy_maxminref8 = wrangle_data(
    option="yoy", country_coverage="uct", maxminshocks_reflength=8
)
df_full_yoy_maxminref6 = wrangle_data(
    option="yoy", country_coverage="uct", maxminshocks_reflength=6
)
df_full_yoy_ratesinlevels = wrangle_data(
    option="yoy_ratesinlevels", country_coverage="uct"
)
# Save processed output
df_yoy.to_parquet(path_data + "data_macro_yoy" + ".parquet")
df_yoy_maxminref8.to_parquet(path_data + "data_macro_yoy_maxminref8" + ".parquet")
df_yoy_maxminref6.to_parquet(path_data + "data_macro_yoy_maxminref6" + ".parquet")
df_yoy_ratesinlevels.to_parquet(path_data + "data_macro_yoy_ratesinlevels" + ".parquet")
df_large_yoy.to_parquet(path_data + "data_macro_large_yoy" + ".parquet")
df_large_yoy_maxminref8.to_parquet(
    path_data + "data_macro_large_yoy_maxminref8" + ".parquet"
)
df_large_yoy_maxminref6.to_parquet(
    path_data + "data_macro_large_yoy_maxminref6" + ".parquet"
)
df_large_yoy_ratesinlevels.to_parquet(
    path_data + "data_macro_large_yoy_ratesinlevels" + ".parquet"
)
df_full_yoy.to_parquet(path_data + "data_macro_full_yoy" + ".parquet")
df_full_yoy_maxminref8.to_parquet(
    path_data + "data_macro_full_yoy_maxminref8" + ".parquet"
)
df_full_yoy_maxminref6.to_parquet(
    path_data + "data_macro_full_yoy_maxminref6" + ".parquet"
)
df_full_yoy_ratesinlevels.to_parquet(
    path_data + "data_macro_full_yoy_ratesinlevels" + ".parquet"
)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
