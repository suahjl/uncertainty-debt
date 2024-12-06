# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import scatterplot
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


def wrangle_data():
    # prelims
    df = pd.read_parquet(path_data + "data_micro_quarterly_raw.parquet")
    # drop entity name for space
    del df["entity"]
    # convert public private to dummy
    df = df.rename(columns={"type": "public"})
    df.loc[df["public"] == "Public Company", "public"] = 1
    df.loc[df["public"] == "Private Company", "public"] = 0
    # convert status to dummy
    df = df.rename(columns={"status": "operating"})
    df.loc[
        (
            (df["operating"] == "Operating Subsidiary")
            | (df["operating"] == "Operating")
        ),
        "operating",
    ] = 1
    df.loc[
        ~(df["operating"] == 1),
        "operating",
    ] = 0
    # interpolate
    # df = df.groupby("id").apply(lambda group: group.interpolate(method="linear"))  # this thing takes forever (consider vectorising)
    cols_identifiers = ["id", "country", "quarter", "public", "operating"]
    for col in [i for i in df.columns if i not in cols_identifiers]:
        df[col] = df.groupby("id")[col].transform(lambda group: group.interpolate(method="linear"))
    df = df.reset_index(drop=True)
    # del df["level_1"]
    # somehow some values are on different rows for the same i and t
    cols_value = [i for i in df.columns if i not in cols_identifiers]
    df = df.groupby(cols_identifiers)[cols_value].mean().reset_index(drop=False)
    # from accounting to econs
    df["capex"] = -1 * df["capex"]
    # create new variables
    df["debtrevenue"] = 100 * (
        df["debt"] / df.groupby("id")["revenue"].rolling(4).sum().reset_index(drop=True)
    )
    # df["debtebitda"] = 100 * (df["debt"] / df["ebitda"])
    df["debtcapex"] = 100 * (
        df["debt"] / df.groupby("id")["capex"].rolling(4).sum().reset_index(drop=True)
    )
    # keep some as levels
    cols_unchanged = [
        # "debtebitda",
        "debtrevenue",
        "debtcapex",
    ]
    for col in cols_unchanged:
        df[col + "_ref"] = df[col].copy()
    # convert to YoY
    cols_yoy = [
        "debt",
        # "ebitda",
        "revenue",
        "capex",
    ]
    for col in cols_yoy:
        df[col] = 100 * ((df[col] / df[col].shift(1)) - 1)
    cols_yoy_diff = [
        # "debtebitda",
        "debtrevenue",
        "debtcapex",
    ]
    for col in cols_yoy_diff:
        df[col] = df[col] - df[col].shift(1)
    # output
    return df


# %%
# II --- Wrangle data
df = wrangle_data()

# %%
# III --- Export
df.to_parquet(path_data + "data_micro_quarterly_yoy.parquet")  # , compression="brotli")

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
