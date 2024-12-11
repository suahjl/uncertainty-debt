# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, fe_reg, lineplot
import localprojections as lp
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast
from tabulate import tabulate
import warnings
import plotly.graph_objects as go
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import itertools

time_start = time.time()


# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
tel_config = os.getenv("TEL_CONFIG")
pd.options.mode.chained_assignment = None
warnings.filterwarnings(
    "ignore"
)  # MissingValueWarning when localprojections implements shift operations


# %%
# I --- Functions
def do_everything(
    cut_off_start_quarter: int,
    cut_off_end_quarter: int,
    col_entity: str,
    col_country_micro: str,
    col_country_macro: str,
    cols_endog_micro: list[str],
    cols_endog_macro: list[str],
    cols_exog_macro: list[str],
    cols_threshold: list[str],
    threshold_option: str,
    file_suffixes: str,
    input_df_suffix: str,
    countries_drop: list[str] = None,
    trim_extreme_ends_perc: float = None,
):
    # A --- Load + prep micro
    # Print key input and output attributes
    print(
        "Now analysing:\n"
        + "data input suffix: "
        + input_df_suffix
        + "\noutput suffix: "
        + file_suffixes
    )
    # Prelims
    col_time = "quarter"
    # Load micro data
    df = pd.read_parquet(
        path_data + "data_micro_quarterly_" + input_df_suffix + ".parquet"
    )
    # PeriodQ
    df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("Q")
    # Testing with US
    # df = df[df["country"] == "united_states"]
    # Creditor firms?
    # df = df[df["debtebitda_ref"] >= 0]
    # Keep operating firms
    print(
        str(len(df["id"].unique()))
        + " firms; "
        + str(len(df.loc[df["operating"] == 1, "id"].unique()))
        + " still operating"
    )
    df = df[df["operating"] == 1].copy()
    # Drop extreme values
    if (trim_extreme_ends_perc is not None) and (trim_extreme_ends_perc > 0):
        df.loc[
            (
                (
                    df[cols_threshold[0]]
                    < df[cols_threshold[0]].quantile(trim_extreme_ends_perc / 2)
                )
                | (
                    df[cols_threshold[0]]
                    > df[cols_threshold[0]].quantile(1 - (trim_extreme_ends_perc / 2))
                )
            ),
            "_extreme",
        ] = 1
        df.loc[
            (
                (
                    df[cols_threshold[0]]
                    >= df[cols_threshold[0]].quantile(trim_extreme_ends_perc / 2)
                )
                & (
                    df[cols_threshold[0]]
                    <= df[cols_threshold[0]].quantile(1 - (trim_extreme_ends_perc / 2))
                )
            ),
            "_extreme",
        ] = 0
        df_extreme = df.groupby(col_entity)["_extreme"].max().reset_index()
        print(
            str(len(df_extreme[df_extreme["_extreme"] == 1]))
            + " firms with extreme values"
        )
        del df["_extreme"]
        df = df.merge(df_extreme, on=col_entity, how="left")
        df = df[df["_extreme"] == 0].copy()
    elif trim_extreme_ends_perc is None:
        pass
    # Trim columns
    df = df[
        [col_entity]
        + [col_time]
        + [col_country_micro]
        + cols_endog_micro
        + cols_threshold
    ]
    print(df)
    # Drop NA
    df = df.dropna()
    # Drop entities with insufficient data points
    micro_id_minquarter = df.groupby(col_entity)[col_time].first().reset_index()
    micro_id_minquarter.loc[
        micro_id_minquarter[col_time] <= cut_off_start_quarter, "keep"
    ] = 1
    micro_id_minquarter = micro_id_minquarter[micro_id_minquarter["keep"] == 1]
    micro_id_minquarter = list(micro_id_minquarter[col_entity])
    print(
        "Firm-level dataframe has "
        + str(len(micro_id_minquarter))
        + " firms with "
        + col_time
        + " coverage longer than "
        + str(cut_off_start_quarter)
    )
    df = df[df[col_entity].isin(micro_id_minquarter)]

    # B --- Load + prep macro
    # Load macro data for exog block
    df_macro = pd.read_parquet(path_data + "data_macro_yoy.parquet")
    # Convert to periodQ
    df_macro[col_time] = pd.to_datetime(df_macro[col_time].astype("str")).dt.to_period(
        "Q"
    )  # fixed to quarter
    df_macro = df_macro.groupby([col_country_macro, col_time]).mean().reset_index()
    # Trim dates for harmonisation
    df_macro = df_macro[
        (df_macro[col_time] >= cut_off_start_quarter)
        & (df_macro[col_time] <= cut_off_end_quarter)
    ]

    # C --- Deal with global variables
    # Keep global variables as exog (use australia since it's the first one)
    df_macro_glob = df_macro.loc[
        df_macro[col_country_macro] == "australia", [col_time] + cols_exog_macro
    ].copy()
    df_macro_glob = df_macro_glob.reset_index(drop=True)
    # Merge
    df = df.merge(
        df_macro_glob, on=[col_time], how="outer"
    )  # use outer to avoid gaps in each panel
    # Sort
    df = df.sort_values(by=[col_entity, col_time], ascending=[True, True])
    df = df.reset_index(drop=True)

    # D --- Back to country-level variables
    # Keep country level data as endog
    df_macro_country = df_macro[[col_country_macro] + [col_time] + cols_endog_macro]
    # Merge
    if col_country_macro == col_country_micro:
        pass
    else:
        df_macro_country = df_macro_country.rename(
            columns={col_country_macro: col_country_micro}
        )
    df = df.merge(
        df_macro_country, on=[col_country_micro, col_time], how="left"
    )  # use left

    # E --- Clean up consolidated dataset
    # Drop countries
    if countries_drop is None:
        pass
    elif countries_drop is not None:
        df = df[~df["country"].isin(countries_drop)].copy()
    # Sort
    df = df.sort_values(by=[col_entity, col_time], ascending=[True, True])
    df = df.reset_index(drop=True)
    # Deal with infs
    df = df.replace(np.inf, np.nan)
    df = df.replace(-np.inf, np.nan)
    for col in cols_endog_micro + cols_threshold:
        df[col] = (
            df.groupby(col_entity)[col].fillna(method="ffill").reset_index(drop=True)
        )
    # Drop NA again (end points)
    df = df.dropna()
    n_analysis = len(df[col_entity].unique())
    # Turn t into numeric
    df[col_time] = df.groupby(col_entity).cumcount()
    # Print N
    print("Final dataframe has " + str(n_analysis) + " firms")

    # F --- Deal with threshold variable
    if threshold_option == "pols_minaicc":
        aicc_file_name = (
            path_output
            + "micro_quarterly_thresholdselection_"
            + "modwith_"
            + cols_endog_macro[0]
            + "_"
            + cols_endog_macro[1]
            + file_suffixes
            + ".csv"
        )
        df_aicc = pd.read_csv(aicc_file_name)
        # find optimal threshold
        threshold_optimal = df_aicc.loc[
            df_aicc["aicc"] == df_aicc["aicc"].min(),
            cols_threshold[0] + "_threshold",
        ].reset_index(drop=True)[0]
        print(
            "Optimal threshold for "
            + cols_threshold[0]
            + " is "
            + str(threshold_optimal)
        )
        # create threshold variable
        df.loc[df[cols_threshold[0]] >= threshold_optimal, "threshold_on"] = 1
        df.loc[df[cols_threshold[0]] < threshold_optimal, "threshold_on"] = 0

    # ----  EVERYTHING ABOVE NEEDS TO MATCH WITH THE LP ANALYSIS SO THAT THE DATA IS EXACTLY THE SAME ----

    # G --- Descriptive stats
    # Change country names
    df[col_country_micro] = df[col_country_micro].str.replace("_", " ").str.title()
    # Columns of interest
    cols_to_analyse = cols_endog_micro + cols_threshold
    # Calculate N
    df_describe = df.groupby(col_country_micro)[col_entity].nunique()
    df_describe = pd.DataFrame(df_describe)
    df_describe.columns = ["N"]
    # Calculate N percentage of total
    df_describe["N%"] = 100 * (df_describe["N"] / n_analysis)
    df_describe["N%"] = df_describe["N%"].round(2)
    df_describe = df_describe.astype("str")
    df_describe["N%"] = df_describe["N%"] + "%"
    # Input T
    df_describe["T"] = df[col_time].max()
    # Add analytical columns
    for col in cols_to_analyse:
        df_describe_analyse = df.groupby(col_country_micro)[col].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
        df_describe_analyse = df_describe_analyse[["10%", "25%", "50%", "75%", "90%"]]
        df_describe_analyse = df_describe_analyse.rename(
            columns={
                "10%": "10th percentile",
                "25%": "25th percentile",
                "50%": "50th percentile",
                "75%": "75th percentile",
                "90%": "90th percentile",
            }
        )
        df_describe_analyse = df_describe.merge(
            df_describe_analyse, left_index=True, right_index=True
        )
        cols_percentiles = [i for i in df_describe_analyse.columns if "percentile" in i]
        df_describe_analyse[cols_percentiles] = df_describe_analyse[
            cols_percentiles
        ].round(1)
        df_describe_analyse = df_describe_analyse.reset_index(drop=False)
        df_describe_analyse.to_csv(
            path_output
            + "micro_quarterly_"
            + file_suffixes
            + "modwith_"
            + cols_endog_macro[0]
            + "_"
            + cols_endog_macro[1]
            + "_"
            + "tabpercentiles"
            + "_"
            + col
            + ".csv",
            index=False,
        )


# %%
# II --- Set up
col_entity = "id"
col_country_micro = "country"
col_country_macro = "country"
cols_endog_micro = ["debt", "capex", "revenue"]
cols_endog_macro = ["maxminepu", "maxminstir", "stir", "gdp", "corecpi"]
cols_exog_macro = ["maxminbrent"]
cols_threshold = [
    "debtrevenue_ref"
]  # can only handle 1 for now (no reason to handle more)
cut_off_start_quarter = "2007Q1"  # "2007Q1"
cut_off_end_quarter = "2024Q1"
threshold_option = "pols_minaicc"  # "pols_minaicc"  "manual"  p75
trim_extreme_ends_perc = None  # 0.01  0.05  0.1

# %%
# II.A --- Base analysis
do_everything(
    cut_off_start_quarter=cut_off_start_quarter,
    cut_off_end_quarter=cut_off_end_quarter,
    col_entity=col_entity,
    col_country_micro=col_country_micro,
    col_country_macro=col_country_macro,
    cols_endog_micro=cols_endog_micro,
    cols_endog_macro=cols_endog_macro,
    cols_exog_macro=cols_exog_macro,
    cols_threshold=cols_threshold,
    threshold_option=threshold_option,
    file_suffixes="",  # "abc_" or ""
    input_df_suffix="yoy",
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        # "colombia",  # 2006 Q4
        # "germany",  # 2006 Q1
        "sweden",  # ends 2020 Q3 --- epu
        "nigeria",
    ],
    trim_extreme_ends_perc=trim_extreme_ends_perc,
)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
