# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, fe_reg, heatmap
import localprojections as lp
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast
from tabulate import tabulate
import ruptures as rpt
from chow_test import chow_test
import warnings

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
path_ceic = "./ceic/"
tel_config = os.getenv("TEL_CONFIG")
t_start = date(1990, 1, 1)

pd.options.mode.chained_assignment = None
warnings.filterwarnings(
    "ignore"
)  # MissingValueWarning when localprojections implements shift operations

heatmaps_y_fontsize = 12
heatmaps_x_fontsize = 12
heatmaps_title_fontsize = 12
heatmaps_annot_fontsize = 12


# %%
# I --- Functions
def quadrant_thresholdsearch_fe(
    data: pd.DataFrame,
    y_col: str,
    threshold_input_cols: list[str],
    new_threshold_col_names: list[str],
    x_interactedwith_threshold_col: str,
    other_x_cols: list[str],
    threshold_ranges: list[list[float]],
    threshold_range_skip: float,
    entity_col: str,
    time_col: str,
):
    # deep copy
    df = data.copy()
    # main frame to keep log likelihoods
    df_loglik = pd.DataFrame(columns=["threshold", "loglik"])
    df_aicc = pd.DataFrame(columns=["threshold", "aicc"])
    # iterate through threshold candidates
    for threshold0 in np.arange(
        threshold_ranges[0][0], threshold_ranges[0][1], threshold_range_skip
    ):  # first threshold variable
        for threshold1 in np.arange(
            threshold_ranges[1][0], threshold_ranges[1][1], threshold_range_skip
        ):  # second threshold variable
            # For inspection
            print(
                "Checking "
                + threshold_input_cols[0]
                + ": "
                + str(threshold0)
                + " and "
                + threshold_input_cols[1]
                + ": "
                + str(threshold1)
            )
            # create threshold variables
            df.loc[
                df[threshold_input_cols[0]] <= threshold0, new_threshold_col_names[0]
            ] = 1  # first
            df.loc[
                df[threshold_input_cols[0]] > threshold0, new_threshold_col_names[0]
            ] = 0
            df.loc[
                df[threshold_input_cols[1]] <= threshold1, new_threshold_col_names[1]
            ] = 1  # second
            df.loc[
                df[threshold_input_cols[1]] > threshold1, new_threshold_col_names[1]
            ] = 0
            # interactions
            df[x_interactedwith_threshold_col + "_" + new_threshold_col_names[0]] = (
                df[x_interactedwith_threshold_col] * df[new_threshold_col_names[0]]
            )  # first
            df[x_interactedwith_threshold_col + "_" + new_threshold_col_names[1]] = (
                df[x_interactedwith_threshold_col] * df[new_threshold_col_names[1]]
            )  # second
            df[new_threshold_col_names[0] + "_" + new_threshold_col_names[1]] = (
                df[new_threshold_col_names[0]] * df[new_threshold_col_names[1]]
            )  # threshold interaction
            df[
                x_interactedwith_threshold_col
                + "_"
                + new_threshold_col_names[0]
                + "_"
                + new_threshold_col_names[1]
            ] = (
                df[x_interactedwith_threshold_col]
                * df[new_threshold_col_names[0]]
                * df[new_threshold_col_names[1]]
            )  # triple interaction
            # estimate
            mod, res, params_table, joint_teststats, reg_det = fe_reg(
                df=df,
                y_col=y_col,
                x_cols=other_x_cols
                + [
                    x_interactedwith_threshold_col,
                    new_threshold_col_names[0],  # first
                    new_threshold_col_names[1],  # second
                    x_interactedwith_threshold_col
                    + "_"
                    + new_threshold_col_names[0],  # first
                    x_interactedwith_threshold_col
                    + "_"
                    + new_threshold_col_names[1],  # second
                    new_threshold_col_names[0]
                    + "_"
                    + new_threshold_col_names[1],  # threshold interaction
                    x_interactedwith_threshold_col
                    + "_"
                    + new_threshold_col_names[0]
                    + "_"
                    + new_threshold_col_names[1],  # triple interaction
                ],  # total seven terms
                i_col=entity_col,
                t_col=time_col,
                fixed_effects=True,  # SET TO TRUE for FE, FALSE for POLS
                time_effects=False,
                cov_choice="robust",
            )
            # log likelihood
            df_loglik_sub = pd.DataFrame(
                {
                    threshold_input_cols[0] + "_threshold": [threshold0],
                    threshold_input_cols[1] + "_threshold": [threshold1],
                    "loglik": [res.loglik],
                }
            )
            df_loglik = pd.concat([df_loglik, df_loglik_sub], axis=0)  # top down
            # AICc
            df_aicc_sub = pd.DataFrame(
                {
                    threshold_input_cols[0] + "_threshold": [threshold0],
                    threshold_input_cols[1] + "_threshold": [threshold1],
                    "aicc": [
                        (-2 * res.loglik + 2 * res.df_model)
                        + (
                            (2 * res.df_model * (res.df_model + 1))
                            / (res.entity_info.total - res.df_model - 1)
                        )
                    ],
                }
            )
            df_aicc = pd.concat([df_aicc, df_aicc_sub], axis=0)  # top down
    # find optimal threshold
    # threshold_optimal = df_loglik.loc[
    #     df_loglik["loglik"] == df_loglik["loglik"].max(), "threshold"
    # ].reset_index(drop=True)[0]
    threshold0_optimal = df_aicc.loc[
        df_aicc["aicc"] == df_aicc["aicc"].min(), threshold_input_cols[0] + "_threshold"
    ].reset_index(drop=True)[0]
    threshold1_optimal = df_aicc.loc[
        df_aicc["aicc"] == df_aicc["aicc"].min(), threshold_input_cols[1] + "_threshold"
    ].reset_index(drop=True)[0]
    print(df_aicc)
    # estimate optimal model
    df.loc[
        df[threshold_input_cols[0]] <= threshold0_optimal, new_threshold_col_names[0]
    ] = 1  # first
    df.loc[
        df[threshold_input_cols[0]] > threshold0_optimal, new_threshold_col_names[0]
    ] = 0
    df.loc[
        df[threshold_input_cols[1]] <= threshold1_optimal, new_threshold_col_names[1]
    ] = 1  # second
    df.loc[
        df[threshold_input_cols[1]] > threshold1_optimal, new_threshold_col_names[1]
    ] = 0
    # interactions
    df[x_interactedwith_threshold_col + "_" + new_threshold_col_names[0]] = (
        df[x_interactedwith_threshold_col] * df[new_threshold_col_names[0]]
    )  # first
    df[x_interactedwith_threshold_col + "_" + new_threshold_col_names[1]] = (
        df[x_interactedwith_threshold_col] * df[new_threshold_col_names[1]]
    )  # second
    df[new_threshold_col_names[0] + "_" + new_threshold_col_names[1]] = (
        df[new_threshold_col_names[0]] * df[new_threshold_col_names[1]]
    )  # threshold interaction
    df[
        x_interactedwith_threshold_col
        + "_"
        + new_threshold_col_names[0]
        + "_"
        + new_threshold_col_names[1]
    ] = (
        df[x_interactedwith_threshold_col]
        * df[new_threshold_col_names[0]]
        * df[new_threshold_col_names[1]]
    )  # triple interaction
    mod, res, params_table, joint_teststats, reg_det = fe_reg(
        df=df,
        y_col=y_col,
        x_cols=other_x_cols
        + [
            x_interactedwith_threshold_col,
            new_threshold_col_names[0],  # first
            new_threshold_col_names[1],  # second
            x_interactedwith_threshold_col + "_" + new_threshold_col_names[0],  # first
            x_interactedwith_threshold_col + "_" + new_threshold_col_names[1],  # second
            new_threshold_col_names[0]
            + "_"
            + new_threshold_col_names[1],  # threshold interaction
            x_interactedwith_threshold_col
            + "_"
            + new_threshold_col_names[0]
            + "_"
            + new_threshold_col_names[1],  # triple interaction
        ],
        i_col=entity_col,
        t_col=time_col,
        fixed_effects=True,
        time_effects=False,
        cov_choice="robust",
    )
    # output
    return params_table, threshold0_optimal, threshold1_optimal, df_loglik, df_aicc


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
# ------- LOOP ------
list_shock_prefixes = ["max", "min", "maxmin"]
# list_mp_variables = [i + "ltir" for i in list_shock_prefixes]
# list_uncertainty_variables = [i + "epu" for i in list_shock_prefixes]
list_mp_variables = ["maxminltir"]  # maxminltir
list_uncertainty_variables = ["maxminepu"]  # maxepu
for mp_variable in tqdm(list_mp_variables):
    for uncertainty_variable in tqdm(list_uncertainty_variables):
        print("\nMP variable is " + mp_variable)
        print("Uncertainty variable is " + uncertainty_variable)
        # II --- Load data
        df = pd.read_parquet(path_data + "data_macro_yoy.parquet")
        # III --- Additional wrangling
        # Groupby ref
        cols_groups = ["country", "quarter"]
        # Trim columns
        cols_all_endog = [
            # "epu",
            "ltir",
            # "hhdebt",  # _ngdp
            # "corpdebt",  # _ngdp
            # "govdebt",  # _ngdp
            # "gdp",  # urate gdp
            # "capflows_ngdp",
            "corecpi",  # corecpi cpi
            "reer",
        ]
        cols_all_exog = ["maxminbrent"]  # maxminltir
        cols_threshold = ["hhdebt_ngdp_ref", "govdebt_ngdp_ref"]
        df = df[
            cols_groups
            + cols_all_endog
            + cols_all_exog
            + cols_threshold
            + [uncertainty_variable]
            + [mp_variable]
            + ["gdp"]
        ].copy()
        # Check when the panel becomes balanced
        check_balance_timing(input=df)
        check_balance_endtiming(input=df)
        # Trim more countries
        # if "ltir" in mp_variable:
        countries_drop = [
            "india",  # 2016 Q3
            "denmark",  # ends 2019 Q3
            "china",  # 2007 Q4 and potentially exclusive case
            "chile",  #  2010Q1
            "colombia",  # 2005 Q4
            "germany",  # 2006 Q1
        ]  # 17 countries
        # elif "stgby" in mp_variable:
        #     countries_drop = [
        #         "australia",  # 2014 Q2
        #         "belgium",  # ends 2022 Q3
        #         "india",  # 2012 Q1
        #         "china",  # 2009 Q3 and potentially exclusive case
        #         "colombia",  # 2006 Q4
        #         "germany",  # 2015 Q1
        #         "sweden",  # ends 2020 Q3 --- epu
        #         "mexico",  # ends 2023 Q1 --- epu
        #         "chile",  # ends 2022 Q2  (doesn't have ltir?)
        #     ]  # 10 countries
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
        threshold_variables = ["hhdebt_ngdp", "govdebt_ngdp"]

        # Reset index
        df = df.reset_index(drop=True)
        # Numeric time
        df["time"] = df.groupby("country").cumcount()
        del df["quarter"]
        # Set multiindex
        # df = df.set_index(["country", "time"])

        # IV --- Analysis
        # estimate model
        (
            params_table_fe,
            threshold0_optimal_fe,
            threshold1_optimal_fe,
            df_loglik_fe,
            df_aicc_fe,
        ) = quadrant_thresholdsearch_fe(
            data=df,
            y_col="gdp",
            threshold_input_cols=cols_threshold,
            new_threshold_col_names=[i + "_threshold" for i in threshold_variables],
            x_interactedwith_threshold_col=uncertainty_variable,
            other_x_cols=cols_all_endog + cols_all_exog + [mp_variable],
            threshold_ranges=[
                [
                    int(
                        df.groupby("country")[threshold_variables[0] + "_ref"]
                        .quantile(0.2)
                        .median()  # min()
                    ),
                    int(
                        df.groupby("country")[threshold_variables[0] + "_ref"]
                        .quantile(0.8)
                        .median()  # max()
                    )
                    + int(1),
                ],
                [
                    int(
                        df.groupby("country")[threshold_variables[1] + "_ref"]
                        .quantile(0.2)
                        .median()  # min()
                    ),
                    int(
                        df.groupby("country")[threshold_variables[1] + "_ref"]
                        .quantile(0.8)
                        .median()  # max()
                    )
                    + int(1),
                ],
            ],
            threshold_range_skip=0.5,
            entity_col="country",
            time_col="time",
        )
        file_name = (
            path_output
            + "reg_quadrant_thresholdselection_ltir_reduced_fe_"
            + "modwith_"
            + uncertainty_variable
            + "_"
            + mp_variable
        )
        chart_title = (
            "FE regression "
            + "\n (optimal thresholds: "
            + threshold_variables[0]
            + " = "
            + str(threshold0_optimal_fe)
            + " and "
            + threshold_variables[1]
            + " = "
            + str(threshold1_optimal_fe)
            + ")"
        )
        print(
            "optimal thresholds: "
            + threshold_variables[0]
            + " = "
            + str(threshold0_optimal_fe)
            + " and "
            + threshold_variables[1]
            + " = "
            + str(threshold1_optimal_fe)
        )
        df_opt_threshold = pd.DataFrame(
            {
                threshold_variables[0]: [threshold0_optimal_fe],
                threshold_variables[1]: [threshold1_optimal_fe],
            }
        )
        df_opt_threshold.to_csv(file_name + "_opt_threshold" + ".csv", index=False)
        fig = heatmap(
            input=params_table_fe,
            mask=False,
            colourmap="vlag",
            outputfile=file_name + ".png",
            title=chart_title,
            lb=params_table_fe.min().min(),
            ub=params_table_fe.max().max(),
            format=".4f",
            show_annot=True,
            y_fontsize=heatmaps_y_fontsize,
            x_fontsize=heatmaps_x_fontsize,
            title_fontsize=heatmaps_title_fontsize,
            annot_fontsize=heatmaps_annot_fontsize,
        )
        df_aicc_fe.to_parquet(file_name + "_aiccsearch" + ".parquet")
        df_aicc_fe.to_csv(file_name + "_aiccsearch" + ".csv", index=False)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
