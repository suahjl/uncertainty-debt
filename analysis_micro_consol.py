# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, fe_reg
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
    cut_off_start_year: int,
    cut_off_end_year: int,
    col_entity: str,
    col_country_micro: str,
    col_country_macro: str,
    cols_endog_micro: list[str],
    cols_endog_macro: list[str],
    cols_exog_macro: list[str],
    cols_endog_ordered: list[str],
    cols_threshold: list[str],
    threshold_option: str,
    col_y_reg_threshold_selection: str,
    threshold_ranges: list[float],
    threshold_range_skip: float,
    col_x_reg_interacted_with_threshold: list[str],
    shocks_to_plot: list[str],
    lp_horizon: int,
    file_suffixes: str,
    input_df_suffix: str,
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
    col_time = "year"
    # Load micro data
    df = pd.read_parquet(path_data + "data_micro_" + input_df_suffix + ".parquet")
    # Testing with US
    # df = df[df["country"] == "united_states"]
    # Creditor firms?
    # df = df[df["debtebitda_ref"] >= 0]
    # Drop extreme values
    df = df[
        (df[cols_threshold[0]] >= df[cols_threshold[0]].quantile(0.005))
        & (df[cols_threshold[0]] <= df[cols_threshold[0]].quantile(0.995))
    ]
    # Trim columns
    df = df[
        [col_entity]
        + [col_time]
        + [col_country_micro]
        + cols_endog_micro
        + cols_threshold
    ]
    # Drop NA
    df = df.dropna()
    # Drop entities with insufficient data points
    micro_id_minyear = df.groupby("id")[col_time].min().reset_index()
    micro_id_minyear.loc[micro_id_minyear[col_time] <= cut_off_start_year, "keep"] = (
        True
    )
    micro_id_minyear = micro_id_minyear[micro_id_minyear["keep"] == True]
    micro_id_minyear = list(micro_id_minyear["id"])
    print(
        "Firm-level dataframe has "
        + str(len(micro_id_minyear))
        + " firms with "
        + col_time
        + " coverage longer than "
        + str(cut_off_start_year)
    )
    df = df[df["id"].isin(micro_id_minyear)]

    # B --- Load + prep macro
    # Load macro data for exog block
    df_macro = pd.read_parquet(path_data + "data_macro_yoy.parquet")
    # Convert macro data into annual freq
    df_macro = df_macro.rename(columns={"quarter": col_time})
    df_macro[col_time] = pd.to_datetime(
        df_macro[col_time].astype("str")
    ).dt.year  # fixed to year
    df_macro = df_macro.groupby([col_country_macro, col_time]).mean().reset_index()
    # Trim dates for harmonisation
    df_macro = df_macro[
        (df_macro[col_time] >= cut_off_start_year)
        & (df_macro[col_time] <= cut_off_end_year)
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
    df = df.sort_values(by=["id", col_time], ascending=[True, True])
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
    # Sort
    df = df.sort_values(by=["id", col_time], ascending=[True, True])
    df = df.reset_index(drop=True)
    # Deal with infs
    df = df.replace(np.inf, np.nan)
    df = df.replace(-np.inf, np.nan)
    for col in cols_endog_micro + cols_threshold:
        df[col] = df.groupby("id")[col].fillna(method="ffill").reset_index(drop=True)
    # Drop NA again (end points)
    df = df.dropna()
    n_analysis = len(df["id"].unique())
    print("Final dataframe has " + str(n_analysis) + " firms")

    # F --- Deal with threshold variable
    if threshold_option == "p75":
        df.loc[
            df[cols_threshold[0]] >= df[cols_threshold[0]].quantile(0.75),
            "threshold_on",
        ] = 1
        df.loc[
            df[cols_threshold[0]] < df[cols_threshold[0]].quantile(0.75), "threshold_on"
        ] = 0
    elif threshold_option == "pols_minaicc":
        aicc_file_name = (
            path_output
            + "micro_thresholdselection_"
            + "modwith_"
            + cols_endog_ordered[0]
            + "_"
            + cols_endog_ordered[1]
            + file_suffixes
            + ".csv"
        )
        if not os.path.isfile(aicc_file_name):
            # base columns on lhs and rhs
            cols_x_reg_threshold_selection = (
                cols_endog_micro + cols_endog_macro + cols_exog_macro
            )
            cols_x_reg_threshold_selection = [
                col
                for col in cols_x_reg_threshold_selection
                if col not in [col_y_reg_threshold_selection]
            ]
            # main frame to keep log likelihoods
            df_loglik = pd.DataFrame(
                columns=[cols_threshold[0] + "_threshold", "loglik"]
            )
            df_aicc = pd.DataFrame(columns=[cols_threshold[0] + "_threshold", "aicc"])
            # loop over threshold candidates
            for threshold in np.arange(
                threshold_ranges[0], threshold_ranges[1], threshold_range_skip
            ):
                # print threshold candidate
                print("Checking AICc for " + cols_threshold[0] + ": " + str(threshold))
                # separate data
                df_threshold = df.copy()
                # create threshold variable
                df_threshold.loc[
                    df_threshold[cols_threshold[0]] > threshold,
                    cols_threshold[0] + "_abovethreshold",
                ] = 1
                df_threshold.loc[
                    df_threshold[cols_threshold[0]] <= threshold,
                    cols_threshold[0] + "_abovethreshold",
                ] = 0
                # rhs terms
                cols_x_reg_threshold_selection = cols_x_reg_threshold_selection + [
                    cols_threshold[0] + "_abovethreshold"
                ]  # X + Y + XY
                # interaction terms
                for col in col_x_reg_interacted_with_threshold:
                    df_threshold[col + "_" + cols_threshold[0] + "_abovethreshold"] = (
                        df_threshold[col]
                        * df_threshold[cols_threshold[0] + "_abovethreshold"]
                    )
                    cols_x_reg_threshold_selection += [
                        col + "_" + cols_threshold[0] + "_abovethreshold"
                    ]  # add all interaction terms
                # estimate the model
                mod, res, params_table, joint_teststats, reg_det = fe_reg(
                    df=df_threshold,
                    y_col=col_y_reg_threshold_selection,
                    x_cols=cols_x_reg_threshold_selection,
                    i_col=col_entity,
                    t_col=col_time,
                    fixed_effects=True,
                    time_effects=False,
                    cov_choice="robust",
                )
                # log likelihood
                df_loglik_sub = pd.DataFrame(
                    {
                        cols_threshold[0] + "_threshold": [threshold],
                        "loglik": [res.loglik],
                    }
                )
                df_loglik = pd.concat([df_loglik, df_loglik_sub], axis=0)  # top down
                # AICc
                df_aicc_sub = pd.DataFrame(
                    {
                        cols_threshold[0] + "_threshold": [threshold],
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
            # save file
            df_aicc.to_csv(
                aicc_file_name,
                index=False,
            )
        elif os.path.isfile(aicc_file_name):
            df_aicc = pd.read_csv(aicc_file_name)
        # find optimal threshold
        threshold_optimal = df_aicc.loc[
            df_aicc["aicc"] == df_aicc["aicc"].min(),
            cols_threshold[0] + "_threshold",
        ].reset_index(drop=True)[0]
        print(df_aicc)
        print(
            "Optimal threshold for "
            + cols_threshold[0]
            + " is "
            + str(threshold_optimal)
        )
        # create threshold variable
        df.loc[df[cols_threshold[0]] >= threshold_optimal, "threshold_on"] = 1
        df.loc[df[cols_threshold[0]] < threshold_optimal, "threshold_on"] = 0

    # X --- Estimate LP model
    # Convert to multiindex
    df = df.set_index([col_entity] + [col_time])
    # Run threshold LP
    irf_on, irf_off = lp.ThresholdPanelLPX(
        data=df,
        Y=cols_endog_ordered,
        X=cols_exog_macro,
        threshold_var="threshold_on",
        response=cols_endog_ordered,
        horizon=lp_horizon,
        lags=1,
        varcov="robust",
        ci_width=0.8,
    )
    # Plot IRFs
    for shock in shocks_to_plot:
        for response in tqdm(cols_endog_micro):
            fig = lp.ThresholdIRFPlot(
                irf_threshold_on=irf_on,
                irf_threshold_off=irf_off,
                response=[response],
                shock=[shock],
                n_columns=1,
                n_rows=1,
                maintitle="Response of "
                + response
                + " to "
                + shock
                + " shocks; threshold variable: "
                + cols_threshold[0]
                + "; N="
                + str(n_analysis),
                show_fig=False,
                save_pic=False,
                annot_size=12,
                font_size=16,
            )
            fig.write_image(
                path_output
                + "micro_panelthresholdlp_"
                + file_suffixes
                + "modwith_"
                + cols_endog_ordered[0]
                + "_"
                + cols_endog_ordered[1]
                + "_shock"
                + shock
                + "_response"
                + response
                + ".png",
                height=768,
                width=1366,
            )


# %%
# II --- Set up
col_entity = "id"
col_country_micro = "country"
col_country_macro = "country"
cols_endog_micro = ["debt", "capex", "revenue"]
cols_endog_macro = ["maxminepu", "maxminstir", "stir", "gdp", "corecpi"]
cols_endog_ordered = [
    "maxminepu",
    "maxminstir",
    "stir",
    "gdp",
    "corecpi",
    "debt",
    "capex",
    "revenue",
]
cols_exog_macro = ["maxminbrent"]
cols_threshold = [
    "debtebitda_ref"
]  # can only handle 1 for now (no reason to handle more)
shocks_to_plot = ["maxminepu", "maxminstir"]  # uncertainty, mp
lp_horizon = 5
cut_off_start_year = 2000
cut_off_end_year = 2023
threshold_option = "pols_minaicc"
col_y_reg_threshold_selection = "capex"
threshold_ranges = [0, 750]
threshold_range_skip = 5
col_x_reg_interacted_with_threshold = ["maxminepu"]

# %%
# II.A --- Base analysis
do_everything(
    cut_off_start_year=cut_off_start_year,
    cut_off_end_year=cut_off_end_year,
    col_entity=col_entity,
    col_country_micro=col_country_micro,
    col_country_macro=col_country_macro,
    cols_endog_micro=cols_endog_micro,
    cols_endog_macro=cols_endog_macro,
    cols_exog_macro=cols_exog_macro,
    cols_endog_ordered=cols_endog_ordered,
    cols_threshold=cols_threshold,
    threshold_option=threshold_option,
    col_y_reg_threshold_selection=col_y_reg_threshold_selection,
    threshold_ranges=threshold_ranges,
    threshold_range_skip=threshold_range_skip,
    col_x_reg_interacted_with_threshold=col_x_reg_interacted_with_threshold,
    shocks_to_plot=shocks_to_plot,
    lp_horizon=lp_horizon,
    file_suffixes="",  # "abc_" or ""
    input_df_suffix="yoy",
)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
