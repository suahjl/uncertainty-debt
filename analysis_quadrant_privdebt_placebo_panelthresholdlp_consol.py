# %%
import pandas as pd
import numpy as np
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
import warnings
import plotly.graph_objects as go

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


# %%
# I --- Do everything function
def do_everything_quadrant_panelthresholdlp(
    cols_endog_after_shocks: list[str],
    cols_all_exog: list[str],
    list_mp_variables: list[str],
    list_uncertainty_variables: list[str],
    cols_threshold: list[str],
    threshold_variables: list[str],
    countries_drop: list[str],
    file_suffixes: str,  # format: "abc_" or ""
    input_df_suffix="yoy",
):
    # Nested functions
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
        return max_quarter_by_country

    # Loop to estimate
    for mp_variable in tqdm(list_mp_variables):
        for uncertainty_variable in tqdm(list_uncertainty_variables):
            print("\nMP variable is " + mp_variable)
            print("Uncertainty variable is " + uncertainty_variable)
            # II --- Load data
            df = pd.read_parquet(
                path_data + "data_macro_" + input_df_suffix + ".parquet"
            )
            # III --- Additional wrangling
            # Groupby ref
            cols_groups = ["country", "quarter"]
            # Retrieve full list of endogenous variables
            cols_all_endog = [
                uncertainty_variable,
                mp_variable,
            ] + cols_endog_after_shocks
            # Trim columns
            df = df[
                cols_groups + cols_all_endog + cols_all_exog + cols_threshold
            ].copy()
            # Check when the panel becomes balanced
            check_balance_timing(input=df)
            df_pre_balance = check_balance_endtiming(input=df)
            df_pre_balance.to_csv(
                path_output
                + "quadrant_privdebt_placebo_prebalancecountries_"
                + file_suffixes
                + "modwith_"
                + uncertainty_variable
                + "_"
                + mp_variable
                + ".csv",
                index=False,
            )
            # Trim more countries
            df = df[~df["country"].isin(countries_drop)].copy()
            # Check again when panel becomes balanced
            check_balance_timing(input=df)
            df_post_balance = check_balance_endtiming(input=df)
            df_post_balance.to_csv(
                path_output
                + "quadrant_privdebt_placebo_postbalancecountries_"
                + file_suffixes
                + "modwith_"
                + uncertainty_variable
                + "_"
                + mp_variable
                + ".csv",
                index=False,
            )
            # Timebound
            df["date"] = pd.to_datetime(df["quarter"]).dt.date
            df = df[(df["date"] >= t_start)]
            del df["date"]
            # Drop NA
            df = df.dropna(axis=0)

            def find_quadrant_thresholds(
                df: pd.DataFrame,
                threshold_variables: list[str],
                option: str,
                param_choices: list[float],
            ):
                if option == "placebo":
                    # Create random states for threshold variables 1 and 2
                    # Number of rows and columns
                    num_rows = len(df)
                    num_columns = 2
                    # Randomly assign a column index for the '1' in each row
                    column_indices = np.random.randint(0, num_columns, size=num_rows)
                    # Create a 2D array of zeros
                    df_random = np.zeros((num_rows, num_columns), dtype=int)
                    # Use advanced indexing to set the correct positions to 1
                    df_random[np.arange(num_rows), column_indices] = 1
                    # Create df to concat
                    df_random = pd.DataFrame(
                        df_random,
                        columns=[
                            threshold_variables[0] + "_above_threshold",
                            threshold_variables[1] + "_above_threshold",
                        ],
                    )
                    df = pd.concat([df, df_random], axis=1)
                    # quadrant 1 (0,0)
                    df.loc[
                        (
                            (df[threshold_variables[0] + "_above_threshold"] == 0)
                            & (df[threshold_variables[1] + "_above_threshold"] == 0)
                        ),
                        threshold_variables[0]
                        + "0"
                        + "_"
                        + threshold_variables[1]
                        + "0",
                    ] = 1
                    df.loc[
                        ~(
                            (df[threshold_variables[0] + "_above_threshold"] == 0)
                            & (df[threshold_variables[1] + "_above_threshold"] == 0)
                        ),
                        threshold_variables[0]
                        + "0"
                        + "_"
                        + threshold_variables[1]
                        + "0",
                    ] = 0
                    # quadrant 2 (1,0)
                    df.loc[
                        (
                            (df[threshold_variables[0] + "_above_threshold"] == 1)
                            & (df[threshold_variables[1] + "_above_threshold"] == 0)
                        ),
                        threshold_variables[0]
                        + "1"
                        + "_"
                        + threshold_variables[1]
                        + "0",
                    ] = 1
                    df.loc[
                        ~(
                            (df[threshold_variables[0] + "_above_threshold"] == 1)
                            & (df[threshold_variables[1] + "_above_threshold"] == 0)
                        ),
                        threshold_variables[0]
                        + "1"
                        + "_"
                        + threshold_variables[1]
                        + "0",
                    ] = 0
                    # quadrant 3 (0,1)
                    df.loc[
                        (
                            (df[threshold_variables[0] + "_above_threshold"] == 0)
                            & (df[threshold_variables[1] + "_above_threshold"] == 1)
                        ),
                        threshold_variables[0]
                        + "0"
                        + "_"
                        + threshold_variables[1]
                        + "1",
                    ] = 1
                    df.loc[
                        ~(
                            (df[threshold_variables[0] + "_above_threshold"] == 0)
                            & (df[threshold_variables[1] + "_above_threshold"] == 1)
                        ),
                        threshold_variables[0]
                        + "0"
                        + "_"
                        + threshold_variables[1]
                        + "1",
                    ] = 0
                    # quadrant 4 (1,1)
                    df.loc[
                        (
                            (df[threshold_variables[0] + "_above_threshold"] == 1)
                            & (df[threshold_variables[1] + "_above_threshold"] == 1)
                        ),
                        threshold_variables[0]
                        + "1"
                        + "_"
                        + threshold_variables[1]
                        + "1",
                    ] = 1
                    df.loc[
                        ~(
                            (df[threshold_variables[0] + "_above_threshold"] == 1)
                            & (df[threshold_variables[1] + "_above_threshold"] == 1)
                        ),
                        threshold_variables[0]
                        + "1"
                        + "_"
                        + threshold_variables[1]
                        + "1",
                    ] = 0
                    print(
                        "Threshold of "
                        + threshold_variables[0]
                        + " is "
                        + str(param_choices[0])
                    )
                    print(
                        "Threshold of "
                        + threshold_variables[1]
                        + " is "
                        + str(param_choices[1])
                    )
                return df

            # Reset index
            df = df.reset_index(drop=True)

            # Threshold
            df = find_quadrant_thresholds(
                df=df,
                threshold_variables=threshold_variables,
                option="placebo",
                param_choices=[0, 0],
            )

            # Reset index
            df = df.reset_index(drop=True)
            print(df)
            # Count how many countries
            print("Number of countries included: " + str(len(df["country"].unique())))
            # Numeric time
            df["time"] = df.groupby("country").cumcount()
            del df["quarter"]
            # Set multiindex
            df = df.set_index(["country", "time"])

            # IV --- Analysis
            irf_consol = pd.DataFrame(
                columns=[
                    threshold_variables[0] + "_above_threshold",
                    threshold_variables[1] + "_above_threshold",
                    "Shock",
                    "Response",
                    "Horizon",
                    "Mean",
                    "LB",
                    "UB",
                ]
            )
            for var0, var0_nice in zip([0, 1], ["below threshold", "above threshold"]):
                for var1, var1_nice in zip(
                    [0, 1], ["below threshold", "above threshold"]
                ):
                    # estimate model
                    irf_on, irf_off = lp.ThresholdPanelLPX(
                        data=df,
                        Y=cols_all_endog,
                        X=cols_all_exog,
                        threshold_var=threshold_variables[0]
                        + str(var0)
                        + "_"
                        + threshold_variables[1]
                        + str(var1),
                        response=cols_all_endog,
                        horizon=12,
                        lags=1,
                        varcov="kernel",
                        ci_width=0.8,
                    )
                    irf_on.to_parquet(
                        path_output
                        + "quadrant_privdebt_placebo_panelthresholdlp_"
                        + file_suffixes
                        + "irf_on_"
                        + threshold_variables[0]
                        + str(var0)
                        + "_"
                        + threshold_variables[1]
                        + str(var1)
                        + "modwith_"
                        + uncertainty_variable
                        + "_"
                        + mp_variable
                        + ".parquet"
                    )
                    irf_off.to_parquet(
                        path_output
                        + "quadrant_privdebt_placebo_panelthresholdlp_"
                        + file_suffixes
                        + "irf_off_"
                        + threshold_variables[0]
                        + str(var0)
                        + "_"
                        + threshold_variables[1]
                        + str(var1)
                        + "modwith_"
                        + uncertainty_variable
                        + "_"
                        + mp_variable
                        + ".parquet"
                    )
                    # plot irf
                    for shock in [uncertainty_variable, mp_variable]:
                        fig = lp.ThresholdIRFPlot(
                            irf_threshold_on=irf_on,
                            irf_threshold_off=irf_off,
                            response=cols_all_endog,
                            shock=[shock],
                            n_columns=3,
                            n_rows=3,
                            maintitle="IRFs of "
                            + shock
                            + " shocks when "
                            + threshold_variables[0]
                            + " is "
                            + var0_nice
                            + " and "
                            + threshold_variables[1]
                            + " is "
                            + var1_nice,
                            show_fig=False,
                            save_pic=False,
                            annot_size=12,
                            font_size=12,
                        )
                        # save irf (need to use kaleido==0.1.0post1)
                        fig.write_image(
                            path_output
                            + "quadrant_privdebt_placebo_panelthresholdlp_"
                            + file_suffixes
                            + "irf_"
                            + threshold_variables[0]
                            + str(var0)
                            + "_"
                            + threshold_variables[1]
                            + str(var1)
                            + "modwith_"
                            + uncertainty_variable
                            + "_"
                            + mp_variable
                            + "_"
                            + "shock"
                            + shock
                            + ".png",
                            height=768,
                            width=1366,
                        )
                        # only output and inflation
                        for col in ["gdp", "corecpi"]:
                            fig = lp.ThresholdIRFPlot(
                                irf_threshold_on=irf_on,
                                irf_threshold_off=irf_off,
                                response=[col],
                                shock=[shock],
                                n_columns=1,
                                n_rows=1,
                                maintitle="IRFs of "
                                + shock
                                + " shocks when "
                                + threshold_variables[0]
                                + " is "
                                + var0_nice
                                + " and "
                                + threshold_variables[1]
                                + " is "
                                + var1_nice,
                                show_fig=False,
                                save_pic=False,
                                annot_size=12,
                                font_size=16,
                            )
                            fig.write_image(
                                path_output
                                + "quadrant_privdebt_placebo_panelthresholdlp_"
                                + file_suffixes
                                + "irf_"
                                + threshold_variables[0]
                                + str(var0)
                                + "_"
                                + threshold_variables[1]
                                + str(var1)
                                + "modwith_"
                                + uncertainty_variable
                                + "_"
                                + mp_variable
                                + "_"
                                + "shock"
                                + shock
                                + "_"
                                + "response"
                                + col
                                + ".png",
                                height=768,
                                width=1366,
                            )
                    # consolidate IRFs of all quadrants for combined plots later (only the red lines where H = 1)
                    irf_on[threshold_variables[0] + "_above_threshold"] = (
                        var0  # new columns to indicate if thresholdvar0 > tau
                    )
                    irf_on[threshold_variables[1] + "_above_threshold"] = (
                        var1  # new columns to indicate if thresholdvar1 > tau
                    )
                    # merge
                    irf_consol = pd.concat([irf_consol, irf_on], axis=0)
            irf_consol.to_parquet(
                path_output
                + "quadrant_privdebt_placebo_panelthresholdlp_"
                + file_suffixes
                + "irf_"
                + "modwith_"
                + uncertainty_variable
                + "_"
                + mp_variable
                + ".parquet"
            )

            # Replot IRFs variable by variable for all 4 regimes
            def plot_quadrant_irf(show_ci: bool):
                quadrant_colours = [
                    "black",
                    "lightgrey",
                    "cadetblue",
                    "red",
                ]  # 00, 10, 01, 11
                quadrant_width = [3, 2, 2, 3]  # 00, 10, 01, 11
                for shock in [uncertainty_variable, mp_variable]:
                    for endog in cols_all_endog:
                        fig = go.Figure()  # 4 lines per chart
                        quadrant_count = 0
                        for var0, var0_nice in zip(
                            [0, 1], ["below threshold", "above threshold"]
                        ):
                            for var1, var1_nice in zip(
                                [0, 1], ["below threshold", "above threshold"]
                            ):
                                # subset
                                irf_sub = irf_consol[
                                    (
                                        (irf_consol["Shock"] == shock)
                                        & (irf_consol["Response"] == endog)
                                        & (
                                            irf_consol[
                                                threshold_variables[0]
                                                + "_above_threshold"
                                            ]
                                            == var0
                                        )
                                        & (
                                            irf_consol[
                                                threshold_variables[1]
                                                + "_above_threshold"
                                            ]
                                            == var1
                                        )
                                    )
                                ].copy()
                                # mean irf
                                fig.add_trace(
                                    go.Scatter(
                                        x=irf_sub["Horizon"],
                                        y=irf_sub["Mean"],
                                        name=threshold_variables[0]
                                        + " "
                                        + var0_nice
                                        + " and "
                                        + threshold_variables[1]
                                        + " "
                                        + var1_nice,
                                        mode="lines",
                                        line=dict(
                                            color=quadrant_colours[quadrant_count],
                                            width=quadrant_width[quadrant_count],
                                            dash="solid",
                                        ),
                                    )
                                )
                                if show_ci:
                                    # lower bound
                                    fig.add_trace(
                                        go.Scatter(
                                            x=irf_sub["Horizon"],
                                            y=irf_sub["LB"],
                                            name="",
                                            mode="lines",
                                            line=dict(
                                                color=quadrant_colours[quadrant_count],
                                                width=1,
                                                dash="dash",
                                            ),
                                        )
                                    )
                                    # upper bound
                                    fig.add_trace(
                                        go.Scatter(
                                            x=irf_sub["Horizon"],
                                            y=irf_sub["UB"],
                                            name="",
                                            mode="lines",
                                            line=dict(
                                                color=quadrant_colours[quadrant_count],
                                                width=1,
                                                dash="dash",
                                            ),
                                        )
                                    )
                                # next
                                quadrant_count += 1
                        # format
                        fig.add_hline(
                            y=0,
                            line_dash="solid",
                            line_color="darkgrey",
                            line_width=1,
                        )
                        fig.update_layout(
                            title="Panel threshold LP IRF: Response of "
                            + endog
                            + " to "
                            + shock
                            + " shocks",
                            plot_bgcolor="white",
                            hovermode="x unified",
                            showlegend=True,
                            font=dict(color="black", size=12),
                        )
                        # save image
                        if show_ci:
                            file_ci_suffix = "_withci"
                        elif not show_ci:
                            file_ci_suffix = ""
                        fig.write_image(
                            path_output
                            + "quadrant_privdebt_placebo_panelthresholdlp_"
                            + file_suffixes
                            + "irf_"
                            + "modwith_"
                            + uncertainty_variable
                            + "_"
                            + mp_variable
                            + "_"
                            + "shock"
                            + shock
                            + "_"
                            + "response"
                            + endog
                            + file_ci_suffix
                            + ".png",
                            height=768,
                            width=1366,
                        )

            plot_quadrant_irf(show_ci=False)
            plot_quadrant_irf(show_ci=True)


# %%
# II --- Do everything
# Some objects for quick ref later
cols_endog_long = [
    "privdebt",
    "govdebt",  # _ngdp
    "gdp",  # urate gdp
    "corecpi",  # corecpi cpi
    "reer",
]
cols_endog_short = [
    "gdp",  # urate gdp
    "corecpi",  # corecpi cpi
    "reer",
]
cols_threshold_priv_gov_ref = ["privdebt_ngdp_ref", "govdebt_ngdp_ref"]
cols_threshold_priv_gov = ["privdebt_ngdp", "govdebt_ngdp"]

# %%
# II.A --- With EPU
# With STIR
do_everything_quadrant_panelthresholdlp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_threshold=cols_threshold_priv_gov_ref,
    threshold_variables=cols_threshold_priv_gov,
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "colombia",  # 2006 Q4
        "germany",  # 2006 Q1
        "sweden",  # ends 2020 Q3 --- epu
    ],
    file_suffixes="",  # format: "abc_" or ""
)
# # With STIR (reduced)
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["stir"] + cols_endog_short,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminstir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "colombia",  # 2006 Q4
#         "germany",  # 2006 Q1
#         "sweden",  # ends 2020 Q3 --- epu
#     ],
#     file_suffixes="reduced_",  # format: "abc_" or ""
# )

# # With M2
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["m2"] + cols_endog_long,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminm2"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2012 Q1
#         "china",  # 2007 Q1 and potentially exclusive case
#         "chile",  # 2010 Q1 and potentially exclusive case
#         "colombia",  # 2005 Q4
#         "singapore",  # 2005 Q1
#     ],
#     file_suffixes="m2_",  # format: "abc_" or ""
# )
# # With M2 (reduced)
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["m2"] + cols_endog_short,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminm2"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2012 Q1
#         "china",  # 2007 Q1 and potentially exclusive case
#         "chile",  # 2010 Q1 and potentially exclusive case
#         "colombia",  # 2005 Q4
#         "singapore",  # 2005 Q1
#     ],
#     file_suffixes="m2_reduced_",  # format: "abc_" or ""
# )

# # With LTIR
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["ltir"] + cols_endog_long,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminltir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "chile",  #  2010Q1
#         "colombia",  # 2005 Q4
#         "germany",  # 2006 Q1
#     ],
#     file_suffixes="ltir_",  # format: "abc_" or ""
# )
# # With LTIR (reduced)
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["ltir"] + cols_endog_short,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminltir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "chile",  #  2010Q1
#         "colombia",  # 2005 Q4
#         "germany",  # 2006 Q1
#     ],
#     file_suffixes="ltir_reduced_",  # format: "abc_" or ""
# )

# # With oneway STIR shocks
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["stir"] + cols_endog_long,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxstir", "minstir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "colombia",  # 2006 Q4
#         "germany",  # 2006 Q1
#         "sweden",  # ends 2020 Q3 --- epu
#     ],
#     file_suffixes="",  # format: "abc_" or ""
# )
# # With oneway STIR shocks (reduced)
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["stir"] + cols_endog_short,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxstir", "minstir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "colombia",  # 2006 Q4
#         "germany",  # 2006 Q1
#         "sweden",  # ends 2020 Q3 --- epu
#     ],
#     file_suffixes="reduced_",  # format: "abc_" or ""
# )

# # With oneway M2 shocks
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["m2"] + cols_endog_long,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxm2", "minm2"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2012 Q1
#         "china",  # 2007 Q1 and potentially exclusive case
#         "chile",  # 2010 Q1 and potentially exclusive case
#         "colombia",  # 2005 Q4
#         "singapore",  # 2005 Q1
#     ],
#     file_suffixes="m2_",  # format: "abc_" or ""
# )
# # With oneway M2 shocks (reduced)
# do_everything_quadrant_panelthresholdlp(
#     cols_endog_after_shocks=["m2"] + cols_endog_short,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxm2", "minm2"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_threshold=cols_threshold_priv_gov_ref,
#     threshold_variables=cols_threshold_priv_gov,
#     countries_drop=[
#         "india",  # 2012 Q1
#         "china",  # 2007 Q1 and potentially exclusive case
#         "chile",  # 2010 Q1 and potentially exclusive case
#         "colombia",  # 2005 Q4
#         "singapore",  # 2005 Q1
#     ],
#     file_suffixes="m2_reduced_",  # format: "abc_" or ""
# )


# %%
# II.B --- With WUI
# With STIR
do_everything_quadrant_panelthresholdlp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminwui"],
    cols_threshold=cols_threshold_priv_gov_ref,
    threshold_variables=cols_threshold_priv_gov,
    countries_drop=[
        "argentina",
        "china",
        "germany",
        "india",
        "indonesia",
        "israel",
        "malaysia",
        "turkey",
        "thailand",
        "denmark",
        "norway",
        "sweden",
    ],
    file_suffixes="",  # format: "abc_" or ""
    input_df_suffix="large_yoy",  # different data set
)

# %%
# II.C --- With UCT
# With STIR
do_everything_quadrant_panelthresholdlp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminuct"],
    cols_threshold=cols_threshold_priv_gov_ref,
    threshold_variables=cols_threshold_priv_gov,
    countries_drop=[
        "thailand",
        "turkey",
        "malaysia",
        "israel",
        "indonesia",
        "india",
        "germany",
        "china",
        "argentina",
        "sweden",
        "denmark",
    ],
    file_suffixes="",  # format: "abc_" or ""
    input_df_suffix="full_yoy",  # different data set
)


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
