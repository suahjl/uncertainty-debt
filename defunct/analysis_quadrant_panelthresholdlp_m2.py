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
import ruptures as rpt
from chow_test import chow_test
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
# I --- Functions
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
# list_mp_variables = [i + "m2" for i in list_shock_prefixes]
# list_uncertainty_variables = [i + "epu" for i in list_shock_prefixes]
list_mp_variables = ["maxminm2"]  # maxminm2
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
            uncertainty_variable,
            # "epu",
            mp_variable,
            "m2",
            "hhdebt",  # _ngdp
            "corpdebt",  # _ngdp
            "govdebt",  # _ngdp
            "gdp",  # urate gdp
            # "capflows_ngdp",
            "corecpi",  # corecpi cpi
            "reer",
        ]
        cols_all_exog = ["maxminbrent"]  # maxminm2
        cols_threshold = ["hhdebt_ngdp_ref", "govdebt_ngdp_ref"]
        df = df[cols_groups + cols_all_endog + cols_all_exog + cols_threshold].copy()
        # Check when the panel becomes balanced
        check_balance_timing(input=df)
        check_balance_endtiming(input=df)
        # Trim more countries
        # if "m2" in mp_variable:
        countries_drop = [
            "india",  # 2012 Q1
            "china",  # 2007 Q1 and potentially exclusive case
            "chile",  # 2010 Q1 and potentially exclusive case
            "colombia",  # 2005 Q4
            "singapore",  # 2005 Q1
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
        #         "chile",  # ends 2022 Q2  (doesn't have m2?)
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

        def find_quadrant_thresholds(
            df: pd.DataFrame,
            threshold_variables: list[str],
            option: str,
            param_choices: list[float],
        ):
            if option == "dumb":
                df.loc[
                    df[threshold_variables[0] + "_ref"] >= param_choices[0],
                    threshold_variables[0] + "_above_threshold",
                ] = 1
                df.loc[
                    df[threshold_variables[0] + "_ref"] < param_choices[0],
                    threshold_variables[0] + "_above_threshold",
                ] = 0
                df.loc[
                    df[threshold_variables[1] + "_ref"] >= param_choices[1],
                    threshold_variables[0] + "_above_threshold",
                ] = 1
                df.loc[
                    df[threshold_variables[1] + "_ref"] < param_choices[1],
                    threshold_variables[1] + "_above_threshold",
                ] = 0
                # quadrant 1 (0,0)
                df.loc[
                    (
                        (df[threshold_variables[0] + "_above_threshold"] == 0)
                        & (df[threshold_variables[1] + "_above_threshold"] == 0)
                    ),
                    threshold_variables[0] + "0" + "_" + threshold_variables[1] + "0",
                ] = 1
                df.loc[
                    ~(
                        (df[threshold_variables[0] + "_above_threshold"] == 0)
                        & (df[threshold_variables[1] + "_above_threshold"] == 0)
                    ),
                    threshold_variables[0] + "0" + "_" + threshold_variables[1] + "0",
                ] = 0
                # quadrant 2 (1,0)
                df.loc[
                    (
                        (df[threshold_variables[0] + "_above_threshold"] == 1)
                        & (df[threshold_variables[1] + "_above_threshold"] == 0)
                    ),
                    threshold_variables[0] + "1" + "_" + threshold_variables[1] + "0",
                ] = 1
                df.loc[
                    ~(
                        (df[threshold_variables[0] + "_above_threshold"] == 1)
                        & (df[threshold_variables[1] + "_above_threshold"] == 0)
                    ),
                    threshold_variables[0] + "1" + "_" + threshold_variables[1] + "0",
                ] = 0
                # quadrant 3 (0,1)
                df.loc[
                    (
                        (df[threshold_variables[0] + "_above_threshold"] == 0)
                        & (df[threshold_variables[1] + "_above_threshold"] == 1)
                    ),
                    threshold_variables[0] + "0" + "_" + threshold_variables[1] + "1",
                ] = 1
                df.loc[
                    ~(
                        (df[threshold_variables[0] + "_above_threshold"] == 0)
                        & (df[threshold_variables[1] + "_above_threshold"] == 1)
                    ),
                    threshold_variables[0] + "0" + "_" + threshold_variables[1] + "1",
                ] = 0
                # quadrant 4 (1,1)
                df.loc[
                    (
                        (df[threshold_variables[0] + "_above_threshold"] == 1)
                        & (df[threshold_variables[1] + "_above_threshold"] == 1)
                    ),
                    threshold_variables[0] + "1" + "_" + threshold_variables[1] + "1",
                ] = 1
                df.loc[
                    ~(
                        (df[threshold_variables[0] + "_above_threshold"] == 1)
                        & (df[threshold_variables[1] + "_above_threshold"] == 1)
                    ),
                    threshold_variables[0] + "1" + "_" + threshold_variables[1] + "1",
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
            elif option == "reg_thresholdselection":
                df_opt_threshold = pd.read_csv(
                    path_output
                    + "reg_quadrant_thresholdselection_m2_fe_"
                    + "modwith_"
                    + uncertainty_variable
                    + "_"
                    + mp_variable
                    + "_opt_threshold"
                    + ".csv"
                )
                opt_threshold0 = df_opt_threshold.iloc[0, 0]
                opt_threshold1 = df_opt_threshold.iloc[0, 1]
                # first
                df.loc[
                    df[threshold_variables[0] + "_ref"] >= opt_threshold0,
                    threshold_variables[0] + "_above_threshold",
                ] = 1
                df.loc[
                    df[threshold_variables[0] + "_ref"] < opt_threshold0,
                    threshold_variables[0] + "_above_threshold",
                ] = 0
                # second
                df.loc[
                    df[threshold_variables[1] + "_ref"] >= opt_threshold1,
                    threshold_variables[1] + "_above_threshold",
                ] = 1
                df.loc[
                    df[threshold_variables[1] + "_ref"] < opt_threshold1,
                    threshold_variables[1] + "_above_threshold",
                ] = 0
                # quadrant 1 (0,0)
                df.loc[
                    (
                        (df[threshold_variables[0] + "_above_threshold"] == 0)
                        & (df[threshold_variables[1] + "_above_threshold"] == 0)
                    ),
                    threshold_variables[0] + "0" + "_" + threshold_variables[1] + "0",
                ] = 1
                df.loc[
                    ~(
                        (df[threshold_variables[0] + "_above_threshold"] == 0)
                        & (df[threshold_variables[1] + "_above_threshold"] == 0)
                    ),
                    threshold_variables[0] + "0" + "_" + threshold_variables[1] + "0",
                ] = 0
                # quadrant 2 (1,0)
                df.loc[
                    (
                        (df[threshold_variables[0] + "_above_threshold"] == 1)
                        & (df[threshold_variables[1] + "_above_threshold"] == 0)
                    ),
                    threshold_variables[0] + "1" + "_" + threshold_variables[1] + "0",
                ] = 1
                df.loc[
                    ~(
                        (df[threshold_variables[0] + "_above_threshold"] == 1)
                        & (df[threshold_variables[1] + "_above_threshold"] == 0)
                    ),
                    threshold_variables[0] + "1" + "_" + threshold_variables[1] + "0",
                ] = 0
                # quadrant 3 (0,1)
                df.loc[
                    (
                        (df[threshold_variables[0] + "_above_threshold"] == 0)
                        & (df[threshold_variables[1] + "_above_threshold"] == 1)
                    ),
                    threshold_variables[0] + "0" + "_" + threshold_variables[1] + "1",
                ] = 1
                df.loc[
                    ~(
                        (df[threshold_variables[0] + "_above_threshold"] == 0)
                        & (df[threshold_variables[1] + "_above_threshold"] == 1)
                    ),
                    threshold_variables[0] + "0" + "_" + threshold_variables[1] + "1",
                ] = 0
                # quadrant 4 (1,1)
                df.loc[
                    (
                        (df[threshold_variables[0] + "_above_threshold"] == 1)
                        & (df[threshold_variables[1] + "_above_threshold"] == 1)
                    ),
                    threshold_variables[0] + "1" + "_" + threshold_variables[1] + "1",
                ] = 1
                df.loc[
                    ~(
                        (df[threshold_variables[0] + "_above_threshold"] == 1)
                        & (df[threshold_variables[1] + "_above_threshold"] == 1)
                    ),
                    threshold_variables[0] + "1" + "_" + threshold_variables[1] + "1",
                ] = 0
                print(
                    "optimal thresholds: "
                    + threshold_variables[0]
                    + " = "
                    + str(opt_threshold0)
                    + " and "
                    + threshold_variables[1]
                    + " = "
                    + str(opt_threshold1)
                )
            return df

        # Threshold
        threshold_variables = ["hhdebt_ngdp", "govdebt_ngdp"]
        df = find_quadrant_thresholds(
            df=df,
            threshold_variables=threshold_variables,
            option="reg_thresholdselection",
            param_choices=[0, 0],
        )

        # Reset index
        df = df.reset_index(drop=True)
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
            for var1, var1_nice in zip([0, 1], ["below threshold", "above threshold"]):
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
                    + "quadrant_panelthresholdlp_m2_irf_on_"
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
                    + "quadrant_panelthresholdlp_m2_irf_off_"
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
                        + "quadrant_panelthresholdlp_m2_irf_"
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
                            + "quadrant_panelthresholdlp_m2_irf_"
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
            + "quadrant_panelthresholdlp_m2_irf_"
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
                                            threshold_variables[0] + "_above_threshold"
                                        ]
                                        == var0
                                    )
                                    & (
                                        irf_consol[
                                            threshold_variables[1] + "_above_threshold"
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
                        + "quadrant_panelthresholdlp_m2_irf_"
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
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
