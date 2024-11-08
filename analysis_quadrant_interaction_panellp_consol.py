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
from itertools import combinations

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
def do_everything_quadrant_interaction_panellp(
    cols_endog_after_shocks: list[str],
    cols_all_exog: list[str],
    list_mp_variables: list[str],
    list_uncertainty_variables: list[str],
    cols_state_dependency: list[str], 
    state_dependency_nice_for_title: str,  # HH debt, Gov debt
    countries_drop: list[str],
    file_suffixes: str,  # format: "abc_" or ""
    beta_values_to_simulate: list[list[float]],
    irf_colours_for_each_beta: list[str]
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

    def generate_n_interactions(df: pd.DataFrame, labels_to_be_interacted: list[str]):
        # A --- Create labels
        result = []
        # Add individual elements
        result.extend(labels_to_be_interacted)
        # Add pairwise combinations
        result.extend(
            "_".join(pair) for pair in combinations(labels_to_be_interacted, 2)
        )
        # Add triplet combinations
        result.extend(
            "_".join(triplet) for triplet in combinations(labels_to_be_interacted, 3)
        )

        # B --- Create new columns in df
        # Loop over pairs
        for pair in combinations(labels_to_be_interacted, 2):
            col_name = "_".join(pair)
            df[col_name] = df[pair[0]] * df[pair[1]]

        # Loop over triples
        for triple in combinations(labels_to_be_interacted, 3):
            col_name = "_".join(triple)
            df[col_name] = df[triple[0]] * df[triple[1]] * df[triple[2]]
        # C --- Output
        return df, result

    def irf_interaction_only_wide(
        irf: pd.DataFrame,
        b1_label: str,  # the shock of interest
        b4_label: str,
        b5_label: str,
        b7_label: str,
        beta_ints: list[list[float]],
    ):
        # Triple interacted model: y = a + b1A + b2B + b3C + b4AB + b5AC + b6BC + b7ABC
        # subset b1 and b3
        irf_b1 = irf[irf["Shock"] == b1_label].copy()
        irf_b4 = irf[irf["Shock"] == b4_label].copy()
        irf_b5 = irf[irf["Shock"] == b5_label].copy()
        irf_b7 = irf[irf["Shock"] == b7_label].copy()
        # rename all column labels for Mean, LB and UB for easy merging later
        irf_b1 = irf_b1.rename(
            columns={
                "Mean": "Mean_" + b1_label,
                "LB": "LB_" + b1_label,
                "UB": "UB_" + b1_label,
            }
        )
        irf_b4 = irf_b4.rename(
            columns={
                "Mean": "Mean_" + b4_label,
                "LB": "LB_" + b4_label,
                "UB": "UB_" + b4_label,
            }
        )
        irf_b5 = irf_b5.rename(
            columns={
                "Mean": "Mean_" + b5_label,
                "LB": "LB_" + b5_label,
                "UB": "UB_" + b5_label,
            }
        )
        irf_b7 = irf_b7.rename(
            columns={
                "Mean": "Mean_" + b7_label,
                "LB": "LB_" + b7_label,
                "UB": "UB_" + b7_label,
            }
        )
        del irf_b4["Shock"]
        del irf_b5["Shock"]
        del irf_b7["Shock"]
        # merge left
        irf_interactions = irf_b1.merge(irf_b4, on=["Response", "Horizon"])
        irf_interactions = irf_interactions.merge(irf_b5, on=["Response", "Horizon"])
        irf_interactions = irf_interactions.merge(irf_b7, on=["Response", "Horizon"])
        irf_interactions["Shock"] = (
            b1_label + "_mult_" + b4_label + "_mult_" + b5_label + "_mult_" + b7_label
        )
        # generate new irfs
        for beta_int in beta_ints:
            for irf_moment in ["Mean", "LB", "UB"]:
                irf_interactions[
                    irf_moment
                    + "_b4"
                    + str(beta_int[0])
                    + "_b5"
                    + str(beta_int[1])
                    + "_b7"
                    + str(beta_int[0] * beta_int[1])
                ] = (
                    irf_interactions[irf_moment + "_" + b1_label]
                    + (beta_int[0] * irf_interactions[irf_moment + "_" + b4_label])
                    + (beta_int[1] * irf_interactions[irf_moment + "_" + b5_label])
                    + (
                        beta_int[0]
                        * beta_int[1]
                        * irf_interactions[irf_moment + "_" + b7_label]
                    )
                )
        # clean house
        for irf_moment in ["Mean", "LB", "UB"]:
            del irf_interactions[irf_moment + "_" + b1_label]
            del irf_interactions[irf_moment + "_" + b4_label]
            del irf_interactions[irf_moment + "_" + b5_label]
            del irf_interactions[irf_moment + "_" + b7_label]
        irf_interactions = irf_interactions[
            ~(
                (irf_interactions["Response"] == b4_label)
                | (irf_interactions["Response"] == b5_label)
                | (irf_interactions["Response"] == b7_label)
            )
        ]
        # export
        return irf_interactions

    def plot_triple_interaction_wide_irf(
        irf: pd.DataFrame,
        shock_variable: str,  # only used in title and file suffix
        response_variable: str,
        show_ci: bool,
        beta_ints: list[list[float]],  # [[60.5, 91], [30, 30]]
        beta_int_colours: list[str],
        interacted_variable_label_nice: str,  # taken from b1_label + "_mult_" + b4_label + "_mult_" + b5_label + "_mult_" + b7_label
    ):
        fig = go.Figure()  # 4 lines per chart
        irf_count = 0
        for beta_int in tqdm(beta_ints):
            # subset
            irf_sub = irf[irf["Response"] == response_variable].copy()
            # mean irf
            fig.add_trace(
                go.Scatter(
                    x=irf_sub["Horizon"],
                    y=irf_sub[
                        "Mean_"
                        + "b4"
                        + str(beta_int[0])
                        + "_b5"
                        + str(beta_int[1])
                        + "_b7"
                        + str(beta_int[0] * beta_int[1])
                    ],
                    name=interacted_variable_label_nice + " = " + ", ".join([str(i) for i in beta_int]),
                    mode="lines",
                    line=dict(
                        color=beta_int_colours[irf_count],
                        dash="solid",
                    ),
                )
            )
            if show_ci:
                # lower bound
                fig.add_trace(
                    go.Scatter(
                        x=irf_sub["Horizon"],
                        y=irf_sub[
                            "LB_"
                            + +"_b4"
                            + str(beta_int[0])
                            + "_b5"
                            + str(beta_int[1])
                            + "_b7"
                            + str(beta_int[0] * beta_int[1])
                        ],
                        name="",
                        mode="lines",
                        line=dict(
                            color=beta_int_colours[irf_count],
                            width=1,
                            dash="dash",
                        ),
                    )
                )
                # upper bound
                fig.add_trace(
                    go.Scatter(
                        x=irf_sub["Horizon"],
                        y=irf_sub[
                            "UB_"
                            + +"_b4"
                            + str(beta_int[0])
                            + "_b5"
                            + str(beta_int[1])
                            + "_b7"
                            + str(beta_int[0] * beta_int[1])
                        ],
                        name="",
                        mode="lines",
                        line=dict(
                            color=beta_int_colours[irf_count],
                            width=1,
                            dash="dash",
                        ),
                    )
                )
            # next
            irf_count += 1
        # format
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="darkgrey",
            line_width=1,
        )
        fig.update_layout(
            title="Panel LP IRF: Response of "
            + response_variable
            + " to "
            + shock_variable 
            + " shocks "
            + "conditional on "
            + interacted_variable_label_nice,
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
            + "quadrant_interaction_panellp_"
            + file_suffixes
            + "irf_"
            + "modwith_"
            + uncertainty_variable
            + "_"
            + mp_variable
            + "_"
            + "shock"
            + shock_variable
            + "_"
            + "response"
            + response_variable
            + file_ci_suffix
            + ".png",
            height=768,
            width=1366,
        )

    # Loop to estimate
    for mp_variable in tqdm(list_mp_variables):
        for uncertainty_variable in tqdm(list_uncertainty_variables):
            print("\nMP variable is " + mp_variable)
            print("Uncertainty variable is " + uncertainty_variable)
            # II --- Load data
            df = pd.read_parquet(path_data + "data_macro_yoy.parquet")
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
                cols_groups + cols_all_endog + cols_all_exog + cols_state_dependency
            ].copy()
            # Check when the panel becomes balanced
            check_balance_timing(input=df)
            check_balance_endtiming(input=df)
            # Create interaction terms

            # Trim more countries
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

            # Reset index
            df = df.reset_index(drop=True)
            # Numeric time
            df["time"] = df.groupby("country").cumcount()
            del df["quarter"]
            # Set multiindex
            df = df.set_index(["country", "time"])

            # IV --- Analysis
            df, cols_interaction = generate_n_interactions(
                df=df,
                labels_to_be_interacted=[uncertainty_variable] + [mp_variable]
                + cols_state_dependency
            )
            irf = lp.PanelLPX(
                data=df,
                Y=cols_all_endog + cols_interaction,
                X=cols_all_exog,
                response=cols_all_endog + cols_interaction,
                horizon=12,
                lags=1,
                varcov="kernel",
                ci_width=0.8,
            )  # actual model is estimated only once
            for shock in [uncertainty_variable, mp_variable]:
                irf_interaction = irf_interaction_only_wide(
                    irf=irf,
                    b1_label=shock,
                    b4_label=shock + "_" + cols_state_dependency[0],
                    b5_label=shock + "_" + cols_state_dependency[1],
                    b7_label=shock + "_" + cols_state_dependency[0] + "_" + cols_state_dependency[1],
                    beta_ints=beta_values_to_simulate,
                )  # interaction terms are calculated twice
                for endog in cols_all_endog:
                    plot_triple_interaction_wide_irf(
                        irf=irf_interaction,
                        response_variable=endog,
                        shock_variable=shock,  # this way, the file name will only say if mp or unc shocks were used
                        show_ci=False,
                        beta_ints=beta_values_to_simulate,  # [[60.5, 91], [30, 30]]
                        beta_int_colours=irf_colours_for_each_beta,  # ["blue", "red]
                        interacted_variable_label_nice=state_dependency_nice_for_title
                    )  # IRF plots are generated for all variables per shock
 

# %%
# II --- Do everything    
# Some objects for quick ref later
cols_endog_long = [
        "hhdebt",  # _ngdp
        "corpdebt",  # _ngdp
        "govdebt",  # _ngdp
        "gdp",  # urate gdp
        "corecpi",  # corecpi cpi
        "reer",
]
cols_endog_short = [
    # "hhdebt",  # _ngdp
    # "corpdebt",  # _ngdp
    # "govdebt",  # _ngdp
    "gdp",  # urate gdp
    "corecpi",  # corecpi cpi
    "reer",
]
cols_threshold_hh_gov = ["hhdebt_ngdp_ref", "govdebt_ngdp_ref"]
debt_values_combos=[
    [30, 60],  # HH low, gov low
    # [60, 60],  # HH med, gov low
    [90, 60],  # HH high, gov low
    # [30, 90],  # HH low, gov med
    # [60, 90],  # HH med, gov med
    # [90, 90],  # HH high, gov med
    [30, 120],  # HH low, gov high
    # [60, 120],  # HH med, gov high
    [90, 120],  # HH high, gov high
]
# debt_values_combos_irf_line_colours=[
#     "lightblue",
#     "blue",
#     "darkblue",
#     "lightgrey", 
#     "grey",
#     "black",
#     "pink",
#     "red",
#     "crimson",
# ]
debt_values_combos_irf_line_colours=[
    "grey",
    "pink",
    "blue",
    "crimson"
]
testing = True
if testing:
    # With STIR
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["stir"] + cols_endog_long,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxminstir"],
        list_uncertainty_variables=["maxminepu"],
        cols_state_dependency=cols_threshold_hh_gov, 
        state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
        countries_drop=[
            "india",  # 2016 Q3
            "denmark",  # ends 2019 Q3
            "china",  # 2007 Q4 and potentially exclusive case
            "colombia",  # 2006 Q4
            "germany",  # 2006 Q1
            "sweden",  # ends 2020 Q3 --- epu
        ],
        file_suffixes="",  # format: "abc_" or ""
        beta_values_to_simulate=debt_values_combos,
        irf_colours_for_each_beta=debt_values_combos_irf_line_colours
    )

elif not testing:
    # With STIR
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["stir"] + cols_endog_long,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxminstir"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
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
    # With STIR (reduced)
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["stir"] + cols_endog_short,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxminstir"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
        countries_drop=[
            "india",  # 2016 Q3
            "denmark",  # ends 2019 Q3
            "china",  # 2007 Q4 and potentially exclusive case
            "colombia",  # 2006 Q4
            "germany",  # 2006 Q1
            "sweden",  # ends 2020 Q3 --- epu
        ],
        file_suffixes="reduced_",  # format: "abc_" or ""
    )

    # With M2
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["m2"] + cols_endog_long,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxminm2"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
        countries_drop=[
            "india",  # 2012 Q1
            "china",  # 2007 Q1 and potentially exclusive case
            "chile",  # 2010 Q1 and potentially exclusive case
            "colombia",  # 2005 Q4
            "singapore",  # 2005 Q1
        ],
        file_suffixes="m2_",  # format: "abc_" or ""
    )
    # With M2 (reduced)
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["m2"] + cols_endog_short,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxminm2"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
        countries_drop=[
            "india",  # 2012 Q1
            "china",  # 2007 Q1 and potentially exclusive case
            "chile",  # 2010 Q1 and potentially exclusive case
            "colombia",  # 2005 Q4
            "singapore",  # 2005 Q1
        ],
        file_suffixes="m2_reduced_",  # format: "abc_" or ""
    )

    # With LTIR
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["ltir"] + cols_endog_long,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxminltir"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
        countries_drop=[
            "india",  # 2016 Q3
            "denmark",  # ends 2019 Q3
            "china",  # 2007 Q4 and potentially exclusive case
            "chile",  #  2010Q1
            "colombia",  # 2005 Q4
            "germany",  # 2006 Q1
        ],
        file_suffixes="ltir_",  # format: "abc_" or ""
    )
    # With LTIR (reduced)
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["ltir"] + cols_endog_short,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxminltir"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
        countries_drop=[
            "india",  # 2016 Q3
            "denmark",  # ends 2019 Q3
            "china",  # 2007 Q4 and potentially exclusive case
            "chile",  #  2010Q1
            "colombia",  # 2005 Q4
            "germany",  # 2006 Q1
        ],
        file_suffixes="ltir_reduced_",  # format: "abc_" or ""
    )

    # With oneway STIR shocks
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["stir"] + cols_endog_long,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxstir", "minstir"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
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
    # With oneway STIR shocks (reduced)
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["stir"] + cols_endog_short,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxstir", "minstir"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
        countries_drop=[
            "india",  # 2016 Q3
            "denmark",  # ends 2019 Q3
            "china",  # 2007 Q4 and potentially exclusive case
            "colombia",  # 2006 Q4
            "germany",  # 2006 Q1
            "sweden",  # ends 2020 Q3 --- epu
        ],
        file_suffixes="reduced_",  # format: "abc_" or ""
    )

    # With oneway M2 shocks
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["m2"] + cols_endog_long,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxm2", "minm2"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
        countries_drop=[
            "india",  # 2012 Q1
            "china",  # 2007 Q1 and potentially exclusive case
            "chile",  # 2010 Q1 and potentially exclusive case
            "colombia",  # 2005 Q4
            "singapore",  # 2005 Q1
        ],
        file_suffixes="m2_",  # format: "abc_" or ""
    )
    # With oneway M2 shocks (reduced)
    do_everything_quadrant_interaction_panellp(
        cols_endog_after_shocks=["m2"] + cols_endog_short,
        cols_all_exog=["maxminbrent"],
        list_mp_variables=["maxm2", "minm2"],
        list_uncertainty_variables=["maxminepu"],
        cols_threshold=cols_threshold_hh_gov,
        countries_drop=[
            "india",  # 2012 Q1
            "china",  # 2007 Q1 and potentially exclusive case
            "chile",  # 2010 Q1 and potentially exclusive case
            "colombia",  # 2005 Q4
            "singapore",  # 2005 Q1
        ],
        file_suffixes="m2_reduced_",  # format: "abc_" or ""
    )

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
