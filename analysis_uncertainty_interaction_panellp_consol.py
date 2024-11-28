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
pd.options.mode.chained_assignment = None
warnings.filterwarnings(
    "ignore"
)  # MissingValueWarning when localprojections implements shift operations


# %%
# I --- Do everything function
def do_everything_single_interaction_panellp(
    cols_endog_after_shocks: list[str],
    cols_all_exog: list[str],
    list_mp_variables: list[str],
    list_uncertainty_variables: list[str],
    cols_state_dependency: list[str],
    state_dependency_nice_for_title: str,  # HH debt, Gov debt
    countries_drop: list[str],
    file_suffixes: str,  # format: "abc_" or ""
    beta_values_to_simulate: list[float],  # [80, 160]
    irf_colours_for_each_beta: list[str],
    t_start: date = date(1990, 1, 1),
    t_end: date = None,
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

    def generate_n_interactions(df: pd.DataFrame, labels_to_be_interacted: list[str]):
        # A --- Create labels
        result = []
        # Add individual elements
        result.extend(labels_to_be_interacted)
        # Add pairwise combinations
        result.extend(
            "_".join(pair) for pair in combinations(labels_to_be_interacted, 2)
        )

        # B --- Create new columns in df
        # Loop over pairs
        for pair in combinations(labels_to_be_interacted, 2):
            col_name = "_".join(pair)
            df[col_name] = df[pair[0]] * df[pair[1]]

        # C --- Output
        return df, result

    def irf_interaction_only_wide(
        irf: pd.DataFrame,
        b1_label: str,  # the shock of interest
        b3_label: str,
        beta_ints: list[float],
    ):
        # Standard interaction model: y = a + b1A + b2B + b3AB
        # subset b1 and b3
        irf_b1 = irf[irf["Shock"] == b1_label].copy()
        irf_b3 = irf[irf["Shock"] == b3_label].copy()
        # rename all column labels for Mean, LB and UB for easy merging later
        irf_b1 = irf_b1.rename(
            columns={
                "Mean": "Mean_" + b1_label,
                "LB": "LB_" + b1_label,
                "UB": "UB_" + b1_label,
            }
        )
        irf_b3 = irf_b3.rename(
            columns={
                "Mean": "Mean_" + b3_label,
                "LB": "LB_" + b3_label,
                "UB": "UB_" + b3_label,
            }
        )
        del irf_b3["Shock"]
        # merge left
        irf_interactions = irf_b1.merge(irf_b3, on=["Response", "Horizon"])
        irf_interactions["Shock"] = b1_label + "_mult_" + b3_label
        # generate new irfs
        for beta_int in beta_ints:
            for irf_moment in ["Mean", "LB", "UB"]:
                irf_interactions[irf_moment + "_b3" + str(beta_int)] = (
                    irf_interactions[irf_moment + "_" + b1_label]
                    + (beta_int * irf_interactions[irf_moment + "_" + b3_label])
                )
        # clean house
        for irf_moment in ["Mean", "LB", "UB"]:
            del irf_interactions[irf_moment + "_" + b1_label]
            del irf_interactions[irf_moment + "_" + b3_label]
        irf_interactions = irf_interactions[
            ~((irf_interactions["Response"] == b3_label))
        ]
        # export
        return irf_interactions

    def plot_interaction_wide_irf(
        irf: pd.DataFrame,
        shock_variable: str,  # only used in title and file suffix
        response_variable: str,
        show_ci: bool,
        beta_ints: list[float],  # [80, 160]
        beta_int_colours: list[str],
        interacted_variable_label_nice: str,  # taken from b1_label + "_mult_" + b3_label
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
                    y=irf_sub["Mean_" + "b3" + str(beta_int)],
                    name=interacted_variable_label_nice
                    + " = "
                    + str(beta_int),
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
                        y=irf_sub["LB_" + "b3" + str(beta_int)],
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
                        y=irf_sub["UB_" + "b3" + str(beta_int)],
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
            + "uncertainty_interaction_panellp_"
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
            if t_end is None:
                pass
            else:
                df = df[(df["date"] <= t_end)]
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
                labels_to_be_interacted=[uncertainty_variable]
                + [mp_variable]
                + cols_state_dependency,
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
                    b3_label=shock + "_" + cols_state_dependency[0],
                    beta_ints=beta_values_to_simulate,
                )  # interaction terms are calculated twice
                for endog in cols_all_endog:
                    plot_interaction_wide_irf(
                        irf=irf_interaction,
                        response_variable=endog,
                        shock_variable=shock,  # this way, the file name will only say if mp or unc shocks were used
                        show_ci=False,
                        beta_ints=beta_values_to_simulate,  # [[60.5, 91], [30, 30]]
                        beta_int_colours=irf_colours_for_each_beta,  # ["blue", "red]
                        interacted_variable_label_nice=state_dependency_nice_for_title,
                    )  # IRF plots are generated for all variables per shock


# %%
# II --- Some objects for quick ref later
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
cols_threshold_epu = ["epu_ref"]
epu_values_combos = [80, 160, 250]
epu_values_combos_irf_line_colours = ["grey", "red", "crimson"]

# %%
# III --- Do everything
# With STIR
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
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
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)
# With STIR (reduced)
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "colombia",  # 2006 Q4
        "germany",  # 2006 Q1
        "sweden",  # ends 2020 Q3 --- epu
    ],
    file_suffixes="reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)

# With M2
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["m2"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
    countries_drop=[
        "india",  # 2012 Q1
        "china",  # 2007 Q1 and potentially exclusive case
        "chile",  # 2010 Q1 and potentially exclusive case
        "colombia",  # 2005 Q4
        "singapore",  # 2005 Q1
    ],
    file_suffixes="m2_",  # format: "abc_" or ""
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)
# With M2 (reduced)
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["m2"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminm2"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
    countries_drop=[
        "india",  # 2012 Q1
        "china",  # 2007 Q1 and potentially exclusive case
        "chile",  # 2010 Q1 and potentially exclusive case
        "colombia",  # 2005 Q4
        "singapore",  # 2005 Q1
    ],
    file_suffixes="m2_reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)

# With LTIR
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["ltir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminltir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "chile",  #  2010Q1
        "colombia",  # 2005 Q4
        "germany",  # 2006 Q1
    ],
    file_suffixes="ltir_",  # format: "abc_" or ""
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)
# With LTIR (reduced)
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["ltir"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminltir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "chile",  #  2010Q1
        "colombia",  # 2005 Q4
        "germany",  # 2006 Q1
    ],
    file_suffixes="ltir_reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)

# With oneway STIR shocks
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxstir", "minstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
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
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)
# With oneway STIR shocks (reduced)
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxstir", "minstir"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
    countries_drop=[
        "india",  # 2016 Q3
        "denmark",  # ends 2019 Q3
        "china",  # 2007 Q4 and potentially exclusive case
        "colombia",  # 2006 Q4
        "germany",  # 2006 Q1
        "sweden",  # ends 2020 Q3 --- epu
    ],
    file_suffixes="reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)

# With oneway M2 shocks
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["m2"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxm2", "minm2"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
    countries_drop=[
        "india",  # 2012 Q1
        "china",  # 2007 Q1 and potentially exclusive case
        "chile",  # 2010 Q1 and potentially exclusive case
        "colombia",  # 2005 Q4
        "singapore",  # 2005 Q1
    ],
    file_suffixes="m2_",  # format: "abc_" or ""
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)
# With oneway M2 shocks (reduced)
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["m2"] + cols_endog_short,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxm2", "minm2"],
    list_uncertainty_variables=["maxminepu"],
    cols_state_dependency=cols_threshold_epu,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
    countries_drop=[
        "india",  # 2012 Q1
        "china",  # 2007 Q1 and potentially exclusive case
        "chile",  # 2010 Q1 and potentially exclusive case
        "colombia",  # 2005 Q4
        "singapore",  # 2005 Q1
    ],
    file_suffixes="m2_reduced_",  # format: "abc_" or ""
    beta_values_to_simulate=epu_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
)

# %%
# IV.2 --- Do everything but exclude country by country
# list_countries_master = [
#     "australia",
#     "belgium",
#     "canada",
#     "china",
#     "colombia",
#     "denmark",
#     "france",
#     "germany",
#     "greece",
#     "india",
#     "ireland",
#     "italy",
#     "japan",
#     "mexico",
#     "netherlands",
#     "russian_federation",
#     "singapore",
#     "spain",
#     "sweden",
#     "united_states",
# ]

# # STIR
# base_countries_to_drop = [
#     "india",  # 2016 Q3
#     "denmark",  # ends 2019 Q3
#     "china",  # 2007 Q4 and potentially exclusive case
#     "colombia",  # 2006 Q4
#     "germany",  # 2006 Q1
#     "sweden",  # ends 2020 Q3 --- epu
# ]
# base_countries_included = [
#     i for i in list_countries_master if i not in base_countries_to_drop
# ]
# for country_to_exclude in tqdm(base_countries_included):
#     do_everything_single_interaction_panellp(
#         cols_endog_after_shocks=["stir"] + cols_endog_long,
#         cols_all_exog=["maxminbrent"],
#         list_mp_variables=["maxminstir"],
#         list_uncertainty_variables=["maxminepu"],
#         cols_state_dependency=cols_threshold_epu,
#         state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
#         countries_drop=base_countries_to_drop + [country_to_exclude],
#         file_suffixes="ex" + country_to_exclude + "_",  # format: "abc_" or ""
#         beta_values_to_simulate=epu_values_combos,
#         irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
#     )

# # STIR (reduced)
# base_countries_included = [
#     i for i in list_countries_master if i not in base_countries_to_drop
# ]
# for country_to_exclude in tqdm(base_countries_included):
#     do_everything_single_interaction_panellp(
#         cols_endog_after_shocks=["stir"] + cols_endog_short,
#         cols_all_exog=["maxminbrent"],
#         list_mp_variables=["maxminstir"],
#         list_uncertainty_variables=["maxminepu"],
#         cols_state_dependency=cols_threshold_epu,
#         state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
#         countries_drop=base_countries_to_drop + [country_to_exclude],
#         file_suffixes="reduced_ex" + country_to_exclude + "_",  # format: "abc_" or ""
#         beta_values_to_simulate=epu_values_combos,
#         irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
#     )

# %%
# IV.3 --- Do everything but exclude certain time periods
# Post-GFC
# With STIR
# do_everything_single_interaction_panellp(
#     cols_endog_after_shocks=["stir"] + cols_endog_long,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminstir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_state_dependency=cols_threshold_epu,
#     state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "colombia",  # 2006 Q4
#         "germany",  # 2006 Q1
#         "sweden",  # ends 2020 Q3 --- epu
#     ],
#     file_suffixes="postgfc_",  # format: "abc_" or ""
#     beta_values_to_simulate=epu_values_combos,
#     irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
#     t_start=date(2012, 1, 1),
# )

# # With STIR (reduced)
# do_everything_single_interaction_panellp(
#     cols_endog_after_shocks=["stir"] + cols_endog_short,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminstir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_state_dependency=cols_threshold_epu,
#     state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "colombia",  # 2006 Q4
#         "germany",  # 2006 Q1
#         "sweden",  # ends 2020 Q3 --- epu
#     ],
#     file_suffixes="reduced_postgfc_",  # format: "abc_" or ""
#     beta_values_to_simulate=epu_values_combos,
#     irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
#     t_start=date(2012, 1, 1),
# )

# # Pre-COVID
# # With STIR
# do_everything_single_interaction_panellp(
#     cols_endog_after_shocks=["stir"] + cols_endog_long,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminstir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_state_dependency=cols_threshold_epu,
#     state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "colombia",  # 2006 Q4
#         "germany",  # 2006 Q1
#         "sweden",  # ends 2020 Q3 --- epu
#     ],
#     file_suffixes="precovid_",  # format: "abc_" or ""
#     beta_values_to_simulate=epu_values_combos,
#     irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
#     t_end=date(2019, 12, 31),
# )

# # With STIR (reduced)
# do_everything_single_interaction_panellp(
#     cols_endog_after_shocks=["stir"] + cols_endog_short,
#     cols_all_exog=["maxminbrent"],
#     list_mp_variables=["maxminstir"],
#     list_uncertainty_variables=["maxminepu"],
#     cols_state_dependency=cols_threshold_epu,
#     state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
#     countries_drop=[
#         "india",  # 2016 Q3
#         "denmark",  # ends 2019 Q3
#         "china",  # 2007 Q4 and potentially exclusive case
#         "colombia",  # 2006 Q4
#         "germany",  # 2006 Q1
#         "sweden",  # ends 2020 Q3 --- epu
#     ],
#     file_suffixes="reduced_precovid_",  # format: "abc_" or ""
#     beta_values_to_simulate=epu_values_combos,
#     irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
#     t_end=date(2019, 12, 31),
# )


# %%
# V --- Do everything but with WUI
cols_threshold_wui = ["wui_ref"]
wui_values_combos = [0.2, 0.4, 0.7]
# With STIR
do_everything_single_interaction_panellp(
    cols_endog_after_shocks=["stir"] + cols_endog_long,
    cols_all_exog=["maxminbrent"],
    list_mp_variables=["maxminstir"],
    list_uncertainty_variables=["maxminwui"],
    cols_state_dependency=cols_threshold_wui,
    state_dependency_nice_for_title="HH debt, Gov debt",  # HH debt, Gov debt
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
    beta_values_to_simulate=wui_values_combos,
    irf_colours_for_each_beta=epu_values_combos_irf_line_colours,
    input_df_suffix="large_yoy",  # different data set
)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%