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
from plotly.subplots import make_subplots

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


# Define channels function
def IRFPlotChannels(
    irf,
    response,
    shock,
    channels,
    channel_colours,
    n_columns,
    n_rows,
    maintitle="Local Projections Model: Propagation Channels",
    show_fig=False,
    save_pic=False,
    out_path="",
    out_name="",
):
    if (len(response) * len(shock)) > (n_columns * n_rows):
        raise NotImplementedError(
            "Number of subplots (n_columns * n_rows) is smaller than number of IRFs to be plotted (n)"
        )
    # Set number of rows and columns
    n_col = n_columns
    n_row = n_rows
    # Generate titles first
    list_titles = []
    for r in response:
        for s in shock:
            subtitle = [s + " -> " + r]
            list_titles = list_titles + subtitle
    # Main plot settings
    fig = make_subplots(rows=n_row, cols=n_col, subplot_titles=list_titles)
    # Subplot loops
    count_col = 1
    count_row = 1
    legend_count = 0
    for r in response:
        for s in shock:
            d = irf.loc[(irf["Response"] == r) & (irf["Shock"] == s)]
            d["Zero"] = 0  # horizontal line
            # Set legend
            if legend_count == 0:
                showlegend_bool = True
            elif legend_count > 0:
                showlegend_bool = False
            # Zero
            fig.add_trace(
                go.Scatter(
                    x=d["Horizon"],
                    y=d["Zero"],
                    mode="lines",
                    line=dict(color="grey", width=1, dash="solid"),
                    showlegend=False,
                ),
                row=count_row,
                col=count_col,
            )
            # Total
            fig.add_trace(
                go.Scatter(
                    x=d["Horizon"],
                    y=d["Total"],
                    mode="lines",
                    line=dict(color="black", width=3, dash="solid"),
                    showlegend=False,
                ),
                row=count_row,
                col=count_col,
            )
            # Add channels
            for c, c_colour in zip(channels, channel_colours):
                fig.add_trace(
                    go.Bar(
                        x=d["Horizon"],
                        y=d[c],
                        name=c,
                        marker=dict(color=c_colour),
                        showlegend=showlegend_bool,
                    ),
                    row=count_row,
                    col=count_col,
                )
            count_col += 1  # move to next
            if count_col <= n_col:
                pass
            elif count_col > n_col:
                count_col = 1
                count_row += 1
            # No further legends
            legend_count += 1
    fig.update_annotations(font_size=11)
    fig.update_layout(
        title=maintitle,
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=True,
        barmode="relative",
        font=dict(color="black", size=11),
    )
    if show_fig == True:
        fig.show()
    if save_pic == True:
        fig.write_image(out_path + out_name + ".png", height=1080, width=1920)
        fig.write_html(out_path + out_name + ".html")
    return fig


# %%
# ------- LOOP ------
list_shock_prefixes = ["max", "min", "maxmin"]
# list_mp_variables = [i + "stir" for i in list_shock_prefixes]
# list_uncertainty_variables = [i + "epu" for i in list_shock_prefixes]
list_mp_variables = ["maxminstir"]  # maxminstir
list_uncertainty_variables = ["maxminepu"]  # maxepu
for mp_variable in list_mp_variables:
    for uncertainty_variable in list_uncertainty_variables:
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
            "stir",
            "hhdebt",  # _ngdp
            "corpdebt",  # _ngdp
            "govdebt",  # _ngdp
            "gdp",  # urate gdp
            # "capflows_ngdp",
            "corecpi",  # corecpi cpi
            "reer",
        ]
        colours_all_endog = [
            "crimson",
            # "epu",
            "darkgreen",
            "green",
            "darkmagenta",  # _ngdp
            "magenta",  # _ngdp
            "sandybrown",  # _ngdp
            "darkblue",  # urate gdp
            # "",
            "orange",  # corecpi cpi
            "cadetblue",
            "lightgrey",  # for own shock
        ]
        cols_all_exog = ["maxminbrent"]  # maxminstir
        # cols_threshold = ["hhdebt_ngdp_ref"]
        df = df[cols_groups + cols_all_endog + cols_all_exog].copy()
        # Check when the panel becomes balanced
        # check_balance_timing(input=df)
        # check_balance_endtiming(input=df)
        # Trim more countries
        # if "stir" in mp_variable:
        countries_drop = [
            "india",  # 2016 Q3
            "denmark",  # ends 2019 Q3
            "china",  # 2007 Q4 and potentially exclusive case
            "colombia",  # 2006 Q4
            "germany",  # 2006 Q1
            "sweden",  # ends 2020 Q3 --- epu
            # "mexico",  # ends 2023 Q1 --- ngdp (keep if %yoy for debt and not %diff_ngdp)
            # "russia",  # basket case
        ]  # 14 countries
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
        #         "chile",  # ends 2022 Q2  (doesn't have stir?)
        #     ]  # 10 countries
        df = df[~df["country"].isin(countries_drop)].copy()
        # Check again when panel becomes balanced
        # check_balance_timing(input=df)
        # check_balance_endtiming(input=df)
        # Timebound
        df["date"] = pd.to_datetime(df["quarter"]).dt.date
        df = df[(df["date"] >= t_start)]
        del df["date"]
        # Drop NA
        df = df.dropna(axis=0)
        # Threshold
        # Reset index
        df = df.reset_index(drop=True)
        # Numeric time
        df["time"] = df.groupby("country").cumcount()
        del df["quarter"]
        # Set multiindex
        df = df.set_index(["country", "time"])

        # IV --- Analysis
        # Generate list of list of endog and exog variables to rotate between
        nested_list_endog = []
        nested_list_exog = []
        for col_endog_loc in range(len(cols_all_endog)):
            # endogs
            sublist_endog = (
                cols_all_endog[:col_endog_loc] + cols_all_endog[col_endog_loc + 1 :]
            )
            nested_list_endog.append(sublist_endog)  # generate list of list
            # exogs
            sublist_exog = cols_all_exog.copy()
            sublist_exog = sublist_exog + [cols_all_endog[col_endog_loc]]
            nested_list_exog.append(sublist_exog)  # generate list of list

        # Estimate model
        count_irf = 0
        for list_endog, list_exog in tqdm(zip(nested_list_endog, nested_list_exog)):
            irf = lp.PanelLPX(
                data=df,
                Y=list_endog,
                X=list_exog,
                response=list_endog,
                horizon=12,
                lags=1,
                varcov="kernel",
                ci_width=0.8,
            )
            # remove CIs from reference IRFs
            for col in ["UB", "LB"]:
                del irf[col]
            # rename irf
            irf = irf.rename(
                columns={"Mean": list_exog[-1]}
            )  # since there are no base exog variables, otherwise take last
            # consolidate
            if count_irf == 0:
                irf_consol = irf.copy()
            elif count_irf > 0:
                irf_consol = irf_consol.merge(
                    irf, on=["Shock", "Response", "Horizon"], how="outer"
                )
            # next
            count_irf += 1
        # Load reference IRF from another script
        irf_total = pd.read_parquet(
            path_output
            + "panellp_irf_"
            + "modwith_"
            + uncertainty_variable
            + "_"
            + mp_variable
            + ".parquet"
        )
        # Rename reference IRFs
        irf_total = irf_total.rename(columns={"Mean": "Total"})
        # Remove CIs from reference IRFs
        for col in ["UB", "LB"]:
            del irf_total[col]
        # Merge with LPX frame
        irf_consol = irf_consol.merge(
            irf_total, on=["Shock", "Response", "Horizon"], how="outer"
        )
        # Compute channel sizes
        for col in cols_all_endog:
            irf_consol[col] = irf_consol["Total"] - irf_consol[col]
        irf_consol["Own"] = irf_consol["Total"] - irf_consol[cols_all_endog].sum(axis=1)
        # Generate output
        irf_consol.to_parquet(
            path_output
            + "panellp_irf_channels_"
            + "modwith_"
            + uncertainty_variable
            + "_"
            + mp_variable
            + ".parquet"
        )
        # plot irf
        for shock in [uncertainty_variable, mp_variable]:
            fig = IRFPlotChannels(
                irf=irf_consol,
                response=cols_all_endog,
                shock=[shock],
                channels=cols_all_endog + ["Own"],
                channel_colours=colours_all_endog,
                n_columns=3,
                n_rows=3,
                maintitle="Decomposition Propagation Channels of IRFs of "
                + shock
                + " shocks"
                + " (exog: "
                + ", ".join(cols_all_exog)
                + ")",
                show_fig=False,
                save_pic=False,
            )
            fig.write_image(
                path_output
                + "panellp_irf_channels_"
                + "modwith_"
                + uncertainty_variable
                + "_"
                + mp_variable
                + "_"
                + "shock"
                + shock
                + ".png",
                height=1080,
                width=1920,
            )

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
