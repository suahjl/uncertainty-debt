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
            # "hhdebt",  # _ngdp
            # "corpdebt",  # _ngdp
            # "govdebt",  # _ngdp
            "gdp",  # urate gdp
            # "capflows_ngdp",
            "corecpi",  # corecpi cpi
            "reer",
        ]
        cols_all_exog = ["maxminbrent"]  # maxminm2
        cols_threshold = ["hhdebt_ngdp_ref"]
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
        # Threshold
        threshold_variable = "hhdebt_ngdp"

        def find_threshold(
            df: pd.DataFrame, threshold_variable: str, option: str, param_choice: float
        ):
            if option == "dumb":
                df.loc[
                    df[threshold_variable + "_ref"] >= param_choice,
                    threshold_variable + "_above_threshold",
                ] = 1
                df.loc[
                    df[threshold_variable + "_ref"] < param_choice,
                    threshold_variable + "_above_threshold",
                ] = 0
                print("Threshold is " + str(param_choice))
            elif option == "global_quantile":
                df.loc[
                    df[threshold_variable + "_ref"]
                    >= df[threshold_variable + "_ref"].quantile(param_choice),
                    threshold_variable + "_above_threshold",
                ] = 1
                df.loc[
                    df[threshold_variable + "_ref"]
                    < df[threshold_variable + "_ref"].quantile(param_choice),
                    threshold_variable + "_above_threshold",
                ] = 0
                print(
                    "Threshold is "
                    + str(df[threshold_variable + "_ref"].quantile(param_choice))
                )
            elif option == "country_quantile":
                ref = pd.DataFrame(
                    df.groupby("country")[threshold_variable + "_ref"].quantile(0.8)
                ).reset_index()
                ref = ref.rename(
                    columns={
                        threshold_variable + "_ref": threshold_variable + "_threshold"
                    }
                )
                df = df.merge(ref, how="left", on="country")
                df.loc[
                    df[threshold_variable + "_ref"]
                    >= df[threshold_variable + "_threshold"],
                    threshold_variable + "_above_threshold",
                ] = 1
                df.loc[
                    df[threshold_variable + "_ref"]
                    < df[threshold_variable + "_threshold"],
                    threshold_variable + "_above_threshold",
                ] = 0
            elif option == "reg_thresholdselection":
                df_opt_threshold = pd.read_csv(
                    path_output
                    + "reg_thresholdselection_m2_reduced_fe_"
                    + "modwith_"
                    + uncertainty_variable
                    + "_"
                    + mp_variable
                    + "_opt_threshold"
                    + ".csv"
                )
                opt_threshold = df_opt_threshold.iloc[0, 0]
                df.loc[
                    df[threshold_variable + "_ref"] >= opt_threshold,
                    threshold_variable + "_above_threshold",
                ] = 1
                df.loc[
                    df[threshold_variable + "_ref"] < opt_threshold,
                    threshold_variable + "_above_threshold",
                ] = 0
                print(
                    "optimal threshold: "
                    + threshold_variable
                    + " = "
                    + str(opt_threshold)
                )
            return df

        df = find_threshold(
            df=df,
            threshold_variable="hhdebt_ngdp",
            option="reg_thresholdselection",
            param_choice=0,
        )

        # Reset index
        df = df.reset_index(drop=True)
        # Numeric time
        df["time"] = df.groupby("country").cumcount()
        del df["quarter"]
        # Set multiindex
        df = df.set_index(["country", "time"])

        # IV --- Analysis
        # estimate model
        irf_on, irf_off = lp.ThresholdPanelLPX(
            data=df,
            Y=cols_all_endog,
            X=cols_all_exog,
            threshold_var=threshold_variable + "_above_threshold",
            response=cols_all_endog,
            horizon=12,
            lags=1,
            varcov="kernel",
            ci_width=0.8,
        )
        # plot irf
        for shock in [uncertainty_variable, mp_variable]:
            fig = lp.ThresholdIRFPlot(
                irf_threshold_on=irf_on,
                irf_threshold_off=irf_off,
                response=cols_all_endog,
                shock=[shock],
                n_columns=3,
                n_rows=2,
                maintitle="IRFs of "
                + shock
                + " shocks when "
                + threshold_variable
                + " is above threshold"
                + " (exog: "
                + ", ".join(cols_all_exog)
                + ")",
                show_fig=False,
                save_pic=False,
                annot_size=12,
                font_size=12,
            )
            # save irf (need to use kaleido==0.1.0post1)
            fig.write_image(
                path_output
                + "panelthresholdlp_m2_reduced_irf_"
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

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
