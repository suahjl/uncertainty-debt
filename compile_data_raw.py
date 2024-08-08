#  %%
"""
Important note:
CPI and Core CPI YoY growth have been replaced with quarterly series for wider and longer coverage.
Have not checked other variables for similar variation in coverage.
Added ST gov bond yields to manual data set (not included in API spreadsheet)
"""

# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, get_data_from_api_ceic
import statsmodels.tsa.api as smt
from statsmodels.tsa.ar_model import ar_select_order
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
path_ceic = "./ceic/"
tel_config = os.getenv("TEL_CONFIG")
t_start = date(1990, 1, 1)
manual_data = ast.literal_eval(os.getenv("MANUAL_DOWNLOAD_DATA"))

# %%
# I --- Load data from CEIC
if not manual_data:
    seriesids_all = pd.read_csv(path_ceic + "ceic_macro" + ".csv")
    count_col = 0
    df_debug = pd.DataFrame()
    # seriesids_all = seriesids_all[["epu"]]
    for col in tqdm(list(seriesids_all.columns)):
        try:
            # subset column by column
            seriesids = seriesids_all[col].dropna()
            seriesids = seriesids.astype("str")
            seriesids = [i.replace(".0", "") for i in seriesids]  # remove trailing decimals
            seriesids = [re.sub("[^0-9]+", "", i) for i in list(seriesids)]  # keep only number
            seriesids = [int(i) for i in seriesids]  # convert into list of int
            # pull from ceic one by one
            print("Now downloading " + col)
            print(",".join([str(i) for i in seriesids]))
            df_sub = get_data_from_api_ceic(
                series_ids=seriesids, start_date=t_start, historical_extension=False
            )
            # wrangle
            df_sub = df_sub.reset_index()
            df_sub = df_sub.rename(columns={"date": "date", "country": "country", "value": col})
            # collapse into quarterly
            df_sub["quarter"] = pd.to_datetime(df_sub["date"]).dt.to_period("Q")
            df_sub = df_sub.groupby(["quarter", "country"])[col].mean().reset_index(drop=False)
            df_sub = df_sub[["quarter", "country", col]]
            # merge
            if count_col == 0:
                df = df_sub.copy()
            elif count_col > 0:
                df = df.merge(df_sub, on=["quarter", "country"], how="outer")
            # next
            count_col += 1
        except:
            df_debug = pd.concat([df_debug, seriesids_all[col]], axis=1)
    df = df.sort_values(by=["country", "quarter"], ascending=[True, True])
    df = df.reset_index(drop=True)

    df_debug.to_csv("./documents/ceic_error_series.csv", index=False)

elif manual_data:

    # Function to convert raw CEIC into wide df, then long df
    def process_ceic_df(df, col_label):
        """
        ---Note--- 
        For some reason, manually downloaded csv files from CEIC are malformed.
        If the "Series Remarks" attribute / row is included, they are not properly
        comma-separated, hence pd.read_csv() won't be able to parse the file correctly.
        Those rows need to be manually deleted as a txt file (not excel csv, which will
        distort the raw downloaded timepoints).
        Where possible, juse use pyceic or the http json api.
        """

        # deal with the ceic csv mess
        df.columns = df.iloc[0]  # first row is always "Region"
        df = df.rename(columns={"Region": "date"})
        timepoints_pattern = r'^\d{2}/\d{4}$'  # mm/yyyy in str
        df = df[df["date"].str.match(timepoints_pattern, na=False)]  # keep only timepoints
        df["quarter"] = pd.to_datetime(df["date"].astype("str")).dt.to_period("Q")
        del df["date"]

        # convert into long form
        df = df.melt(id_vars=['quarter'], var_name='country', value_name=col_label)
        df[col_label] = df[col_label].astype("float")
        df = df.groupby(["country", "quarter"])[col_label].mean().reset_index(drop=False)

        # format country labels
        df["country"] = df["country"].str.replace(r"[^A-Za-z0-9]+", '_', regex=True).str.lower()

        # reset indices
        df = df.reset_index(drop=True)

        # output
        return df
    
    def process_ceic_df_global(df, col_label, daily):
        """
        ---Note--- 
        For some reason, manually downloaded csv files from CEIC are malformed.
        If the "Series Remarks" attribute / row is included, they are not properly
        comma-separated, hence pd.read_csv() won't be able to parse the file correctly.
        Those rows need to be manually deleted as a txt file (not excel csv, which will
        distort the raw downloaded timepoints).
        Where possible, juse use pyceic or the http json api.
        """

        # deal with the ceic csv mess
        df.columns = ["date", col_label]  # only 2 columns
        if daily:
            timepoints_pattern = r'^\d{2}/\d{2}/\d{4}$'  # dd/mm/yyyy in str
        elif not daily:   
            timepoints_pattern = r'^\d{2}/\d{4}$'  # mm/yyyy in str
        df = df[df["date"].str.match(timepoints_pattern, na=False)]  # keep only timepoints
        df["quarter"] = pd.to_datetime(df["date"].astype("str")).dt.to_period("Q")
        del df["date"]

        # labels
        df[col_label] = df[col_label].astype("float")
        df = df.groupby(["quarter"])[col_label].mean().reset_index(drop=False)

        # reset indices
        df = df.reset_index(drop=True)

        # output
        return df

    # Collect file names (changed into txt files)
    list_filenames = [f for f in os.listdir(path_data + "ceic_manual_download/") if f.endswith(".txt")]
    # Collect column labels 
    list_col_labels = [os.path.splitext(f)[0] for f in list_filenames]
    # Load each csv file and process
    op_count = 0
    for filename, col_label in tqdm(zip(list_filenames, list_col_labels)):
        print("Processing " + col_label)
        df_sub = pd.read_csv(path_data + "ceic_manual_download/" +  filename)
        df_sub = process_ceic_df(df=df_sub, col_label=col_label)
        if op_count == 0:
            df = df_sub.copy()
        elif op_count > 0:
            df = df.merge(df_sub, on=["country", "quarter"], how="outer")
        op_count += 1
    # Load global variables
    for col, daily_or_not in zip(["brent"], [True]):
        df_global = pd.read_csv(path_data + "ceic_manual_download/global/" + col + ".txt")
        df_global = process_ceic_df_global(df=df_global, col_label=col, daily=daily_or_not)
        df_global = df_global[["quarter", col]].copy()
        df = df.merge(df_global, on="quarter", how="outer")
    # Sort
    df = df.sort_values(by=["country", "quarter"], ascending=[True, True])


# %%
# II --- Export
df["quarter"] = df["quarter"].astype("str")
df.to_parquet(path_data + "data_macro_raw.parquet") 

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
