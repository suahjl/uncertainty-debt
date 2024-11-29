# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from helper import telsendmsg, telsendimg, telsendfiles
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast
import re

time_start = time.time()

# %%
# I --- Preliminaries
path_data = "./data/"
path_spg = path_data + "spg_manual_download/"
path_output = "./output/"


# %%
# II --- Functions
def process_spg_and_concat():
    count_df = 0
    # gather file names
    xlsx_filenames = [f for f in os.listdir(path_spg) if f.endswith(".xlsx")]
    for xlsx_filename in tqdm(xlsx_filenames):
        xlsx_filepath = os.path.join(path_spg + xlsx_filename)
        # load file iteratively
        df = pd.read_excel(xlsx_filepath, sheet_name="Sheet1")
        # ensure that we only have entity, entity id, country code and company type
        df.columns = list(df.iloc[3, :])
        if "SP_GEOGRAPHY" in df.columns:
            del df["SP_GEOGRAPHY"]
        # column names
        df.columns = list(df.iloc[3, 0:4]) + list(df.iloc[4, 4:])
        df = df.iloc[6:, :]
        # convert into panel
        df = pd.wide_to_long(
            df=df,
            stubnames="FY",
            i=["SP_ENTITY_NAME", "SP_ENTITY_ID", "SP_COUNTRY_CODE", "SP_COMPANY_TYPE"],
            j="year",
        )
        df = df.reset_index()
        # rename + use regex to figure out which col label this is
        pattern = r"_(\w+)\."
        match = re.search(pattern, xlsx_filename)
        col_name = match.group(1)
        df = df.rename(
            columns={
                "SP_ENTITY_NAME": "entity",
                "SP_ENTITY_ID": "id",
                "SP_COUNTRY_CODE": "cc",
                "SP_COMPANY_TYPE": "type",
                "FY": col_name,
            }
        )
        # merge / concat
        cols_identifiers = ["entity", "id", "cc", "type", "year"]
        if count_df == 0:
            df_full = df.copy()
        elif count_df > 0:
            if col_name in df_full.columns:
                df_full = df_full.merge(
                    df, how="outer", on=cols_identifiers + [col_name]
                )
            else:
                df_full = df_full.merge(df, how="outer", on=cols_identifiers)
        count_df += 1
    # output
    return df_full


# %%
# III --- Processing
df = process_spg_and_concat()
dict_dtypes = {
    "entity": "str",
    "id": "str",
    "cc": "str",
    "type": "str",
    "year": "int",
    "debt": "float",
    "debtebitda": "float",
    "debttangibleequity": "float",
    "revenue": "float",
}
df = df.replace("NM", np.nan)
df = df.astype(dict_dtypes)
df.to_parquet(path_data + "data_micro_raw.parquet", compression="brotli")

# %%
# X --- Notify
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
