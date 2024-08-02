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

# %%
# I --- Load data from CEIC
seriesids_all = pd.read_csv(path_ceic + "ceic_macro" + ".csv")
count_col = 0
seriesids_all = seriesids_all[["epu"]]
for col in tqdm(list(seriesids_all.columns)):
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
    df_sub["quarter"] = pd.to_datetime(df_sub["date"]).dt.to_period("q")
    df_sub = df_sub.groupby(["quarter", "country"])[col].mean().reset_index(drop=False)
    df_sub = df_sub[["quarter", "country", col]]
    # merge
    if count_col == 0:
        df = df_sub.copy()
    elif count_col > 0:
        df = df.merge(df_sub, on=["quarter", "country"], how="outer")
    # next
    count_col += 1
df = df.sort_values(by=["country", "quarter"], ascending=[True, True])
df = df.reset_index(drop=True)

# %%
# II --- Export
df["quarter"] = df["quarter"].astype("str")
df.to_parquet(path_data + "data_macro.parquet") 

# %%
# X --- Notify
# telsendmsg(conf=tel_config, msg="uncertainty-debt --- compile_data: COMPLETED")

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
