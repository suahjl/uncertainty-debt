# %%
from helper import telsendmsg, telsendfiles, telsendimg
import time
import os
from dotenv import load_dotenv

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
tel_config = os.getenv("TEL_CONFIG")

# %%
# I --- Which scripts to run in succession?
# %%
import compile_data_macro_raw
import process_macro_data

# %%
import compile_data_micro_raw
import compile_data_micro_quarterly_raw
import process_micro_data
import process_micro_quarterly_data


# %%
import descriptive_lineplot_ratesinlevels


# %%
import analysis_reg_quadrant_privdebt_thresholdselection_consol

# %%
import analysis_quadrant_privdebt_panelthresholdlp_consol

# %%
import descriptive_scatter_regime_ratesinlevels
import descriptive_special_charts

# %%
import compile_output_for_chartpack
import compile_output_for_paper


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
