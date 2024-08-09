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
import compile_data_raw

# %%
import process_macro_data

# %%
import descriptive_lineplot
import descriptive_scatter
import descriptive_scatter_regime

# %%
import analysis_panelthresholdlp
import analysis_panellp
import analysis_cbycthresholdlp

# %%
import analysis_panelthresholdlp_ber
import analysis_panellp_ber
import analysis_cbycthresholdlp_ber



# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%