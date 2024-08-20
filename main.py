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

# %%
import descriptive_lineplot_ratesinlevels
import descriptive_scatter_ratesinlevels

# %%
import analysis_reg_quadrant_thresholdselection
import analysis_reg_quadrant_thresholdselection_reduced
import analysis_reg_quadrant_thresholdselection_ltir
import analysis_reg_quadrant_thresholdselection_ltir_reduced
import analysis_reg_quadrant_thresholdselection_m2
import analysis_reg_quadrant_thresholdselection_m2_reduced

# %%
import analysis_quadrant_panelthresholdlp
import analysis_quadrant_panelthresholdlp_reduced
import analysis_quadrant_panelthresholdlp_ltir
import analysis_quadrant_panelthresholdlp_ltir_reduced
import analysis_quadrant_panelthresholdlp_m2
import analysis_quadrant_panelthresholdlp_m2_reduced
import analysis_quadrant_thresholdlp_us_jln

# %%
import descriptive_scatter_regime
import descriptive_scatter_regime_ratesinlevels
import descriptive_special_charts

# %%
import compile_output_for_paper


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
