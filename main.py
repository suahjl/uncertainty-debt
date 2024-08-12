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
import analysis_reg_thresholdselection
import analysis_panelthresholdlp
import analysis_panellp
# import analysis_cbycthresholdlp

# %%
import descriptive_scatter_regime
import descriptive_scatter_regime_ratesinlevels

# %%
import analysis_reg_thresholdselection_reduced
import analysis_panelthresholdlp_reduced
import analysis_panellp_reduced
# import analysis_cbycthresholdlp_reduced

# %%
import analysis_reg_thresholdselection_ltir
import analysis_panelthresholdlp_ltir
import analysis_panellp_ltir
# import analysis_cbycthresholdlp_ltir

# %%
import analysis_reg_thresholdselection_ltir_reduced
import analysis_panelthresholdlp_ltir_reduced
import analysis_panellp_ltir_reduced
# import analysis_cbycthresholdlp_ltir_reduced

# %%
import analysis_reg_thresholdselection_ber
import analysis_panelthresholdlp_ber
import analysis_panellp_ber
# import analysis_cbycthresholdlp_ber

# %%
import analysis_reg_thresholdselection_m2
import analysis_panelthresholdlp_m2
import analysis_panellp_m2
# import analysis_cbycthresholdlp_m2

# %%
import analysis_reg_thresholdselection_m2_reduced
import analysis_panelthresholdlp_m2_reduced
import analysis_panellp_m2_reduced
# import analysis_cbycthresholdlp_m2_reduced


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
