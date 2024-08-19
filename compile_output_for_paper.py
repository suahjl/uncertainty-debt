# %%
# Compiles only files and place them in the ./output/for_paper/ directory (not gitignored)

# %%
import re
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast
import shutil

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_output = "./output/"
path_for_paper = "./output/for_paper/"

# %%
# I --- Function


def copy_files(input_path, input_file_names, output_path):
    # Loop through each file in the list
    for input_file_name in input_file_names:
        # Check if the file exists
        if os.path.isfile(input_path + input_file_name):
            # Copy the file to the destination directory
            shutil.copy(input_path + input_file_name, output_path)
            print(f"Copied {input_path + input_file_name} to {output_path}")
        else:
            print(f"File not found: {input_path + input_file_name}")


# %%
# II --- Compile selected files
selected_file_names = [
    "stacked_area_lineplot_globaldebt",
    "lineplot_gepu",
    "scatter_regime_ratesinlevels_pooled_gdp_against_maxminepu",
    "scatter_regime_ratesinlevels_pooled_corecpi_against_maxminepu",
    "scatter_regime_ratesinlevels_pooled_cpi_against_maxminepu",
    "lineplot_hhdebt_ngdp_ref",
    "lineplot_corpdebt_ngdp_ref",
    "lineplot_govdebt_ngdp_ref",
    "lineplot_hhdebt",
    "lineplot_corpdebt",
    "lineplot_govdebt",
    "lineplot_maxminepu",
    "lineplot_maxminstir",
    "lineplot_ratesinlevels_epu",
    "lineplot_ratesinlevels_stir",
    "panelthresholdlp_irf_modwith_maxminepu_maxminstir_shockmaxminepu",
    "panelthresholdlp_irf_modwith_maxminepu_maxminstir_shockmaxminstir",
    "panelthresholdlp_irf_on_modwith_maxminepu_maxminstir_shockmaxminepu",
    "panelthresholdlp_irf_on_modwith_maxminepu_maxminstir_shockmaxminstir",
    "panelthresholdlp_irf_off_modwith_maxminepu_maxminstir_shockmaxminepu",
    "panelthresholdlp_irf_off_modwith_maxminepu_maxminstir_shockmaxminstir",
    "panelthresholdlp_ltir_irf_modwith_maxminepu_maxminltir_shockmaxminepu",
    "panelthresholdlp_ltir_irf_modwith_maxminepu_maxminltir_shockmaxminltir",
    "panelthresholdlp_reduced_irf_modwith_maxminepu_maxminstir_shockmaxminepu",
    "panelthresholdlp_reduced_irf_modwith_maxminepu_maxminstir_shockmaxminstir",
    "panelthresholdlp_ltir_reduced_irf_modwith_maxminepu_maxminltir_shockmaxminepu",
    "panelthresholdlp_ltir_reduced_irf_modwith_maxminepu_maxminltir_shockmaxminltir",
    "panelthresholdlp_m2_irf_modwith_maxminepu_maxminm2_shockmaxminepu",
    "panelthresholdlp_m2_irf_modwith_maxminepu_maxminm2_shockmaxminm2",
    "panelthresholdlp_ber_irf_modwith_maxminepu_maxminstir_shockmaxminepu",
    "panelthresholdlp_ber_irf_modwith_maxminepu_maxminstir_shockmaxminstir",
    "panellp_irf_modwith_maxminepu_maxminstir_shockmaxminepu",
    "panellp_irf_modwith_maxminepu_maxminstir_shockmaxminstir",
    "thresholdlp_irf_us_jln_modwith_maxminus_jln_maxminstir_shockmaxminus_jln",
    "thresholdlp_irf_us_jln_modwith_maxminus_jln_maxminstir_shockmaxminstir",
]
selected_file_names = [i + ".png" for i in selected_file_names]
copy_files(
    input_path=path_output,
    input_file_names=selected_file_names,
    output_path=path_for_paper,
)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
