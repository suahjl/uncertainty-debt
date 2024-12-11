# %%
# Compiles only some files (non png) and place them in the ./output/tables_for_papers/ directory (not gitignored)

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
path_for_paper = "./tables_for_papers/"

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


def clear_directory(directory_to_clear):
    for filename in os.listdir(directory_to_clear):
        file_path = os.path.join(directory_to_clear, filename)
        if os.path.isfile(file_path):  # Check if it's a file
            os.remove(file_path)


def compile_filenames_regex(pattern, file_path):
    file_list = [file for file in os.listdir(file_path) if re.match(pattern, file)]
    return file_list


# %%
# II --- Compile selected files
list_regex = [
    r"^[\w]+_prebalancecountries_[\w+]+.csv",
    r"^[\w]+_postbalancecountries_[\w+]+.csv",
    r"^micro_quarterly_modwith_[\w]+_tabpercentiles_[\w]+.csv",
]
individual_file_names = [
    ""
]
individual_file_names = [i + ".csv" for i in individual_file_names]
file_names_from_regex = []
for regex in list_regex:
    file_names_from_regex += compile_filenames_regex(
        pattern=regex,
        file_path=path_output,
    )
selected_file_names = file_names_from_regex  # + individual_file_names 
clear_directory(directory_to_clear=path_for_paper)
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
