# %%
# Compiles only some files and place them in the ./output/charts_for_papers/ directory (not gitignored)

# %%
import re
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast
import shutil
from helper import pil_img2pdf_manualextension
from pypdf import PdfMerger, PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_output = "./output/"
path_for_paper = "./charts_for_papers/"

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


def create_watermark(text, filename):
    """Create a single-page PDF with the watermark text."""
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 9)  # font size
    c.setFillColorRGB(0, 0, 0, alpha=1)  # RGB, opacity
    c.drawString(10, 10, text)  # Adjust position (X, Y, text)
    c.save()


def watermark_pdf(input_pdf, output_pdf, watermarks):
    """Apply watermarks to a PDF, each page with a different watermark."""
    reader = PdfReader(input_pdf + ".pdf")
    writer = PdfWriter()

    for i, page in enumerate(reader.pages):
        # Create a unique watermark file
        watermark_filename = f"watermark_{i}.pdf"
        create_watermark(watermarks[i], watermark_filename)

        # Read the watermark
        watermark_reader = PdfReader(watermark_filename)
        watermark_page = watermark_reader.pages[0]

        # Merge the watermark with the page
        page.merge_page(watermark_page)
        writer.add_page(page)

        # Clean up the temporary watermark file
        os.remove(watermark_filename)

    # Write the watermarked PDF
    with open(output_pdf + ".pdf", "wb") as output_file:
        writer.write(output_file)


# %%
# II --- Compile selected files
list_regex = [
    r"^thresholdselection_fe_modwith_[\w]+_aicc_contour.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_[\w]+_shockmaxminepu_responsegdp.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_privdebt_ngdp[\w]+_govdebt_ngdp[\w]+modwith_[\w]+_shockmaxminepu_responsegdp.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_[\w]+_shockmaxminwui_responsegdp.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_privdebt_ngdp[\w]+_govdebt_ngdp[\w]+modwith_[\w]+_shockmaxminwui_responsegdp.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_[\w]+_shockmaxminuct_responsegdp.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_privdebt_ngdp[\w]+_govdebt_ngdp[\w]+modwith_[\w]+_shockmaxminuct_responsegdp.png",
    r"^micro_quarterly_panelthresholdlp_modwith_[\w]+_shockmaxminepu_responsecapex.png",
    r"^heatmap_cbyc_[\w]+",
]
individual_file_names = [
    "stacked_area_lineplot_globaldebt",
    "stacked_area_lineplot_globalgdp_breakdown",
    "lineplot_gepu",
    "scatter_global_gdpvol_epu",
    "scatter_cbyc_gdpvol_epu",
    "scatter_global_gdpvol_wui",
    "scatter_cbyc_gdpvol_wui",
    "scatter_global_gdpvol_uct",
    "scatter_cbyc_gdpvol_uct",
    "scatter_regime_ratesinlevels_privdebt_ngdp_gdp_against_maxminepu",
    "scatter_regime_ratesinlevels_govdebt_ngdp_gdp_against_maxminepu",
    "histogram_debt_ngdp",
    "lineplot_ratesinlevels_privdebt_ngdp_ref",
    "lineplot_ratesinlevels_govdebt_ngdp_ref",
    "lineplot_ratesinlevels_maxminepu",
    "lineplot_ratesinlevels_epu",
    "lineplot_ratesinlevels_maxminstir",
    "lineplot_ratesinlevels_stir",
    "lineplot_ratesinlevels_privdebt",
    "lineplot_ratesinlevels_govdebt",
    "lineplot_ratesinlevels_gdp",
    "lineplot_ratesinlevels_corecpi",
    "lineplot_ratesinlevels_reer",
    "lineplot_ratesinlevels_maxminbrent",
]
individual_file_names = [i + ".png" for i in individual_file_names]
file_names_from_regex = []
for regex in list_regex:
    file_names_from_regex += compile_filenames_regex(
        pattern=regex,
        file_path=path_output,
    )
selected_file_names = individual_file_names + file_names_from_regex
clear_directory(directory_to_clear=path_for_paper)
copy_files(
    input_path=path_output,
    input_file_names=selected_file_names,
    output_path=path_for_paper,
)
pdfname = "all_charts_for_paper"
selected_file_names = [path_output + i for i in selected_file_names]
pil_img2pdf_manualextension(
    list_images=selected_file_names, pdf_name=path_for_paper + pdfname
)
watermark_pdf(
    input_pdf=path_for_paper + pdfname,
    output_pdf=path_for_paper + pdfname,
    watermarks=selected_file_names,
)

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
