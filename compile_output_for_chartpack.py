# %%
# Compiles only some files and place them in the ./output/charts_for_chartpacks/ directory (not gitignored)
# Packaged in several pdf files for easy reference

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
path_for_chartpack = "./charts_for_chartpacks/"

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
# II --- Compile selected files (can instead softcode, but hardcoding makes it easier to define specific set)
list_regex = [
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_[\w]+_shockmaxminepu_response[\w]+.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_[\w]+_shockmaxminwui_response[\w]+.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_[\w]+_shockmaxminuct_response[\w]+.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_maxminepu_[\w]+_shockmaxminstir_response[\w]+.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_maxminwui_[\w]+_shockmaxminstir_response[\w]+.png",
    r"^quadrant_privdebt_panelthresholdlp_[\w]+_modwith_maxminuct_[\w]+_shockmaxminstir_response[\w]+.png",
    r"^micro_quarterly_panelthresholdlp_modwith_[\w]+_shockmaxminepu_response[\w]+.png",
    r"^micro_quarterly_panelthresholdlp_modwith_[\w]+_shockmaxminstir_response[\w]+.png",
    r"^micro_quarterly_panelthresholdlp_exjpus_modwith_[\w]+_shockmaxminepu_response[\w]+.png",
    r"^micro_quarterly_panelthresholdlp_exjpus_modwith_[\w]+_shockmaxminstir_response[\w]+.png",
    r"^micro_panelthresholdlp_modwith_[\w]+_shockmaxminepu_response[\w]+.png",
    r"^micro_panelthresholdlp_modwith_[\w]+_shockmaxminstir_response[\w]+.png",
]
list_pdfnames = [
    "quadrant_privdebt_panelthresholdlp_shockmaxminepu",
    "quadrant_privdebt_panelthresholdlp_shockmaxminwui",
    "quadrant_privdebt_panelthresholdlp_shockmaxminuct",
    "quadrant_privdebt_panelthresholdlp_modwith_maxminepu_shockmaxminstir",
    "quadrant_privdebt_panelthresholdlp_modwith_maxminwui_shockmaxminstir",
    "quadrant_privdebt_panelthresholdlp_modwith_maxminuct_shockmaxminstir",
    "micro_quarterly_panelthresholdlp_shockmaxminepu",
    "micro_quarterly_panelthresholdlp_shockmaxminstir",
    "micro_quarterly_panelthresholdlp_exjpus_shockmaxminepu",
    "micro_quarterly_panelthresholdlp_exjpus_shockmaxminstir",
    "micro_panelthresholdlp_shockmaxminepu",
    "micro_panelthresholdlp_shockmaxminstir",
]
clear_directory(directory_to_clear=path_for_chartpack)
for regex, pdfname in zip(list_regex, list_pdfnames):
    selected_file_names = compile_filenames_regex(
        pattern=regex,
        file_path=path_output,
    )
    # print(selected_file_names)
    copy_files(
        input_path=path_output,
        input_file_names=selected_file_names,
        output_path=path_for_chartpack,
    )
    selected_file_names = [path_for_chartpack + i for i in selected_file_names]
    pil_img2pdf_manualextension(
        list_images=selected_file_names, pdf_name=path_for_chartpack + pdfname
    )
    watermark_pdf(
        input_pdf=path_for_chartpack + pdfname,
        output_pdf=path_for_chartpack + pdfname,
        watermarks=selected_file_names,
    )


# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
