import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter

# Scherrer equation
# Scherrer size (nm) from FWHM (A^-1):
def scherrer_size(fwhm, k_shape=0.9):
    size = (2 * np.pi * k_shape / fwhm) / 10 / (4 * np.pi / 3)**(1/3)
    return size

# Filtering function based on R_squared and PCOV
def is_good_fit(row, r2_threshold=0.9, pcov_threshold=1e4):
    if row['R_squared'] < r2_threshold:
        return False
    try:
        pcov_array = np.array(ast.literal_eval(row['PCOV']))
        if np.any(np.abs(pcov_array) > pcov_threshold):
            return False
    except Exception:
        return False
    return True

# Select Excel file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Excel file", filetypes=[("Excel files", "*.xlsx")])

# Output directory
output_dir = os.path.splitext(file_path)[0] + "_plots"
os.makedirs(output_dir, exist_ok=True)

# Read all sheets
xls = pd.ExcelFile(file_path)
sheets = xls.sheet_names

# User-defined HKLs to plot (must match sheet names exactly)
# hkls_to_plot = ['(001)', '(011)', '(002)', '(012)', '(022)']
hkls_to_plot = [ '(001)', '(011)', '(002)']

colors = plt.cm.tab10.colors

# # Plot FWHM for selected HKLs
# plt.figure(figsize=(8, 6))
# for i, sheet in enumerate(hkls_to_plot):
#     if sheet not in sheets:
#         print(f"Sheet {sheet} not found in file.")
#         continue
    
#     df = pd.read_excel(xls, sheet_name=sheet)
#     df = df.sort_values('Time (s)')
#     df['Keep'] = df.apply(is_good_fit, axis=1)
#     df_filtered = df[df['Keep']]

#     if len(df_filtered) == 0:
#         print(f"No good fits for {sheet}")
#         continue

#     time = df_filtered['Time (s)'].to_numpy()
#     fwhm = df_filtered['FWHM (A^-1)'].to_numpy()

#     color = colors[i % len(colors)]
#     plt.scatter(time, fwhm, label=sheet, color=color)

#     if len(time) >= 5:
#         fwhm_smooth = savgol_filter(fwhm, window_length=5, polyorder=2)
#         # plt.plot(time, fwhm_smooth, '-', color=color)

# plt.xlabel('Time (s)')
# plt.ylabel('FWHM (A$^{-1}$)')
# plt.title('Nanofibers S4')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'Selected_FWHM_vs_Time.png'))
# plt.show()

# Plot Scherrer size for selected HKLs
plt.figure(figsize=(8, 6))
for i, sheet in enumerate(hkls_to_plot):
    if sheet not in sheets:
        continue
    
    df = pd.read_excel(xls, sheet_name=sheet)
    df = df.sort_values('Time (s)')
    df['Keep'] = df.apply(is_good_fit, axis=1)
    df_filtered = df[df['Keep']]

    if len(df_filtered) == 0:
        continue

    time = df_filtered['Time (s)'].to_numpy()
    fwhm = df_filtered['FWHM (A^-1)'].to_numpy()
    size = scherrer_size(fwhm)

    color = colors[i % len(colors)]
    plt.scatter(time, size, label=sheet, color=color)

    if len(time) >= 5:
        size_smooth = savgol_filter(size, window_length=5, polyorder=2)
        # plt.plot(time, size_smooth, '-', color=color)

plt.xlabel('Time (s)')
plt.ylabel('Scherrer Size (nm)')
title = tk.simpledialog.askstring("Plot Title", "Enter plot title:", initialvalue="Nanofibers")
plt.title(title)
plt.grid()
plt.ylim(10, 20)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Scherrer_size_vs_Time.png'))
plt.show()

print("Selected HKL FWHM and Scherrer size plots generated successfully!")
