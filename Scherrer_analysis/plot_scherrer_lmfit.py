import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter
import matplotlib as mpl

mpl.rcParams.update(
    {
        # 1) pick Arial for all sans-serif text…
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        # 2) make mathtext use Arial as well
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "rm",
        # 3) still your other style settings
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 14,
        "figure.figsize": (6, 8),
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "axes.grid": False,
        "savefig.dpi": 300,
        # if you had usetex on, turn it off so mathtext takes over:
        "text.usetex": False,
    }
)

# Select Excel file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select LMFIT Excel file", filetypes=[("Excel files", "*.xlsx")]
)

# Output directory
output_dir = os.path.splitext(file_path)[0] + "_plots"
os.makedirs(output_dir, exist_ok=True)

# Read all sheets
xls = pd.ExcelFile(file_path)
sheets = xls.sheet_names

# User-defined HKLs to plot (must match sheet names exactly)
hkls_to_plot = ["(001)", "(011)", "(111)"]
# hkls_to_plot = ['(011)']

colors = plt.cm.tab10.colors

# Plot FWHM for selected HKLs
# plt.figure(figsize=(8, 6))
# for i, sheet in enumerate(hkls_to_plot):
#     if sheet not in sheets:
#         print(f"Sheet {sheet} not found in file.")
#         continue

#     df = pd.read_excel(xls, sheet_name=sheet)
#     df = df.sort_values('Time (s)')
#     df_filtered = df[df['R_squared'] >= 0.9]

#     if len(df_filtered) == 0:
#         print(f"No good fits for {sheet}")
#         continue

#     time = df_filtered['Time (s)'].to_numpy()
#     fwhm = df_filtered['FWHM (A^-1)'].to_numpy()

#     color = colors[i % len(colors)]
#     plt.scatter(time, fwhm, label=sheet, color=color)

#     if len(time) >= 5:
#         fwhm_smooth = savgol_filter(fwhm, window_length=5, polyorder=2)
#         plt.plot(time, fwhm_smooth, '-', color=color)

# plt.xlabel('Time (s)')
# plt.ylabel('FWHM (A$^{-1}$)')
# plt.title('FWHM vs Time (LMFIT Results)')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'Selected_FWHM_vs_Time.png'))
# plt.show()

# Plot Scherrer size for selected HKLs
average_size = pd.DataFrame()
time_points = []

plt.figure(figsize=(8, 6))
for i, sheet in enumerate(hkls_to_plot):
    if sheet not in sheets:
        continue

    df = pd.read_excel(xls, sheet_name=sheet)
    df = df.sort_values("Time (s)")
    df_filtered = df[df["R_squared"] >= 0.9]

    if len(df_filtered) == 0:
        continue

    time = df_filtered["Time (s)"].to_numpy()
    size = df_filtered["Scherrer Size (nm)"].to_numpy()
    average_size = pd.concat(
        [average_size, df_filtered["Scherrer Size (nm)"].to_frame().rename(columns={"Scherrer Size (nm)": sheet})],
        axis=1,
    )
    if i == 0:
        time_points = pd.DataFrame(time, columns=["Time (s)"])

    color = colors[i % len(colors)]
    # plt.scatter(time, size, label=sheet, color=color)

    if len(time) >= 5:
        size_smooth = savgol_filter(size, window_length=7, polyorder=3)
        plt.scatter(
            time,
            size_smooth,
            color=color,
            label=sheet,
        )

plt.xlabel("Time (s)")
plt.ylabel("Scherrer Size (nm)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Selected_Scherrer_vs_Time.png"))

import seaborn as sns


# Align time_points and average_size, handling missing data
# average_size = average_size.reindex(time_points.index)
average_size = average_size.mean(axis=1, skipna=True)



fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x=time_points["Time (s)"], y=average_size, ax=ax)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Average Scherrer Size (nm)")
# ax.set_ylim(0, 30)
# ax.set_xlim(0, 375)
plt.grid()
plt.tight_layout()

output_path=os.path.join(output_dir, "Average_Scherrer_Size_vs_Time.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()


print("Plots from LMFIT output generated successfully!")
