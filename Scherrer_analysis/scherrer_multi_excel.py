import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
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


def compute_average_scherrer(file_path: str, sheets_to_include=None, r2_threshold=0.93):
    """
    Reads an Excel file, computes the average Scherrer size vs time across specified sheets.
    If sheets_to_include is None, all sheets are used.
    Returns a pandas Series indexed by Time (s).
    """
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    if sheets_to_include:
        sheets = [s for s in sheets_to_include if s in sheets]
    # collect per-sheet series
    series_list = []
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        if (
            "R_squared" not in df.columns
            or "Time (s)" not in df.columns
            or "Scherrer Size (nm)" not in df.columns
        ):
            continue
        df = df.sort_values("Time (s)")
        df = df[df["R_squared"] >= r2_threshold]
        if df.empty:
            continue
        time = df["Time (s)"].values
        size = df["Scherrer Size (nm)"].values
        # smooth if enough points
        if len(size) >= 9:
            size = savgol_filter(size, window_length=9, polyorder=3)
        series = pd.Series(data=size, index=time)
        series_list.append(series)
    if not series_list:
        return None
    # align and average
    df_all = pd.concat(series_list, axis=1)
    avg = df_all.mean(axis=1, skipna=True)
    avg = avg.sort_index()
    return avg


def main():
    # Launch file dialog to select multiple Excel files
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # # Ask user to select one or more Scherrer Excel files, using a dialog to ask if they want to select another file
    # file_paths = []
    # while True:
    #     path = filedialog.askopenfilename(
    #         title="Select Scherrer Excel file",
    #         filetypes=[("Excel files", "*.xlsx;*.xls")],
    #         initialdir=os.getcwd()
    #     )
    #     if not path:
    #         break
    #     file_paths.append(path)
    #     if not tk.messagebox.askyesno("Continue", "Do you want to select another file?"):
    #         break
    # if not file_paths:
    #     print("No files selected, exiting.")
    #     return
    # print(file_paths)
    file_paths = [
        "C:/Users/Aj/Documents/GitHub/MultiModalAnalysis/MAPI_Control_S1_18_30min_GIWAXS_raw_FittingResults.xlsx",
        "C:/Users/Aj/Documents/GitHub/MultiModalAnalysis/MAPI_1pct_AVA_S1_18_30min_GIWAXS_raw_FittingResults.xlsx",
        # "C:/Users/Aj/Documents/GitHub/MultiModalAnalysis/1pct_AVA_S1_GIWAXS_raw_FittingResults.xlsx",
        "C:/Users/Aj/Documents/GitHub/MultiModalAnalysis/MAPI_1pct_AVAI_S1_18_5min_GIWAXS_raw_FittingResults.xlsx",
        "C:/Users/Aj/Documents/GitHub/MultiModalAnalysis/AVACL_GIWAXS_raw_FittingResults.xlsx",
    ]
    # Compute averages for each file
    results = {}
    for path in file_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        sheets_to_include = ["(001)", "(011)", "(003)", "(002)", "(022)", "(111)"]
        avg_series = compute_average_scherrer(
            path, sheets_to_include=sheets_to_include, r2_threshold=0.95
        )

        if avg_series is None:
            print(f"  No valid data in {base}, skipping.")
            continue
        results[base] = avg_series
        print(f"  Processed {base}, {len(avg_series)} time points.")
    if not results:
        print("No data to plot after processing all files.")
        return

    # Plot all average curves on one figure
    colors = ["#080808", "#0262F3", "#863AB9", "#0A9628", "#56CCF2"]
    markers = ["s", "o", "^", "v"]
    labels = ("Pristine $MAPbI_3$", r"$1\%$ 5-AVA", r"$1\%$ 5-AVAI", r"$1\%$ 5-AVACl")
    plt.figure(figsize=(8, 6))
    for label, series, marker in zip(labels, results.values(), markers):
        series = series[series.index > 80]
        plt.plot(
            series.index,
            series.values,
            marker=marker,
            linestyle="-",
            label=label,
            color=colors.pop(0),
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Average Scherrer Size (nm)")

    plt.tight_layout()
    plt.tick_params(axis="both", direction="in", which="both", top=True, right=True)
    plt.legend(loc="upper right", fontsize=12)
    plt.xlim(0, 375)  # Fixed x-axis limit
    plt.ylim(0, 30)  # Adjust y-axis limit as needed
    # Save to same directory as first file
    out_dir = os.path.dirname(file_paths[0])
    out_path = os.path.join(out_dir, "Combined_Average_Scherrer.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Combined plot saved to: {out_path}")


if __name__ == "__main__":
    main()
