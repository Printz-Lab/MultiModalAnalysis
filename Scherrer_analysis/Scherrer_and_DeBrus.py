import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import savgol_filter
import matplotlib as mpl
from pprint import pprint

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


def fwhm_correction_instrumental_broadening(measured_fwhm: np.ndarray) -> np.ndarray:
    """
    Corrects the measured FWHM in q-space by removing the instrumental broadening.
    """
    wavelength = 1.2398
    instrument_fwhm_rad = np.deg2rad(0.1)  # Example instrumental FWHM in radians
    instrument_dq = (2 * np.pi / wavelength) * instrument_fwhm_rad
    corrected_fwhm = np.sqrt(
        np.clip(measured_fwhm**2 - instrument_dq**2, 0.000001, None)
    )
    # plt.figure(figsize=(8, 6))
    # plt.plot(measured_fwhm, label='Measured FWHM')
    # plt.plot(corrected_fwhm, label='Corrected FWHM', linestyle='--')
    # plt.ylabel('FWHM (A^-1)')
    # plt.show()
    return corrected_fwhm


def calculate_scherrer_size(
    fwhm: np.ndarray, wavelength: float = 1.2398, k: float = 0.9
) -> np.ndarray:
    """
    Calculates the Scherrer size from the FWHM in q-space.
    """
    if np.any(fwhm <= 0):
        raise ValueError("FWHM values must be positive for Scherrer size calculation.")
    scherrer_size = k * 2 * np.pi / fwhm / 10 / 2  # (4 * np.pi / 3)**(1/3)
    return scherrer_size


def savgol_with_clipping(y, window_length=9, polyorder=3, n_sigma=3):
    """
    Apply a Savitzky–Golay filter but clip any points that deviate
    more than n_sigma * std(residual) back to the smoothed curve.

    Parameters
    ----------
    y : 1D array
    window_length : int, odd
    polyorder : int < window_length
    n_sigma : float
      How many residual‐sigmas to tolerate before clipping.

    Returns
    -------
    y_clean : 1D array
    """
    # 1) smooth
    y_smooth = savgol_filter(y, window_length, polyorder)
    # 2) residuals & sigma
    subset_y = y[100:200]
    y_smooth_subset = y_smooth[100:200]
    resid = y - y_smooth
    resid_subset = subset_y - y_smooth_subset
    sigma = np.std(resid_subset)
    # sigma = np.std(resid)
    # 3) find spikes
    spike_mask = np.abs(resid) > (n_sigma * sigma)
    # 4) clip: only those points get replaced
    y_clean = y_smooth.copy()
    y_clean[spike_mask] = np.nan
    return y_clean


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
        fwhm = df["FWHM (A^-1)"].to_numpy()
        fwhm = fwhm_correction_instrumental_broadening(fwhm)
        size = calculate_scherrer_size(fwhm)
        time = df["Time (s)"].values
        # size = df["Scherrer Size (nm)"].values
        # smooth if enough points
        if len(size) >= 9:
            size = savgol_with_clipping(size, window_length=9, polyorder=3, n_sigma=3)
        series = pd.Series(data=size, index=time).dropna()
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
    # Scherrer_file_paths = []
    # while True:
    #     path = filedialog.askopenfilename(
    #         title="Select Scherrer Excel file",
    #         filetypes=[("Excel files", "*.xlsx;*.xls")],
    #         initialdir=os.getcwd()
    #     )
    #     if not path:
    #         break
    #     Scherrer_file_paths.append(path)
    #     if not messagebox.askyesno("Continue", "Do you want to select another file?"):
    #         break
    # if not Scherrer_file_paths:
    #     print("No files selected, exiting.")
    #     return
    # pprint(Scherrer_file_paths)
    Scherrer_file_paths = [
        "K:/printz_Apr2024/MAPI_YL/working_files/MAPI_Control_S1_18_30min/GIWAXS/MAPI_Control_S1_18_30min_GIWAXS_raw_LMFIT_FittingResults.xlsx",
        "K:/printz_Apr2024/MAPI_YL/working_files/MAPI_1pct_AVA_S1_18_30min/GIWAXS/MAPI_1pct_AVA_S1_18_30min_GIWAXS_raw_LMFIT_FittingResults.xlsx",
        "K:/printz_Apr2024/MAPI_YL/working_files/MAPI_1pct_AVAI_S1_18_5min/GIWAXS/MAPI_1pct_AVAI_S1_18_5min_GIWAXS_raw_LMFIT_FittingResults.xlsx",
        "K:/printz_Apr2024/MAPI_YL/working_files/MAPI_1pct_AVACl_S1_18_5min/GIWAXS/_GIWAXS_raw_LMFIT_FittingResults.xlsx",
    ]
    # Compute averages for each file
    Scherrer_results = {}
    for path in Scherrer_file_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        sheets_to_include = ["(001)", "(011)", "(111)"]
        avg_series = compute_average_scherrer(
            path, sheets_to_include=sheets_to_include, r2_threshold=0.95
        )

        if avg_series is None:
            print(f"  No valid data in {base}, skipping.")
            continue
        Scherrer_results[base] = avg_series
        print(f"  Processed {base}, {len(avg_series)} time points.")
    if not Scherrer_results:
        print("No data to plot after processing all files.")
        return

    # DeBrus_File_paths = []
    # while True:
    #     path = filedialog.askopenfilename(
    #         title="Select DeBrus CSV file",
    #         filetypes=[("CSV files", "*.csv")]
    #     )
    #     if not path:
    #         break
    #     DeBrus_File_paths.append(path)
    #     if not messagebox.askyesno("Continue", "Do you want to select another file?"):
    #         break
    # if not DeBrus_File_paths:
    #     print("No files selected, exiting.")
    #     return
    # pprint(DeBrus_File_paths)

    DeBrus_File_paths = [
        "K:/printz_Apr2024/MAPI_YL/working_files/MAPI_Control_S1_18_30min/output/PL_FitResults_analysis.csv",
        "K:/printz_Apr2024/MAPI_YL/working_files/MAPI_1pct_AVA_S1_18_30min/output/PL_FitResults_analysis.csv",
        "K:/printz_Apr2024/MAPI_YL/working_files/MAPI_1pct_AVAI_S1_18_5min/output/PL_FitResults_analysis.csv",
        "K:/printz_Apr2024/MAPI_YL/working_files/MAPI_1pct_AVACl_S1_18_5min/output/PL_FitResults_analysis.csv",
    ]

    DeBrus_results = {}
    for path in DeBrus_File_paths:
        # Extract unique part from the parent directory name
        base = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if not os.path.exists(path):
            print(f"  File {path} does not exist, skipping.")
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"  No data in {base}, skipping.")
                continue
            # Process the DataFrame as needed
            DeBrus_results[base] = df
            print(f"  Processed {base}, {len(df)} rows.")
        except Exception as e:
            print(f"  Error processing {base}: {e}")

    # Plot all average curves on one figure
    colors_scherrer = ["#080808", "#0262F3", "#863AB9", "#0A9628", "#56CCF2"]
    colors_debrus = ["#292929", "#013F9C", "#673588", "#064714", "#2C6B80"]
    markers = ["s", "o", "^", "v"]
    labels = ("Pristine $MAPbI_3$ SCH", r"$1\%$ 5-AVA SCH", r"$1\%$ 5-AVAI SCH", r"$1\%$ 5-AVACl SCH")
    labels_debrus = (
        "Pristine $MAPbI_3$ DB", r"$1\%$ 5-AVA DB", r"$1\%$ 5-AVAI DB", r"$1\%$ 5-AVACl DB"
    )
    ymax = 25

    plt.figure(figsize=(8, 6))
    
    # Plot Scherrer results
    for label, series, marker, color in zip(labels, Scherrer_results.values(), markers, colors_scherrer):
        series = series[series.index > 80]
        plt.plot(
            series.index,
            series.values,
            marker=marker,
            linestyle="-",
            label=label,
            color=color,
            alpha=0.8
        )
    # Plot DeBrus results using the same colors
    for label, df, marker, color in zip(labels_debrus, DeBrus_results.values(), markers, colors_debrus):
        time = df["Time (s)"]
        radii = df["Radius (nm)"]
        plt.plot(
            time,
            radii,
            marker=marker,
            linestyle="--",
            label=label,
            color=color,
            alpha=0.5
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Average Radius (nm)")

    plt.tight_layout()
    plt.tick_params(axis="both", direction="in", which="both", top=True, right=True)
    plt.legend()

    plt.xlim(0, 375)  # Fixed x-axis limit
    plt.ylim(0, ymax)  # Adjust y-axis limit as needed
    # Save to same directory as first file
    out_dir = os.path.dirname(Scherrer_file_paths[0])
    out_path = os.path.join(out_dir, "Combined_Average_Scherrer.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Combined plot saved to: {out_path}")


if __name__ == "__main__":
    main()
