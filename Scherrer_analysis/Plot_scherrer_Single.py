import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter

import matplotlib as mpl
import seaborn as sns

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

def plot_data(
    df: pd.DataFrame,
    x: str,
    y_scat: str,
    y_line: str,
    hue: str,
    markers: list[str],
    palette: list[str],
    xlim: tuple[float, float],
    xticks: np.ndarray[float],
    xmticks: np.ndarray[float],
    xlabel: str,
    ylim: tuple[float, float],
    yticks: np.ndarray[float],
    ymticks: np.ndarray[float],
    ylabel: str,
    ax: plt.Axes,
) -> None:
    """Plot the data on a given axis

    Args:
        df (pd.DataFrame): The DataFrame to plot from
        x (str): column to plot on x-axis
        y (str): column to plot on y-axis
        hue (str): column specifiying hue
        markers (list[str]): Markers for plot
        palette (list[str]): color palette for the plot
        xlim (tuple[float, float]): x-axis limits
        xticks (np.ndarray[float]): x-axis ticks
        xmticks (np.ndarray[float]): x-axis minor ticks
        xlabel (str): x-axis label
        ylim (tuple[float, float]): y-axis limits
        yticks (np.ndarray[float]): y-axis ticks
        ymticks (np.ndarray[float]): y-axis minor ticks
        ylabel (str): y-axis label
        ax (plt.Axes): matplotlib axes to plot on
    """
    sns.scatterplot(
        df,
        x=x,
        y=y_scat,
        hue=hue,
        style=hue,
        palette=palette,
        markers=markers,
        ax=ax,
        legend=False,
    )

    sns.lineplot(
        df,
        x=x,
        y=y_line,
        hue=hue,
        style=hue,
        palette=palette,
        ax=ax,
        legend=True,
        sort=False,
    )

    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticks(xmticks, minor=True)
    # Lim last so ticks won't override
    ax.set_xlim(xlim)
    if xlabel == "":
        ax.set_xticklabels([""] * len(xticks))

    ax.set_ylabel(ylabel)
    ax.set_yticks(yticks)
    ax.set_yticks(ymticks, minor=True)
    ax.set_ylim(ylim)
    if ylabel == "":
        ax.set_yticklabels([""] * len(yticks))

            # clear existing ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # OR set auto ticks
    ax.tick_params(axis='both', which='both')
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    plt.tight_layout()



def fwhm_correction_instrumental_broadening(
    measured_fwhm: np.ndarray, sheet: str, q: float
) -> np.ndarray:
    """
    Corrects the measured FWHM in q-space by removing the instrumental broadening.
    """
    wavelength = 1.2398
    instrument_fwhm_theta = .1  # Example instrumental FWHM in radians
    theta = q / (4 * np.pi / wavelength)
    print(theta)
    measured_fwhm_theta = measured_fwhm * (4 * np.pi / wavelength)/np.cos(theta) 
    print(measured_fwhm_theta[100])
    print(instrument_fwhm_theta)

    corrected_fwhm_theta = np.sqrt(np.clip(measured_fwhm_theta**2 - instrument_fwhm_theta**2, 0.00000001, None))
    corrected_fwhm = corrected_fwhm_theta * np.cos(theta) * wavelength / (4 * np.pi)
    plt.figure(figsize=(8, 6))
    plt.plot(measured_fwhm, label='Measured FWHM')
    plt.plot(corrected_fwhm, label='Corrected FWHM', linestyle='--')
    plt.legend()
    plt.title(f"FWHM Correction for {sheet}")
    plt.ylabel('FWHM (A^-1)')
    plt.show()
    return corrected_fwhm

import numpy as np

# Example usage:
# measured = [0.15, 0.12, 0.18]  # degrees
# inst = 0.10                   # degrees
# corr = correct_fwhm(measured, inst)
# print("Corrected FWHM (radians):", corr)


def calculate_scherrer_size(
    fwhm: np.ndarray, wavelength: float = 1.2398, k: float = 0.85
) -> np.ndarray:
    """
    Calculates the Scherrer size from the FWHM in q-space.
    """
    if np.any(fwhm <= 0):
        raise ValueError("FWHM values must be positive for Scherrer size calculation.")
    scherrer_size = k * 2 * np.pi / fwhm / 10 / (4 * np.pi / 3)**(1/3)
    return scherrer_size

# Define a default set of markers and colors
tpl_markers = ["o", "s", "^", "d", "x"]
tpl_palette = ["#080808", "#1c73dd", "#3aa56d", "#872ec4", "#56CCF2"]

def plot_scherrer_size(file_path: str, hkls_to_plot: list[str], output_dir: str = None):
    """
    Reads an LMFIT Excel file, computes Scherrer sizes, and creates two styled plots:
      1) All original HKL curves
      2) The average curve (tolerant to missing data)
    """
    if output_dir is None:
        output_dir = os.path.splitext(file_path)[0] + "_plots"
    os.makedirs(output_dir, exist_ok=True)

    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    processed = []  # store filtered & smoothed series for each HKL

    for sheet in hkls_to_plot:
        if sheet not in sheets:
            print(f"Sheet {sheet} not found in file.")
            continue

        df = pd.read_excel(xls, sheet_name=sheet)
        # drop any rows with non‑numeric times
        df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors="coerce")
        df = df.dropna(subset=["Time (s)"])

        df_filt = df[df["R_squared"] >= 0.9].copy()
        # --- NEW ---
        # sort by time before you extract arrays
        df_filt.sort_values("Time (s)", inplace=True)
        df_filt = df[df["R_squared"] >= 0.9]
        if df_filt.empty:
            continue
        # fwhm = df_filt["FWHM (A^-1)"].to_numpy()
        # q = df_filt["Center q (A^-1)"].to_numpy()
        # q = np.mean(q)  # average q for the sheet
        # fwhm = fwhm_correction_instrumental_broadening(fwhm, sheet, q)
        # size = calculate_scherrer_size(fwhm)
        time = df_filt["Time (s)"].to_numpy()
        size = df_filt["Scherrer Size (nm)"].to_numpy()
        if len(size) >= 9:
            size = savgol_filter(size, window_length=9, polyorder=3)
        # create a tiny DataFrame and sort its index
        df_proc = (
            pd.DataFrame({"Time (s)": time, sheet: size})
            .set_index("Time (s)")
            .sort_index()
        )
        processed.append(df_proc)

    if not processed:
        print("No valid data to plot.")
        return

    # Combine into DataFrame for both original and average
    all_df = pd.concat(processed, axis=1).sort_index()

    
    # 1) Styled plot of original curves
    df_orig = all_df.reset_index().melt(
        id_vars=["Time (s)"], var_name="HKL", value_name="Scherrer Size"
    )
    df_orig = df_orig[df_orig["Time (s)"] >= 80]
    fig_o, ax_o = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax_o.tick_params(direction="in", which="both", right=True, top=True)
    n = df_orig["HKL"].nunique()
    markers = [tpl_markers[i % len(tpl_markers)] for i in range(n)]
    colors = [tpl_palette[i % len(tpl_palette)] for i in range(n)]

    # Determine x/y limits and ticks
    x_min = 0
    x_max = 375  # Fixed limit for original plot
    mask = (df_orig["Time (s)"] >= x_min+100) & (df_orig["Time (s)"] <= x_max)
    y_max = np.nanmax(df_orig.loc[mask, "Scherrer Size"]) * 1.1
    y_min = 0
    x_ticks = np.linspace(x_min, x_max, 5)
    x_mticks = np.linspace(x_min, x_max, 25)
    y_ticks = np.linspace(y_min, y_max, 6)
    y_mticks = np.linspace(y_max, y_max, 30)

    plot_data(
        df_orig,
        x="Time (s)",
        y_scat="Scherrer Size",
        y_line="Scherrer Size",
        hue="HKL",
        markers=markers,
        palette=colors,
        xlim=(x_min, x_max),
        xticks=x_ticks,
        xmticks=x_mticks,
        ylim=(y_min, y_max),
        yticks=y_ticks,
        ymticks=y_mticks,
        xlabel="Time (s)",
        ylabel="Radius (nm)",
        ax=ax_o,
    )
    orig_path = os.path.join(output_dir, "Original_Scherrer_Styled_vs_Time.png")
    fig_o.savefig(orig_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 2) Compute and plot the average curve
    avg_series = all_df.mean(axis=1)
    df_avg = pd.DataFrame({
        "Time (s)": avg_series.index.values,
        "Scherrer Size": avg_series.values,
        "HKL": ["Average"] * len(avg_series),
    })
    df_avg = df_avg[df_avg["Time (s)"] >= 80]
    fig_a, ax_a = plt.subplots(figsize=(8, 6))
    ax_a.tick_params(direction="in", which="both", right=True, top=True)


    x_min_a = 0
    x_max_a = 375  # Fixed limit for average plot
    # Find min/max within bounds of x_min_a and x_max_a
    mask = (df_avg["Time (s)"] >= x_min_a+100) & (df_avg["Time (s)"] <= x_max_a)
    y_max_a = np.nanmax(df_avg.loc[mask, "Scherrer Size"]) * 1.1
    y_min_a = 0
    xticks_a = np.linspace(x_min_a, x_max_a, 5)
    xmticks_a = np.linspace(x_min_a, x_max_a, 25)
    yticks_a = np.linspace(y_min_a, y_max_a, 6)
    ymticks_a = np.linspace(y_max_a, y_max_a, 30)

    plot_data(
        df_avg,
        x="Time (s)",
        y_scat="Scherrer Size",
        y_line="Scherrer Size",
        hue="HKL",
        markers=[tpl_markers[0]],
        palette=[tpl_palette[0]],
        xlim=(x_min_a, x_max_a),
        xticks=xticks_a,
        xmticks=xmticks_a,
        ylim=(y_min_a, y_max_a),
        yticks=yticks_a,
        ymticks=ymticks_a,
        xlabel="Time (s)",
        ylabel="Average Radius (nm)",
        ax=ax_a,
    )
    
    avg_path = os.path.join(output_dir, "Average_Scherrer_Styled_vs_Time.png")
    fig_a.savefig(avg_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Original and average plots generated successfully!")

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw() ; root.wm_attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select LMFIT Excel file", filetypes=[("Excel files", "*.xlsx")]
    )
    hkls_to_plot = ["(001)", "(011)", "(111)"] #can add more HKLs here
    plot_scherrer_size(file_path, hkls_to_plot)
