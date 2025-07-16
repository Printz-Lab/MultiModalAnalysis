import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter
# Import plot_data from template_plottin
from template_plotting import plot_data


# Define a default set of markers and colors
tpl_markers = ["o", "s", "^", "d", "x"]
tpl_palette = ["#080808", "#0262F3", "#0A9628", "#863AB9", "#56CCF2"]

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

        df = pd.read_excel(xls, sheet_name=sheet).sort_values("Time (s)")
        df_filt = df[df["R_squared"] >= 0.93]
        if df_filt.empty:
            continue

        time = df_filt["Time (s)"].to_numpy()
        size = df_filt["Scherrer Size (nm)"].to_numpy()
        if len(size) >= 9:
            size = savgol_filter(size, window_length=9, polyorder=3)

        processed.append(pd.DataFrame({"Time (s)": time, sheet: size}).set_index("Time (s)"))

    if not processed:
        print("No valid data to plot.")
        return

    # Combine into DataFrame for both original and average
    all_df = pd.concat(processed, axis=1)

    # 1) Styled plot of original curves
    df_orig = all_df.reset_index().melt(
        id_vars=["Time (s)"], var_name="curve", value_name="Scherrer Size"
    )
    fig_o, ax_o = plt.subplots(figsize=(8, 6))
    ax_o.tick_params(direction="in", which="both", right=True, top=True)
    n = df_orig["curve"].nunique()
    markers = [tpl_markers[i % len(tpl_markers)] for i in range(n)]
    colors = [tpl_palette[i % len(tpl_palette)] for i in range(n)]

    # Determine x/y limits and ticks
    x_min = 0
    x_max = 600  # Fixed limit for original plot
    mask = (df_orig["Time (s)"] >= x_min) & (df_orig["Time (s)"] <= x_max)
    y_max = np.nanmax(df_orig.loc[mask, "Scherrer Size"]) * 1.1
    y_min = np.nanmin(df_orig.loc[mask, "Scherrer Size"]) * 0.9
    x_ticks = np.linspace(x_min, x_max, 5)
    x_mticks = np.linspace(x_min, x_max, 25)
    y_ticks = np.linspace(y_min, y_max, 6)
    y_mticks = np.linspace(y_max, y_max, 30)

    plot_data(
        df_orig,
        x="Time (s)",
        y_scat="Scherrer Size",
        y_line="Scherrer Size",
        hue="curve",
        markers=markers,
        palette=colors,
        xlim=(x_min, x_max),
        xticks=x_ticks,
        xmticks=x_mticks,
        ylim=(y_min, y_max),
        yticks=y_ticks,
        ymticks=y_mticks,
        xlabel="Time (s)",
        ylabel="Scherrer Size (nm)",
        ax=ax_o,
    )
    plt.tight_layout()
    orig_path = os.path.join(output_dir, "Original_Scherrer_Styled_vs_Time.png")
    fig_o.savefig(orig_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 2) Compute and plot the average curve
    avg_series = all_df.mean(axis=1)
    df_avg = pd.DataFrame({
        "Time (s)": avg_series.index.values,
        "Scherrer Size": avg_series.values,
        "curve": ["Average"] * len(avg_series),
    })
    fig_a, ax_a = plt.subplots(figsize=(8, 6))
    ax_a.tick_params(direction="in", which="both", right=True, top=True)


    x_min_a = 0
    x_max_a = 600  # Fixed limit for average plot
    # Find min/max within bounds of x_min_a and x_max_a
    mask = (df_avg["Time (s)"] >= x_min_a) & (df_avg["Time (s)"] <= x_max_a)
    y_max_a = np.nanmax(df_avg.loc[mask, "Scherrer Size"]) * 1.1
    y_min_a = np.nanmin(df_avg.loc[mask, "Scherrer Size"]) * 0.9
    xticks_a = np.linspace(x_min_a, x_max_a, 5)
    xmticks_a = np.linspace(x_min_a, x_max_a, 25)
    yticks_a = np.linspace(y_min_a, y_max_a, 6)
    ymticks_a = np.linspace(y_max_a, y_max_a, 30)

    plot_data(
        df_avg,
        x="Time (s)",
        y_scat="Scherrer Size",
        y_line="Scherrer Size",
        hue="curve",
        markers=[tpl_markers[0]],
        palette=[tpl_palette[0]],
        xlim=(x_min_a, x_max_a),
        xticks=xticks_a,
        xmticks=xmticks_a,
        ylim=(y_min_a, y_max_a),
        yticks=yticks_a,
        ymticks=ymticks_a,
        xlabel="Time (s)",
        ylabel="Average Scherrer Size (nm)",
        ax=ax_a,
    )
    plt.tight_layout()
    avg_path = os.path.join(output_dir, "Average_Scherrer_Styled_vs_Time.png")
    fig_a.savefig(avg_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Original and average plots generated successfully!")

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw() ; root.wm_attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select LMFIT Excel file", filetypes=[("Excel files", "*.xlsx")]
    )
    hkls_to_plot = ["(001)", "(003)", "(002)", "(022)"]
    plot_scherrer_size(file_path, hkls_to_plot)
