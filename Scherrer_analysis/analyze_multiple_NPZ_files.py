import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the analysis functions from your existing script
# Ensure scherrer_lmfit.py is in the same folder or on PYTHONPATH
from Scherrer_LmFit import (
    _suppress_uncertainty_warnings,
    get_lattice_from_cif,
    generate_cubic_hkl_q,
    match_peaks_to_hkl,
    fit_peak,
    process_frame
)

# Initialize warnings filter
_suppress_uncertainty_warnings()


def analyze_npz(npz_file: str, a_lattice: float, output_root: str, plot_every_n: int = 20):
    """
    Runs the full GIWAXS Scherrer/LmFit pipeline on a single .npz file.
    Saves fit plots and writes results to an Excel file in output_root.
    Returns the path to the saved Excel file.
    """
    data = np.load(npz_file)
    q = data['q']
    time = data['time']
    intensity = data['intensity']

    # Build theoretical HKL peaks
    hkl_peaks = generate_cubic_hkl_q(a_lattice)
    # Use middle frame to detect actual peaks
    mid_idx = len(time) // 2
    from scipy.signal import find_peaks
    I_mid = intensity[mid_idx]
    idxs, _ = find_peaks(I_mid, height=np.max(I_mid)*0.1, distance=5)
    q_peaks = q[idxs]
    matched_peaks = match_peaks_to_hkl(q_peaks, hkl_peaks)

    # Prepare output directories
    base = os.path.splitext(os.path.basename(npz_file))[0]
    out_dir = os.path.join(output_root, base + '_fit_plots')
    os.makedirs(out_dir, exist_ok=True)

    # Run fits in parallel
    all_results = {}
    prev_params = {}
    with ProcessPoolExecutor() as executor:
        futures = []
        for frame_idx in range(intensity.shape[0]):
            I_frame = intensity[frame_idx]
            futures.append(
                executor.submit(
                    process_frame,
                    frame_idx,
                    q,
                    time,
                    I_frame,
                    matched_peaks,
                    prev_params,
                    out_dir,
                    plot_every_n
                )
            )
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {base}"):
            frame_idx, frame_dict = future.result()
            for hkl, res in frame_dict.items():
                all_results.setdefault(hkl, []).append(res)
                # update prev_params for next fits
                if res['R_squared'] > 0.9:
                    prev_params[hkl] = res
                else:
                    prev_params[hkl] = None

    # Write combined results to Excel
    excel_path = os.path.join(output_root, base + '_FittingResults.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        for hkl, recs in all_results.items():
            df = pd.DataFrame(recs).sort_values('Time (s)')
            df.to_excel(writer, sheet_name=hkl, index=False)

    return excel_path


def main():
    # GUI for selecting multiple NPZ files
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Ensure dialog is on top

    npz_files = []
    # Ask user to select one or more NPZ files
    npz_file = filedialog.askopenfilename(
        title="Select one or more GIWAXS .npz files",
        filetypes=[("NPZ files", "*.npz")]
    )
    #ask if they want to select more files
    while npz_file:
        npz_files.append(npz_file)
        if not messagebox.askyesno("Select more files", "Do you want to select another NPZ file?"):
            break
        npz_file = filedialog.askopenfilename(
            title="Select another GIWAXS .npz file",
            filetypes=[("NPZ files", "*.npz")]
        )
    
    if not npz_files:
        print("No files selected. Exiting.")
        return

    # Ask for lattice parameter
    use_cif = messagebox.askyesno("Lattice Parameter", "Read lattice parameter from CIF file?")
    if use_cif:
        cif_file = filedialog.askopenfilename(
            title="Select CIF file", filetypes=[("CIF files", "*.cif")]
        )
        if not cif_file:
            print("No CIF selected, exiting.")
            return
        a_lattice = get_lattice_from_cif(cif_file)
        messagebox.showinfo("Lattice Parameter", f"Using a = {a_lattice:.4f} Å from CIF.")
    else:
        a_lattice = simpledialog.askfloat(
            "Lattice Parameter",
            "Enter cubic lattice parameter a (Å):",
            minvalue=1.0,
            maxvalue=20.0
        )
        if a_lattice is None:
            print("No lattice parameter given, exiting.")
            return

    # Choose output directory
    output_root = filedialog.askdirectory(
        title="Select folder to save output"
    ) or os.getcwd()

    # Process each NPZ
    results = []
    for npz in npz_files:
        print(f"Analyzing {os.path.basename(npz)}...")
        excel_file = analyze_npz(npz, a_lattice, output_root)
        print(f"Results saved to {excel_file}")
        results.append(excel_file)

    print("All files processed.")


if __name__ == "__main__":
    main()
