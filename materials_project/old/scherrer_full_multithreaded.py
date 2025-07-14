import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
import os
from pymatgen.io.cif import CifParser
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Pseudo-Voigt function
def pseudo_voigt(q, amplitude, center, sigma, fraction, background):
    gaussian = amplitude * (1 - fraction) * np.exp(- (q - center)**2 / (2 * sigma**2))
    lorentzian = amplitude * fraction * (sigma**2) / ((q - center)**2 + sigma**2)
    return gaussian + lorentzian + background

# FWHM calculation (more accurate)
def fwhm_pseudo_voigt(sigma, fraction):
    FWHM_G =  2 * np.sqrt(2 * np.log(2)) * sigma
    FWHM_L = 2.0 * sigma
    FWHM_total = (1 - fraction) * FWHM_G + fraction * FWHM_L
    return FWHM_total

# Function to read lattice parameter from CIF
def get_lattice_from_cif(cif_path):
    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]
    lattice = structure.lattice
    return lattice.a

# Generate cubic HKLs
def generate_cubic_hkl_q(a, max_hkl=4):
    hkl_list = []
    for h in range(0, max_hkl+1):
        for k in range(0, max_hkl+1):
            for l in range(0, max_hkl+1):
                if (h, k, l) == (0, 0, 0):
                    continue
                d_hkl = a / np.sqrt(h**2 + k**2 + l**2)
                q_hkl = 2 * np.pi / d_hkl
                hkl_list.append((h, k, l, q_hkl))
    return sorted(hkl_list, key=lambda x: x[3])

# Match found peaks to HKLs
def match_peaks_to_hkl(q_peaks, hkl_peaks, tolerance=0.05):
    matches = []
    for qp in q_peaks:
        closest = None
        min_diff = tolerance
        for h, k, l, q_hkl in hkl_peaks:
            diff = abs(qp - q_hkl)
            if diff < min_diff:
                closest = (h, k, l, qp)
                min_diff = diff
        if closest:
            matches.append(closest)
    return matches

# Function to process a single frame

def process_frame(frame_idx, q, time, I, hkl_peaks, window, npz_file, save_plot_interval=20, save_hkl_vlines_interval=20):
    frame_results = {}
    peak_indices, _ = find_peaks(I, height=np.max(I)*0.05, distance=5)
    q_peaks = q[peak_indices]
    matched_peaks = match_peaks_to_hkl(q_peaks, hkl_peaks)
    
    # Plot full 1D profile with HKL vlines every N frames
    if frame_idx % save_hkl_vlines_interval == 0:
        vline_subfolder = os.path.splitext(npz_file)[0] + '_VlinePlots'
        os.makedirs(vline_subfolder, exist_ok=True)
        plt.figure(figsize=(8,5))
        plt.plot(q, I, label=f'Frame {frame_idx}')
        for (h, k, l, q_target) in matched_peaks:
            plt.axvline(q_target, color='r', linestyle='--', alpha=0.5)
            plt.text(q_target, np.max(I)*0.9, f'({h}{k}{l})', rotation=90, fontsize=8, ha='center')
        plt.xlabel('q (A$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(f'1D Pattern with HKLs Frame {frame_idx}')
        plt.legend()
        plt.grid()
        vline_plot_file = os.path.join(vline_subfolder, f'Frame{frame_idx}_HKL_Vlines.png')
        plt.savefig(vline_plot_file)
        plt.close()
    
    for (h, k, l, q_target) in matched_peaks:
        mask = (q >= q_target - window) & (q <= q_target + window)
        if np.sum(mask) < 5:
            continue
        q_fit = q[mask]
        I_fit = I[mask]
        amplitude_guess = np.max(I_fit)
        center_guess = q_target
        sigma_guess = 0.005
        fraction_guess = 0.5
        background_guess = np.min(I_fit)
        p0 = [amplitude_guess, center_guess, sigma_guess, fraction_guess, background_guess]
        try:
            popt, pcov = curve_fit(pseudo_voigt, q_fit, I_fit, p0=p0, maxfev=10000)
            fwhm = fwhm_pseudo_voigt(popt[2], popt[3])
            k_shape = 0.9
            scherrer_size = k_shape * 2 * np.pi / fwhm / 10  # in nm
            
            # Calculate R-squared
            residuals = I_fit - pseudo_voigt(q_fit, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((I_fit - np.mean(I_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            result = {
                'Frame': frame_idx,
                'Time (s)': time[frame_idx],
                'Center q (A^-1)': popt[1],
                'Sigma': popt[2],
                'Fraction': popt[3],
                'Background': popt[4],
                'FWHM (A^-1)': fwhm,
                'Scherrer Size (nm)': scherrer_size,
                'R_squared': r_squared,
                'PCOV': str(pcov.tolist())
            }
            hkl_str = f"({h}{k}{l})"
            if hkl_str not in frame_results:
                frame_results[hkl_str] = []
            frame_results[hkl_str].append(result)
            
            # Save fit plot every Nth frame
            if frame_idx % save_plot_interval == 0:
                fit_subfolder = os.path.splitext(npz_file)[0] + '_FitPlots'
                os.makedirs(fit_subfolder, exist_ok=True)
                plt.figure(figsize=(6,4))
                plt.plot(q_fit, I_fit, 'bo', label='Data')
                plt.plot(q_fit, pseudo_voigt(q_fit, *popt), 'r-', label='Fit')
                plt.title(f'Frame {frame_idx} HKL ({h}{k}{l})')
                plt.xlabel('q (A$^{-1}$)')
                plt.ylabel('Intensity (a.u.)')
                plt.legend()
                plt.grid()
                plot_filename = os.path.join(fit_subfolder, f'Frame{frame_idx}_HKL({h}{k}{l}).png')
                plt.savefig(plot_filename)
                plt.close()
            
        except RuntimeError:
            continue
    return frame_results

# Start full-frame processing
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    npz_file = filedialog.askopenfilename(title="Select GIWAXS .npz file", filetypes=[("NPZ files", "*.npz")])
    choice = messagebox.askyesno("Lattice Parameter", "Read lattice parameter from CIF file?")

    if choice:
        cif_file = filedialog.askopenfilename(title="Select CIF file", filetypes=[("CIF files", "*.cif")])
        a_lattice = get_lattice_from_cif(cif_file)
        messagebox.showinfo("Lattice Parameter", f"Read a = {a_lattice:.4f} Å from CIF.")
    else:
        a_lattice = simpledialog.askfloat("Lattice parameter", "Enter cubic lattice parameter a (Angstrom):", minvalue=1.0, maxvalue=10.0)

    data = np.load(npz_file)
    q = data['q']
    time = data['time']
    intensity = data['intensity']

    window = 0.03  # fixed fitting window
    hkl_peaks = generate_cubic_hkl_q(a_lattice)
    all_results = {}

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_frame, frame_idx, q, time, intensity[frame_idx, :], hkl_peaks, window, npz_file, 20, 20): frame_idx for frame_idx in range(intensity.shape[0])}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            frame_results = future.result()
            for hkl_str, results in frame_results.items():
                if hkl_str not in all_results:
                    all_results[hkl_str] = []
                all_results[hkl_str].extend(results)

    # Write to Excel file
    out_excel = os.path.splitext(npz_file)[0] + '_FittingResults.xlsx'
    with pd.ExcelWriter(out_excel) as writer:
        for hkl, results in all_results.items():
            df = pd.DataFrame(results)
            df_sorted = df.sort_values('Time (s)')
            df_sorted.to_excel(writer, sheet_name=hkl, index=False)
    print(f"Saved full fitting results to {out_excel}")