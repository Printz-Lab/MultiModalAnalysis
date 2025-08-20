import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
import os
from pymatgen.io.cif import CifParser
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# Pseudo-Voigt function
def pseudo_voigt(q, amplitude, center, sigma, fraction, background):
    gaussian = amplitude * (1 - fraction) * np.exp(- (q - center)**2 / (2 * sigma**2))
    lorentzian = amplitude * fraction * (sigma**2) / ((q - center)**2 + sigma**2)
    return gaussian + lorentzian + background

# FWHM calculation
def fwhm_pseudo_voigt(sigma, fraction):
    return 2 * sigma * ((1 - fraction) + fraction)

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

    wavelength = 12.398 / 10  # 10 keV
    window = 0.03  # fixed fitting window

    hkl_peaks = generate_cubic_hkl_q(a_lattice)
    all_results = {}

    for frame_idx in range(intensity.shape[0]):
        I = intensity[frame_idx, :]
        peak_indices, _ = find_peaks(I, height=np.max(I)*0.05, distance=5)
        q_peaks = q[peak_indices]
        matched_peaks = match_peaks_to_hkl(q_peaks, hkl_peaks)
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
                # Plot raw data and fit
                plt.figure(figsize=(6,4))
                plt.plot(q_fit, I_fit, 'bo', label='Data')
                plt.plot(q_fit, pseudo_voigt(q_fit, *popt), 'r-', label='Fit')
                plt.title(f'Frame {frame_idx} HKL ({h}{k}{l})')
                plt.xlabel('q (A$^{-1}$)')
                plt.ylabel('Intensity (a.u.)')
                plt.legend()
                plt.grid()

                # Save plot
                plot_filename = os.path.splitext(npz_file)[0] + f'_Frame{frame_idx}_HKL({h}{k}{l}).png'
                plt.savefig(plot_filename)
                plt.close()

                result = {
                    'Frame': frame_idx,
                    'Time (s)': time[frame_idx],
                    'Center q (A^-1)': popt[1],
                    'Sigma': popt[2],
                    'Fraction': popt[3],
                    'Background': popt[4],
                    'FWHM (A^-1)': fwhm
                }
                hkl_str = f"({h}{k}{l})"
                if hkl_str not in all_results:
                    all_results[hkl_str] = []
                all_results[hkl_str].append(result)
            except RuntimeError:
                continue

    # Write to Excel file
    out_excel = os.path.splitext(npz_file)[0] + '_FittingResults.xlsx'
    with pd.ExcelWriter(out_excel) as writer:
        for hkl, results in all_results.items():
            df = pd.DataFrame(results)
            df.to_excel(writer, sheet_name=hkl, index=False)
    print(f"Saved full fitting results to {out_excel}")
