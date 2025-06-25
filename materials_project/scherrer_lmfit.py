import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import os
from pymatgen.io.cif import CifParser
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import lmfit

# Pseudo-Voigt model
def pseudo_voigt(q, amplitude, center, sigma, fraction, background):
    gaussian = amplitude * (1 - fraction) * np.exp(- (q - center)**2 / (2 * sigma**2))
    lorentzian = amplitude * fraction * (sigma**2) / ((q - center)**2 + sigma**2)
    return gaussian + lorentzian + background

# FWHM calculation
def fwhm_pseudo_voigt(sigma, fraction):
    FWHM_G = 2 * np.sqrt(2 * np.log(2)) * sigma
    FWHM_L = 2.0 * sigma
    return (1 - fraction) * FWHM_G + fraction * FWHM_L

# Read lattice parameter from CIF
def get_lattice_from_cif(cif_path):
    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]
    return structure.lattice.a

# Generate cubic HKLs

def generate_cubic_hkl_q(a, max_hkl=4):
    hkl_list = []
    for h in range(max_hkl+1):
        for k in range(max_hkl+1):
            for l in range(max_hkl+1):
                if (h, k, l) == (0, 0, 0):
                    continue
                d_hkl = a / np.sqrt(h**2 + k**2 + l**2)
                q_hkl = 2 * np.pi / d_hkl
                hkl_list.append((h, k, l, q_hkl))
    return sorted(hkl_list, key=lambda x: x[3])

# Match detected peaks to HKLs
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

# Fit single peak using lmfit
def fit_peak(q_fit, I_fit, q_target):
    model = lmfit.Model(pseudo_voigt)
    params = model.make_params(
        amplitude=np.max(I_fit),
        center=q_target,
        sigma=0.005,
        fraction=0.5,
        background=np.min(I_fit)
    )
    params['fraction'].set(min=0, max=1)
    params['sigma'].set(min=0)
    params['center'].set(min=q_target - 0.03, max=q_target + 0.03)

    result = model.fit(I_fit, params, q=q_fit)
    return result

# Process a single frame
def process_frame(frame_idx, q, time, I, matched_peaks, npz_file, window=0.03):
    frame_results = {}
    for (h, k, l, q_target) in matched_peaks:
        mask = (q >= q_target - window) & (q <= q_target + window)
        if np.sum(mask) < 5:
            continue
        q_fit = q[mask]
        I_fit = I[mask]
        try:
            fit_result = fit_peak(q_fit, I_fit, q_target)
            popt = fit_result.best_values
            fwhm = fwhm_pseudo_voigt(popt['sigma'], popt['fraction'])
            k_shape = 0.9
            scherrer_size = k_shape * 2 * np.pi / fwhm / 10

            result = {
                'Frame': frame_idx,
                'Time (s)': time[frame_idx],
                'Amplitude': popt['amplitude'],
                'Center q (A^-1)': popt['center'],
                'Sigma': popt['sigma'],
                'Fraction': popt['fraction'],
                'Background': popt['background'],
                'FWHM (A^-1)': fwhm,
                'Scherrer Size (nm)': scherrer_size,
                'R_squared': fit_result.rsquared,
                'RedChi': fit_result.redchi,
                'AIC': fit_result.aic,
                'BIC': fit_result.bic
            }
            hkl_str = f"({h}{k}{l})"
            if hkl_str not in frame_results:
                frame_results[hkl_str] = []
            frame_results[hkl_str].append(result)
        except Exception:
            continue
    return frame_results

# Main pipeline
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    npz_file = filedialog.askopenfilename(title="Select GIWAXS .npz file", filetypes=[("NPZ files", "*.npz")])
    choice = messagebox.askyesno("Lattice Parameter", "Read lattice parameter from CIF file?")
    if choice:
        # cif_file = filedialog.askopenfilename(title="Select CIF file", filetypes=[("CIF files", "*.cif")])
        cif_file = r'materials_project\CH3NH3PbI3_cubic.cif'
        a_lattice = get_lattice_from_cif(cif_file)
        messagebox.showinfo("Lattice Parameter", f"Read a = {a_lattice:.4f} Å from CIF.")
    else:
        a_lattice = simpledialog.askfloat("Lattice parameter", "Enter cubic lattice parameter a (Angstrom):", minvalue=1.0, maxvalue=10.0)

    data = np.load(npz_file)
    q = data['q']
    time = data['time']
    intensity = data['intensity']

    hkl_peaks = generate_cubic_hkl_q(a_lattice)
    middle_idx = len(time) // 2
    I_middle = intensity[middle_idx, :]
    peak_indices, _ = find_peaks(I_middle, height=np.max(I_middle)*0.05, distance=5)
    q_peaks = q[peak_indices]
    matched_peaks = match_peaks_to_hkl(q_peaks, hkl_peaks)

    all_results = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_frame, frame_idx, q, time, intensity[frame_idx, :], matched_peaks, npz_file): frame_idx for frame_idx in range(intensity.shape[0])}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            frame_results = future.result()
            for hkl_str, results in frame_results.items():
                if hkl_str not in all_results:
                    all_results[hkl_str] = []
                all_results[hkl_str].extend(results)

    out_excel = os.path.splitext(npz_file)[0] + '_LMFIT_FittingResults.xlsx'
    with pd.ExcelWriter(out_excel) as writer:
        for hkl, results in all_results.items():
            df = pd.DataFrame(results)
            df_sorted = df.sort_values('Time (s)')
            df_sorted.to_excel(writer, sheet_name=hkl, index=False)
    print(f"Saved full lmfit fitting results to {out_excel}")
