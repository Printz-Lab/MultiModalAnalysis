import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import os
from pymatgen.io.cif import CifParser
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tqdm import tqdm
import lmfit

# Gaussian model
def gaussian(q, amplitude, center, sigma, background):
    return amplitude * np.exp(- (q - center)**2 / (2 * sigma**2)) + background

# FWHM calculation for Gaussian
def fwhm_gaussian(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma

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

# Fit single peak using lmfit Gaussian model
def fit_peak(q_fit, I_fit, q_target, prev_params=None):
    model = lmfit.Model(gaussian)
    default_params = model.make_params(
        amplitude=np.max(I_fit),
        center=q_target,
        sigma=0.01,
        background=np.min(I_fit)
    )
    default_params['amplitude'].set(min=0)
    default_params['center'].set(min=q_target - 0.02, max=q_target + 0.02)
    default_params['sigma'].set(min=0.00005, max=0.05)

    if prev_params is not None:
        try:
            default_params['amplitude'].set(value=prev_params['amplitude'])
            default_params['center'].set(value=prev_params['center'])
            default_params['sigma'].set(value=prev_params['sigma'])
            default_params['background'].set(value=prev_params['background'])
        except Exception:
            pass
    result = model.fit(I_fit, default_params, q=q_fit)
    return result

# Compute R-squared
def calculate_r_squared(I_fit, fit_result):
    residuals = fit_result.residual
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((I_fit - np.mean(I_fit))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Main pipeline
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

    hkl_peaks = generate_cubic_hkl_q(a_lattice)
    middle_idx = len(time) // 2
    I_middle = intensity[middle_idx, :]
    peak_indices, _ = find_peaks(I_middle, height=np.max(I_middle)*0.05, distance=5)
    q_peaks = q[peak_indices]
    matched_peaks = match_peaks_to_hkl(q_peaks, hkl_peaks)

    all_results = {}
    prev_fit_params = {}
    prev_r_squared = {}

    base_output = os.path.splitext(npz_file)[0]
    vline_dir = base_output + '_VlinePlots'
    fit_dir = base_output + '_FitPlots'
    os.makedirs(vline_dir, exist_ok=True)
    os.makedirs(fit_dir, exist_ok=True)

    for frame_idx in tqdm(range(intensity.shape[0]), desc="Processing frames"):
        I = intensity[frame_idx, :]

        if frame_idx % 20 == 0:
            plt.figure(figsize=(8, 5))
            plt.plot(q, I, label=f'Frame {frame_idx}')
            for (h, k, l, q_target) in matched_peaks:
                plt.axvline(q_target, color='r', linestyle='--', alpha=0.5)
                plt.text(q_target, np.max(I)*0.9, f'({h}{k}{l})', rotation=90, fontsize=8, ha='center')
            plt.xlabel('q (A$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            plt.title(f'GIWAXS 1D pattern Frame {frame_idx}')
            plt.grid()
            plt.savefig(os.path.join(vline_dir, f'Frame_{frame_idx}_vlines.png'))
            plt.close()

        frame_results = {}
        for (h, k, l, q_target) in matched_peaks:
            mask = (q >= q_target - 0.03) & (q <= q_target + 0.03)
            if np.sum(mask) < 5:
                continue
            q_fit = q[mask]
            I_fit = I[mask]
            hkl_str = f"({h}{k}{l})"

            prev_params = None
            if prev_r_squared.get(hkl_str, 1.0) >= 0.8:
                prev_params = prev_fit_params.get(hkl_str, None)

            try:
                fit_result = fit_peak(q_fit, I_fit, q_target, prev_params)
                popt = fit_result.best_values
                fwhm = fwhm_gaussian(popt['sigma'])
                scherrer_size = 0.9 * 2 * np.pi / fwhm / 10
                r_squared = calculate_r_squared(I_fit, fit_result)

                result = {
                    'Frame': frame_idx,
                    'Time (s)': time[frame_idx],
                    'Amplitude': popt['amplitude'],
                    'Center q (A^-1)': popt['center'],
                    'Sigma': popt['sigma'],
                    'Background': popt['background'],
                    'FWHM (A^-1)': fwhm,
                    'Scherrer Size (nm)': scherrer_size,
                    'R_squared': r_squared,
                    'RedChi': fit_result.redchi,
                    'AIC': fit_result.aic,
                    'BIC': fit_result.bic
                }

                if hkl_str not in frame_results:
                    frame_results[hkl_str] = []
                frame_results[hkl_str].append(result)

                prev_fit_params[hkl_str] = popt
                prev_r_squared[hkl_str] = r_squared

                if frame_idx % 20 == 0:
                    plt.figure(figsize=(6,4))
                    plt.plot(q_fit, I_fit, 'bo', label='Data')
                    plt.plot(q_fit, fit_result.best_fit, 'r-', label='Fit')
                    plt.title(f'Frame {frame_idx} HKL ({h}{k}{l})')
                    plt.xlabel('q (A$^{-1}$)')
                    plt.ylabel('Intensity')
                    plt.legend()
                    plt.grid()
                    plt.savefig(os.path.join(fit_dir, f'Frame_{frame_idx}_HKL_{h}{k}{l}.png'))
                    plt.close()
            except Exception:
                continue

        for hkl_str, results in frame_results.items():
            if hkl_str not in all_results:
                all_results[hkl_str] = []
            all_results[hkl_str].extend(results)

    out_excel = base_output + '_LMFIT_Gaussian_FittingResults.xlsx'
    with pd.ExcelWriter(out_excel) as writer:
        for hkl, results in all_results.items():
            df = pd.DataFrame(results)
            df_sorted = df.sort_values('Time (s)')
            df_sorted.to_excel(writer, sheet_name=hkl, index=False)

    print(f"Saved full LMFIT Gaussian fitting results to {out_excel}")
