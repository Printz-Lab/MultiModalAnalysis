import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

# Approximate FWHM for pseudo-Voigt
def fwhm_pseudo_voigt(sigma, fraction):
    return 2 * sigma * ((1 - fraction) + fraction)

# Scherrer equation
def scherrer(fwhm, wavelength, two_theta, K=0.9):
    theta_rad = np.radians(two_theta / 2)
    beta_rad = np.radians(fwhm)
    return K * wavelength / (beta_rad * np.cos(theta_rad))

# Compute multiplicity for cubic structure
def cubic_multiplicity(h, k, l):
    indices = sorted([abs(h), abs(k), abs(l)])
    unique = len(set(indices))
    if indices.count(0) == 3:
        return 1
    elif indices.count(0) == 2:
        return 6
    elif indices.count(0) == 1:
        if indices[1] == indices[2]:
            return 12
        else:
            return 24
    elif unique == 1:
        return 8
    elif unique == 2:
        return 24
    else:
        return 48

# Compute expected q positions for cubic structure with intensities
def generate_cubic_hkl_q(a, max_hkl=4):
    hkl_list = []
    for h in range(0, max_hkl+1):
        for k in range(0, max_hkl+1):
            for l in range(0, max_hkl+1):
                if (h, k, l) == (0, 0, 0):
                    continue
                d_hkl = a / np.sqrt(h**2 + k**2 + l**2)
                q_hkl = 2 * np.pi / d_hkl
                mult = cubic_multiplicity(h, k, l)
                hkl_list.append((h, k, l, q_hkl, mult))
    hkl_list = sorted(hkl_list, key=lambda x: -x[4])  # Sort by multiplicity (intensity)
    return hkl_list[:10]  # Keep top 10 most intense reflections

# Read lattice parameter from CIF
def get_lattice_from_cif(cif_path):
    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]
    lattice = structure.lattice
    return lattice.a

# Function to process a single frame
def process_frame(frame_idx, q, I, time, hkl_peaks, wavelength, search_window, npz_path):
    frame_results = []
    for h, k, l, q_exp, mult in hkl_peaks:
        mask = (q >= q_exp - search_window) & (q <= q_exp + search_window)
        if np.sum(mask) < 5:
            continue
        q_fit = q[mask]
        I_fit = I[mask]
        amplitude_guess = np.max(I_fit)
        center_guess = q_fit[np.argmax(I_fit)]
        sigma_guess = 0.005
        fraction_guess = 0.5
        background_guess = np.min(I_fit)
        p0 = [amplitude_guess, center_guess, sigma_guess, fraction_guess, background_guess]
        try:
            popt, _ = curve_fit(pseudo_voigt, q_fit, I_fit, p0=p0, maxfev=5000)
            amplitude, center, sigma, fraction, background = popt
            fwhm = fwhm_pseudo_voigt(sigma, fraction)
            two_theta = 2 * np.degrees(np.arcsin(center * wavelength / (4 * np.pi)))
            size = scherrer(np.degrees(fwhm), wavelength, two_theta)
            frame_results.append({
                'Frame': frame_idx,
                'Time (s)': time[frame_idx],
                'HKL': f'({h}{k}{l})',
                'q_expected (A^-1)': q_exp,
                'q_fit (A^-1)': center,
                'FWHM (A^-1)': fwhm,
                'Size (nm)': size
            })
            # Save fit plot for debugging
            plt.figure()
            plt.plot(q_fit, I_fit, 'bo', label='Data')
            plt.plot(q_fit, pseudo_voigt(q_fit, *popt), 'r-', label='Fit')
            plt.title(f'Frame {frame_idx} HKL ({h}{k}{l})')
            plt.xlabel('q (A$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            plt.legend()
            plt.grid()
            fit_plot_file = os.path.splitext(npz_path)[0] + f'_Frame{frame_idx}_HKL_{h}{k}{l}.png'
            plt.savefig(fit_plot_file)
            plt.close()
        except RuntimeError:
            continue
    return frame_results

# Main pipeline
def analyze_giwaxs(npz_path, a_lattice):
    wavelength = 12.398 / 10  # for 10 keV, in Angstrom
    search_window = 0.03  # Å⁻¹
    data = np.load(npz_path)
    q = data['q']
    time = data['time']
    intensity = data['intensity']

    hkl_peaks = generate_cubic_hkl_q(a_lattice, max_hkl=4)
    results = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_frame, frame_idx, q, intensity[frame_idx, :], time, hkl_peaks, wavelength, search_window, npz_path): frame_idx for frame_idx in range(intensity.shape[0])}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            frame_results = future.result()
            results.extend(frame_results)

    df = pd.DataFrame(results)
    out_csv = os.path.splitext(npz_path)[0] + '_Scherrer.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved Scherrer analysis to {out_csv}")

    # Plot time evolution for each HKL
    for hkl in df['HKL'].unique():
        df_hkl = df[df['HKL'] == hkl]
        plt.figure()
        plt.plot(df_hkl['Time (s)'], df_hkl['Size (nm)'], 'o-')
        plt.xlabel('Time (s)')
        plt.ylabel('Crystallite Size (nm)')
        plt.title(f'Size Evolution for {hkl}')
        plt.grid()
        plot_file = os.path.splitext(npz_path)[0] + f'_Size_{hkl}.png'
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved plot for {hkl} to {plot_file}")

    # Plot raw data with predicted HKLs for middle frame
    middle_idx = intensity.shape[0] // 2
    plt.figure(figsize=(8, 5))
    plt.plot(q, intensity[middle_idx, :], label=f'Frame {middle_idx}')
    for h, k, l, q_exp, mult in hkl_peaks:
        plt.axvline(q_exp, color='red', linestyle='--', alpha=0.5)
        plt.text(q_exp, plt.ylim()[1]*0.9, f'({h}{k}{l})', rotation=90, va='top', fontsize=8)
    plt.xlabel('q (A$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Raw Data with Predicted HKL positions')
    plt.legend()
    plt.grid()
    overlay_file = os.path.splitext(npz_path)[0] + '_HKL_overlay.png'
    plt.savefig(overlay_file)
    plt.close()
    print(f"Saved HKL overlay plot to {overlay_file}")

    return df

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

    df = analyze_giwaxs(npz_file, a_lattice)
