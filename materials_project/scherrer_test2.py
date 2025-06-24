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

# Start interactive fitting
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

    frame_idx = simpledialog.askinteger("Frame selection", f"Total frames: {intensity.shape[0]}", initialvalue=intensity.shape[0]//2)
    I = intensity[frame_idx, :]

    hkl_peaks = generate_cubic_hkl_q(a_lattice)

    # Find peaks in the raw data
    peak_indices, _ = find_peaks(I, height=np.max(I)*0.05, distance=5)
    q_peaks = q[peak_indices]

    matched_peaks = match_peaks_to_hkl(q_peaks, hkl_peaks)

    print("Matched HKL peaks:")
    for i, (h, k, l, q_found) in enumerate(matched_peaks):
        print(f"{i}: ({h}{k}{l}) matched at q={q_found:.3f} A^-1")

    hkl_idx = simpledialog.askinteger("HKL selection", "Enter index of HKL to fit:", initialvalue=0)
    h, k, l, q_target = matched_peaks[hkl_idx]

    window = simpledialog.askfloat("Fitting window", "q-window (+/- A^-1):", initialvalue=0.03)
    mask = (q >= q_target - window) & (q <= q_target + window)
    q_fit = q[mask]
    I_fit = I[mask]

    plt.figure(figsize=(8, 5))
    plt.plot(q, I, label='Raw data')
    plt.axvline(q_target, color='r', linestyle='--', label=f'Matched ({h}{k}{l})')
    plt.xlabel('q (A$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.title(f'Frame {frame_idx}')
    plt.show()

    amplitude_guess = np.max(I_fit)
    center_guess = q_target
    sigma_guess = 0.005
    fraction_guess = 0.5
    background_guess = np.min(I_fit)
    p0 = [amplitude_guess, center_guess, sigma_guess, fraction_guess, background_guess]

    try:
        popt, pcov = curve_fit(pseudo_voigt, q_fit, I_fit, p0=p0, maxfev=10000)
        fit_I = pseudo_voigt(q_fit, *popt)

        plt.figure(figsize=(8, 5))
        plt.plot(q_fit, I_fit, 'bo', label='Data')
        plt.plot(q_fit, fit_I, 'r-', label='Fit')
        plt.xlabel('q (A$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.title(f'Fit for Frame {frame_idx} ({h}{k}{l})')
        plt.grid()
        plt.show()

        print("Fitted parameters:")
        print(f"Center: {popt[1]:.4f} A^-1")
        print(f"Sigma: {popt[2]:.5f} A^-1")
        print(f"Fraction (Lorentzian): {popt[3]:.3f}")
        print(f"Background: {popt[4]:.2f}")

    except RuntimeError:
        print("Fit failed!")