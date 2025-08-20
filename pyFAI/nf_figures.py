import numpy as np
from matplotlib import lines
from matplotlib.pyplot import subplots
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyFAI.calibrant import Calibrant
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import fabio
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks
from pyFAI import units
import tkinter as tk
from tkinter import filedialog
from pyFAI.io.ponifile import PoniFile
from pyFAI.gui.jupyter import subplots, display, plot1d, plot2d

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
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 14,
        "figure.figsize": (8, 6),
        "axes.linewidth": 3,
        "xtick.color": "black",  # tick marks white
        "ytick.color": "black",
        "xtick.labelcolor": "black",  # tick labels black
        "ytick.labelcolor": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 9,
        "ytick.major.size": 9,
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

# ask user to select poni file
poni_files = [
    r"E:\MAPI_sean\MAPI_sean_control_S1_30_tube_5min\MAPI_sean_control_S1_30_tube_5min_00222_refined.poni",
    r"E:\nanofibers\MAPbI3_nanofibers_S4\MAPbI3_nanofibers_S4_00209_refined.poni"
]
image_paths = [
    r"E:\MAPI_sean\MAPI_sean_control_S1_30_tube_5min\MAPI_sean_control_S1_30_tube_5min_00375.tif",
    r"E:\nanofibers\MAPbI3_nanofibers_S4\MAPbI3_nanofibers_S4_00375.tif",
]
fig, axs = subplots(2, 1, figsize=(8, 8), sharex=True)
axs[1].set_xlabel("q (A$^{-1}$)")
labels = ["$MAPbI_3$ Control", "$MAPbI_3$ Nanofibers"]
for i, image_path in enumerate(image_paths):
    poni_file = poni_files[i]
    poni = PoniFile(data=poni_file)
    calibrant_file = r"pyFAI\MAPbI3_Calibrant.D"
    calibrant = Calibrant(wavelength=poni.wavelength)
    calibrant.load_file(calibrant_file)
    ax = axs[i]

    ai = AzimuthalIntegrator()
    ai.load(poni_file)
    ai.set_rot3(180 * np.pi / 180)

    method = ("full", "csr", "opencl")
    image = fabio.open(image_path).data
    res1d = ai.integrate1d_ng(
        data=image, npt=1000, method=method, unit="q_A^-1", radial_range=(0.5, 4)
    )
    calibrant_file = r"C:\Users\raglo\OneDrive - University of Arizona\Documents\GitHub\MultiModalAnalysis\pyFAI\MAPbI3_Calibrant.D"
    calibrant = Calibrant(wavelength=ai.wavelength)
    calibrant.load_file(calibrant_file)
    colors = ["#1E1919E0", "#cf0000"]
    q_vals, intensity_vals = res1d
    # intensity_vals = intensity_vals / np.max(intensity_vals)  # Normalize intensity
    offset = [0, 0]
    ax.plot(q_vals, intensity_vals + offset[i], color=colors[i], label=labels[i])
    
    ax.legend(loc="upper right", fontsize=14)
    ax.set_ylim(0, np.max(intensity_vals) * 1.1)

    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"")
    # Estimate background using a simple rolling minimum filter

    background = minimum_filter1d(intensity_vals, size=50)
    intensity_bgsub = intensity_vals - background

    peaks, _ = find_peaks(
        intensity_bgsub, height=np.max(intensity_bgsub) * 0.03, distance=10
    )

    peak_labelss = [
        [
            # "",
            # "2D",
            # "$PbI_2$",
            "100",
            "110",
            "111",
            "200",
            "ITO",
            "210",
            "211",
            "220",
            "221",
            "222",
            "",
            "",
            "",
            "",
        ],
        [
            "",
            "100",
            "110",
            "111",
            "200",
            "ITO",
            "210",
            "211",
            "220",
            "221",
            "222",
            "",
            "",
            "",
        ],
    ]

    for ii, peak in enumerate(peaks):
        peak_labels = peak_labelss[i]
        ax.annotate(
            peak_labels[ii],
            (q_vals[peak], intensity_vals[peak] + offset[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color="#890303",
        )

    # ax.scatter(q_vals[peaks[1:]], intensity_vals[peaks[1:]], color="#890303", zorder=5)
plt.tight_layout()

plt.show(   )
