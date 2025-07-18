import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
import fabio
import pyFAI
from pyFAI.io.ponifile import PoniFile
from pyFAI.calibrant import Calibrant
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.gui.jupyter import subplots, display, plot1d, plot2d
from PIL import Image
from PIL.TiffTags import TAGS
from datetime import datetime
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib as mpl

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
        "figure.figsize": (8, 6),
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


# ========== Helper: Extract timestamp from .tif ==========
def extract_datetime_from_tif(tif_path):
    with Image.open(tif_path) as img:
        metadata = {TAGS.get(k, k): img.tag[k] for k in img.tag.keys()}
        raw_datetime = metadata.get("DateTime")
        if raw_datetime:
            dt_str = raw_datetime[0].split('\x00')[0]
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        else:
            raise ValueError(f"No valid DateTime in: {tif_path}")

# ========== Step 1: GUI input selection ==========
def select_inputs():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder_path = filedialog.askdirectory(title="Select Folder with TIF Files", mustexist=True)
    if not folder_path:
        raise ValueError("No folder selected.")

    poni_files = glob.glob(os.path.join(folder_path, "*.poni"))
    if poni_files:
        poni_path = poni_files[0]
    else:
        poni_path = filedialog.askopenfilename(
            title="Select PONI File",
            filetypes=[("PONI files", "*.poni")],
            initialdir=folder_path
        )

    if not poni_path:
        raise ValueError("No PONI file selected.")

    sample_name = simpledialog.askstring("Sample Name", "Enter the sample name for saving the output file:",
                                         initialvalue=os.path.basename(folder_path))
    if not sample_name:
        raise ValueError("No sample name provided.")

    return folder_path, poni_path, sample_name

# ========== Step 2: GIWAXS integration ==========
def integrate_giwaxs(folder_path, poni_path, method = ("bbox", "csr", "cython")):
    tif_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
    if len(tif_files) < 2:
        raise ValueError("Need at least two .tif files to infer timing.")

    # Extract timestamps
    frame_times = [extract_datetime_from_tif(tif) for tif in tif_files]
    start_time = frame_times[0]
    frame_times = [(dt - start_time).total_seconds() for dt in frame_times]

    # Setup integrator
    poni = PoniFile(data=poni_path)
    pixel_to_um = 172.0
    full_length = 981 * pixel_to_um / 1_000_000
    print(poni.dist)
    ai = AzimuthalIntegrator(
        dist=poni.dist,
        poni1=poni.poni1,
        poni2=full_length - poni.poni2,
        wavelength=poni.wavelength,
        rot1=np.pi - poni.rot1,
        rot2=np.pi - poni.rot2,
        rot3=poni.rot3,
        detector=pyFAI.detector_factory("Pilatus1M", {"orientation": 4}),
    )

    all_intensities = []
    q_vals = None

    for i, tif_file in enumerate(tqdm(tif_files, desc="Integrating TIFs")):
        img = fabio.open(tif_file)
        img_array = img.data
        mask = img_array > 4e9
        res = ai.integrate1d_ng(img_array, 1000, mask=mask, unit="q_A^-1", method=method)
        if i == 0:
            q_vals = res.radial
        all_intensities.append(res.intensity)

    return np.array(q_vals), np.array(frame_times), np.array(all_intensities)

# ========== Step 3: Save and plot ==========
def save_and_plot(q_vals, frame_times, intensities, save_path, sample_name):
    npz_filename = os.path.join(save_path, f"{sample_name}_GIWAXS_raw.npz")
    np.savez_compressed(npz_filename, q=q_vals, time=frame_times, intensity=intensities)
    print(f"Saved integrated GIWAXS data to {npz_filename}")

    fig = plt.figure(figsize=(7, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes((left, bottom, width, height))

    # add the contour plot and a colorbar
    cp = plt.contourf(frame_times, q_vals, intensities.T, levels=100, cmap='viridis')
    plt.colorbar(cp, location='left', label='Intensity (a.u.)')
#     pcm = ax.pcolormesh(
#     frame_times, q_vals,
#     intensities.T,
#     shading="auto"
# )
#     fig.colorbar(pcm, ax=ax, label="Intensity", location='left')

    # define axis names, ticks, etc.
    q_min, q_max = (0.8, 4)
    # y_ticks = np.linspace(q_min, q_max, 10)  # number of tickmarks
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Q $(\AA^{-1})$')
    # ax.set_yticks(y_ticks)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim(q_min, q_max)
    # ax.set_title(sample_name)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, str(sample_name) + '_GIWAXS_Plot'), dpi=300, bbox_inches="tight")
    plt.show()

def plot_Chi_2theta(image_file, poni_file, ax=None, label = None, calibrant_file=None):
    """ Plot the 2D image in Chi-2theta space using pyFAI.
    Parameters:
        image (numpy.ndarray): 2D array representing the image data.
        poni_file (str): Path to the .poni file for calibration.
    """
    image = fabio.open(image_file).data
    # Load the calibration file
    ai = AzimuthalIntegrator()
    ai.load(poni_file)
    ai.set_rot3(180 * np.pi / 180)
    print(ai)
    # image = np.clip(image, 0, None)  # Clip the image to avoid negative values
    alpha_i = 2*(2*np.pi)/360 # 2 degrees in radians
    
    res2d = ai.integrate2d(image, 1024, 1024, method=("no", "csr", "cython")) #convert to q_r and q_z
    if label is None:
        label = "GIWAXS Image in Chi-2theta Space"

    if calibrant_file:
        plot2d(res2d, 
            label=label, ax=ax)
        result = res2d

        poni = PoniFile(data=poni_file)
        calibrant = Calibrant(wavelength=poni.wavelength) 
        calibrant.load_file(calibrant_file)
        img = result.intensity
        pos_rad = result.radial
        pos_azim = result.azimuthal
        from pyFAI import units
        x_values = None
        twotheta = np.array([i for i in calibrant.get_2th() if i])  # in radian
        unit = result.unit
        unit = unit[0]
        if unit == units.TTH_DEG:
            x_values = np.rad2deg(twotheta)
        elif unit == units.TTH_RAD:
            x_values = twotheta
        elif unit == units.Q_A:
            x_values = (4.e-10 * np.pi / calibrant.wavelength) * np.sin(.5 * twotheta)
        elif unit == units.Q_NM:
            x_values = (4.e-9 * np.pi / calibrant.wavelength) * np.sin(.5 * twotheta)
        if x_values is not None:
            for x in x_values:
                line = lines.Line2D([x, x], [pos_azim.min(), pos_azim.max()],
                                    color='red', linestyle='--', linewidth=1)
                ax.add_line(line)
                

    return res2d