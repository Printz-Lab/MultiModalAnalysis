import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import fabio
import pyFAI
from pyFAI.io.ponifile import PoniFile
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.gui.jupyter import subplots, display, plot1d, plot2d
from PIL import Image
from PIL.TiffTags import TAGS
from datetime import datetime
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, simpledialog

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
        res = ai.integrate1d_ng(img_array, 1000, mask=mask, unit="q_A^-1")
        if i == 0:
            q_vals = res.radial
        all_intensities.append(res.intensity)

    return np.array(q_vals), np.array(frame_times), np.array(all_intensities)

# ========== Step 3: Save and plot ==========
def save_and_plot(q_vals, frame_times, intensities, save_path, sample_name):
    npz_filename = os.path.join(save_path, f"{sample_name}_GIWAXS_raw.npz")
    np.savez_compressed(npz_filename, q=q_vals, time=frame_times, intensity=intensities)
    print(f"Saved integrated GIWAXS data to {npz_filename}")

    plt.figure(figsize=(10, 6))
    extent = [frame_times[0], frame_times[-1], q_vals[0], q_vals[-1]]
    plt.imshow(
        intensities.T,
        aspect='auto',
        extent=extent,
        origin='lower',
        cmap='viridis'
    )
    plt.colorbar(label='Intensity (a.u.)')
    plt.ylabel('q (Å$^{-1}$)')
    plt.xlabel('Time (s)')
    plt.title(f'{sample_name} GIWAXS Heatmap')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, f"{sample_name}_GIWAXS_heatmap.png"))

def plot_Chi_2theta(image_file, poni_file):
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

    plot2d(res2d, 
           label="GIWAXS Image in Chi-2theta Space")