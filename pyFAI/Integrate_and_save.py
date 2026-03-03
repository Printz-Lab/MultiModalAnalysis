import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
import fabio
import pyFAI
from pyFAI import units
import json
from pyFAI.io.ponifile import PoniFile
from pyFAI.calibrant import Calibrant
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.integrator.fiber import FiberIntegrator
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
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 14,
        "figure.figsize": (8, 6),
        "axes.linewidth": 3,
        "xtick.color": "white",      # tick marks white
        "ytick.color": "white",
        "xtick.labelcolor": "black", # tick labels black
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

def twotheta_to_q_components(tth_deg, λ_A, chi_deg, αi_deg):
    tth = np.deg2rad(tth_deg)
    θf  = tth/2
    k   = 4*np.pi/λ_A * np.sin(θf)  # wavevector magnitude in 1/Angstrom
    αi  = np.deg2rad(αi_deg)
    

    # exact GI geometry:
    Qr = k * np.sin(θf) * np.cos(np.deg2rad(chi_deg))
    Qz = k * (np.cos(θf + αi))

    return Qr, Qz
# ========== Helper: Extract timestamp from .tif ==========
def extract_datetime_from_tif(tif_path):
    with Image.open(tif_path) as img:
        metadata = {TAGS.get(k, k): img.tag[k] for k in img.tag.keys()}
        raw_datetime = metadata.get("DateTime")
        if raw_datetime:
            dt_str = raw_datetime[0].split("\x00")[0]
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        else:
            raise ValueError(f"No valid DateTime in: {tif_path}")


# ========== Step 1: GUI input selection ==========
def select_inputs():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder_path = filedialog.askdirectory(
        title="Select Folder with TIF Files", mustexist=True
    )
    if not folder_path:
        raise ValueError("No folder selected.")

    poni_files = glob.glob(os.path.join(folder_path, "*.poni"))
    if poni_files:
        poni_path = poni_files[0]
    else:
        poni_path = filedialog.askopenfilename(
            title="Select PONI File",
            filetypes=[("PONI files", "*.poni")],
            initialdir=folder_path,
        )

    if not poni_path:
        raise ValueError("No PONI file selected.")

    sample_name = simpledialog.askstring(
        "Sample Name",
        "Enter the sample name for saving the output file:",
        initialvalue=os.path.basename(folder_path),
    )
    if not sample_name:
        raise ValueError("No sample name provided.")

    return folder_path, poni_path, sample_name


# ========== Step 2: GIWAXS integration ==========
def integrate_giwaxs(folder_path, poni_path, method=("bbox", "csr", "cython")):
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
        res = ai.integrate1d_ng(
            img_array, 1000, mask=mask, unit="q_A^-1", method=method
        )
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
    intensities = intensities**0.25  # square root for better visualization
    cp = plt.contourf(frame_times, q_vals, intensities.T, levels=100, cmap="viridis")
    cb = fig.colorbar(cp, location="left")
    cb.locator = mpl.ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    cb.set_label("Gamma-Adjusted Intensity \n $\\gamma = 0.25$")
    #     pcm = ax.pcolormesh(
    #     frame_times, q_vals,
    #     intensities.T,
    #     shading="auto"
    # )
    #     fig.colorbar(pcm, ax=ax, label="Intensity", location='left')

    # define axis names, ticks, etc.
    q_min, q_max = (0.8, 3.5)  # adjust these limits as needed
    # y_ticks = np.linspace(q_min, q_max, 10)  # number of tickmarks
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Q $(\AA^{-1})$")
    # ax.set_yticks(y_ticks)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim(q_min, q_max)
    # ax.set_title(sample_name)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, str(sample_name) + "_GIWAXS_Plot"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_Chi_2theta(image_file, poni_file, ax=None, label=None, calibrant_file=None, hkls =None):
    """Plot the 2D image in Chi-2theta space using pyFAI.
    Parameters:
        image (numpy.ndarray): 2D array representing the image data.
        poni_file (str): Path to the .poni file for calibration.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    image = fabio.open(image_file).data
    # Load the calibration file
    ai = AzimuthalIntegrator()
    ai.load(poni_file)
    ai.set_rot3(180 * np.pi / 180)
    print(ai)
    # image = np.clip(image, 0, None)  # Clip the image to avoid negative values
    alpha_i = 2 * (2 * np.pi) / 360  # 2 degrees in radians

    res2d = ai.integrate2d(
        image, 1024, 1024, method=("no", "csr", "opencl"),
        unit = (units.Q_A, units.CHI_DEG),  # convert to chi and 2theta
    )  # convert to chi and 2theta
    if label is None:
        label = "GIWAXS Image in Chi-2theta Space"
    plot2d(res2d, label=label, ax=ax)
    im = ax.images[0]
    im.set_cmap('viridis')

    if hkls is not None: 
        records = json.load(open(hkls, 'r'))                
        d2hkls = {
        rec["d"]: [f"({hkl[0]})" for hkl in rec["hkl"]]
        for rec in records
        }

    if calibrant_file:

        result = res2d

        poni = PoniFile(data=poni_file)
        calibrant = Calibrant(wavelength=poni.wavelength)
        calibrant.load_file(calibrant_file)
        img = result.intensity
        pos_rad = result.radial
        pos_azim = result.azimuthal
    

        x_values = None
        twotheta = np.array([i for i in calibrant.get_2th() if i])  # in radian
        d = calibrant.get_dSpacing()
        unit = result.unit
        unit = unit[0]
        if unit == units.TTH_DEG:
            x_values = np.rad2deg(twotheta)
        elif unit == units.TTH_RAD:
            x_values = twotheta
        elif unit == units.Q_A:
            x_values = (4.0e-10 * np.pi / calibrant.wavelength) * np.sin(0.5 * twotheta)
        elif unit == units.Q_NM:
            x_values = (4.0e-9 * np.pi / calibrant.wavelength) * np.sin(0.5 * twotheta)
        if x_values is not None:
            for x, d_spacing, i in zip(x_values, d, range(len(x_values))):
                print(f"d = {d_spacing:.4f} Å at x = {x:.2f} {unit}")
                line = lines.Line2D(
                    [x, x],
                    [pos_azim.min(), pos_azim.max()],
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.2
                )
                ax.add_line(line)
                if hkls is not None:
                    hkl = d2hkls[d_spacing][-1]
                    if hkl is not None:
                        ax.text(x, pos_azim.max() -10*(i+3), f"{hkl}", color="white", fontsize=16, ha='center', va='top')

    return res2d


def plot_QR_QZ(image_file, poni_file, ax=None, label=None, calibrant_file=None, hkls=None):
    """Plot the 2D image in Qr-Qz space using pyFAI.
    Parameters:
        image (numpy.ndarray): 2D array representing the image data.
        poni_file (str): Path to the .poni file for calibration.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    image = fabio.open(image_file).data
    # Load the calibration file
    poni = PoniFile(data=poni_file)

    # fi = FiberIntegrator(dist=poni.dist, poni1=poni.poni1, poni2=full_length -poni.poni2,
    #                  wavelength=poni.wavelength,
    #                  rot1=np.pi - poni.rot1, rot2=np.pi - poni.rot2, rot3=poni.rot3,
    #                  detector=pyFAI.detector_factory("Pilatus1M", {"orientation": 4}),
    #                 )
    fi = FiberIntegrator()
    fi.load(poni_file)
    alpha_i = np.deg2rad(2)  # 2 degrees in radians
    print(alpha_i)
    fi.reset_integrator(incident_angle=2, tilt_angle=0, sample_orientation=6) #this doesn't do anything, but it let's us display the FI object 
    fi.set_rot3(180 * np.pi / 180)

    print(fi)
    # image = np.clip(image, 0, None)  # Clip the image to avoid negative values

    res2d = fi.integrate2d_grazing_incidence(
        image,
        npt_ip=1024,
        npt_oop=1024,
        method=("no", "csr", "opencl"),
        unit_ip="qip_A^-1",
        unit_oop="qoop_A^-1",
        incident_angle=alpha_i,
        tilt_angle=0,
        sample_orientation=6
    )  # convert to q_r and q_z
    if label is None:
        label = "GIWAXS Image in Qip-Qoop Space"
    plot2d(res2d, label=label, ax=ax)
    im = ax.images[0]
    im.set_cmap('viridis')
    ax.set_xlabel(r"$q_r$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$q_z$ ($\AA^{-1}$)")
    if hkls is not None:
        records = json.load(open(hkls, 'r'))                
        d2hkls = {
            rec["d"]: [f"({hkl[0]})" for hkl in rec["hkl"]]
            for rec in records
        }
    if calibrant_file:
        # build Qr/Qz grids
        qr = res2d.radial        # shape (n_ip,)
        qz = res2d.azimuthal     # shape (n_oop,)
        Qr_grid, Qz_grid = np.meshgrid(qr, qz)

        # total-Q map
        Qtot = np.hypot(Qr_grid, Qz_grid)

        # grab d‑spacings → q = 2π/d
        poni = PoniFile(data=poni_file)
        cal  = Calibrant(wavelength=poni.wavelength)
        cal.load_file(calibrant_file)
        d_list = np.array(cal.get_dSpacing())
        q_peaks = 2*np.pi / d_list

        # contour at each q_peak
        for q0, d in zip(q_peaks, d_list):
            cs =ax.contour(
                Qr_grid,
                Qz_grid,
                Qtot,
                levels=[q0],
                colors="red",
                linestyles="--",
                linewidths=1,
                alpha= 0.2,
            )
            seg_list = cs.allsegs[0]       # index 0 because we only had one level
            if not seg_list:
                continue                   # nothing found, skip
            first_path = seg_list[0]       # the first continuous segment (an array of shape (M,2))

            # reference segment endpoints:
            P1 = np.array([0.64, 0.8])
            P2 = np.array([1.45, 3.1])
            v  = P2 - P1
            vv = np.dot(v, v)

            # assume `seg` is your (N×2) contour array, and `hkl_txt` its label
            # 1) project each seg point onto P1→P2 (clamped to the segment)
            w  = first_path - P1                  # shape (N,2)
            t  = np.clip((w @ v) / vv, 0, 1)  # shape (N,)
            proj = P1 + np.outer(t, v)     # shape (N,2)

            # 2) find the closest vertex
            d2 = np.sum((first_path - proj)**2, axis=1)
            i_min = np.argmin(d2)
            x0, y0 = first_path[i_min]
            offset = .03
            x_txt = x0 + offset * v[0]
            y_txt = y0 + offset * v[1]
            dx, dy = np.gradient(first_path[:, 0]), np.gradient(first_path[:, 1])
            angle = np.degrees(np.arctan2(dy[i_min], dx[i_min])) + 4
            # print(angle)

            ax.text(
                x_txt,
                y_txt,
                f"{d2hkls[d][-1]}",
                ha ="center",
                va ="center",
                rotation = angle,
                color="white",
                fontsize=18,
            )
    return res2d



# ========== Main execution block ==========
if __name__ == "__main__":
    folder_path = "E:/MAPI_sean/MAPI_sean_control_S1_30_tube_5min"
    poni_file = (
        "E:/MAPI_sean/MAPI_sean_control_S1_30_tube_5min/MAPI_sean_control_S1_30_tube_5min_00222_refined.poni"
    )
    sample_name = ""
    calibrant_file = "pyFAI/MAPbI3_Calibrant.D"
    image_path = "E:/MAPI_sean/MAPI_sean_control_S1_30_tube_5min/MAPI_sean_control_S1_30_tube_5min_00500.tif"
    hkl_path = "pyFAI/cubic_MAPbI3_dhkl.json"
    fig, ax = subplots(1, 1, figsize=(8, 8))

    result1 = plot_Chi_2theta(image_path, poni_file, ax=ax, label=sample_name, calibrant_file=calibrant_file, hkls=hkl_path)  # Plot in Chi-2theta space


    fig, ax = subplots(1, 1, figsize=(8, 8))
    res = plot_QR_QZ(image_path, poni_file, label=sample_name, calibrant_file=calibrant_file, ax=ax, hkls=hkl_path)  # Plot in Qr-Qz space
    ax.set_xlim(0, max(res.radial))
    im = ax.images[0]
    im.set_cmap('viridis')
    plt.show()
