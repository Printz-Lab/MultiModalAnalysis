# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:04:24 2024

@author: Tim Kodalle
"""

from PIL import Image
import glob
from tkinter.filedialog import askdirectory, askopenfilenames
import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm

def compute_average_intensity(image, roi):
    """
    Compute the average intensity of pixels within the region of interest (ROI).
    
    Args:
        image (numpy.ndarray): The input image.
        roi (numpy.ndarray): A binary mask representing the region of interest.
        
    Returns:
        float: The average intensity of pixels within the ROI.
    """
    # Apply the ROI mask to the image
    roi_pixels = image[roi]
    
    # Compute the average intensity
    average_intensity = np.mean(roi_pixels)
    
    return average_intensity

def create_roi(image_shape, roi_coordinates):
    """
    Create a binary mask representing the region of interest (ROI).
    
    Args:
        image_shape (tuple): The shape of the image (height, width).
        roi_coordinates (tuple): The coordinates of the ROI (start_y, start_x, end_y, end_x).
        
    Returns:
        numpy.ndarray: A binary mask representing the ROI.
    """
    # Create an empty binary mask
    roi = np.zeros(image_shape, dtype=np.uint8)
    
    # Set the ROI region to 1
    for i in range(0, len(roi_coordinates)):
        start_y, start_x, end_y, end_x = roi_coordinates[i]
        roi[start_y:end_y, start_x:end_x] = 1
    
    return roi


# =================Part 1: Loading and Generating Data======================
# Loading data
giwaxsFile = askopenfilenames()

dataOld = pd.read_csv(giwaxsFile[0], sep='\s+', header=0, names=np.array(
    ['image_num', 'twotheta', 'twotheta_cuka', 'dspacing', 'qvalue', 'intensity', 'frame_number', 'izero',
      'date', 'time', 'AMPM']))

path = askdirectory()
files = sorted(glob.glob(path + "\*.tif"))
# Get number of files
nFiles = len(files)
creationTimes = []
iZero = []

# Define the coordinates of the region of interest (start_y, start_x, end_y, end_x)
roi_coordinates = [(1, 1, 100, 100)]

# Load an example TIFF image
image_path = files[0]
image = tifffile.imread(image_path)

# Create the region of interest (ROI) mask
roi_mask = create_roi(image.shape, roi_coordinates)

# Plot the image with the ROI marked
plt.figure(figsize=(8, 6))
plt.imshow(image, cmap='gray', norm=colors.LogNorm(vmin=0.1, vmax=100))
plt.imshow(roi_mask, cmap='jet', alpha=0.5)  # Overlay the ROI mask
plt.title('Image with ROI')
plt.colorbar(label='Intensity')
plt.show()

linesPerImage = int(len(dataOld.qvalue) / nFiles)

for i in tqdm(range(0, nFiles)):
    curImage = tifffile.imread(files[i])
    tempiZero = compute_average_intensity(curImage, roi_mask)
    for ii in range(0,linesPerImage):
        # Compute the average intensity within the ROI
        iZero.append(tempiZero)
    # Get correct timestamp
    with Image.open(files[i]) as img:        
        for iii in range(0,linesPerImage):
            creationTimes.append(img.tag_v2[306].split(' ')[1].split("\x00")[0])

dataNew = dataOld.drop(['AMPM', 'time', 'izero'], axis=1)
dataNew.insert(8, "time", creationTimes, True)
dataNew.insert(9, "iZero", iZero, True) 
newCols = ['image_num', 'twotheta', 'twotheta_cuka', 'dspacing', 'qvalue', 'intensity', 'frame_number', 'iZero',
  'date', 'time']
dataNew = dataNew[newCols]


dataNew.to_csv(path + "_output.dat", sep = '\t', index = False)
