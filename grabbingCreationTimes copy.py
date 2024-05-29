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
import time
import os

def compute_average_intensity(image, roi):
    """
    Compute the average intensity of pixels within the region of interest (ROI).
    
    Args:
        image (numpy.ndarray): The input image.
        roi (numpy.ndarray): A binary mask representing the region of interest.
        
    Returns:
        float: The average intensity of pixels within the ROI.
    """
    # Sum the pixel values in the region of interest (this is much faster than indexing and using np.mean)
    sum_intensity = np.sum(image * roi)
    
    # Count the number of pixels in the region of interest
    count_pixels = np.sum(roi)
    
    # Compute the average intensity
    average_intensity = sum_intensity / count_pixels
    
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
print(giwaxsFile)
dataOld = pd.read_csv(giwaxsFile[0], sep='\s+', header=0, names=np.array(
    ['image_num', 'twotheta', 'twotheta_cuka', 'dspacing', 'qvalue', 'intensity', 'frame_number', 'izero',
      'date', 'time', 'AMPM']))

path = askdirectory()
print(path)
files = sorted(glob.glob(path + "\*.tif"))
# Get number of files
nFiles = len(files)
print(nFiles)
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
print(linesPerImage)
#read all tiff files at once
imagestack = tifffile.imread(files)

for i in tqdm(range(0, nFiles)):
    tempiZero = compute_average_intensity(imagestack[i,:,:], roi_mask)
    tif = tifffile.TiffFile(files[i])
    timestamp = tif.pages[0].tags['DateTime'].value

    # Create np arrays with the same length as the number of lines per image, but skip the for loop used previously
    tempiZero_array = np.full( linesPerImage, float(tempiZero))
    timestamp_array = np.full( linesPerImage, timestamp) 
    iZero.append(tempiZero_array)
    creationTimes.append(timestamp_array)

# Flatten the lists
creationTimes = np.concatenate(creationTimes)
iZero = np.concatenate(iZero)

dataNew = dataOld.drop(['AMPM', 'time', 'izero'], axis=1)
dataNew.insert(8, "time", creationTimes, True)
dataNew.insert(9, "iZero", iZero, True) 
newCols = ['image_num', 'twotheta', 'twotheta_cuka', 'dspacing', 'qvalue', 'intensity', 'frame_number', 'iZero',
  'date', 'time']
dataNew = dataNew[newCols]


filepath = os.path.splitext(giwaxsFile[0])[0]
dataNew.to_csv(filepath + "_output.dat", sep='\t', index=False)
