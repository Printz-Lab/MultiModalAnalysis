import os
import shutil
import tkinter as tk
from tkinter import filedialog

# Use tkinter to prompt the user to select the logfiles
root = tk.Tk()
root.withdraw()
logfiles= filedialog.askopenfilenames()
for file in logfiles:
    Spectrums_folder = os.path.basename(file).replace(".txt", " Spectrums")
    Spectrums_folder = os.path.join(os.path.dirname(file), Spectrums_folder)
    pl_subfolder = "PL"
    logfiles_subfolder = "Logfile"
    # Create the subfolders if they don't exist
    os.makedirs(os.path.join(Spectrums_folder, pl_subfolder), exist_ok=True)
    os.makedirs(os.path.join(Spectrums_folder, logfiles_subfolder), exist_ok=True)

    # Move all PL text files to the PL subfolder
    for PlFile in os.listdir(Spectrums_folder):
        if PlFile.endswith(".txt"):
            Spectrums_path = os.path.join(Spectrums_folder, PlFile)
            destination_path = os.path.join(Spectrums_folder, pl_subfolder, PlFile)
            shutil.move(Spectrums_path, destination_path)

    # move the log file to the Logfile subfolder
    destination_path = os.path.join(Spectrums_folder, logfiles_subfolder, os.path.basename(file))
    shutil.move(file, destination_path)