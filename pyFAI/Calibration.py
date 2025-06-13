import subprocess
import tkinter as tk
from tkinter import filedialog
import fabio
from pyFAI.calibrant import Calibrant
from pyFAI.geometry import Geometry
from pyFAI.detectors import detector_factory
from pyFAI.goniometer import SingleGeometry
from pyFAI.gui import jupyter
import matplotlib.pyplot as plt


def giwaxsCalibration(default_poni_file_path=r'pyFAI\ITO_test.poni', calibrant=r'pyFAI\ito_calibrant.D', energy='10'):
    
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)
    
    # Path to the default calibration file
    
    calibrant_image_path = tk.filedialog.askopenfilename(title="Select the calibrant image")
    if not calibrant_image_path:
        print("No calibrant image selected. Exiting.")
        return
    # default_poni_file_path = r'pyFAI\ITO_test.poni'
    # default_poni_file_path = r'pyFAI\MAPI_1pct_AVAI_S1_18_5min_refined.poni'
    # calibrant = r'pyFAI\ito_calibrant.D'
    # energy = '10'
    
    calibCommand = [
    'pyFAI-calib2',
    '--poni', default_poni_file_path,
    '--c', calibrant,
    '--e', energy,
    calibrant_image_path
    ]
    
    # Launch the pyFAI calibration GUI using command line
    print("Launching pyFAI calibration GUI. This will take a few seconds...")
    subprocess.run(calibCommand, capture_output=True, text=True)
    
    return 

def refine_calibration(sampleName, image_path, initial_poni, calibrant_file, refined_poni):
    """
    Refine calibration using an initial .poni file and a custom calibrant.
    

    Parameters:
        image_path (str): Path to the calibration image.
        initial_poni (str): Path to the initial .poni file.
        calibrant_file (str): Path to the custom calibrant file.
        refined_poni (str): Path to save the refined .poni file.
    """
    # Load the calibration image
    image = fabio.open(image_path).data

    # Load the custom calibrant
    calibrant = Calibrant(filename=calibrant_file)

    # Load the initial geometry
    initial = Geometry()
    initial.load(initial_poni)
    pilatus = detector_factory("Pilatus1M")
  
    # (Optional) Add custom refinement logic if necessary.
    sg = SingleGeometry("Recalibration of Sample " + sampleName, image, calibrant=calibrant, detector=pilatus, geometry=initial)
    sg.extract_cp(max_rings=5)
    sg.geometry_refinement.refine2(fix=["rot1", "rot2", "rot3", "wavelength"])
    sg.get_ai()
    
    # Verify refinement
    ax = jupyter.display(sg=sg)
    # plt.show()

    # Save refined .poni file
    sg.geometry_refinement.save(refined_poni)
    
    return 

if __name__ == "__main__":
    # Example usage
    giwaxsCalibration()
    