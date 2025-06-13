#%%
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

with MPRester(api_key="j6I5M4p81ujtyq16TyFw629cCo637cX9") as mpr:
    # first retrieve the relevant structure
    structure = mpr.get_structure_by_material_id("mp-995214")

# important to use the conventional structure to ensure
# that peaks are labelled with the conventional Miller indices
sga = SpacegroupAnalyzer(structure)
conventional_structure = sga.get_conventional_standard_structure()

# this example shows how to obtain an XRD diffraction pattern
# these patterns are calculated on-the-fly from the structure
calculator = XRDCalculator(wavelength="CuKa")
pattern = calculator.get_pattern(conventional_structure)
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

with MPRester(api_key="j6I5M4p81ujtyq16TyFw629cCo637cX9") as mpr:
    # first retrieve the relevant structure
    structure = mpr.get_structure_by_material_id("mp-995214")

# important to use the conventional structure to ensure
# that peaks are labelled with the conventional Miller indices
sga = SpacegroupAnalyzer(structure)
conventional_structure = sga.get_conventional_standard_structure()

#%%
# this example shows how to obtain an XRD diffraction pattern
# these patterns are calculated on-the-fly from the structure
calculator = XRDCalculator(wavelength=1.239841984332002)
calculator.show_plot(conventional_structure)

#%%
pattern = calculator.get_pattern(conventional_structure)
print(pattern)
# the pattern is a DiffractionPattern object
# it contains the 2-theta values, intensities, and Miller indices
import matplotlib.pyplot as plt
plt.plot(pattern.x, pattern.y)
plt.xlabel("2-theta (degrees)")
plt.ylabel("Intensity (a.u.)")
plt.title("XRD Pattern for mp-995214")
plt.xticks(rotation=45)
plt.grid()
plt.show()
