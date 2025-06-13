#%%

from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
# calculator.show_plot(conventional_structure)

#%%
pattern = calculator.get_pattern(conventional_structure)
two_thetas = []
intensities = []
all_hkls = []
qs = []
for peak in zip(pattern.x, pattern.y, pattern.d_hkls, pattern.hkls):
    two_theta, intensity, d_hkl, hkls = peak
    if intensity > 10:  # filter low intensity
        miller_list = [hkl['hkl'] for hkl in hkls]  # list of contributing planes
        two_thetas.append(two_theta)
        intensities.append(intensity)
        all_hkls.append(miller_list)
        q = 2 * np.pi / d_hkl  # calculate the scattering vector magnitude 
        qs.append(q)
       
# Sort by intensity
sorted_indices = np.argsort(intensities)[::-1]
two_thetas = np.array(two_thetas)[sorted_indices]
qs = np.array(qs)[sorted_indices]
intensities = np.array(intensities)[sorted_indices]
all_hkls = [all_hkls[i] for i in sorted_indices]

df = pd.DataFrame({
    "2-theta (degrees)": two_thetas,
    "q (Å⁻¹)": qs,
    "Intensity": intensities,
    "Miller indices": all_hkls
})
df.to_csv("xrd_pattern_mp-995214.csv", index=False)
# Print peak info
for i, (q, intensity, hkl_list, theta) in enumerate(zip(qs, intensities, all_hkls, two_thetas)):
    print(f"Peak {i+1}: q = {q:.2f} Å⁻¹, Intensity = {intensity:.2f}, 2-theta = {theta:.2f}°, Miller indices = {hkl_list}")

# Plot intensity vs q
plt.figure(figsize=(10, 5))
plt.vlines(qs, [0]*len(intensities), intensities)
for x, y, hkl_list in zip(qs, intensities, all_hkls):
    label = ','.join([str(hkl) for hkl in hkl_list])
    plt.text(x, y + 1, label, rotation=90, ha='center', va='bottom', fontsize=8)

plt.xlabel("q (Å⁻¹)")
plt.ylabel("Intensity (a.u.)")
plt.title("Simulated XRD Pattern (q-space) for mp-995214")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
