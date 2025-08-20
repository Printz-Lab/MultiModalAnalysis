import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from bokeh.palettes import Viridis256, Greys256
from bokeh.plotting import figure, output_file, save
from bokeh.models import LinearColorMapper, ColorBar, NumericInput, LinearAxis, Range1d, HoverTool, CheckboxGroup, CustomJS
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.layouts import layout
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

def plotStacked(genParams, sampleName, savePath, q, timeGIWAXS, intGIWAXS, energyPL, timePL, intPL, logData, logTimeEndIdx):
    xlim = (0, 375)
    if genParams['PL']:
        # define subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9,9), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
        
        # PL plot
        # removing negative points from data (important for log plot)
        intPL = np.where(intPL < 1, 1, intPL)
        # intPL = intPL / intPL.max()  # normalize to max value
        # intPL = np.log(intPL)
        # i_max = logIntPL.max()
        plt.setp(ax1.get_xticklabels())
        intPL = intPL**0.25  # square root for better visualization
        # Create non-uniform levels to emphasize lower intensities
        max_int = intPL.max()

        cp1 = ax1.contourf(timePL, energyPL, intPL, 100,cmap='plasma')
        cbax1 = fig.add_axes([0.89, 0.66, 0.03, 0.3])
        cb1 = fig.colorbar(cp1, ax=ax1, cax=cbax1)
        cb1.locator = mpl.ticker.MaxNLocator(nbins=5)
        cb1.update_ticks()
        
        cb1.set_label('Gamma-Adjusted Intensity \n $\\gamma = 0.25$', fontsize=16, )
        ax1.set_ylabel('Energy (eV)')
    
        # Inset graph for PL plot
        inset = False # set True if zoomed inset is desired, change to False otherwise
        if inset:
            ax_new = plt.axes([.68, .86, .14, .09]) # create inset axes with dimensions [left, bottom, width, height]
            ax_new.contourf(timePL, energyPL, intPL/i_max, np.linspace(0.2/i_max,1, 100), cmap = plt.get_cmap('gist_heat')) # copy code line for larger plot
            # modify tick colors on both axes
            ax_new.tick_params(axis='x', colors='white')
            ax_new.tick_params(axis='y', colors='white')
            # add border around plot - currently need to add an invisible plot to allow connector lines to be added later
            # maybe this could be replaced somehow to improve processing speed/clarity???
            border = plt.axes([.6, .9, .25, .13]) # dimensions of plot border [left, bottom, width, height]
            border.contourf(timePL, energyPL, intPL/i_max, np.linspace(0.2/i_max,1, 100), cmap = plt.get_cmap('gist_heat'), alpha=0)
            # set tick colors and limits for border axes
            border.tick_params(axis='x', colors='none')
            border.tick_params(axis='y', colors='none')
            border.set_xlim(73, 74)
            border.set_ylim(1.55, 1.9)
            # add the actual border
            border.spines['bottom'].set_color('1')
            border.spines['top'].set_color('1')
            border.spines['right'].set_color('1')
            border.spines['left'].set_color('1')
            border.patch.set_facecolor('none')
            border.patch.set_edgecolor("1")
            # set limits for inset axes
            ax_new.set_xlim(72.5,74.5)
            ax_new.set_ylim(1.5,1.9)
            # set tick label spacing
            ax_new.set_yticks(np.arange(1.5,1.9,0.2))
            ax_new.set_xticks(np.arange(73,76,1))
            # add four connector lines between corners of the original and inset plots
            mark_inset(ax1, border, loc1=1, loc2=3, fc="none", ec="1")
            mark_inset(ax1, border, loc1=2, loc2=4, fc="none", ec="1")
            
        giwaxsBarPos = [0.89, 0.3, 0.03, 0.3]
            
    else:
        # define subplots
        fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(6, 5.4), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        giwaxsBarPos = [0.89, 0.425, 0.03, 0.54]

    # GIWAXS plot
    # define ranges and limits
    i_max = intGIWAXS.max()
    i_min = intGIWAXS.min()
    intGIWAXS = intGIWAXS**0.25  # square root for better visualization
    
    cp2 = ax2.contourf(timeGIWAXS, q, intGIWAXS.T, 100, cmap=plt.get_cmap('viridis'))
    cbax2 = fig.add_axes(giwaxsBarPos)
    cb2 = fig.colorbar(cp2, ax = ax2, cax=cbax2)
    cb2.locator = mpl.ticker.MaxNLocator(nbins=5)
    cb2.update_ticks()
    cb2.set_label('Gamma-Adjusted Intensity \n $\\gamma = 0.25$', fontsize=16)
    ax2.set_ylabel(r'q ($\AA^{-1}$)')

    # Logging plot
    ax3.plot(logData.Time, logData.Pyrometer, 'r-')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'Temperature ($^{\circ}$C)', color='r')
    # ax3.set_ylim([0, 105])
    if not genParams['TempOld']:
        ax4 = ax3.twinx()
        ax4.plot(logData.Time, logData.Spin_Motor, 'b-')
        ax4.set_ylabel(r'Spin speed (rpm)', color='b')
        plt.subplots_adjust(right=0.88, top=0.97, bottom = 0.1, hspace=0.1)
    else:
        plt.subplots_adjust(right=0.88, top=0.97, bottom = 0.1, hspace=0.1)

    # General settings for the figure
    ax3.set_xlim(xlim) # assumption that logging was terminated last, change to different axis otherwise
    # supress fig.subtitle(sample_name) if no plot title is desired
    # fig.subtitle(sample_name)
    plt.savefig(os.path.join("", sampleName + '_Stacked_Plot.png'), dpi = 300, bbox_inches = "tight")
    # plt.show()
    
    return fig

pkl_files = [
        "D:/nanofibers/MAPbI3_nanofibers_S1/output/MAPbI3_nanofibers_S1.pkl",
        "D:/nanofibers/MAPbI3_nanofibers_S2/output/MAPbI3_nanofibers_S2.pkl",
        "D:/nanofibers/MAPbI3_nanofibers_S3/output/MAPbI3_nanofibers_S3.pkl",
        "D:/nanofibers/MAPbI3_nanofibers_S4/output/MAPbI3_nanofibers_S4.pkl",
        "D:/nanofibers/MAPbI3_nanofibers_S5/output/MAPbI3_nanofibers_S5.pkl",
    ]

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        testObj = pickle.load(f)
    
    # Assuming testObj has the necessary attributes
    if hasattr(testObj, 'genParams') and hasattr(testObj, 'sampleName') and hasattr(testObj, 'outputPath'):
        plotStacked(testObj.genParams, testObj.sampleName, testObj.outputPath, 
                    testObj.giwaxsQPost, testObj.giwaxsTimePost, testObj.giwaxsIntensityPost, 
                    testObj.plEnergyPost, testObj.plTimePost, testObj.plIntensityPost, 
                    testObj.logDataPost, testObj.logTimeEndIdx)
    else:
        print(f"Skipping {pkl_file}: Missing required attributes.")