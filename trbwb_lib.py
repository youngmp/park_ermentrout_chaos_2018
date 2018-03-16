import sys
import numpy as np
from scipy.interpolate import interp1d


def get_mean_currents(f):
    """
    given a frequency, get the current.
    
    """
    
    # load data
    ftb_data = np.loadtxt('tbfi2.dat')
    fwb_data = np.loadtxt('wbfi2.dat')
    
    # define interpolating functions for frequency-current
    tbfi = interp1d(ftb_data[:,0],ftb_data[:,1])
    wbfi = interp1d(fwb_data[:,0],fwb_data[:,1])

    # define inverse interpolating functions for frequency-current
    # well-defind because the FI curves are monotonic
    tbfi_inv = interp1d(ftb_data[:,1],ftb_data[:,0])
    wbfi_inv = interp1d(fwb_data[:,1],fwb_data[:,0])

    itb_mean = tbfi_inv(f)
    iwb_mean = wbfi_inv(f)

    return itb_mean,iwb_mean
    

def mean_field_rhs():
    """
    mean field rhs

    stb' = eps*(-stb + fx(stb,swb))/mux
    swb' = eps*(-swb + fy(stb,swb))/muy
    """
    pass
    


def update_progress(progress):
    # from https://stackoverflow.com/questions/3160699/python-progress-bar
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"

    
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), np.round(progress,3)*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
