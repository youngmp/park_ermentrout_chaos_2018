import numpy as np
import matplotlib.pylab as mp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

tbfi = np.loadtxt('tbfi.dat')
tbfi2 = np.loadtxt('tbfi2.dat')

mp.figure()
mp.plot(tbfi[:,0],tbfi[:,1])
mp.plot(tbfi2[:,0],tbfi2[:,1])
mp.show()

