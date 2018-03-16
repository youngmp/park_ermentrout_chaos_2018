"""
plot fi curves of trb,wb
"""

import numpy as np
import matplotlib.pyplot as plt


wb = np.loadtxt('wbfi.dat')
trb = np.loadtxt('tbfi.dat')


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(wb[:,0],wb[:,1],color='blue',label='wb')
ax.plot(trb[:,0],trb[:,1],color='green',label='trb')


plt.show()

