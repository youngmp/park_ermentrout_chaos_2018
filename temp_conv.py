import numpy as np
import matplotlib.pylab as mp


def f(x):
    """
    period 1
    """
    return np.mod(x,1.)

def g(x):
    """
    period 2
    """
    return np.mod(x/2.,1.)

# brute force conv
sint = np.linspace(0,2.,5000)
ds = 1.*sint[-1]/len(sint)

tot = 0
for i in range(len(sint)):
    tot += f(sint-sint[i])*g(sint[i])

tot *= ds


tot2 = np.real(np.fft.ifft(np.fft.fft(g(sint))*np.fft.fft(f(sint))))*ds

mp.figure()
mp.plot(sint,tot)
mp.plot(sint,tot2)
#mp.plot(sint,f(sint))
#mp.plot(sint,g(sint))
mp.show()
