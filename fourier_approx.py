"""
Copyright (c) 2016, Youngmin Park, Bard Ermentrout
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ympark1988@gmail.com

Plot data and compute Fourier coefficients

(Thank you Mario for the help and inspiration)

TODO:
1. Let user choose coefficients up to x% of the L2 norm
2. Add code to write coefficients to file
"""

import trbwb_phase as tbwbp
import thetaslowmod_phase as tsmp
from thetaslowmod_lib import *
from trbwb_lib import *



try:
    import matplotlib.pylab as mp
    matplotlib_module = True
except ImportError:
    print "You must have matplotlib installed to generate plots"
    matplotlib_module = False

try:
    import numpy as np
    #np.random.seed(0)
    numpy_module = True
except ImportError:
    print "You must have numpy installed to run this script"
    numpy_module = False

try:
    from scipy.optimize import brentq
    scipy_module = True
except ImportError:
    print "You must have numpy installed to run this script"
    scipy_module = False

def manual_ift(bc,freqc,idxc,N):
    """
    manual inverse fourier transform
    bc: nonzero output from np.fft.fft
    ffreqc: corresponding freq. component
    idxc: corresponding index of freq. component
    """
    # define domain
    n = np.linspace(0,N-1,N)
    tot = 0

    c = 0 # counter
    # for select k, compute value at each n
    outstring = ''
    outstring_d = ''
    
    coeff_table_1 = ''
    coeff_table_2 = ''

    for k in idxc:
        if k > N/2:
            k -= N

        tot += np.real(bc[c])*np.cos(k*2*np.pi*n/N) - np.imag(bc[c])*np.sin(k*2*np.pi*n/N)

        # format fourier series
        if np.sign(np.real(bc[c]))<0:
            coeff1 = '-'+str(np.abs(np.real(bc[c])))+'*'
        else:
            coeff1 = '+'+str(np.abs(np.real(bc[c])))+'*'

        if np.sign(-np.imag(bc[c]))<0:
            coeff2 = '-'+str(np.abs(-np.imag(bc[c])))+'*'
        else:
            coeff2 = '+'+str(np.abs(-np.imag(bc[c])))+'*'
        
        if k >= 0:
            coeff_table_1 += '[\'$a_'+str(k)+'$\', '+str(2*np.real(bc[c]))+'],\n'
            coeff_table_2 += '[\'$b_'+str(k)+'$\', '+str(-2*np.imag(bc[c]))+'],\n'

        #print +', b_'+str(k)+'='+coeff2
        outstring += coeff1+'cos('+str(k)+'*x)'+coeff2+'sin('+str(k)+'*x)'

        # format derivative
        if np.sign(np.real(bc[c])*k)<0:
            coeff1 = '+'+str(np.abs(k))+'*'+str(np.abs(np.real(bc[c])))+'*'
        else:
            coeff1 = '-'+str(np.abs(k))+'*'+str(np.abs(np.real(bc[c])))+'*'


        if np.sign(-np.imag(bc[c])*k)<0:
            coeff2 = '-'+str(np.abs(k))+'*'+str(np.abs(-np.imag(bc[c])))+'*'
        else:
            coeff2 = '+'+str(np.abs(k))+'*'+str(np.abs(-np.imag(bc[c])))+'*'

        
        outstring_d += coeff1+'sin('+str(k)+'*x)'+coeff2+'cos('+str(k)+'*x)'
        
        c += 1
    
    print coeff_table_1
    print coeff_table_2

    print 'fourier series'
    print outstring

    print 'fourier series derivative'
    print outstring_d
    
    return tot

def manual_ift_sin(bc,freqc,idxc,N):
    """
    manual inverse fourier transform for sine terms only
    bc: nonzero output from np.fft.fft
    ffreqc: corresponding freq. component
    idxc: corresponding index of freq. component
    """
    # define domain
    n = np.linspace(0,N-1,N)
    tot = 0

    c = 0 # counter
    # for select k, compute value at each n
    for k in idxc:
        #print np.imag(bc[c])
        tot += np.imag(bc[c])*np.sin(k*2*np.pi*n/N)
        #print k,np.imag(bc[c]),'sin'
        c += 1
    return tot

def manual_ift_cos(bc,freqc,idxc,N):
    """
    manual inverse fourier transform for cosine terms only
    bc: nonzero output from np.fft.fft
    ffreqc: corresponding freq. component
    idxc: corresponding index of freq. component
    """
    # define domain
    n = np.linspace(0,N-1,N)
    tot = 0

    c = 0 # counter
    # for select k, compute value at each n
    for k in idxc:
        tot += np.real(bc[c])*np.cos(k*2*np.pi*n/N)
        #print k,np.real(bc[c]),'cos'
        c += 1

    return tot


def amp_cutoff(x,n,fcoeff):
    # goal: find ideal x s.t. sum(coeff_array_idx) = n
    """
    fcoeff: output from np.fft.fft
    x: cutoff for magnitude of fourier coefficient
    n: desired number of fourier coefficients
    """
    coeff_array_idx = np.absolute(fcoeff) > x
    return sum(coeff_array_idx) - n

def main():
    # load/define data


    #dat = np.loadtxt("hxx_fixed_a1=0.1_b1=1.0_c1=1.1_a2=0.1_b2=1.0_c2=1.1_eps=0.01_mux=1.0_muy=1.0.dat")
    #dat = dat[:,1]
    #dat2 = np.sin(np.linspace(0,10,N))
    #dom = np.linspace(0,1000,1000)


    
    # fourier coefficients for phase model theta model
    if True:
        mux = 1.
        muy = 1.
        #muy = 2.62

        eps = .01

        a1=.1;b1=1.;c1=1.1
        a2=a1;b2=b1;c2=c1


        sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)

        phase = tsmp.Phase(use_last=False,
                           save_last=False,
                           run_full=False,
                           recompute_h=True,
                           run_phase=True,
                           recompute_slow_lc=False,
                           a1=a1,b1=b1,c1=c1,
                           a2=a2,b2=b2,c2=c2,
                           T=0,dt=.05,mux=mux,muy=muy,
                           eps=eps,
                           thxin=[0,0],thyin=[0,0],sx0=sx0,sy0=sy0)

        sx = phase.sxa_fn(0)
        sy = phase.sya_fn(0)

        #domxx,dat = phase.generate_h(sx,sy,choice='xx',return_domain=True)
        #domxy,dat = phase.generate_h_inhom(sx,sy,choice='xy')
        #domyx,dat = phase.generate_h_inhom(sx,sy,choice='yx')
        domyy,dat = phase.generate_h(sx,sy,choice='yy',return_domain=True)

    # Fourier coefficients of the Traub+Ca/Wang-Buzsaki phase model
    if False:
        
        gee_phase=10.;gei_phase=24.
        gie_phase=13.;gii_phase=10.

        eps = .0025
        mux_phase=1.;muy_phase=1.#23.15
        
        sx0=.05
        sy0=.05

        phs_init_trb = [0.,.1]
        phs_init_wb = [.3,-.2]
        
        # determine current data (good candidates: f=0.05,0.064)    
        fFixed=.05
        itb_mean,iwb_mean = get_mean_currents(fFixed)
        
        itb_phase=itb_mean-fFixed*(gee_phase-gei_phase)
        iwb_phase=iwb_mean-fFixed*(gie_phase-gii_phase)

        T_phase=10;dt_phase=.01
        
        phase = tbwbp.Phase(use_last=False,
                      save_last=False,
                      recompute_beta=True,
                      
                      recompute_h=True,
                      sbar=fFixed,
                      T=T_phase,dt=dt_phase,
                      
                      itb_mean=itb_mean,iwb_mean=iwb_mean,
                      gee=gee_phase,gei=gei_phase,
                      gie=gie_phase,gii=gii_phase,
                      
                      mux=mux_phase,muy=muy_phase,eps=eps,
                      phs_init_trb=phs_init_trb,
                      phs_init_wb=phs_init_wb,
                      
                      sx0=sx0,sy0=sy0,
                      #use_mean_field_data=[],
                      verbose=True)

        sx = sx0
        sy = sy0

        #domxx,dat = phase.generate_h(sx,sy,choice='xx',return_domain=True)
        #domxy,dat = phase.generate_h(sx,sy,choice='xy',return_domain=True)
        #domyx,dat = phase.generate_h(sx,sy,choice='yx',return_domain=True)
        #domyy,dat = phase.generate_h(sx,sy,choice='yy',return_domain=True)

        
    N = len(dat)
    dom = np.linspace(0,N,N)
    #print np.shape(dat), np.shape(dat2)

    # get Fourier transform and frequencies
    fcoeff = np.fft.fft(dat)
    ffreq = np.fft.fftfreq(dat.size)

    # find cutoff x for desired number of coefficients
    n = 2 # desired # of coefficients
    x = brentq(amp_cutoff,0,np.amax(np.abs(fcoeff)),args=(n,fcoeff))
    
    # array corresponding to desired coefficients
    coeff_array_idx = np.absolute(fcoeff) > x

    # build list of desired coefficients
    b = fcoeff*coeff_array_idx

    # extract corresponding frequencies
    freq = ffreq*coeff_array_idx   
    
    # build lits of only desired coeff & freq
    bc = fcoeff[coeff_array_idx]/N
    freqc = ffreq[coeff_array_idx]
    
    idxc = np.nonzero(coeff_array_idx)[0]
    #print bc,freqc,idxc
    # come back to time domain
    
    c = np.fft.ifft(b)
    # or
    c2 = manual_ift(bc,freqc,idxc,N)

    # for sine/cosine component only:
    c3 = manual_ift_sin(bc,freqc,idxc,N)
    c4 = manual_ift_cos(bc,freqc,idxc,N)

    print 'max pointwise error (sine + cosine):', np.amax(np.abs(dat-c2))
    print 'max pointwise error (sine):', np.amax(np.abs(dat-c3))
    print 'max pointwise error (cosine):', np.amax(np.abs(dat-c4))
    
    # add option to write coefficients to file
    if True:
        pass
    
    if matplotlib_module:
        mp.figure()
        mp.plot(dat,label='original')
        mp.plot(c2,label='sin+cos')
        #mp.plot(c)
        #mp.plot(c2)
        #mp.plot(c3,label='sin')
        #mp.plot(c4,label='cos')

        #mp.pl
        #mp.ylim(-2,2)

        mp.legend()
        mp.show()
    
if __name__ == "__main__":
    main()
