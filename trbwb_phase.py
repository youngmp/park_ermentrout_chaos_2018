"""
slow coupling to weak fast coupling between WB and Traub Ca. 

phase code, complements trbwb_full.py

Notes:
There are two manually-determined modes:
(1) constant mean synaptic variables. In this case, we choose input parameters such that the frequencies of all oscillators in and between populations are sufficiently identical. To ensure that the numerical integration of H functions is consistent, we use linearly interpolated FI functions and the inverses (inverses exist due to monotonicity of the FI curve in the parameter ranges we use).
(2) slowly varying mean synaptic variables. In this case, we generate the H functions dynamically. The existence of a slowly varying oscillation must be determined manually.

todo:
-allow user to import mean field data.
-run mean field first before running phase model. keep the oscillation existence and limit cycle finder code, but need to have available the raw mean field dynamics as well.

complete:
-let N be defined by number of coordinates in init arrays.
-create generate H function in both cases.

""" 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as mp
import scipy as sp

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sys import stdout

import sys
import os
import getopt

from trbwb_lib import *
from xppcall import xpprun, read_numerics, read_pars
#from trbwb_full import Traub_Wb as tw

np.random.seed(10)

cos = np.cos
sin = np.sin
pi = np.pi
sqrt = np.sqrt
exp = np.exp


greens = ['#00ff04','#0d9800','#144700','#6ff662','#449f29']
off_greens = []

blues = ['#0000FF','#0099FF','#0033FF','#00CCFF','#0066FF']
off_blues = []


class Phase(object):
    """
    run the phase model of N excitatory Traub neurons slowly coupled to N inhibitory Wang-Buszaki neurons. This simulation is independent of the full model
    """
    
    def __init__(self,
                 recompute_beta=False,
                 recompute_h=False,
                 use_last=False,
                 save_last=False,
                 mux=1.,muy=1.,
                 itb_mean=6.793934,iwb_mean=0.3590794,
                 gee=102,gei=117.1,
                 gie=20,gii=11,
                 T=500,dt=.01,
                 eps=.01,
                 integration_N=2**15,
                 sbar=0.05,
                 phs_init_trb=None,
                 phs_init_wb=None,
                 verbose=True,
                 sx0=0,sy0=0):

        """
        sbar: mean frequency at fixed point (exists independent of the existence of slow mean field oscillations)
        mux,muy: synaptic parameters
        itb,iwb,gee,gei,gie,gii: coupling parameters (input current)
        T,dt: simulation time parameters
        eps: epsilon (see the trbwb_full.py or trbwb2.ode to see where this eps shows up)
        N: N=2 total number of neurons in each excitatory or inhibitory population
        integration_N: total number of intervals in Riemann integrals
        phs_init_trb: initial phase values for Traub neurons (from 0 to 1, normalized later based on starting sbarx,sbary)
        phs_init_wb: inital phase values for WB neurons (from 0 to 1, normalized later based on starting sbarx,sbary)
        use_last: use final value of previous sim
        save_last: save final value of curren sim
        """

        self.recompute_h = recompute_h
        self.recompute_beta = recompute_beta
        self.use_last = use_last
        self.save_last = save_last
        self.verbose = verbose
        
        self.gee=gee;self.gei=gei
        self.gie=gie;self.gii=gii
        
        self.itb_mean=itb_mean;self.iwb_mean=iwb_mean
        self.itb = self.itb_mean - sbar*(self.gee - self.gei)
        self.iwb = self.iwb_mean - sbar*(self.gie - self.gii)

        self.sbarx = sbar
        self.sbary = sbar
        self.freqx_base = self.sbarx
        self.freqy_base = self.sbary
        self.Tx_base = 1./self.freqx_base
        self.Ty_base = 1./self.freqy_base

        self.sx0 = sx0
        self.sy0 = sy0
        
        self.integration_N = integration_N

        assert (len(phs_init_trb) == len(phs_init_wb))

        self.N = len(phs_init_trb)

        self.eps = eps
        self.mux = mux
        self.muy = muy

        self.dt = dt
        self.T = T
        self.TN = int(self.T/self.dt)
        self.tau = np.linspace(0,self.T,self.TN)
        self.t = np.linspace(0,self.T/self.eps,self.TN)

        self.savedir = 'savedir/'
        
        self.paramsfile = '_eps='+str(self.eps)+\
                          '_mux='+str(self.mux)+\
                        '_muy='+str(self.muy)+\
                        '_gee='+str(self.gee)+\
                        '_gei='+str(self.gei)+\
                        '_gie='+str(self.gie)+\
                        '_gii='+str(self.gii)+\
                        '_N='+str(self.N)+\
                        '.dat'

        # set plot title to this string to see params quickly
        self.paramstitle = '_eps='+str(self.eps)+\
                          '_mux='+str(self.mux)+\
                        '_muy='+str(self.muy)+\
                        '_gee='+str(self.gee)+\
                        '_gei='+str(self.gei)+\
                        '_gie='+str(self.gie)+\
                        '_gii='+str(self.gii)+\
                        '_N='+str(self.N)


        self.filename_thx = self.savedir+'thx_init_tb.dat' # inits for all theta^x_i
        self.filename_thy = self.savedir+'thy_init_wb.dat' # inits for all theta^y_i
        self.filename_s = self.savedir+'s_mean_init_tbwb.dat' # inits for mean field s

        # phase on [0,1], since generally we don't know the period to start
        self.phs_init_trb = phs_init_trb
        self.phs_init_wb = phs_init_wb
        
        if (self.phs_init_trb != None) and (self.phs_init_wb != None):
            #rint len(self.phs_init_trb),len(self.phs_init_wb)
            assert ((len(self.phs_init_trb) == self.N) and (len(self.phs_init_wb) == self.N))

            # convert phase from [0,1] to [0,T]
            self.phs_init_trb = np.array(self.phs_init_trb)*self.Tx_base
            self.phs_init_wb = np.array(self.phs_init_wb)*self.Ty_base


        self.load_z() # load h functions,iPRC and define domains on [0,2pi]
        self.load_fi() # load frequency-current functions
        self.load_h_const() # load H functions

        # in the phase model we need to run the mean field beforehand
        self.load_inits()
        self.run_mean_field(self.sx0,self.sy0,self.T,self.dt)
        self.load_beta_coeffs()
        self.run_phase()

    def load_inits(self):
        """
        helper function for def run_phase below. load inits.
        """

        # preallocate
        if self.T == 0:
            self.TN = 1
            self.thx = np.zeros((1,self.N))
            self.thy = np.zeros((1,self.N))

            self.perx = self.Tx_base
            self.pery = self.Ty_base

        else:
            self.TN = len(self.tau)
            self.thx = np.zeros((self.TN,self.N))
            self.thy = np.zeros((self.TN,self.N))

        # determine initial value (use last, use manual)
        file_not_found = False
        while True:

            if self.use_last and not(file_not_found):
                # if user asks to use_last and there exists a saved file...
                if os.path.isfile(self.filename_thx) and\
                   os.path.isfile(self.filename_thy) and\
                   os.path.isfile(self.filename_s):

                    # load thx,thy terms
                    self.thx[0,:] = np.loadtxt(self.filename_thx)
                    self.thy[0,:] = np.loadtxt(self.filename_thy)
                    
                    # load mean s terms
                    sv = np.loadtxt(self.filename_s)
                    self.sx0 = sv[0]
                    self.sy0 = sv[1]

                    print 'using inits:'
                    
                    # found inits, so break.
                    break
                
                else:
                    # if the filenames don't exist, update flag.
                    file_not_found = True
                                
            else:
                print 'using manual inits:'

                self.thx[0,:] = self.phs_init_trb
                self.thy[0,:] = self.phs_init_wb

                self.sx0 = self.sx0
                self.sy0 = self.sy0

                #self.sx[0] = self.sx0#1.017038158211262644
                #self.sy[0] = self.sy0#1.000222753692209698

                break

        """
        # display inits for all variables, independent of the above while loop
        for i in range(self.N):
            print 'thx_'+str(i)+'='+str(self.thx[0,i]), '\t thy_'+str(i)+'='+str(self.thy[0,i])
        print 'sx,sy='+str(self.sx[0])+','+str(self.sy[0])
        """
        
    def run_phase(self):
        """
        run the phase model
        """

        print 'Running phase model with T =',self.T

        # initial conditions
        #self.thx[0,:] = self.phs_init_trb
        #self.thy[0,:] = self.phs_init_wb

        # integration loop
        i = np.arange(0,self.N)
        for k in range(self.TN-1):

            if self.verbose and (k%10 == 0):
                frac = 1.*(k+1)/(self.TN-1)
                update_progress(frac) # display integration progress
                
            #    stdout.write("\r  ... simulating phase... %d%%" % int((100.*(k+1)/(self.TN-1))))
            #    stdout.flush()

            sumxx=np.zeros(self.N);sumyy=np.zeros(self.N)
            sumxy=np.zeros(self.N);sumyx=np.zeros(self.N)

            # mean s values
            #sx = self.sxa_data[k]
            #sy = self.sya_data[k]

            sx = self.sxa_fn(self.tau[k]/self.eps)
            sy = self.sya_fn(self.tau[k]/self.eps)

            freqx = self.freqx_fn(self.tau[k]/self.eps)
            freqy = self.freqy_fn(self.tau[k]/self.eps)

            Tx = self.perx_fn(self.tau[k]/self.eps)
            Ty = self.pery_fn(self.tau[k]/self.eps)


            for j in range(self.N):
                
                """
                dom11,tot11 = self.generate_h(sx,sy,choice='xx',return_domain=True)
                dom12,tot12 = self.generate_h(sx,sy,choice='xy',return_domain=True)
                dom21,tot21 = self.generate_h(sx,sy,choice='yx',return_domain=True)
                dom22,tot22 = self.generate_h(sx,sy,choice='yy',return_domain=True)

                inxx = np.mod(self.thx[k,j] - self.thx[k,i],Tx)
                inxy = np.mod(self.thy[k,j] - self.thx[k,i],Ty) # mod y
                inyx = np.mod(self.thx[k,j] - self.thy[k,i],Tx) # mod x
                inyy = np.mod(self.thy[k,j] - self.thy[k,i],Ty)

                sumxx += np.interp(inxx,dom11,tot11)
                sumxy += np.interp(inxy,dom12,tot12)
                sumyx += np.interp(inyx,dom21,tot21)
                sumyy += np.interp(inyy,dom22,tot22)
                """
                

                inxx = np.mod(self.thx[k,j] - self.thx[k,i],self.Tx_base)
                inxy = np.mod(self.thy[k,j] - self.thx[k,i],self.Ty_base)

                inyx = np.mod(self.thx[k,j] - self.thy[k,i],self.Tx_base)
                inyy = np.mod(self.thy[k,j] - self.thy[k,i],self.Ty_base)
                
                sumxx += self.h11(inxx)
                sumxy += self.h12(inxy)

                sumyx += self.h21(inyx)
                sumyy += self.h22(inyy)

            betaxx = (sx-self.sbarx)*self.beta_xx_coeff/self.Tx_base
            betaxy = (sy-self.sbary)*self.beta_xy_coeff/self.Tx_base
            betayx = (sx-self.sbarx)*self.beta_yx_coeff/self.Tx_base
            betayy = (sy-self.sbary)*self.beta_yy_coeff/self.Tx_base

            #print betaxx,betaxy,betayx,betayy

            #print sumyx,sumyy

            # division by self.N is accounted for in definition of self.h
            thxprime = (sumxx + sumxy)/(1.*self.N)
            thyprime = (sumyx + sumyy)/(1.*self.N)

            #if np.abs(thxprime[0]-thyprime[0])<1e-5:
            if (k%400 == 0) and False:
                mp.figure()
                x = np.linspace(0,self.Tx_base,500)
                #mp.plot(x,.5*(self.h11(np.mod(-x,self.Tx_base))-self.h11(x)) + self.h12(np.mod(-x,self.Tx_base)) - self.h12(0) )
                #mp.scatter(np.mod(self.thx[k,1]-self.thx[k,0],self.Tx_base),0)
                #mp.scatter(np.mod(self.thy[k,0]-self.thx[k,0],self.Tx_base),0)
                mp.plot(self.h11(x))
                mp.plot(self.h12(x))
                mp.plot(self.h21(x))
                mp.plot(self.h22(x))

                mp.show()

            #print ,thxprime

            self.thx[k+1,i] = self.thx[k,i] + self.dt*(thxprime + (betaxx + betaxy)/self.eps)
            self.thy[k+1,i] = self.thy[k,i] + self.dt*(thyprime + (betayx + betayy)/self.eps)


            self.sxa[k+1] = self.sxa_fn(self.t[k+1])
            self.sya[k+1] = self.sya_fn(self.t[k+1])
            
            if False:
                print self.sxa_fn(self.t[k+1])

            self.freqx[k+1] = freqx
            self.freqy[k+1] = freqy

            self.perx[k+1] = Tx
            self.pery[k+1] = Ty

        if self.save_last and self.T > 0:
            np.savetxt(self.filename_thx,self.thx[-1,:])
            np.savetxt(self.filename_thy,self.thy[-1,:])
            np.savetxt(self.filename_s,np.array([sx,sy]))

        if self.N == 2:
            self.diff1 = np.mod(self.thx[:,1] - self.thx[:,0]+self.perx_data/2.,self.perx_data)-self.perx_data/2.
            self.diff2 = np.mod(self.thy[:,1] - self.thy[:,0]+self.pery_data/2.,self.pery_data)-self.pery_data/2.
        self.diff3 = np.mod(self.thy[:,0] - self.thx[:,0]+self.pery_data/2.,self.pery_data)-self.pery_data/2.
        #rawdiff = np.mod(self.thy[:,0] - self.thx[:,0],self.pery_data)
        #self.diff3 = np.mod(rawdiff+self.pery_data/2.,self.pery_data)-self.pery_data/2.


        if self.T == 0:
            self.thx_unnormed = np.zeros((1,2))
            self.thy_unnormed = np.zeros((1,2))
        else:
            self.thx_unnormed = np.zeros(np.shape(self.thx))
            self.thy_unnormed = np.zeros(np.shape(self.thy))

            for i in range(self.N):
                self.thx_unnormed[:,i] = self.thx[:,i]*self.perx/self.Tx_base
                self.thy_unnormed[:,i] = self.thy[:,i]*self.pery/self.Ty_base

        print 

    def load_fi(self):
        """
        load FI functions
        """

        # load FI functions
        # load the frequency data
        self.tbfi_data = np.loadtxt('tbfi2.dat')
        self.wbfi_data = np.loadtxt('wbfi.dat')

        # convert to frequency functions
        self.tbfi = interp1d(self.tbfi_data[:,0],self.tbfi_data[:,1])
        self.wbfi = interp1d(self.wbfi_data[:,0],self.wbfi_data[:,1])


    def load_z(self):
        """
        load iPRC functions and define domains
        these data files are called/used in def zx() and def zy().
        """
        print 'loading iPRCs'

        # load/define iPRCs
        
        self.zxdata = np.loadtxt('tb_adj_f='+str(self.sbarx)+'.dat')
        self.zydata = np.loadtxt('wb_adj_f='+str(self.sbary)+'.dat')

        # in *4.dat, the periods are 20 (frequency of 0.05)

        # define interpolation on [0,T]
        self.zxfn = interp1d(self.zxdata[:,0],self.zxdata[:,1])
        self.zyfn = interp1d(self.zydata[:,0],self.zydata[:,1])


    def load_beta_coeffs(self):
        """
        load or compute beta function coefficients

        """
        
        file_not_found = False

        print 'loading or recomputing beta functions'

        self.beta_coeffs_file = self.savedir+"beta_coeffs_tbwb"+self.paramsfile
        #self.b2file = self.savedir+"b2"+self.paramsfile

        while True:

            if self.recompute_beta or file_not_found:

                # get the coefficients
                self.generate_beta_coeffs()
                xx = self.beta_xx_coeff
                xy = self.beta_xy_coeff
                yx = self.beta_yx_coeff
                yy = self.beta_yy_coeff
                
                # save the coefficients
                np.savetxt(self.beta_coeffs_file,np.array([xx,xy,yx,yy]))
                
                break

            elif os.path.isfile(self.beta_coeffs_file):

                xx,xy,yx,yy = np.loadtxt(self.beta_coeffs_file)
                self.beta_xx_coeff = xx
                self.beta_xy_coeff = xy
                self.beta_yx_coeff = yx
                self.beta_yy_coeff = yy

                break

            else:
                file_not_found = True

    def generate_beta_coeffs(self,integration_N=1000):
        """
        choice: xx,xy,yx, or yy.
        integration_N: number of discretization points in integration

        Generate beta function coefficients (the integrals below)

        The equations are
        \begin{align*}
        \beta^{xx}(\tau) &= \frac{[\bar s^x(\tau) - \bar s]}{\ve T^x}\int_0^{T^x}\F^x_{s_x}(\Phi^x(t),\bar s^k)\z^x(t,\bar s^k)dt,\\
        \beta^{xy}(\tau) &= \frac{[\bar s^y(\tau) - \bar s]}{\ve T^x}\int_0^{T^x}\F^x_{s_y}(\Phi^x(t),\bar s^k)\z^x(t,\bar s^k)dt,\\
        \beta^{yx}(\tau) &= \frac{[\bar s^x(\tau) - \bar s]}{\ve T^y}\int_0^{T^y}\F^y_{s_x}(\Phi^y(t),\bar s^k)\z^y(t,\bar s^k)dt,\\
        \beta^{yy}(\tau) &= \frac{[\bar s^y(\tau) - \bar s]}{\ve T^y}\int_0^{T^y}\F^y_{s_y}(\Phi^y(t),\bar s^k)\z^y(t,\bar s^k)dt,\\
        \end{align*}

        recalling that $F^x_{s^x}(x_j) = \pi b (1+\cos(x_j))$, we evaluate at $\Phi^x(t,\bar s^k)$ and integrate against the iPRC $z^x(t)$.
        
        in the code I often use x0 instead of Phi.
        """

        # integration variables
        s_int_x = np.linspace(0,self.Tx_base,integration_N)
        s_int_y = np.linspace(0,self.Ty_base,integration_N)

        # discretization step
        ds_int_x = self.Tx_base/integration_N
        ds_int_y = self.Ty_base/integration_N

        # RHS derivatives evaluated on limit cycle
        Fx_sx = self.gee
        Fx_sy = -self.gei
        Fy_sx = self.gie
        Fy_sy = -self.gii

        # iPRCs
        zx = self.zx(s_int_x)
        zy = self.zy(s_int_y)

        # integrals
        self.beta_xx_coeff = np.sum(Fx_sx*zx)*ds_int_x
        self.beta_xy_coeff = np.sum(Fx_sy*zx)*ds_int_x
        self.beta_yx_coeff = np.sum(Fy_sx*zy)*ds_int_y
        self.beta_yy_coeff = np.sum(Fy_sy*zy)*ds_int_y


    def load_h_const(self):
        """
        load or compute h dynamic function (two varying frequencies)
        """
        print 'loading or recomputing h functions'
        file_not_found = False

        self.h11file = self.savedir+"h11_tbwb"+self.paramsfile
        self.h12file = self.savedir+"h12_tbwb"+self.paramsfile
        self.h21file = self.savedir+"h21_tbwb"+self.paramsfile
        self.h22file = self.savedir+"h22_tbwb"+self.paramsfile


        #dom,h11 = self.generate_h(self.sxa(t),self.sya(t),'11',return_domain=True)
        #fn = interp1d(dom,h)
        #return fn(theta)


        while True:

            if self.recompute_h or file_not_found:
                # get lookup table for sx,sy given mux,muy, get period of oscillation
                # interpolate to be used as function of time

                #trange = np.linspace(0,self.ta[-1],10) # period of slowmod

                # generate h data files

                #def generate_h(self,sx,sy,choice,return_domain=False):

                """
                dom,h11 = self.generate_h_const(self.sbarx,self.sbary,'11',return_domain=True)
                h12 = self.generate_h_const(self.sbarx,self.sbary,'12',return_domain=False)
                h21 = self.generate_h_const(self.sbarx,self.sbary,'21',return_domain=False)
                h22 = self.generate_h_const(self.sbarx,self.sbary,'22',return_domain=False)
                """

                dom,h11 = self.generate_h(self.sbarx,self.sbary,'11',return_domain=True)
                h12 = self.generate_h(self.sbarx,self.sbary,'12',return_domain=False)
                h21 = self.generate_h(self.sbarx,self.sbary,'21',return_domain=False)
                h22 = self.generate_h(self.sbarx,self.sbary,'22',return_domain=False)


                """
                dom,h11 = self.generate_h(self.sbar,self.sbar,'11',return_domain=True)
                h12 = self.generate_h(self.sbar,self.sbar,'12',return_domain=False)
                h21 = self.generate_h(self.sbar,self.sbar,'21',return_domain=False)
                h22 = self.generate_h(self.sbar,self.sbar,'22',return_domain=False)
                """

                h11dat = np.zeros((len(dom),2))
                h12dat = np.zeros((len(dom),2))
                h21dat = np.zeros((len(dom),2))
                h22dat = np.zeros((len(dom),2))

                h11dat[:,0]=dom;h11dat[:,1]=h11
                h12dat[:,0]=dom;h12dat[:,1]=h12
                h21dat[:,0]=dom;h21dat[:,1]=h21
                h22dat[:,0]=dom;h22dat[:,1]=h22

                self.interpdom = dom

                self.h11 = interp1d(dom,h11)
                self.h12 = interp1d(dom,h12)
                self.h21 = interp1d(dom,h21)
                self.h22 = interp1d(dom,h22)

                print 'saving h fns'
                np.savetxt(self.h11file,h11dat)
                np.savetxt(self.h12file,h12dat)
                np.savetxt(self.h21file,h21dat)
                np.savetxt(self.h22file,h22dat)
                
                break

            elif os.path.isfile(self.h11file) and os.path.isfile(self.h12file) and\
                 os.path.isfile(self.h21file) and os.path.isfile(self.h22file):

                h11dat = np.loadtxt(self.h11file)
                h12dat = np.loadtxt(self.h12file)
                h21dat = np.loadtxt(self.h21file)
                h22dat = np.loadtxt(self.h22file)

                dom=h11dat[:,0];h11=h11dat[:,1]
                dom=h12dat[:,0];h12=h12dat[:,1]
                dom=h21dat[:,0];h21=h21dat[:,1]
                dom=h22dat[:,0];h22=h22dat[:,1]
                
                #print np.shape(dom),np.shape(h11)
                
                self.interpdom = dom
                self.h11 = interp1d(dom,h11)
                self.h12 = interp1d(dom,h12)
                self.h21 = interp1d(dom,h21)
                self.h22 = interp1d(dom,h22)                 

                break

            else:
                file_not_found = True



            
    def run_mean_field(self,sx0,sy0,T,dt,no_eps=False):
        """
        run the mean field model
        """

        print 'Running mean field'

        TN = int(T/dt)
        t = np.linspace(0,T,TN)


        # we have the self.sxa function to retreive the mean field value
        # but we need to record the sx values as well.
        self.sxa = np.zeros(TN) # corresponding sx values used.
        self.sya = np.zeros(TN) # corresponding sy values used.

        self.freqx = np.zeros(TN)
        self.freqy = np.zeros(TN)
        
        self.perx = np.zeros(TN)
        self.pery = np.zeros(TN)


        # if T == 0, skip running some stuff
        if self.T == 0:
            self.tau = np.array([0,1])
            
            self.sxa_data = np.array([self.sbarx,self.sbarx])
            self.sya_data = np.array([self.sbary,self.sbary])

            self.freqx_data = np.array([self.sbarx,self.sbarx])
            self.freqy_data = np.array([self.sbarx,self.sbary])

            self.perx_data = 1./self.freqx_data
            self.pery_data = 1./self.freqy_data

            self.sxa_fn = interp1d(self.tau,self.sxa_data)
            self.sya_fn = interp1d(self.tau,self.sya_data)
            
            self.freqx_fn = interp1d(self.tau,self.freqx_data)
            self.freqy_fn = interp1d(self.tau,self.freqy_data)

            self.perx_fn = interp1d(self.tau,self.perx_data)
            self.pery_fn = interp1d(self.tau,self.pery_data)

                    
        else:
            #print 'itb,gee,gei',self.itb,self.gee,self.gei
            #print 'iwb,gie,gii',self.iwb,self.gie,self.gii
            #print 'eps,mux,muy',self.eps,self.mux,self.muy

            # run
            #ta = np.linspace(0,500,int(500/.05))
            sol = odeint(self.mean_field_rhs,[sx0,sy0],self.tau)

            # slow variable data (rename for convenience)
            self.sxa_data = sol[:,0]
            self.sya_data = sol[:,1]

            # create mean slow variable functions
            self.sxa_fn = interp1d(self.tau/self.eps,self.sxa_data)
            self.sya_fn = interp1d(self.tau/self.eps,self.sya_data)

            # frequency data and functions
            Iexc_data = self.itb+self.gee*self.sxa_data-self.gei*self.sya_data
            Iinh_data = self.iwb+self.gie*self.sxa_data-self.gii*self.sya_data

            # freq/period data
            self.freqx_data = self.tbfi(Iexc_data)
            self.freqy_data = self.wbfi(Iinh_data)

            self.perx_data = 1./self.freqx_data
            self.pery_data = 1./self.freqy_data

            # freq fns
            self.freqx_fn = interp1d(self.tau/self.eps,self.freqx_data)
            self.freqy_fn = interp1d(self.tau/self.eps,self.freqy_data)

            # period fns
            self.perx_fn = interp1d(self.tau/self.eps,self.perx_data)
            self.pery_fn = interp1d(self.tau/self.eps,self.pery_data)
        
            return sol
        
    def mean_field_rhs(self,y,t):
        """
        rhs mean field model for trb-wb system
        y: 2 state variables
        """
        stb = y[0]
        swb = y[1]

        #print self.gee,self.gei,self.gie,self.gii

        Iexc = self.itb+self.gee*stb-self.gei*swb
        Iinh = self.iwb+self.gie*stb-self.gii*swb

        tbfi_prime = (-stb+self.tbfi(Iexc))/self.mux
        wbfi_prime = (-swb+self.wbfi(Iinh))/self.muy

        return np.array([tbfi_prime,wbfi_prime])
        

    def fx(self,t,f):
        """
        spiking term in phase model
        """
        p = 1./f
        return (np.mod(1.-t/p,1.)-1./2)/self.mux

    def fy(self,t,f):
        """
        spiking term in phase model
        """
        p = 1./f
        return (np.mod(1.-t/p,1.)-1./2)/self.muy

    def zx(self,t):
        return self.zxfn(np.mod(t,self.Tx_base))

    def zy(self,t):
        return self.zyfn(np.mod(t,self.Ty_base))

    def generate_h(self,stb,swb,choice,return_domain=False):
        """
        given the two slow variables, generate the h function dependent on the frequencies.

        sx,sy: slow variables
        choice: which h function to generate
        return_domain: returns the domain of integration (useful for generating interpolating functions)
        """


        
        Iexc = self.itb+self.gee*stb-self.gei*swb
        Iinh = self.iwb+self.gie*stb-self.gii*swb

        freqx=self.tbfi(Iexc);freqy=self.wbfi(Iinh)

        

        
        freq = self.freqx_base
        rx=1;ry=1
        px=1;py=1

        if (choice == '11') or (choice == 'xx'):
            fn = self.fx # spiking term
            z = self.zx
            coeff = self.gee
            
        elif (choice == '12') or (choice == 'xy'):
            fn = self.fy
            z = self.zx
            coeff = -self.gei

        elif (choice == '21') or (choice == 'yx'):
            fn = self.fx
            z = self.zy
            coeff = self.gie

        elif (choice == '22') or (choice == 'yy'):
            fn = self.fy
            z = self.zy
            coeff = -self.gii


            
        per = 1./freq
        sint = np.linspace(0,per,self.integration_N) # integration variable
        dx = (sint[-1]-sint[0])/len(sint)

        # cross-correlation using fft
        f1 = coeff*z(sint)
        f2 = fn(sint,freq)

        f1_fft = np.fft.fft(f1).conj()
        f2_fft = np.fft.fft(f2)
        
        tot = np.real(np.fft.ifft(f1_fft*f2_fft))*dx/self.Tx_base

        
        if return_domain:
            return sint,tot
        return tot

    def plot(self,choice,nskip=1,saveplot=False):

        if choice == 'h':
            fig = plt.figure()
            datdom = np.linspace(0,self.tau[-1]/2.,5)
            datx = self.sxa_fn(datdom)
            daty = self.sya_fn(datdom)

            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

            ax1.set_title("hxx")
            ax2.set_title("hxy")
            ax3.set_title("hyx")
            ax4.set_title("hyy")


            for i in range(len(datdom)):
                c = str(.75*i/len(datdom))
                
                domxx,hxx = self.generate_h(self.sbarx,self.sbary,'xx',return_domain=True)
                domxy,hxy = self.generate_h(self.sbarx,self.sbary,'xy',return_domain=True)
                domyx,hyx = self.generate_h(self.sbarx,self.sbary,'yx',return_domain=True)
                domyy,hyy = self.generate_h(self.sbarx,self.sbary,'yy',return_domain=True)

                ax1.plot(domxx,hxx,label='hxx',color=c)
                ax2.plot(domxy,hxy,label='hxy',color=c)
                ax3.plot(domyx,hyx,label='hyx',color=c)
                ax4.plot(domyy,hyy,label='hyy',color=c)

            if saveplot:
                datxx = np.zeros((len(domxx),2))
                datxy = np.zeros((len(domxy),2))
                datyx = np.zeros((len(domyx),2))
                datyy = np.zeros((len(domyy),2))

                datxx[:,0]=domxx;datxx[:,1]=hxx                
                datxy[:,0]=domxy;datxy[:,1]=hxy
                datyx[:,0]=domyx;datyx[:,1]=hyx
                datyy[:,0]=domyy;datyy[:,1]=hyy

                np.savetxt(self.savedir+'tbwb_hxx_fixed'+self.paramsfile,datxx)
                np.savetxt(self.savedir+'tbwb_hyy_fixed'+self.paramsfile,datyy)
                np.savetxt(self.savedir+'tbwb_hxy_fixed'+self.paramsfile,datxy)
                np.savetxt(self.savedir+'tbwb_hyx_fixed'+self.paramsfile,datyx)

        elif choice == 'h_integrand':
            fig = plt.figure()
            ax = fig.add_subplot(111)

            dom = np.linspace(0,self.Tx_base,100)
            ax.plot(dom,self.zx(dom)*self.fx(dom,self.sbarx))
                
        elif choice == 'hodd':
            fig = plt.figure()
            datdom = np.linspace(0,self.tau[-1]/2.,5)
            datx = self.sxa_fn(datdom)
            daty = self.sya_fn(datdom)


            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

            ax1.set_title("hxx")
            ax2.set_title("hxy")
            ax3.set_title("hyx")
            ax4.set_title("hyy")


            for i in range(len(datdom)):
                c = str(.75*i/len(datdom))
                
                domxx,hxx = self.generate_h(datx[i],daty[i],'xx',return_domain=True)
                domxy,hxy = self.generate_h(datx[i],daty[i],'xy',return_domain=True)
                domyx,hyx = self.generate_h(datx[i],daty[i],'yx',return_domain=True)
                domyy,hyy = self.generate_h(datx[i],daty[i],'yy',return_domain=True)

                hxxo = hxx - np.flipud(hxx)
                hxyo = hxy - np.flipud(hxy)
                hyxo = hyx - np.flipud(hyx)
                hyyo = hyy - np.flipud(hyy)
                
                ax1.plot(domxx,hxxo,label='hxx',color=c)
                ax2.plot(domxy,hxyo,label='hxy',color=c)
                ax3.plot(domyx,hyxo,label='hyx',color=c)
                ax4.plot(domyy,hyyo,label='hyy',color=c)

                

        elif choice == 'z':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = np.linspace(0,self.Tx_base,1000)
            y = np.linspace(0,self.Ty_base,1000)
            ax.plot(x,self.zx(x),label='zx')
            ax.plot(y,self.zy(y),label='zy')
            ax.legend()

        elif choice == 'f':
            fig = plt.figure()
            ax = fig.add_subplot(111)

            dom = np.linspace(0,self.Tx_base,100)
            ax.plot(dom,self.fx(dom,self.sbarx),label='qx')
            ax.plot(dom,self.fy(dom,self.sbary),label='qy')

            ax.legend()

        elif choice == 'sol':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.t,np.mod(self.thx[:,0],self.per)+np.random.rand()/5.,label=r'$\theta_1^x$')
            ax.plot(self.t,np.mod(self.thy[:,0],self.per)+np.random.rand()/5.,label=r'$\theta_1^y$')
            if self.N == 2:
                ax.plot(self.t,np.mod(self.thx[:,1],self.per)+np.random.rand()/5.,label=r'$\theta_2^x$')
                ax.plot(self.t,np.mod(self.thy[:,1],self.per)+np.random.rand()/5.,label=r'$\theta_2^y$')
            ax.legend()

        elif choice == 'diff':
            fig = plt.figure()
            # plot phase differences
            ax = fig.add_subplot(111)

            ax.set_title('Theory Phase Difference')
            
            if self.N == 2:
                ax.plot(self.t,self.diff1,label=r'$\theta_2^x-\theta_1^x$',lw=2)
                ax.plot(self.t,self.diff2,label=r'$\theta_2^y-\theta_1^y$',lw=2)

            ax.plot(self.t,self.diff3,label=r'$\theta_1^y-\theta_1^x$',lw=2)

            #ax.set_ylim(-self.per/2.,self.per/2.)
            ax.plot(self.t,self.perx_data/2.,ls='-',color='gray')
            ax.plot(self.t,-self.perx_data/2.,ls='-',color='gray',label='perx')
            ax.plot(self.t,self.pery_data/2.,ls='--',color='gray')
            ax.plot(self.t,-self.pery_data/2.,ls='--',color='gray',label='pery')
            
            ax.legend()

            ax.set_xlim(self.t[0],self.t[-1])
            ax.set_ylim(-np.amax(self.pery_data)/2-1,np.amax(self.pery_data)/2+1)

        elif choice == 'thdiff-unnormed':
            fig = plt.figure()
            ax = fig.add_subplot(111)#fig.add_subplot(121)
            ax.set_title('thji - thjk, raw (theory)')

            #per = 1.
            sxvals = self.sxa_fn(self.t)
            syvals = self.sya_fn(self.t)

            freqx_arr = self.freqx_fn(self.t)
            freqy_arr = self.freqy_fn(self.t)

            perx_arr = 1./freqx_arr
            pery_arr = 1./freqy_arr
            
            for i in range(self.N-1):
                diff1 = (self.thx[:,i+1]-self.thx[:,0])*perx_arr/self.Tx_base
                diff2 = (self.thy[:,i+1]-self.thy[:,0])*pery_arr/self.Ty_base

                diff1 = np.mod(diff1+perx_arr/2.,perx_arr)-perx_arr/2.
                diff2 = np.mod(diff2+pery_arr/2.,pery_arr)-pery_arr/2.

                #diff1 = np.mod(diff1+self.Tx_base/2.,self.Tx_base)-self.Tx_base/2.
                #diff2 = np.mod(diff2+self.Ty_base/2.,self.Ty_base)-self.Ty_base/2.
                
                diff1 = diff1[::nskip]
                diff2 = diff2[::nskip]

                #ax.scatter(self.t[::nskip],diff1,color='blue', edgecolor='none',label='thx'+str(i+2)+'-thx1',alpha=(.5*i+.5)/self.N)
                #ax.scatter(self.t[::nskip],diff2,color='green',edgecolor='none',label='thy'+str(i+2)+'-thy1',alpha=(.5*i+.5)/self.N)
                ax.scatter(self.t[::nskip],diff1,color=blues[i], edgecolor='none',label='thx'+str(i+2)+'-thx1')
                ax.scatter(self.t[::nskip],diff2,color=greens[i],edgecolor='none',label='thy'+str(i+2)+'-thy1')

            
            ax.plot(self.t,-perx_arr/2.,color='gray')
            ax.plot(self.t,-pery_arr/2.,color='gray',label='pery',ls='--')
            ax.plot(self.t,perx_arr/2.,color='gray',label='perx')
            ax.plot(self.t,pery_arr/2.,color='gray',ls='--')

            diff3 = (self.thy[:,0]-self.thx[:,0])*(pery_arr/self.Ty_base)
            diff3 = np.mod(diff3+pery_arr/2.,pery_arr)-pery_arr/2.
            diff3 = diff3[::nskip]

            ax.scatter(self.t[::nskip],diff3,edgecolor='none',color='red',label='thy1-thx1')
            ax.set_xlim(self.t[0],self.t[-1])
            
            ax.legend()



        elif choice == '1':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('hyx(-x)-hxy(x)+O(1)')

            x,hyx = self.generate_h(.05,.05,'yx',return_domain=True)
            hyy = self.generate_h(.05,.05,'yy')
            hxx = self.generate_h(.05,.05,'xx')
            hxy = self.generate_h(.05,.05,'xy')
            
            ax.plot(x,np.flipud(hyx)+hyy[0]-hxx[0]-hxy,label='')


        elif choice == 'slowmod-space':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('Slowmod lc space (mean field)')
            ax.plot(self.sxa_data,self.sya_data)
            ax.scatter(self.sbarx,self.sbary,color='black',label='fixed point')
            ax.legend()

        elif choice == 'slowmod-t':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('Slowmod lc t')
            ax.plot(self.t,self.sxa_data,label='sx')
            ax.plot(self.t,self.sya_data,label='sy')
            ax.legend()
            
        
        elif choice == 'freq':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('frequencies over time')
            ax.plot(self.t,self.freqx_data,label='freqx')
            ax.plot(self.t,self.freqy_data,label='freqy')
            ax.legend()

        elif choice == 'currents':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('currents over time')
            Iexc = self.itb+self.gee*self.sxa_data-self.gei*self.sya_data
            Iinh = self.iwb+self.gie*self.sxa_data-self.gii*self.sya_data
            ax.plot(self.t,Iexc,label='Iexc')
            ax.plot(self.t,Iinh,label='Iinh')
            ax.legend()

        elif choice == 'fi-curves':
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # load FI functions
            # load the frequency data
            #self.tbfi_data = np.loadtxt('tbfi2.dat')
            #self.wbfi_data = np.loadtxt('wbfi.dat')

            # convert to frequency functions
            #self.tbfi = interp1d(self.tbfi_data[:,0],self.tbfi_data[:,1])
            #self.wbfi = interp1d(self.wbfi_data[:,0],self.wbfi_data[:,1])

            
            # convert to frequency functions
            print self.tbfi_data[:,0][0],self.tbfi_data[:,0][-1]
            ax.plot(self.tbfi_data[:,0],self.tbfi(self.tbfi_data[:,0]),label='tb')
            ax.plot(self.wbfi_data[:,0],self.wbfi(self.wbfi_data[:,0]),label='wb')
            
            ax.legend()


        else:
            raise ValueError('Invalid plot choice',str(choice))

def main():
    print 'use trbwb_master.py'


if __name__ == '__main__':
    main()
