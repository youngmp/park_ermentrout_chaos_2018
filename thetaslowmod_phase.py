

"""

network of 2N theta nuerons. N are excitatory, N are inhibitory. Both populations receive self excitation and self inhibition.

Notes:
T=5000,a=1.,b=1.,c=1.,mux=3.,muy=1. synchrony. random initial cond

T=5000,a=1.0,b=1.,c=1.,mux=1.,muy=3. phase locking with initial conditions close to sync

T=5000,a=1.0,b=3.,c=1.,mux=1.,muy=3. phase locking with initial conditions close to sync

T=5000,a=1.0,b=3.,c=1.,mux=3.,muy=1. phase locking? inits close to sync

T=5000,a=1.,b=3.,c=1.,mux=5.,muy=1. sync with inits close to sync.

T=5000,a=1.,b=1.,c=1.,mux=1.,muy=3. x1,x2 sync, y1,y2 sync, but populations phase locked i.e. phi^z != 0 with inits x1=x2=.1,y1=y2=0. With random initial conditions, get some other phase locking. In fact, x1,y2 synchronize, while the others do not.

T=5000,a=0.01,b=1.,c=1.,mux=1.,muy=3. oscillating average in sx,sy, synchrony in xj,yj

T=5000,a=1.0,b=1.,c=3.,mux=1.,muy=3. phase locking with random 

T=5000,a=1.0,b=1.,c=3.,mux=1.,muy=3. phase locking with random initial conditions close to synchrony

TODO:
-Think about OTT-ANTENSON (maybe it won't work)
-try larger N


Done:
-implement save last use last feature.
-local interpolation of H functions
-rework code so that the mean field is run first.

"""

from __future__ import division
from sys import stdout
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.signal import argrelextrema


import numpy as np
import matplotlib.pylab as mp
import matplotlib.pyplot as plt
import sys
import os
import copy

from trbwb_lib import *
from xppcall import xpprun, read_numerics#, read_pars, read_inits
#from thetaslowmod_full import Theta
from thetaslowmod_lib import *

np.random.seed(0)

cos = np.cos
sin = np.sin
pi = np.pi
sqrt = np.sqrt

greens = ['#00ff04','#0d9800','#144700','#6ff662','#449f29']
off_greens = []

blues = ['#0000FF','#0099FF','#0033FF','#00CCFF','#0066FF']
off_blues = []

class Phase(object):
    def __init__(self,run_full=False,
                 run_phase=True,
                 recompute_h=False,

                 recompute_beta=False,
                 recompute_slow_lc=False,
                 save_last=False,

                 use_last=False,
                 verbose=True,
                 use_slowmod_lc=True,

                 force_const_sk=False,
                 T=100,
                 a1=0.,b1=0.,c1=0.,

                 a2=0.,b2=0.,c2=0.,
                 eps=.005,dt=.05,
                 mux=1.,muy=1.,

                 t0=0.,full_T=1000.,
                 integration_N=1024,
                 slowmod_lc_tol=1e-4,

                 thxin=None,thyin=None,
                 sx0=None,sy0=None,
                 
                 heterogeneous_pop=False
    ):

        """
        run_phase: run phase model
        recompute_h: recompute interaction function h
        recompute_beta: reocmpute terms dependent on dsx/dtau, dsy/dtau

        force_fixed_sk: if user knows that the means should be fixed, force the sims to use fixed mean.

        old params
        a1=.5,b1=7.,c1=6.5,
        a2=1.1,b2=25.,c2=25.1,
        """        

        #self.N = N
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1

        self.a2 = a2
        self.b2 = b2
        self.c2 = c2

        self.eps = eps
        self.mux = mux
        self.muy = muy
        self.taux = self.mux/self.eps
        self.tauy = self.muy/self.eps

        self.savedir = 'savedir/'
        self.slowmod_lc_tol = slowmod_lc_tol

        self.filename_thx = self.savedir+'thx_init.dat' # inits for all theta^x_i
        self.filename_thy = self.savedir+'thy_init.dat' # inits for all theta^y_i
        self.filename_s = self.savedir+'s_mean_init.dat' # inits for mean field s
        
        if ((thxin is None) or (thyin is None)
            or (sx0 is None) or (sy0 is None)):
            raise ValueError('Enter inital conditions')

        assert(len(thxin) == len(thyin))
        self.N = len(thxin)

        if (self.N != 2) and (self.N != 1):
            print "warning: full theta sim uses N=2 or N=1. Other values are not yet implemented."


        self.thxin = thxin
        self.thyin = thyin
        self.sx0 = sx0
        self.sy0 = sy0

        self.paramsfile = '_a1='+str(a1)+\
                      '_b1='+str(b1)+\
                        '_c1='+str(c1)+\
                        '_a2='+str(a2)+\
                        '_b2='+str(b2)+\
                        '_c2='+str(c2)+\
                        '_eps='+str(eps)+\
                        '_mux='+str(mux)+\
                        '_muy='+str(muy)+'.dat'

        self.paramstitle = ", a1="+str(self.a1)+\
                           ", b1="+str(self.b1)+\
                        ", c1="+str(self.c1)+\
                        ", a2="+str(self.a2)+\
                        ", b2="+str(self.b2)+\
                        ",\nc2="+str(self.c2)+\
                        ", mux="+str(self.mux)+\
                        ", muy="+str(self.muy)+\
                        ", eps="+str(self.eps)

        self.verbose = verbose
        self.recompute_beta = recompute_beta
        self.recompute_h = recompute_h

        self.run_full = run_full
        self.run_phase = run_phase
        self.recompute_slow_lc = recompute_slow_lc
        self.force_const_sk = force_const_sk

        self.use_slowmod_lc = use_slowmod_lc
        self.save_last = save_last
        self.use_last = use_last
        self.integration_N = integration_N

        self.heterogeneous_pop = heterogeneous_pop

        self.T = T
        self.dt = dt
        self.t = np.linspace(0,self.T,int(self.T/self.dt))
        self.TN = int(self.T/self.dt)

        # get fixed point of mean field
        self.sbarx,self.sbary = get_sbar(self.a1,self.b1,self.c1,
                                         self.a2,self.b2,self.c2)

        # get base frequency/period
        self.freqx_base = np.sqrt(self.a1+self.b1*self.sbarx-self.c1*self.sbary)
        self.freqy_base = np.sqrt(self.a2+self.b2*self.sbarx-self.c2*self.sbary)
        
        self.Tx_base = 1./self.freqx_base
        self.Ty_base = 1./self.freqy_base

        #print self.Tx_base,self.Ty_base
        
        # check if oscillations exist
        self.slow_osc_exist = slow_osc(self.sbarx,self.sbary,
                                       self.a1,self.b1,self.c1,
                                       self.a2,self.b2,self.c2,
                                       self.mux,self.muy)
        

        self.load_beta_coeffs() # load beta functions

        if self.use_slowmod_lc and self.slow_osc_exist:
            print 'testafsdfasdfasdfasdfasfd'
            self.load_slowmod_lc() # get mean field limit cycle            

        # run mean field. run mean field must be here because it depends on def load_inits
        self.run_mean_field(self.sx0,self.sy0,self.T,self.dt)

        # generate h function approximations as a function of mean field using Fourier coeffs
        self.load_h_const()

        # generate heterogeneous terms (B_i, B_j)
        if self.heterogeneous_pop:
            self.load_het_const()

        # run phase model if T > 0. Otherwise define dummy variables to save time/space.
        if self.run_phase and (self.T > 0):

            # load the initial conditions for the phase model taking into account user preferences
            self.load_inits() 

            # run phase model
            self.run_phase_sim()
    
        # if phase model not set to run, generate empty arrays
        elif self.T > 0:
            self.thx = np.zeros((self.TN,self.N))
            self.thy = np.zeros((self.TN,self.N))
            
        # if T == 0, use inits as single-time solutions. This case is for testing/generating other functions outside of simulating the phase model.
        elif self.T == 0:
            self.thx = self.thxin
            self.thy = self.thyin
            
            # we have the self.sxa function to retreive the mean field value
            # but we need to record the sx values as well.

            #self.run_mean_field(self.sx0,self.sy0)

    def run_mean_field(self,sx0,sy0,T,dt,no_eps=False):
        """
        run mean field sim.

        sx0,sy0: initial values of mean field
        T: total time
        dt: time step

        we keep these inputs explicit so we can call/use this function outside of the class.
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

        # if T == 0, skip running stuff
        if (T == 0):
        
            # define interpolation functions
            #self.ta = np.array([0,1])
            self.t = np.array([0,1])

            #print len(self.t),len(np.zeros(len(self.t))+sx0)
            #print np.shape(self.t),np.shape(np.zeros(len(self.t))+sx0)
            self.sxa_fn = interp1d(self.t,np.zeros(len(self.t))+sx0)
            self.sya_fn = interp1d(self.t,np.zeros(len(self.t))+sy0)
            
            fx0 = sqrt(self.a1+self.b1*sx0-self.c1*sy0)
            fy0 = sqrt(self.a2+self.b2*sx0-self.c2*sy0)

            self.freqx_fn = interp1d(self.t,np.zeros(len(self.t))+fx0)
            self.freqy_fn = interp1d(self.t,np.zeros(len(self.t))+fy0)
            
            self.perx_fn = interp1d(self.t,[1./self.freqx_fn(0),1./self.freqx_fn(self.t[-1])])
            self.pery_fn = interp1d(self.t,[1./self.freqy_fn(0),1./self.freqy_fn(self.t[-1])])
            
            self.dsxa_fn = interp1d(self.t,[0,0])
            self.dsya_fn = interp1d(self.t,[0,0])

            self.sx = sx0
            self.sy = sy0
            
        # if T > 0 run mean field
        else:
            #print 'itb,gee,gei',self.itb,self.gee,self.gei
            #print 'iwb,gie,gii',self.iwb,self.gie,self.gii
            #print 'eps,mux,muy',self.eps,self.mux,self.muy

            # run
            #ta = np.linspace(0,500,int(500/.05))
            if no_eps:
                sol = odeint(average_no_eps,[sx0,sy0],t,args=(self.a1,self.b1,self.c1,
                                                       self.a2,self.b2,self.c2,
                                                       self.mux,self.muy))
                
            else:
                sol = odeint(average,[sx0,sy0],t,args=(self.a1,self.b1,self.c1,
                                                       self.a2,self.b2,self.c2,
                                                       self.mux,self.muy,self.eps))

            # slow variable data (rename for convenience)
            self.sxa_data = sol[:,0]
            self.sya_data = sol[:,1]


            #self.sx = sx0
            #self.sy = sy0


            # create mean slow variable functions
            self.sxa_fn = interp1d(t,self.sxa_data)
            self.sya_fn = interp1d(t,self.sya_data)

            # freq/period data
            self.freqx_data = np.sqrt(self.a1+self.b1*self.sxa_data-self.c1*self.sya_data)
            self.freqy_data = np.sqrt(self.a2+self.b2*self.sxa_data-self.c2*self.sya_data)

            self.perx_data = 1./self.freqx_data
            self.pery_data = 1./self.freqy_data

            # freq fns
            self.freqx_fn = interp1d(t,self.freqx_data)
            self.freqy_fn = interp1d(t,self.freqy_data)

            # period fns
            self.perx_fn = interp1d(t,self.perx_data)
            self.pery_fn = interp1d(t,self.pery_data)
        
            return sol



    def load_inits(self):
        """
        helper function for 'run_phase_sim'
        load initial conditions for main sim below.
        """


        # solution arrays
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
                #print self.thx

                # (integrator thinks input phase is between 0 and 1)
                
                self.thx[0,:] = self.thxin
                self.thy[0,:] = self.thyin

                """
                if self.sx0 != self.sbarx:
                    print 'warning, manual sx0 not the same as sbarx. error =',str(np.abs(self.sbarx-self.sx0))
                if self.sy0 != self.sbary:
                    print 'warning, manual sy0 not the same as sbary. error =',str(np.abs(self.sbary-self.sy0))
                """

                break

        self.sxa[0] = self.sx0#1.017038158211262644
        self.sya[0] = self.sy0#1.000222753692209698

        # display inits for all variables, independent of the above while loop
        for i in range(self.N):
            print 'thx_'+str(i)+'='+str(self.thx[0,i]), '\t thy_'+str(i)+'='+str(self.thy[0,i])
        print 'sx,sy='+str(self.sxa[0])+','+str(self.sya[0])


    def run_phase_sim(self):
        """
        run phase model
        """
        
        print 'Running phase_sim'

        # neuron index array
        i = np.arange(0,self.N)

        for k in range(0,self.TN-1):

            if self.verbose:
                if k%100 == 0:
                    frac = 1.*(k+1)/(self.TN-1)
                    update_progress(frac) # display integration progress

            #stdout.write("\r  ... simulating phase... %d%%" % int((100.*(k+1)/(self.TN-1))))
            #stdout.flush()

            sumxx=np.zeros(self.N);sumyy=np.zeros(self.N)
            sumxy=np.zeros(self.N);sumyx=np.zeros(self.N)

            # mean s values
            #sx = self.sxa_fn(np.mod(self.t[k],self.ta[-1]))
            #sy = self.sya_fn(np.mod(self.t[k],self.ta[-1]))

            sx = self.sxa_fn(self.t[k])
            sy = self.sya_fn(self.t[k])

            # derivatives
            #dsx = self.dsxa_fn(np.mod(self.t[k],self.ta[-1]))
            #dsy = self.dsya_fn(np.mod(self.t[k],self.ta[-1]))

            freqx = np.sqrt(self.a1+self.b1*sx-self.c1*sy)
            freqy = np.sqrt(self.a2+self.b2*sx-self.c2*sy)

            Tx = 1./freqx
            Ty = 1./freqy

            for j in range(self.N):

                inxx = np.mod(self.thx[k,j] - self.thx[k,i],self.Tx_base)
                inxy = np.mod(self.thy[k,j] - self.thx[k,i],self.Ty_base) # mod y

                inyx = np.mod(self.thx[k,j] - self.thy[k,i],self.Tx_base) # mod x
                inyy = np.mod(self.thy[k,j] - self.thy[k,i],self.Ty_base)

                sumxx += self.h11(inxx)
                sumxy += self.h12(inxy)

                sumyx += self.h21(inyx)
                sumyy += self.h22(inyy)

            # beta terms
            betaxx = (sx-self.sbarx)*self.beta_xx_coeff/(self.Tx_base)
            betaxy = (sy-self.sbary)*self.beta_xy_coeff/(self.Tx_base)
            betayx = (sx-self.sbarx)*self.beta_yx_coeff/(self.Ty_base)
            betayy = (sy-self.sbary)*self.beta_yy_coeff/(self.Ty_base)

            #print betaxx,betaxy,self.dt*k#,self.eps*self.Gx

            # right hand side
            thxprime = (sumxx + sumxy)/self.N 
            thyprime = (sumyx + sumyy)/self.N 

            if self.heterogeneous_pop:
                self.thx[k+1,i] = self.thx[k,i] + self.dt*(self.eps*thxprime + betaxx + betaxy + self.eps*self.Gx)
                self.thy[k+1,i] = self.thy[k,i] + self.dt*(self.eps*thyprime + betayx + betayy + self.eps*self.Gy)

            else:
                self.thx[k+1,i] = self.thx[k,i] + self.dt*(self.eps*thxprime + betaxx + betaxy)
                self.thy[k+1,i] = self.thy[k,i] + self.dt*(self.eps*thyprime + betayx + betayy)

            self.sxa[k+1] = self.sxa_fn(self.t[k+1])
            self.sya[k+1] = self.sya_fn(self.t[k+1])
            
            if False:
                print self.sxa_fn(self.t[k+1])

            self.freqx[k+1] = freqx
            self.freqy[k+1] = freqy

            self.perx[k+1] = Tx
            self.pery[k+1] = Ty


        self.thx_unnormed = np.zeros(np.shape(self.thx))
        self.thy_unnormed = np.zeros(np.shape(self.thy))

        for i in range(self.N):
            self.thx_unnormed[:,i] = self.thx[:,i]*self.perx/self.Tx_base
            self.thy_unnormed[:,i] = self.thy[:,i]*self.pery/self.Ty_base
            
        # save solutions
        if self.save_last:
            np.savetxt(self.filename_thx,self.thx[-1,:])
            np.savetxt(self.filename_thy,self.thy[-1,:])
            np.savetxt(self.filename_s,np.array([self.sxa[-1],self.sya[-1]]))


        print
        # end run_sim_phase

    def load_slowmod_lc(self):
        """
        load single period (slow time) oscillator
        """

        print 'loading slowmod lc'
        # 3 cols, t, sx, sy. data for  one period.
        self.filename = self.savedir+"slowmod_lc"+self.paramsfile
        #self.slowmod_lc_init_name = self.savedir+"slowmod_lc"+self.paramsfile
        self.filename_grad = self.savedir+"slowmod_lc_grad"+self.paramsfile
        self.filename_freq = self.savedir+"fast_freq_for1_slow_period"+self.paramsfile

        
        
        if os.path.isfile(self.filename) and os.path.isfile(self.filename_grad) and\
           os.path.isfile(self.filename_freq) and not(self.recompute_slow_lc):
            data = np.loadtxt(self.filename)
            data_grad = np.loadtxt(self.filename_grad)
            data_freq = np.loadtxt(self.filename_freq)

            self.ta = data[:,0]
            self.taua = self.ta*self.eps
            self.sxa_fn = interp1d(self.ta,data[:,1])
            self.sya_fn = interp1d(self.ta,data[:,2])

            self.dsxa_fn = interp1d(self.ta,data_grad[:,1])
            self.dsya_fn = interp1d(self.ta,data_grad[:,2])

            self.freqx_fn = interp1d(self.ta,data_freq[:,1])
            self.freqy_fn = interp1d(self.ta,data_freq[:,2])

            self.perx_fn = interp1d(self.ta,1./data_freq[:,1])
            self.pery_fn = interp1d(self.ta,1./data_freq[:,2])

        else:
            #print 'estsdffffffffffffffffffffffffffffffffffffffffffffffff'
            self.generate_slowmod_lc(tol=self.slowmod_lc_tol)

        print 'loaded slowmod lc'

    def generate_slowmod_lc(self,tol=1e-4,max_time=50000,
                            plot_values=True,maxiter=20,dt=.01):
        """
        get slowly varying limit cycle
        tol: tolerance value. use the difference between
            the last solution value and the solution value one (slow) period before. 
        max_time: total integration time per iteration. this value needs to be large enough
            that we catch multiple crossings of a given poincare section.
        """

        # check if slowmod lc exists.
        # if not, start sim at mean + small pert
        # get approximate period using slice trick.

        print 'Finding slow limit cycle with max_time='+str(max_time)+' and maxiter='+str(maxiter) +', dt='+str(dt)
        init = [self.sbarx+.01,self.sbary+.01]
        ta = np.linspace(0,max_time,int(max_time)/dt)
        diff = 10
        
        iterv = 0 # counter for number of times the simulation has re-run
        
        if self.slow_osc_exist: # if slowmod lc exists
            
            while (diff > tol) and iterv < maxiter:
                
                sol = odeint(average,init,ta,args=(self.a1,self.b1,self.c1,
                                                   self.a2,self.b2,self.c2,
                                                   self.mux,self.muy,self.eps))
                if False:
                    plt.figure()
                    plt.plot(sol[:,0])
                    plt.show()
                # after one iteration
                
                sxa = sol[:,0]
                sya = sol[:,1]
                
                # check if LC exists
                # if the magnitude of the starting solution is large relative to the ending
                # solution, then there might be a limit cycle.
                #print np.amax(sxa[:-1-int(100/.05)]-np.mean(sxa[:-1-int(100/.05)]))
                if np.amax(sxa[:-1-int(100/dt)]-np.mean(sxa[:-1-int(100/dt)])) < tol:
                    no_lc_flag = True
                    intxn_idx = np.zeros(len(sxa))
                    intxn_idx[:-10] = 1
                    
                    break

                # find intersection of poincare section
                # poincare section here is defined as max sx and sbar
                
                # get index of max sx
                max_sx_idx = np.argmax(sxa) 
                
                # get corresponding value of sy -- this is the poincare section
                # actually, just use sbary.
                intxn_sy = self.sbary#self.sya[max_sx_idx]-.04
                
                # find where sy crosses section (there should be 2 values)
                intxn_idx = (sya[1:]<=intxn_sy)*(sya[:-1]>intxn_sy) 
                
                # restrict the 2 values to 1 (find where the intersection is also max of sx)
                intxn_idx *= (sxa[1:]<self.sbarx)
                
                print len(sxa),len(intxn_idx)
                print sxa[1:][intxn_idx]
                print np.shape(sxa[1:][intxn_idx])

                diff1 = np.abs(sxa[1:][intxn_idx][-1] - sxa[1:][intxn_idx][-2])
                diff2 = np.abs(sya[1:][intxn_idx][-1] - sya[1:][intxn_idx][-2])
                diff = diff1+diff2

                print 'Iteration '+str(iterv)+'. Slow lc diff,tol=',diff,',',tol
                
                init = sol[-1,:]
                # get intersection, check tol, break if pass.
                iterv += 1

            print 'Note: if the slow limit cycle appears discontinuous in sx-sy space, re-run the slow limit cycle code (recompute_slow_lc=True) with a smaller tolerance (tol)'

            # slow limit cycle init and period
            lc_init = [sxa[1:][intxn_idx][-1],sya[1:][intxn_idx][-1]]
            period = ta[1:][intxn_idx][-1] - ta[1:][intxn_idx][-2]
            
            print 'lcinit,per',lc_init,period
            self.ta = np.linspace(0,period,int(period/dt))
            self.taua = self.ta*self.eps
            sol = odeint(average,lc_init,self.ta,args=(self.a1,self.b1,self.c1,
                                                       self.a2,self.b2,self.c2,
                                                       self.mux,self.muy,self.eps))

            if False:
                #plt.show()
                plt.plot(sol[:,0])
                plt.show()
            
            sxa = sol[:,0]
            sya = sol[:,1]
            
            savedata = np.zeros((len(self.ta),3))
            savedata[:,0] = self.ta
            savedata[:,1] = sxa
            savedata[:,2] = sya
            # interpolate average sx,sy
            self.sxa_1per_fn = interp1d(self.ta,sxa)
            self.sya_1per_fn = interp1d(self.ta,sya)
            
            # get mean frequency of fast system
            f1temp = np.sqrt(self.a1+self.b1*self.sxa_1per_fn(self.ta)-self.c1*self.sya_1per_fn(self.ta))
            f2temp = np.sqrt(self.a2+self.b2*self.sxa_1per_fn(self.ta)-self.c2*self.sya_1per_fn(self.ta))
            
            savedata_freq = np.zeros((len(self.ta),3))
            savedata_freq[:,0] = self.ta
            savedata_freq[:,1] = f1temp
            savedata_freq[:,2] = f2temp
            
            # interpolate average freqx,freqy
            self.freqx_1per_fn = interp1d(self.ta,f1temp)
            self.freqy_1per_fn = interp1d(self.ta,f2temp)
            
            
            # get derivatives
            #dt = np.gradient(self.ta)

            dsxa = np.gradient(self.sxa_1per_fn(self.ta),dt)
            dsya = np.gradient(self.sya_1per_fn(self.ta),dt)

            if False:
                mp.figure()
                mp.plot(self.ta,dsxa)
                mp.plot(self.ta,dsya)

                mp.figure()
                mp.plot(self.ta,self.sxa_1per_fn(self.ta))
                mp.plot(self.ta,self.sya_1per_fn(self.ta))

                mp.show()

            savedata_grad = np.zeros((len(self.ta),3))
            savedata_grad[:,0] = self.ta
            savedata_grad[:,1] = dsxa
            savedata_grad[:,2] = dsya
            
            self.dsxa_1per_fn = interp1d(self.ta,dsxa)
            self.dsya_1per_fn = interp1d(self.ta,dsya)
            
            # save all data
            np.savetxt(self.filename,savedata)
            np.savetxt(self.filename_grad,savedata_grad)
            np.savetxt(self.filename_freq,savedata_freq)


    def load_beta_coeffs(self):
        """
        load or compute beta function coefficients

        """
        
        file_not_found = False

        print 'loading or recomputing beta functions'

        self.beta_coeffs_file = self.savedir+"beta_coeffs"+self.paramsfile
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


        print self.beta_xx_coeff,self.beta_xy_coeff,self.beta_yx_coeff,self.beta_yy_coeff

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
        Fx_sx = pi*self.b1*(1+cos(x0(s_int_x,self.freqx_base)))
        Fx_sy = -pi*self.c1*(1+cos(x0(s_int_x,self.freqx_base)))
        Fy_sx = pi*self.b2*(1+cos(x0(s_int_y,self.freqy_base)))
        Fy_sy = -pi*self.c2*(1+cos(x0(s_int_y,self.freqy_base)))

        # iPRCs
        zx = z(s_int_x,self.freqx_base)
        zy = z(s_int_y,self.freqy_base)

        # integrals
        self.beta_xx_coeff = np.sum(Fx_sx*zx)*ds_int_x
        self.beta_xy_coeff = np.sum(Fx_sy*zx)*ds_int_x
        self.beta_yx_coeff = np.sum(Fy_sx*zy)*ds_int_y
        self.beta_yy_coeff = np.sum(Fy_sy*zy)*ds_int_y



    def generate_h(self,sx,sy,choice,return_domain=False):
        """
        given the two slow variables, generate h function dependent on frequencies and time
        sx,sy: the slow variables
        """

        Iexc = self.a1+self.b1*sx-self.c1*sy
        Iinh = self.a2+self.b2*sx-self.c2*sy
        freqx=sqrt(Iexc);freqy=sqrt(Iinh)
        
        decimal_truncation = 100000

        ratioy=self.freqy_base/freqy;ratiox=self.freqx_base/freqx

        ry = int(round(ratioy*decimal_truncation)) # r^yT
        py = decimal_truncation # p^y T^y

        rx = int(round(ratiox*decimal_truncation)) # r^x T
        px = decimal_truncation # p^x T^x
        # NEW
        if (choice == '11') or (choice == 'xx'):
            freq = freqx
            freq_base = self.freqx_base
            fn = self.fx # spiking term
            normal = px/(rx*self.Tx_base)
            #normal = freq*px/rx
            
            coeff = self.b1*pi

            
        elif (choice == '22') or (choice == 'yy'):
            freq = freqy
            freq_base = self.freqy_base
            fn = self.fy # spiking term
            normal = py/(ry*self.Ty_base)
            #normal = freq*py/ry
            coeff = -self.c2*pi

        else:
            raise ValueError('Invalid choice:'+str(choice))
            
        per = 1./freq
        #print per
        sint = np.linspace(0,per,self.integration_N) # integration variable
        dx = per/self.integration_N
        
        # cross-correlation using fft        
        f1 = coeff*(1.+cos(x0(sint,freq_base)))*z(sint,freq_base)
        #f1 = coeff*(1.+cos(x0(sint,freq)))*z(sint,freq)
        f2 = fn(sint,freq)
        
        f1_fft = np.fft.fft(f1).conj()
        f2_fft = np.fft.fft(f2)
        
        tot = np.real(np.fft.ifft(f1_fft*f2_fft))*normal*dx

        
        if return_domain:
            return sint,tot

        return tot


        

    def generate_h_inhom(self,sx,sy,choice,return_domain=True,brute=False):
        """
        generate the h functions for the coupling from x to y and y to x.
        sx,sy: mean slow variables
        """
        freqx=sqrt(self.a1+self.b1*sx-self.c1*sy);freqy=sqrt(self.a2+self.b2*sx-self.c2*sy)        
        decimal_truncation = 100000

        ratioy = self.freqy_base/freqy

        ry = int(round(ratioy*decimal_truncation)) # r^yT
        py = decimal_truncation # p^y T^y
        
        ratiox = self.freqx_base/freqx
        rx = int(round(ratiox*decimal_truncation)) # r^x T
        px = decimal_truncation # p^x T^x


        # i.e., for every ry rotations in fixed mean neurons, there are py rotations in neuron y.
        # for every rx rotations in fixed mean neurons, there are px rotations in neuron x
        # for every rx*ry rotations in fixed mean neurons, there are rx*py rotations in neuron y and ry*px rotations in neuron x

        #print 'p,q=',p,q
        #print 'long period ratios',q*Tx/(p*Ty)

        if (choice == '12') or (choice == 'xy'):
            freq = freqx
            freq_base = self.freqx_base
            fn = self.fy
            normal = 1.*py/(ry*self.Tx_base)#q*Tx/p
            #normal = freq*py/ry
            coeff = -self.c1*pi

        elif (choice == '21') or (choice == 'yx'):
            freq = freqy
            freq_base = self.freqy_base
            fn = self.fx
            normal = 1.*px/(rx*self.Ty_base)
            #normal = freq*px/rx
            coeff = self.b2*pi
            
        else:
            raise ValueError('Invalid choice:'+str(choice))

        per = 1./freq
        sint = np.linspace(0,per,self.integration_N) # integration variable
        dx = per/self.integration_N

        # cross-correlation using fft        
        #f1 = coeff*(1.+cos(x0(sint,freq_base)))*z(sint,freq_base)
        f1 = coeff*(1.+cos(x0(sint,freq_base)))*z(sint,freq_base)
        f2 = fn(sint,freq)
        
        f1_fft = np.fft.fft(f1).conj()
        f2_fft = np.fft.fft(f2)
        
        tot = np.real(np.fft.ifft(f1_fft*f2_fft))*normal*dx
        
        if return_domain:
            return sint,tot
        return tot


    def G(self,x,choice):
        """
        heterogeneous function
        choice: Gx0,Gx1,Gy0, or Gy1
        # python unif random vars on [-1,1] with random seed=0. [0.09762701 0.43037873 0.20552675 0.08976637]

        """
        if choice == 'x0':
            return pi*0.09762701*(1+cos(x))
        elif choice == 'x1':
            return pi*0.43037873*(1+cos(x))

        elif choice == 'y0':
            return pi*0.20552675*(1+cos(x))
        elif choice == 'y1':
            return pi*0.08976637*(1+cos(x))

        else:
            raise ValueError('invalid choice for heterogeneous function G: choice='+str(choice))
        

    def load_het_const(self):
        """
        heterogeneous terms added to vector field.
        hard-coded at the moment. see theta2_het_pop.ode
        in the future make it flexible.

        return Gx as 1x2 or 2x1 array. Gx[0] is the first het fn, Gx[1] is the second. same for Gy
        
        """
        
        print 'loading or recomputing vector field heterogeneities'
        file_not_found = False        
        
        per = self.Tx_base
        domain = np.linspace(0,per,self.integration_N)
        dx = per/self.integration_N
        
        
        self.Gx0 = np.sum(self.G(x0(domain,self.freqx_base),'x0')*z(domain,self.freqx_base))*dx/per
        self.Gx1 = np.sum(self.G(x0(domain,self.freqx_base),'x1')*z(domain,self.freqx_base))*dx/per
        self.Gy0 = np.sum(self.G(x0(domain,self.freqx_base),'y0')*z(domain,self.freqx_base))*dx/per
        self.Gy1 = np.sum(self.G(x0(domain,self.freqx_base),'y1')*z(domain,self.freqx_base))*dx/per

        self.Gx = np.array([self.Gx0,self.Gx1])
        self.Gy = np.array([self.Gy0,self.Gy1])
        
        return self.Gx,self.Gy                
                
        
        

    def load_h_const(self):
        """
        load or compute h dynamic function (two varying frequencies)
        """
        print 'loading or recomputing h functions'
        file_not_found = False

        self.h11file = self.savedir+"h11"+self.paramsfile
        self.h12file = self.savedir+"h12"+self.paramsfile
        self.h21file = self.savedir+"h21"+self.paramsfile
        self.h22file = self.savedir+"h22"+self.paramsfile


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
                h12 = self.generate_h_inhom(self.sbarx,self.sbary,'12',return_domain=False)
                h21 = self.generate_h_inhom(self.sbarx,self.sbary,'21',return_domain=False)
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


    def generate_h_const(self,sx,sy,choice,N=None,return_domain=False):
        """
        given the two slow variables, generate h function dependent on frequencies and time
        sx,sy: the slow variables
        """

        freq = sqrt(self.a1 + self.b1*sx -self.c1*sy)

        if N == None:
            N = self.integration_N

        if (choice == '11') or (choice == 'xx'):
            ff=self.fx
            p=self.b1*pi

        elif (choice == '12') or (choice == 'xy'):
            ff=self.fy
            p=-self.c1*pi
        
        elif (choice == '21') or (choice == 'yx'):
            ff=self.fx
            p=self.b2*pi

        elif (choice == '22') or (choice == 'yy'):
            ff=self.fy
            p=-self.c2*pi
        else:
            raise ValueError('Invalid choice:'+str(choice))


        per = 1./freq
        s = np.linspace(0,per,N) # integration variable

        tot = 0
        for i in range(N):
            tot += z(s[i],freq) * \
                   (1.+cos(x0(s[i],freq))) * \
                   ff((s[i]+s),freq)

        dx = s[-1]/N
        coeff = per*per/(2*pi)

        tot *= p*dx*coeff#/self.normalization()[0]
        if return_domain:
            return s,tot

        return tot

    def generate_h_const_coeffs(self,trange,choice,deg=2.,plot_data=False):
        """
        trange: period of slow lc in fast time
        
        f: requency range
        """

        if deg != 2:
            raise ValueError('ERROR: H function coefficient approximation not implemented for deg != 2')
        
        coeffs = np.zeros((len(trange),3))

        for i in range(len(trange)):
            sx = self.sxa(trange[i])
            sy = self.sya(trange[i])
            d = self.generate_h(sx,sy,choice)
            #ax.plot(dom,d,color=str((1.*i/len(parvals))))
            
            cf = self.get_sin_coeffs(d)

            coeffs[i,0]=cf[0]
            coeffs[i,1]=cf[1]
            coeffs[i,2]=cf[2]

        # get poly fit
        d1,d2,d3=np.polyfit(trange,coeffs[:,0],deg)
        e1,e2,e3=np.polyfit(trange,coeffs[:,1],deg)
        f1,f2,f3=np.polyfit(trange,coeffs[:,2],deg)

        if plot_data:
            mp.figure()
            mp.title(choice)

            
            mp.plot(trange,coeffs[:,0])
            mp.plot(trange,coeffs[:,1])

            mp.plot(trange,e1*trange**2+e2*trange+e3)

            mp.plot(trange,coeffs[:,2])

            mp.plot(trange,f1*trange**2+f2*trange+f3)

            mp.figure()
            mp.plot(trange,self.sxa(trange))
            mp.show()            


        return d1,d2,d3,e1,e2,e3,f1,f2,f3



        
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

    def plot(self,choice='h_inhom',brute=False,nskip=1,saveplot=False):
        #fig = plt.figure()
        #ax = plt.subplot(111)
        
        #tt = np.linspace(0,self.ta[-1]/4.,10) # 1/4 period of slowmod
        
        #sx = self.sxa(tt)
        #sy = self.sya(tt)
    
        
        if choice == 'h_inhom':

            datdom = np.linspace(0,self.t [-1]/2.,5)
            datx = self.sxa_fn(datdom)
            daty = self.sya_fn(datdom)
            
            fig = plt.figure()
            
            ax1 = fig.add_subplot(121)
            ax1.set_title("hxy")
            
            ax2 = fig.add_subplot(122)
            ax2.set_title("hyx")
            

            for i in range(len(datdom)):
                c = str(.75*i/len(datdom))
                x = np.linspace(0,1,100)


                domxy,totxy = self.generate_h_inhom(datx[i],daty[i],choice='xy')
                ax1.plot(domxy,totxy,color=c,picker=5)
                #ax1.set_xlim(dom[0],dom[-1])

                #ax1.plot(x,self.h12(x),color=c,picker=5)

                domyx,totyx = self.generate_h_inhom(datx[i],daty[i],choice='yx')
                ax2.plot(domyx,totyx,color=c,picker=5)
                #ax1.set_xlim(dom[0],dom[-1])

                #ax1.plot(x,self.h21(x),color=c,picker=5)

            plt.suptitle("raw generated inhom. h funs"+self.paramstitle+'sxa='+str(datx[i])+',sya='+str(daty[i]))
            plt.subplots_adjust(top=0.85)

            if self.N == 1:
                fig = plt.figure()
                
                ax = fig.add_subplot(111)
                ax.set_title("hyx(-phi)-hxy(phi)")

                domxy,totxy = self.generate_h_inhom(self.sxa_fn(0),self.sya_fn(0),choice='xy',brute=brute)
                domyx,totyx = self.generate_h_inhom(self.sxa_fn(0),self.sya_fn(0),choice='yx',brute=brute)

                ax.plot(domxy,np.flipud(totyx)-totxy)

            if saveplot:
                datxy = np.zeros((len(domxy),2))
                datyx = np.zeros((len(domyx),2))
                
                datxy[:,0]=domxy;datxy[:,1]=totxy
                datyx[:,0]=domyx;datyx[:,1]=totyx
                
                np.savetxt(self.savedir+'hxy_fixed'+self.paramsfile,datxy)
                np.savetxt(self.savedir+'hyx_fixed'+self.paramsfile,datyx)

        elif choice == 'h_hom':
            datdom = np.linspace(0,self.t[-1]/2.,5)
            datx = self.sxa_fn(datdom)
            daty = self.sya_fn(datdom)
            
            fig = plt.figure()
            
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.set_title("hxx")
            ax2.set_title("hyy")

            for i in range(len(datdom)):
                c = str(.75*i/len(datdom))

                x = np.linspace(0,1,100)
                domxx,totxx = self.generate_h(datx[i],daty[i],choice='xx',return_domain=True)
                ax1.plot(domxx,totxx,color=c,picker=5)

                domyy,totyy = self.generate_h(datx[i],daty[i],choice='yy',return_domain=True)
                ax2.plot(domyy,totyy,color=c,picker=5)

            plt.suptitle("raw generated hom. h funs"+self.paramstitle+'sxa='+str(datx[i])+',sya='+str(daty[i]))
            plt.subplots_adjust(top=0.85)

            if saveplot:
                datxx = np.zeros((len(domxx),2))
                datyy = np.zeros((len(domyy),2))
                
                datxx[:,0]=domxx;datxx[:,1]=totxx
                datyy[:,0]=domyy;datyy[:,1]=totyy

                np.savetxt(self.savedir+'hxx_fixed'+self.paramsfile,datxx)
                np.savetxt(self.savedir+'hyy_fixed'+self.paramsfile,datyy)

        elif choice == 'h':
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.set_title("hxx")
            
            ax2 = fig.add_subplot(122)
            ax2.set_title("hyy")

            fig2 = plt.figure()
            ax1b = fig2.add_subplot(121)
            ax1b.set_title("hxy")
            
            ax2b = fig2.add_subplot(122)
            ax2b.set_title("hyx")

            """
            dom,h11 = self.generate_h(self.sbarx,self.sbary,'11',return_domain=True)
            h12 = self.generate_h_inhom(self.sbarx,self.sbary,'12',return_domain=False)
            h21 = self.generate_h_inhom(self.sbarx,self.sbary,'21',return_domain=False)
            h22 = self.generate_h(self.sbarx,self.sbary,'22',return_domain=False)
            """

            ax1.plot(self.interpdom,self.h11(self.interpdom))
            ax2.plot(self.interpdom,self.h22(self.interpdom))

            ax1b.plot(self.interpdom,self.h12(self.interpdom))
            ax2b.plot(self.interpdom,self.h21(self.interpdom))


        elif choice == 'z':

            
            s = np.linspace(0,1./self.sbarx,100) # integration variable

            ax1 = plt.subplot(121)
            ax1.set_title("z1")

            ax1.plot(s,z(s,freq1[0]))

            
            ax2 = plt.subplot(122)
            ax2.set_title("z2")

            ax2.plot(s,z(s,freq2[0]))

        
            ax.legend()

        elif choice == 'slowmod_lc-space':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('oscillations in slow variables (thoery)')
            ax.plot(self.sxa_fn(self.t)[:-10],self.sya_fn(self.t)[:-10],color='black')

        elif choice == 'slowmod_lc-t-1per':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('oscillations in slow variables over time (1 period, theory)')
            ax.plot(self.ta*self.eps,self.sxa_fn(self.ta),label='sx')
            ax.plot(self.ta*self.eps,self.sya_fn(self.ta),label='sy')

            
        elif choice == 'slowmod_lc-t':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('oscillations in slow variables over time (all time, theory)')
            ax.plot(self.t,self.sxa_fn(self.t),label='sx')
            ax.plot(self.t,self.sya_fn(self.t),label='sy')

        elif choice == 'lc':
            s = np.linspace(0,1./self.sbarx,200) # integration variable

            ax1 = plt.subplot(121)
            ax1.set_title("lc1")

            ax1.plot(s,x0(s,freq1[0]))
            
            ax2 = plt.subplot(122)
            ax2.set_title("lc2")

            ax2.plot(s,x0(s,freq2[0]))


        elif choice == 'dlc':
            s = np.linspace(0,2*pi,200) # integration variable

            z = np.linspace(0,1,len(sx))
            color = plt.cm.Greys(z)

            plt.suptitle("Limit Cycle Derivative"+self.paramstitle)

            ax1 = plt.subplot(121)
            ax1.set_title("lc1-deriv-sx(black),sy(red)")

            for i in range(len(freq1)):
                ivar1 = s/(2*pi*freq1[i])
                ax1.plot(s,x0(ivar1,freq1[i],self.b1,self.c1,d='sx'),color='black')
                ax1.plot(s,x0(ivar1,freq1[i],self.b1,self.c1,d='sy'),color='red')
            
            ax2 = plt.subplot(122)
            ax2.set_title("lc2-deriv-sx(black),sy(red)")

            for i in range(len(freq2)):
                ivar2 = s/(2*pi*freq2[i])
                ax2.plot(s,x0(ivar2,freq2[i],self.b2,self.c2,d='sx'),color='black')
                ax2.plot(s,x0(ivar2,freq2[i],self.b2,self.c2,d='sy'),color='red')
            

        elif choice == 'fx':
            freq = 0.966953980291
            ax.set_title("fx"+self.paramstitle+"_freq="+str(freq))

            s = np.linspace(0,100./freq,10000) # integration variable

            ax.plot(s,self.fx(s,freq))

        elif choice == 'fy':
            freq = 0.865447860937
            ax.set_title("fy"+self.paramstitle+"_freq="+str(freq))

            s = np.linspace(0,90./freq,10000) # integration variable

            ax.plot(s,self.fy(s,freq))

            #ax.set_title("fy"+self.paramstitle)
            #ax.plot(self.phi,self.fy(self.phi))
        
        elif choice == 'dhx':
            ax.set_title("dhx"+self.paramstitle)
            ax.plot(self.phi,np.gradient(self.hx(self.phi)))
        
        elif choice == 'dhy':
            ax.set_title("dhy"+self.paramstitle)
            ax.plot(self.phi,np.gradient(self.hy(self.phi)))

        elif choice == 'th-t':
            ax = plt.subplot(121)#fig.add_subplot(121)
            ax.set_title("thx")
            for i in range(self.N):
                ax.plot(self.t,np.mod(self.thx[:,i]+pi,2*pi)-pi,label='thx'+str(i))

            ax.set_ylim(-pi,pi)

            ax2 = plt.subplot(122)#fig.add_subplot(122)
            ax2.set_title("thy")
            #ax2.plot(self.t,np.mod(self.thy[:,0]+pi,2*pi)-pi,label='thy1')
            #ax2.plot(self.t,np.mod(self.thy[:,1]+pi,2*pi)-pi,label='thy2')
            for i in range(self.N):
                ax2.plot(self.t,np.mod(self.thy[:,i]+pi,2*pi)-pi,label='thy'+str(i))

            ax2.set_ylim(-pi,pi)

            plt.suptitle("th-t"+self.paramstitle)
            plt.subplots_adjust(top=0.85)
            
            ax.legend()
            ax2.legend()

        elif choice == 'th-plane':
            z = np.linspace(0,1,len(self.thx[:,0]))
            color = plt.cm.Greys(z)


            ax = plt.subplot(121)#fig.add_subplot(121)
            ax.set_title('thx1 v thy1')
            ax.scatter(self.thx[:,0],self.thy[:,0],color=color)
            #ax.set_title("thx1 v thx2")
            #ax.scatter(self.thx[:,0],self.thx[:,1],color=color)

            #ax.set_xlim(0,self.period)
            #ax.set_ylim(0,self.period)

            ax2 = plt.subplot(122)#fig.add_subplot(122)
            #ax2.set_title("thy1 vs thy2")
            #ax2.scatter(self.thy[:,0],self.thy[:,1],color=color)

            #ax2.set_xlim(0,self.period)
            #ax2.set_ylim(0,self.period)

            plt.suptitle("th-plane"+self.paramstitle)
            plt.subplots_adjust(top=.85)


        elif choice == 'thdiff-raw':
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
                diff1 = self.thx[:,i+1]-self.thx[:,0]
                diff2 = self.thy[:,i+1]-self.thy[:,0]

                #diff1 = np.mod(diff1+perx_arr/2.,perx_arr)-perx_arr/2.
                #diff2 = np.mod(diff2+pery_arr/2.,pery_arr)-pery_arr/2.

                diff1 = np.mod(diff1+self.Tx_base/2.,self.Tx_base)-self.Tx_base/2.
                diff2 = np.mod(diff2+self.Ty_base/2.,self.Ty_base)-self.Ty_base/2.
                
                diff1 = diff1[::nskip]
                diff2 = diff2[::nskip]

                #ax.scatter(self.t[::nskip],diff1,color='blue', edgecolor='none',label='thx'+str(i+2)+'-thx1',alpha=(.5*i+.5)/self.N)
                #ax.scatter(self.t[::nskip],diff2,color='green',edgecolor='none',label='thy'+str(i+2)+'-thy1',alpha=(.5*i+.5)/self.N)
                ax.scatter(self.t[::nskip],diff1,color=blues[i], edgecolor='none',label='thx'+str(i+2)+'-thx1')
                ax.scatter(self.t[::nskip],diff2,color=greens[i],edgecolor='none',label='thy'+str(i+2)+'-thy1')

            
            #ax.plot(self.t,-perx_arr/2.,color='gray')
            #ax.plot(self.t,-pery_arr/2.,color='gray',label='pery',ls='--')
            #ax.plot(self.t,perx_arr/2.,color='gray',label='perx')
            #ax.plot(self.t,pery_arr/2.,color='gray',ls='--')

            diff3 = np.mod(self.thy[:,0]-self.thx[:,0]+self.Ty_base/2.,self.Ty_base)-self.Ty_base/2.
            #diff3 = np.mod(self.thy[:,0]-self.thx[:,0]+pery_arr/2.,pery_arr)-pery_arr/2.
            diff3 = diff3[::nskip]

            ax.scatter(self.t[::nskip],diff3,edgecolor='none',color='red',label='thy1-thx1')
            ax.set_xlim(self.t[0],self.t[-1])
            
            ax.legend()


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



        elif choice == 'thdiffdiff-raw':
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
            


            #diff3 = np.mod(self.thy[:,0]-self.thx[:,0]+pery_arr/2.,pery_arr)-pery_arr/2.
            diff3 = np.mod(self.thy[:,0]-self.thx[:,0]+self.Ty_base/2.,self.Ty_base)-self.Ty_base/2.
            diff3 = diff3[::nskip]

            #ax.scatter(self.t[::nskip],diff3,edgecolor='none',color='red',label='thy1-thx1')
            #ax.set_xlim(self.t[0],self.t[-1])


            for i in range(self.N-1):
                diff1 = self.thx[:,i+1]-self.thx[:,0]
                diff2 = self.thy[:,i+1]-self.thy[:,0]

                diffdiff1 = diff1-diff3
                diffdiff2 = diff2-diff3

                diffdiff1 = np.mod(diffdiff1+perx_arr/2.,perx_arr)-perx_arr/2.
                diffdiff2 = np.mod(diffdiff2+pery_arr/2.,pery_arr)-pery_arr/2.
                
                diffdiff1 = diffdiff1[::nskip]
                diffdiff2 = diffdiff2[::nskip]

                #ax.scatter(self.t[::nskip],diff1,color='blue', edgecolor='none',label='thx'+str(i+2)+'-thx1',alpha=(.5*i+.5)/self.N)
                #ax.scatter(self.t[::nskip],diff2,color='green',edgecolor='none',label='thy'+str(i+2)+'-thy1',alpha=(.5*i+.5)/self.N)
                #ax.scatter(self.t[::nskip],diffdiff1,color=blues[i], edgecolor='none',label='thx'+str(i+2)+'-thx1')
                ax.scatter(diffdiff1,diffdiff2,color=blues[i],edgecolor='none',label='diffdiff1,diffdiff2')

            #ax.plot(self.t,-perx_arr/2.,color='gray')
            #ax.plot(self.t,-pery_arr/2.,color='gray',label='pery',ls='--')
            #ax.plot(self.t,perx_arr/2.,color='gray',label='perx')
            #ax.plot(self.t,pery_arr/2.,color='gray',ls='--')
            
            ax.legend()


        elif choice == 'thdiff-normed':
            fig = plt.figure()
            ax = fig.add_subplot(111)#fig.add_subplot(121)
            ax.set_title('thji - thjk, normalized (theory)')

            #per = 1.
            sxvals = self.sxa_fn(np.mod(self.t,self.ta[-1]))
            syvals = self.sya_fn(np.mod(self.t,self.ta[-1]))

            freqx_arr = self.freqx_fn(np.mod(self.t,self.ta[-1]))
            freqy_arr = self.freqy_fn(np.mod(self.t,self.ta[-1]))

            perx_arr = 1./freqx_arr
            pery_arr = 1./freqy_arr
            
            for i in range(self.N-1):
                diff1 = self.thx[:,i+1]-self.thx[:,0]
                diff2 = self.thy[:,i+1]-self.thy[:,0]

                diff1 = np.mod(diff1+perx_arr/2.,perx_arr)-perx_arr/2.
                diff2 = np.mod(diff2+pery_arr/2.,pery_arr)-pery_arr/2.

                diff1 = diff1[::nskip]
                diff2 = diff2[::nskip]

                ax.scatter(self.t,diff1/perx_arr,color='blue', edgecolor='none',label='thx'+str(i+2)+'-thx1')
                ax.scatter(self.t,diff2/pery_arr,color='green',edgecolor='none',label='thy'+str(i+2)+'-thy1')

            ax.plot(self.t,self.t*0-1/2.,color='gray')
            ax.plot(self.t,self.t*0+1/2.,color='gray',label='perx')
            
            ax.plot(self.t,self.t*0-1/2.,color='gray',label='pery',ls='--')
            ax.plot(self.t,self.t*0+1/2.,color='gray',ls='--')
            
            if (self.sbarx == self.sbary) and not(self.slow_osc_exist):
                diff3 = np.mod(self.phy_normed[:,0]-self.phx_normed[:,0]+perx_arr/2.,perx_arr)-perx_arr/2.
                diff3 /= perx_arr
                diff3 = diff3[::nskip]
                
                ax.scatter(t,diff3,edgecolor='none',color='red',label='thy1-thx1')

            ax.set_xlim(self.t[0],self.t[-1])
            
            ax.legend()


        elif choice == 'n1rhs':
            ax = plt.subplot(111)#fig.add_subplot(121)
            ax.set_title('N=1, rhs')
            
            ax.plot(self.phi,
                    self.h(-self.phi,'21')+self.h(0,'22')
                    -self.h(0,'11')-self.h(self.phi,'12'))
            #ax.plot(self.phi,self.h(-self.phi,'21'))
            #ax.plot(self.phi,self.h(np.zeros(len(self.phi)),'22'))
            #ax.plot(self.phi,-self.h(np.zeros(len(self.phi)),'11'))
            #ax.plot(self.phi,-self.h(self.phi,'12'))


        elif choice == 'normalization':
            ax = plt.subplot(111)
            ax.set_title("normalization condition per parval (should be 1 for all frequency values)")

            normal1,normal2 = self.normalization()
            
            ax.plot(normal1)
            ax.plot(normal2)

        elif choice == 'norm-no-int':
            ax = plt.subplot(121)
            #ax.set_title()
            

            t,normal1,normal2 = self.norm_no_int()

            #z = np.linspace(0,1,len(normal1[:,0]))
            #color = plt.cm.Greys(z)


            ax.set_title("LC 1 normalization")
            for k in range(len(normal1[:,0])):
                color = 1.*k/len(normal1[:,0])
                ax.plot(t,normal1[k,:],color=str(color))
            
            ax2 = plt.subplot(122)
            ax2.set_title("LC 2 normalization")
            for k in range(len(normal1[:,0])):
                color = 1.*k/len(normal2[:,0])
                ax2.plot(t,normal2[k,:],color=str(color))


            plt.suptitle("normalization condition over time, with parm values in grayscale (should be 1 for all time values)")

        elif choice == '1/adj':
            ax = plt.subplot(111)
            ax.plot(self.phi,1/z(self.sbarx*self.phi/(2*pi),self.sbarx))
        
        elif choice == 'dlc/dt':
            ax = plt.subplot(111)

        elif choice == 'beta':
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            sxvals = self.sxa_fn(self.ta)
            syvals = self.sya_fn(self.ta)
            
            ax1.set_title('beta1, beta2 for thetax vars')
            ax1.plot(self.beta(sxvals,syvals,d='sx',choice='x'))
            ax1.plot(self.beta(sxvals,syvals,d='sy',choice='x'))



            ax2.set_title('beta1, beta2 for thetay vars')
            ax2.plot(self.beta(sxvals,syvals,d='sx',choice='y'))
            ax2.plot(self.beta(sxvals,syvals,d='sy',choice='y'))




        elif choice == 'beta_full':
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            sxvals = self.sxa_fn(np.mod(self.t,self.ta[-1]))
            syvals = self.sya_fn(np.mod(self.t,self.ta[-1]))


            dsx = self.dsxa(np.mod(self.t,self.ta[-1]))
            dsy = self.dsya(np.mod(self.t,self.ta[-1]))


            # beta derivatives
            betax1 = -dsx*self.beta(sxvals,syvals,d='sx',choice='x')/self.eps
            betax2 = -dsy*self.beta(sxvals,syvals,d='sy',choice='x')/self.eps

            betay1 = -dsx*self.beta(sxvals,syvals,d='sx',choice='y')/self.eps
            betay2 = -dsy*self.beta(sxvals,syvals,d='sy',choice='y')/self.eps

            #print len(self.t),len(betax1)
            ax1.set_title('beta1, beta2 for thetax vars in rhs')
            ax1.plot(self.t,betax1+betax2)
            #ax1.plot(self.t,betax2)


            ax2.set_title('beta1, beta2 for thetay vars in rhs')
            ax2.plot(self.t,betay1+betay2)
            #ax2.plot(self.t,betay2)

            #thxprime = (sumxx + sumxy)/self.N + betax1 + betax2
            #thyprime = (sumyx + sumyy)/self.N + betay1 + betay2

        elif choice == 'slowmod_lc-diff':
            """
            dsx,dsy
            """
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("slowmod_lc-diff (theory)")

            dsx = self.dsxa(np.mod(self.t,self.ta[-1]))
            dsy = self.dsya(np.mod(self.t,self.ta[-1]))

            ax.plot(self.t,dsx)
            ax.plot(self.t,dsy)
            
            

        else:
            raise ValueError('Invalid plot choice (in class phase): '+str(choice))


        #plt.tight_layout()

        plt.tight_layout()
        #return fig


def main():
    print 'use thetaslowmod_master.py'


if __name__ == "__main__":
    main()
