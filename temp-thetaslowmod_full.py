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
check that sbar(\tau) = freq(\tau)

"""



import numpy as np
import scipy as sp
import matplotlib.pylab as mp
import matplotlib.pyplot as plt
import sys
import os
import getopt
import copy

from sys import stdout
from scipy.interpolate import interp1d
#from scipy.integrate import odeint
from scipy.signal import argrelextrema

from xppcall import xpprun, read_numerics#, read_pars, read_inits
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


class Theta(object):
    """
    full sim. for simulating the original and averaged system.
    """
    def __init__(self,
                 mode='full',
                 run=True,
                 use_last=False,
                 save_last=False,
                 use_ss=False,
                 save_ss=False,
                 use_random=False,
                 with_mean_field=False,
                 use_init_option=None,
                 a1=0.,b1=0.,c1=0.,
                 a2=0.,b2=0.,c2=0.,
                 al=1,
                 eps=.005,
                 mux=1.,muy=1.,
                 t0=0.,T=1000.,dt=.01,
                 xin=None,yin=None,
                 sx0=None,sy0=None,
             ):
        
        # supercrit values
        #a1=.5,b1=7.,c1=6.5,
        #a2=1.1,b2=25.,c2=25.1,

        # some other values
        #a1=.5,b1=7.,c1=6.5,
        #a2=1.1,b2=5.,c2=5.1,

        if (use_init_option == 'manual') and ((xin == None) or (yin == None)):
            raise ValueError('Enter inital conditions')

        self.use_last=use_last
        self.save_last = save_last
        self.use_ss=use_ss
        self.save_ss = save_ss
        self.use_random = use_random
        self.use_init_option = use_init_option
        self.with_mean_field=with_mean_field

        assert(len(xin) == len(yin))
        self.N = len(xin)

        self.xin = xin
        self.yin = yin
        self.sx0 = sx0
        self.sy0 = sy0

        self.mode = mode
        self.run = run

        #self.N = N
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1

        self.a2 = a2
        self.b2 = b2
        self.c2 = c2

        self.al = al

        self.eps = eps
        self.mux = mux
        self.muy = muy
        self.taux = self.mux/self.eps
        self.tauy = self.muy/self.eps

        self.t0 = t0
        self.T = T
        self.dt = dt
        #self.TN = int((self.T-self.t0)/self.dt)
        #self.t = np.linspace(self.t0,self.T,self.TN)

        self.savedir = 'savedir/'

        self.filename_x = self.savedir+'x_init.dat'
        self.filename_y = self.savedir+'y_init.dat'
        self.filename_s = self.savedir+'s_init.dat'
        #print self.filename_x
        # include a,b,c, 

        if (self.N != 2) and (self.N != 1) and (self.N != 3):
            print "warning: full theta sim uses N=2 or N=1. Other values are not yet implemented."
        
        self.paramsfile = '_a1='+str(a1)+\
                      '_b1='+str(b1)+\
                        '_c1='+str(c1)+\
                        '_a2='+str(a2)+\
                        '_b2='+str(b2)+\
                        '_c2='+str(c2)+\
                        '_al='+str(al)+\
                        '_eps='+str(eps)+\
                        '_mux='+str(mux)+\
                        '_muy='+str(muy)+'.dat'

        self.paramstitle = ", a1="+str(self.a1)+\
                           ", b1="+str(self.b1)+\
                        ", c1="+str(self.c1)+\
                        ", a2="+str(self.a2)+\
                        ", b2="+str(self.b2)+\
                        ",\nc2="+str(self.c2)+\
                        ", al="+str(al)+\
                        ", mux="+str(self.mux)+\
                        ", muy="+str(self.muy)+\
                        ", eps="+str(self.eps)

        self.filename_x_ss = self.savedir+'x_ss'+self.paramsfile
        self.filename_y_ss = self.savedir+'y_ss'+self.paramsfile
        self.filename_s_ss = self.savedir+'s_ss'+self.paramsfile

        self.sbarx,self.sbary = get_sbar(self.a1,self.b1,self.c1,self.a2,self.b2,self.c2)

        # check if oscillations exist
        self.slow_osc_exist = slow_osc(self.sbarx,self.sbary,
                                       self.a1,self.b1,self.c1,
                                       self.a2,self.b2,self.c2,
                                       self.mux,self.muy)

        print 'with_mean_field='+str(self.with_mean_field)

        if self.run:
            self.run_full_sim()

        if self.with_mean_field:
            self.run_mean_field()

        self.get_frequency_and_phase_data()
            
    def get_frequency_and_phase_data(self):
        """
        given the simulation, smooth the synaptic variables to approximate the periods
        """
        
        padN = 2010
        kernelSize = 2000
        self.sx_smooth = get_averaged(self.sx,dt=self.dt,
                                      padN=padN,kernelSize=kernelSize,
                                      time_pre=-self.t[padN],time_post=self.t[padN])
        self.sy_smooth = get_averaged(self.sy,dt=self.dt,
                                      padN=padN,kernelSize=kernelSize,
                                      time_pre=-self.t[padN],time_post=self.t[padN])
        
        self.freqx = np.sqrt(self.a1+self.b1*self.sx_smooth-self.c1*self.sy_smooth)
        self.freqy = np.sqrt(self.a2+self.b2*self.sx_smooth-self.c2*self.sy_smooth)

        self.perx = 1./self.freqx
        self.pery = 1./self.freqy

        
        """
        # get spike times
        if self.N == 2:
            self.spike_x1 = self.t[argrelextrema(self.x[:,0],np.greater)[0]]
            self.spike_x2 = self.t[argrelextrema(self.x[:,1],np.greater)[0]]
            self.spike_y1 = self.t[argrelextrema(self.y[:,0],np.greater)[0]]
            self.spike_y2 = self.t[argrelextrema(self.y[:,1],np.greater)[0]]

        elif self.N == 1:
            self.spike_x1 = self.t[argrelextrema(self.x[:,0],np.greater)[0]]
            self.spike_y1 = self.t[argrelextrema(self.y[:,0],np.greater)[0]]

        # get periods by taking time differences between spike times.
        ndiffx = np.diff(self.spike_x1)
        ndiffy = np.diff(self.spike_y1)
    
        # get binned frequencies
        self.fx = 1./ndiffx
        self.fy = 1./ndiffy

        # ndiffx,ndiffy are length len(self.spike_x1)-1, so append the last value to keep the same size
        self.fx = np.append(self.fx,self.fx[-1])
        self.fy = np.append(self.fy,self.fy[-1])

        # create frequency and period functions
        # frequency values are sampled non-uniformly in time...
        self.freqx = interp1d(self.spike_x1,self.fx)
        self.freqy = interp1d(self.spike_y1,self.fy)
        """


        """
        # smooth out the frequency data
        self.freqx = get_averaged(self.fx,dt=ndiffx[0])
        self.freqy = get_averaged(self.fy,dt=ndiffy[0])

        print self.spike_x1[0],self.spike_x1[-1]
        print self.t[0],self.t[-1]

        self.temp_tx = np.linspace(self.spike_x1[0],self.spike_x1[-1],len(self.t))
        self.temp_ty = np.linspace(self.spike_y1[0],self.spike_y1[-1],len(self.t))
        
        self.freqx = self.freqx(self.temp_tx)
        self.freqy = self.freqy(self.temp_ty)
        
        self.perx = 1./self.freqx
        self.pery = 1./self.freqy
        """

        # define solution arrays to fit xpp output
        self.phasex = np.zeros((len(self.t),self.N))
        self.phasey = np.zeros((len(self.t),self.N))
        
        # phase from 0 to T
        for i in range(self.N):
            self.phasex[:,i] = np.arctan(np.tan(self.x[:,i]/2.)/self.freqx)/(self.freqx*pi)
            self.phasey[:,i] = np.arctan(np.tan(self.y[:,i]/2.)/self.freqy)/(self.freqy*pi)
    
    def run_full_sim(self):
        
        # solution arrays
        self.x = np.zeros((1,self.N))
        self.y = np.zeros((1,self.N))
        self.sx = np.zeros(1)
        self.sy = np.zeros(1)
        
        # spike times

        # determine initial value (use last, use manual)
        file_not_found = False
        while True:

            if self.use_last and not(file_not_found):
                # if user asks to use_last and there exists a saved file...
                if os.path.isfile(self.filename_x) and\
                   os.path.isfile(self.filename_y) and\
                   os.path.isfile(self.filename_s):
                    self.x[0,:] = np.loadtxt(self.filename_x)
                    self.y[0,:] = np.loadtxt(self.filename_y)
                    
                    sv = np.loadtxt(self.filename_s)
                    self.sx[0] = sv[0]
                    self.sy[0] = sv[1]

                    #print 'using inits (x1,x2)=('+str(self.x[0,0])+','+str(self.x[0,1])+')'
                    #print '(y1,y2)=('+str(self.y[0,0])+','+str(self.y[0,1])+')'
                    #print 'sx,sy='+str(self.sx[0])+','+str(self.sy[0])
                    break
                else:
                    file_not_found = True
            
            elif self.use_ss and not(file_not_found):
                if os.path.isfile(filename_x_ss) and\
                   os.path.isfile(filename_y_ss) and\
                   os.path.isfile(filename_s_ss):
                    self.x[0,:] = np.loadtxt(self.filename_x_ss)
                    self.y[0,:] = np.loadtxt(self.filename_y_ss)
                    
                    sv = np.loadtxt(self.filename_s_ss)
                    self.sx[0] = sv[0]
                    self.sy[0] = sv[1]
                    break
                else:
                    file_not_found = True

            else:
                print 'using manual inits. (x1,x2)='+str(self.xin)+'. (y1,y2)='+str(self.yin)
                self.x[0,:] = self.xin
                self.y[0,:] = self.yin

                self.sx[0] = self.sx0#1.017038158211262644
                self.sy[0] = self.sy0#1.000222753692209698

                break

        print 'init sx,sy =',self.sx[0],self.sy[0]
        print 'running full_sim through xpp'

        if self.N == 1:
            npa, vn = xpprun('theta1_het.ode',
                             xppname='xppaut',
                             inits={'x1':self.x[0,0],'y1':self.y[0,0],
                                    'sx':self.sx[0], 'sy':self.sy[0]},
                             parameters={'eps':self.eps,
                                         'a1':self.a1,'b1':self.b1,'c1':self.c1,
                                         'a2':self.a2,'b2':self.b2,'c2':self.c2,
                                         'al':self.al,
                                         'dt':self.dt,'total':self.T,'mux':self.mux,'muy':self.muy},
                             clean_after=True)


        elif self.N >= 2:

            # get filename
            n = self.N
            fname = 'theta'+str(n)+'_het.ode'
            
            # get inits
            x0 = []
            y0 = []
            for i in range(self.N):
                x0.append(self.x[0,i])
                y0.append(self.y[0,i])
            
            npa, vn = xpprun(fname,
                             xppname='xppaut',
                             inits={'x':x0,'y':y0,'sx':self.sx[0], 'sy':self.sy[0]},
                             parameters={'eps':self.eps,
                                         'a1':self.a1,'b1':self.b1,'c1':self.c1,
                                         'a2':self.a2,'b2':self.b2,'c2':self.c2,
                                         'al':self.al,
                                         'dt':self.dt,'total':self.T,'mux':self.mux,'muy':self.muy},
                             clean_after=True)

        print 'finished xpp run'

        t = npa[:,0]
        sv = npa[:,1:]

        #num_opts = read_numerics('theta2_het.ode')

        self.t = t
        self.T = self.t[-1]
        #self.dt = float(num_opts['dt'])

        self.x = np.zeros((len(self.t),self.N))
        self.y = np.zeros((len(self.t),self.N))
        self.sx = np.zeros(len(self.t))
        self.sy = np.zeros(len(self.t))

        # save solutions to compatible format
        #print vn
        #print sv[0,:]
        #print np.shape(sv)
        #print sv[0,:]
        
        for i in range(self.N):
            #print i,vn.index('x'+str(i)), sv[:,vn.index('x'+str(i))]

            self.x[:,i] = sv[:,vn.index('x'+str(i))]
            self.y[:,i] = sv[:,vn.index('y'+str(i))]

        #print self.x[:5,0],self.x[:5,1],self.x[:5,2]

        self.sx = sv[:,vn.index('sx')]
        self.sy = sv[:,vn.index('sy')]
        
        if self.save_last:
            np.savetxt(self.filename_x,self.x[-1,:])
            np.savetxt(self.filename_y,self.y[-1,:])
            np.savetxt(self.filename_s,np.array([self.sx[-1],self.sy[-1]]))
        
        if self.save_ss:
            np.savetxt(self.filename_x_ss,self.x[-1,:])
            np.savetxt(self.filename_y_ss,self.y[-1,:])
            np.savetxt(self.filename_s_ss,np.array([self.sx[-1],self.sy[-1]]))
            #return self.wba,trba,swb,strb

    def run_mean_field(self):
        """
        averaged sim
        """
        print 'running mean field...'
        sxa = np.zeros(len(self.t))
        sya = np.zeros(len(self.t))
        
        sxa[0] = self.sx[0]
        sya[0] = self.sy[0]
        
        for i in range(0,len(self.t)-1):
            f1 = self.a1 + self.b1*sxa[i] - self.c1*sya[i]
            f2 = self.a2 + self.b2*sxa[i] - self.c2*sya[i]
            if f1 < 0:
                f1 = 0.
            if f2 < 0:
                f2 = 0.
        
            sxaprime = (1./self.taux)*(-sxa[i] + sqrt(f1))
            syaprime = (1./self.tauy)*(-sya[i] + sqrt(f2))

            sxa[i+1] = sxa[i] + self.dt*sxaprime
            sya[i+1] = sya[i] + self.dt*syaprime

        self.sxa = sxa
        self.sya = sya

        return self.sxa,self.sya



    def plot(self,choice='full',nskip=1):
        fig = plt.figure()

        ts_idx = 0

        z = np.linspace(0,1,len(self.t)+len(self.t)/5)#[:len(self.t)-ts_idx]

        color = plt.cm.Greys(z)

        if choice == 'full-vars':
            
            ax = plt.subplot(111)
            ax.set_title("xi,yi")

            for i in range(self.N):
                ax.plot(self.t[ts_idx:],self.x[ts_idx:,i],label='x'+str(i))
                ax.plot(self.t[ts_idx:],self.y[ts_idx:,i],label='y'+str(i))

            ax.legend()


        elif choice == 'full-s':
            ax2 = plt.subplot(111)
            ax2.set_title("sx,sy (numerics)")
            ax2.plot(self.t[ts_idx:],self.sx[ts_idx:],label='sx',color='blue',alpha=.5)
            ax2.plot(self.t[ts_idx:],self.sy[ts_idx:],label='sy',color='green',alpha=.5)

            ax2.plot(self.t[ts_idx:],self.sx_smooth[ts_idx:],label='sx_smoothed',color='blue',lw=2)
            ax2.plot(self.t[ts_idx:],self.sy_smooth[ts_idx:],label='sy_smoothed',color='green',lw=2)

            if self.with_mean_field:
                ax2.plot(self.t[ts_idx:],self.sxa[ts_idx:],label='sxa')
                ax2.plot(self.t[ts_idx:],self.sya[ts_idx:],label='sya')

            

            ax2.legend()

        elif choice == 'full-s-space':
            ax2 = plt.subplot(111)
            ax2.set_title("sx vs sy (numerics)")
            ax2.plot(self.sx[ts_idx:],self.sy[ts_idx:],label='sx',color='blue',alpha=.5)
            
            ax2.plot(self.sx_smooth[ts_idx:],self.sy_smooth[ts_idx:],label='sx_smoothed',color='black',lw=2)
            ax2.legend()

        elif choice == 'full-s-t':
            ax2 = plt.subplot(111)
            ax2.set_title("sx vs sy (numerics)")
            ax2.plot(self.sx[ts_idx:],label='sx',color='blue',alpha=.5)
            ax2.plot(self.sy[ts_idx:],label='sy',color='green',alpha=.5)
            
            ax2.plot(self.sx_smooth[ts_idx:],label='sx_smoothed',color='black',lw=2)
            ax2.plot(self.sy_smooth[ts_idx:],label='sy_smoothed',color='gray',lw=2)
            ax2.legend()

        elif choice == 'x1-x2':
            ax = plt.subplot(111)
            ax.set_title("x1 vs x2"+self.paramstitle)
            ax.scatter(self.x[ts_idx:,0],self.x[ts_idx:,1],color=color,edgecolor='none',s=20)

        elif choice == 'x1-y2':
            ax = plt.subplot(111)
            ax.set_title("x1 vs y2"+self.paramstitle)
            ax.scatter(self.x[ts_idx:,0],self.y[ts_idx:,1],color=color,edgecolor='none',s=20)

        elif choice == 'y1-x2':
            ax = plt.subplot(111)
            ax.set_title("y1 vs x2"+self.paramstitle)
            ax.scatter(self.y[ts_idx:,0],self.x[ts_idx:,1],color=color,edgecolor='none',s=20)

        elif choice == 'y1-y2':
            ax = plt.subplot(111)
            ax.set_title("y1 vs y2"+self.paramstitle)
            ax.scatter(self.y[ts_idx:,0],self.y[ts_idx:,1],color=color,edgecolor='none',s=20)

        elif choice == 'xj-yj':

            if self.N == 2:
                # plot all combinations of relative phases
                ax11 = plt.subplot(321)
                ax11.set_title("x1 vs x2")
                #ax11.scatter(self.x[ts_idx:,0],self.x[ts_idx:,1],color=color,edgecolor='none')
                ax11.scatter(self.x[ts_idx:,0],self.x[ts_idx:,1],color=color,edgecolor='none')

                ax22 = plt.subplot(322)
                ax22.set_title("y1 vs y2")
                ax22.scatter(self.y[ts_idx:,0],self.y[ts_idx:,1],color=color,edgecolor='none')

                ax12 = plt.subplot(323)
                ax12.set_title("x1 vs y1")
                ax12.scatter(self.x[ts_idx:,0],self.y[ts_idx:,0],color=color,edgecolor='none')

                ax21 = plt.subplot(324)
                ax21.set_title("y1 vs x1")
                #ax21.scatter(self.y[ts_idx:,0],self.x[ts_idx:,0],color=color,edgecolor='none')

                ax12 = plt.subplot(325)
                ax12.set_title("x1 vs y2")
                #ax12.scatter(self.x[ts_idx:,0],self.y[ts_idx:,1],color=color,edgecolor='none')

                ax21 = plt.subplot(326)
                ax21.set_title("y1 vs x2")
                #ax21.scatter(self.y[ts_idx:,0],self.x[ts_idx:,1],color=color,edgecolor='none')

                #plt.subplots_adjust(top=0.85)
                #plt.suptitle("xj-yj"+self.paramstitle)
            elif self.N == 1:
                # plot all combinations of relative phases
                ax11 = plt.subplot(111)
                ax11.set_title("x1 vs y1")
                #ax11.scatter(self.x[ts_idx:,0],self.x[ts_idx:,1],color=color,edgecolor='none')
                ax11.scatter(self.y[ts_idx:,0],self.x[ts_idx:,0],color=color,edgecolor='none')

                #plt.subplots_adjust(top=0.85)
                #plt.suptitle("xj-yj"+self.paramstitle)
            

        elif choice == 'phase-diff-raw':
            ax = plt.subplot(111)
            ax.set_title("phase diff raw (numerics)")
            
            # the sparseness in this graph is due to the frequency approximation. What we end up with is a near-discrete frequency function due to the smoothing of sx, sy.
            #mp.figure()
            #mp.scatter(self.t,self.phasey[:,0]-self.phasex[:,0])
            #mp.show()


            for i in range(self.N-1):

                diff1 = np.mod(self.phasex[:,i+1]-self.phasex[:,0]+self.perx/2.,self.perx)-self.perx/2.
                diff2 = np.mod(self.phasey[:,i+1]-self.phasey[:,0]+self.pery/2.,self.pery)-self.pery/2.
                #print i,str(i+2),(.5*i+.5)/self.N
                #ax.scatter(self.t[::nskip],diff1[::nskip],color='blue',edgecolor='none',label='x'+str(i+2)+'-x1',alpha=(.5*(i+1)+.5)/self.N)
                #ax.scatter(self.t[::nskip],diff2[::nskip],color='green',edgecolor='none',label='y'+str(i+2)+'-y1',alpha=(.5*(i+1)+.5)/self.N)
                ax.scatter(self.t[::nskip],diff1[::nskip],color=blues[i],edgecolor='none',label='x'+str(i+2)+'-x1')
                ax.scatter(self.t[::nskip],diff2[::nskip],color=greens[i],edgecolor='none',label='y'+str(i+2)+'-y1')

            # if now slow oscillations exist and mean frequencies are the same
            diff3 = np.mod(self.phasey[:,0]-self.phasex[:,0]+self.pery/2.,self.pery)-self.pery/2.

            ax.scatter(self.t[::nskip],diff3[::nskip],color='red',edgecolor='none',label='y1-x1')
            ax.legend()

            ax.set_xlim(self.t[0],self.t[-1])
            #ax.set_ylim(-per/2.-.1,per/2.+.1)
            #ax.set_ylim(-.5-.05,.5+.05)

            # plot the antiphase lines
            ax.plot(self.t,-self.perx/2.,color='gray',ls='-')
            ax.plot(self.t,-self.pery/2.,color='gray',ls='--')
            ax.plot(self.t,self.perx/2.,color='gray',ls='-')
            ax.plot(self.t,self.pery/2.,color='gray',ls='--')

            # plot antiphase lines
            #ax.plot(self.t,self.freqx/2.,color='gray',ls='-')
            #ax.plot(self.t,-self.freqx/2.,color='gray',ls='-')
            #ax.plot(self.t,-self.freqy/2.,color='gray',ls='-')
            #ax.plot(self.t,-self.freqy/2.,color='gray',ls='-')

        elif choice == 'phase-diff-normed':
            ax = plt.subplot(111)
            ax.set_title("numerical phase diff normalized")

            nskip = 50

            self.phx_normed = self.phasex
            self.phy_normed = self.phasey

            self.phx_normed[:,0] = self.phx_normed[:,0]/self.perx
            self.phy_normed[:,0] = self.phy_normed[:,0]/self.pery

            for i in range(self.N-1):
                self.phx_normed[:,i+1] = self.phx_normed[:,i+1]/self.perx
                self.phy_normed[:,i+1] = self.phy_normed[:,i+1]/self.pery

                diff1 = np.mod(self.phx_normed[:,i+1]-self.phx_normed[:,0]+1/2.,1)-1/2.
                diff2 = np.mod(self.phy_normed[:,i+1]-self.phy_normed[:,0]+1/2.,1)-1/2.

                #ax.scatter(self.t,diff1*2*pi/per,color='blue',edgecolor='none',label='x'+str(i+2)+'-x1')
                #ax.scatter(self.t,diff2*2*pi/per,color='green',edgecolor='none',label='y'+str(i+2)+'-y1')

                ax.scatter(self.t[::nskip],diff1[::nskip],color='blue',edgecolor='none',label='x'+str(i+2)+'-x1')
                ax.scatter(self.t[::nskip],diff2[::nskip],color='green',edgecolor='none',label='y'+str(i+2)+'-y1')

            # display x vs y if homogeneous frequencies (no oscillations + same mean firing at sbar)
            if (self.sbarx == self.sbary) and not(self.slow_osc_exist):
                diff3 = np.mod(self.phy_normed[:,0]-self.phx_normed[:,0]+1/2.,1)-1/2.
                ax.scatter(self.t[::nskip],diff3[::nskip],color='blue',edgecolor='none',label='y1-x1')

            ax.legend()

            ax.set_xlim(self.t[0],self.t[-1])
            #ax.set_ylim(-per/2.-.1,per/2.+.1)
            #ax.set_ylim(-.5-.05,.5+.05)


            # plot the antiphase lines
            #ax.plot(self.t,np.ones(len(self.t))*.5,color='gray',alpha=.5)
            #ax.plot(self.t,np.ones(len(self.t))*.5,color='gray',alpha=.5)
            
            #for slc in unlink_wrap(diff,[-self.period/2.,self.period/2.]):
            #    ax.plot(self.t[slc],diff[slc],color='blue')

            #ax.scatter()
            """
            self.spike_x1 = np.array(self.spike_x1)
            self.spike_y1 = np.array(self.spike_y1)
            
            per = 1./self.sbar
            nlen = np.amin([len(self.spike_x1),len(self.spike_y1)])

            ndiff1 = 2*pi*(np.mod((self.spike_y1[:nlen] - self.spike_x1[:nlen])/per+.5,1)-.5)

            ax.scatter(self.spike_y1[:nlen],ndiff1)

            ax.plot([self.t[0],self.t[-1]],[pi,pi],color='gray')
            ax.plot([self.t[0],self.t[-1]],[-pi,-pi],color='gray')
            #ax.set_ylim(-self.period/2.-.1,self.period/2.+.1)
            """


        elif choice == 'phx-phy':
            
            #ax.set_title("numerical phase diff")
            nskip = 100

            ax11 = plt.subplot(221)
            ax12 = plt.subplot(222)
            ax21 = plt.subplot(223)
            ax12.set_title('y1 vs y2')
            ax11.set_title("x1 vs x2")
            ax21.set_title('x1 vs y1')

            if self.N >= 2:
                ax11.scatter(self.phasex[:,0],self.phasex[:,1],color='blue',edgecolor='none')
                ax12.scatter(self.phasey[:,0],self.phasey[:,1],color='green',edgecolor='none')

            ax21.scatter(self.phasex[:,0],self.phasey[:,0],color='red',edgecolor='none')
            

        elif choice == 'phase':
            ax = plt.subplot(111)
            ax.set_title("numerical phase")

            #f = 1./self.period
            #phasex = np.arctan(np.tan(self.x/2.)/f)/(f*pi)
            #phasey = np.arctan(np.tan(self.y/2.)/f)/(f*pi)

            for i in range(self.N-1):
                ax.plot(self.t,self.phasex[:,i],label='x'+str(i))
                ax.plot(self.t,self.phasey[:,i],label='y'+str(i))

            #for slc in unlink_wrap(diff,[-self.period/2.,self.period/2.]):
            #    ax.plot(self.t[slc],diff[slc],color='blue')


        elif choice == 'phase-normed':
            ax = plt.subplot(111)
            ax.set_title("numerical phase normalized")

            ax.plot(self.t,self.phasex[:,0]/self.perx,label='x1')
            ax.plot(self.t,self.phasex[:,1]/self.perx,label='x2')

            ax.plot(self.t,self.phasey[:,0]/self.pery,label='y1')
            ax.plot(self.t,self.phasey[:,1]/self.pery,label='y2')

            #for slc in unlink_wrap(diff,[-self.period/2.,self.period/2.]):
            #    ax.plot(self.t[slc],diff[slc],color='blue')


        elif choice == 'phase-shift-normed':
            ax = plt.subplot(111)
            ax.set_title("numerical phase shift normalized")

            ax.plot(self.t,self.phasex[:,0]/self.perx,label='x1')
            ax.plot(self.t,self.phasex[:,1]/self.perx,label='x2')

            ax.plot(self.t,self.phasey[:,0]/self.pery,label='y1')
            ax.plot(self.t,self.phasey[:,1]/self.pery,label='y2')

            #for slc in unlink_wrap(diff,[-self.period/2.,self.period/2.]):
            #    ax.plot(self.t[slc],diff[slc],color='blue')

            ax.legend()
            
        elif choice == 'freq':
            ax = plt.subplot(111)
            ax.set_title("frequency")
            
            ax.plot(self.t,self.freqx,lw=2,alpha=.5,color='blue')
            #ax.plot(self.spike_x1[1:],1./np.diff(self.spike_x1),label='x1 freq',color='blue')
            #ax.plot(self.spike_y1[1:],1./np.diff(self.spike_y1),label='y1 freq')

            ax.plot(self.t,self.freqy,lw=2,alpha=.5,color='green')
            #ax.plot(self.spike_y1[1:],1./np.diff(self.spike_y1),label='y1 freq',color='green')
            #ax.plot(self.spike_y2[1:],1./np.diff(self.spike_y2),label='y2 freq')

            ax.legend()


        elif choice == 'period':
            ax = plt.subplot(111)
            ax.set_title("period")
            ax.plot(self.spike_x1[1:],np.diff(self.spike_x1),label='x1 period')
            ax.plot(self.spike_y1[1:],np.diff(self.spike_y1),label='y1 period')


        else:
            raise ValueError('Invalid plot choice (in class theta): '+str(choice))

        plt.tight_layout()
        return fig


def main():
    print 'use thetaslowmod_master.py'

if __name__ == "__main__":
    main()
