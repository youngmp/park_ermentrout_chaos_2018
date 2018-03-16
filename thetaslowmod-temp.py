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

def usage():
    print "-l, --use-last\t\t: use last data from last sim"
    print "-v, --save-last\t\t: save last data of current sim"
    print "-s, --use-ss\t\t: use last saved steady-state data"
    print "-e, --save-ss\t\t: save solution as steady-state data"
    print "-r, --use-random\t: use random inits"
    print "-h, --help\t\t: help function"
    print "-p, --run-phase\t\t: run phase"
    print "-f, --run-full\t\t: run full"

from sys import stdout
import numpy as np
import matplotlib.pylab as mp
import matplotlib.pyplot as plt
import sys
import os
import getopt
import copy
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.signal import argrelextrema

from xppcall import xpprun, read_numerics#, read_pars, read_inits

np.random.seed(10)

cos = np.cos
sin = np.sin
pi = np.pi
sqrt = np.sqrt

def unlink_wrap(dat, lims=[-np.pi, np.pi], thresh = 0.95):
    # http://stackoverflow.com/questions/27138751/preventing-plot-joining-when-values-wrap-in-matplotlib-plots
    """
    Iterate over contiguous regions of `dat` (i.e. where it does not
    jump from near one limit to the other).

    This function returns an iterator object that yields slice
    objects, which index the contiguous portions of `dat`.

    This function implicitly assumes that all points in `dat` fall
    within `lims`.

    """
    jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))


def average_jac(sbar,a1,b1,c1,a2,b2,c2,mux,muy):
    """
    Jacobian of averaged system at the fixed point sbar
    """
    # WLOG use a1,b1,c1, since a2,b2,c2 are chosen to have the same value
    
    sx = sbar
    sy = sbar
    
    w1 = 0.5/np.sqrt(a1 + b1*sx - c1*sy)
    w2 = 0.5/np.sqrt(a2 + b2*sx - c2*sy)
    
    return np.array([[(-1+b1*w1)/mux,-c1*w1/mux],
                     [b2*w2/muy,(-1-c2*w2)/muy]])


def slow_osc(sbar,a1,b1,c1,a2,b2,c2,mux,muy):
    """
    return true or false
    mux,muy: parameters of slow/averaged system
    sbar: fixed point (assume (sbar,sbar))
    """
    w,v = np.linalg.eig(average_jac(sbar,a1,b1,c1,a2,b2,c2,mux,muy))
    
    # get real parts
    rew1 = np.real(w[0])
    rew2 = np.real(w[1])

    if (rew1*rew2 > 0) and (rew1+rew2 > 0):
        return True
    else:
        return False
    

class Sim(object):
    # maybe use this later for fixed parameters
    pass

class Theta(Sim):
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
                 use_init_option=None,
                 a1=0.,b1=0.,c1=0.,
                 a2=0.,b2=0.,c2=0.,
                 eps=.005,
                 mux=1.,muy=1.,
                 t0=0.,T=1000.,dt=.01,
                 N=2
             ):
        
        # supercrit values
        #a1=.5,b1=7.,c1=6.5,
        #a2=1.1,b2=25.,c2=25.1,

        # some other values
        #a1=.5,b1=7.,c1=6.5,
        #a2=1.1,b2=5.,c2=5.1,


        self.use_last=use_last
        self.save_last = save_last
        self.use_ss=use_ss
        self.save_ss = save_ss
        self.use_random = use_random
        self.use_init_option = use_init_option

        self.mode = mode
        self.run = run

        self.N = N
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

        self.t0 = t0
        self.T = T
        self.dt = dt
        self.TN = int((self.T-self.t0)/self.dt)
        self.t = np.linspace(self.t0,self.T,self.TN)

        self.savedir = 'savedir/'

        self.filename_x = self.savedir+'x_init.dat'
        self.filename_y = self.savedir+'y_init.dat'
        self.filename_s = self.savedir+'s_init.dat'
        #print self.filename_x
        # include a,b,c, 

        if (self.N != 2) and (self.N != 1):
            print "warning: full theta sim uses N=2 or N=1. Other values are not yet implemented."
        
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

        self.filename_x_ss = self.savedir+'x_ss'+self.paramsfile
        self.filename_y_ss = self.savedir+'y_ss'+self.paramsfile
        self.filename_s_ss = self.savedir+'s_ss'+self.paramsfile


        self.sbar = self.get_sbar()
        self.periodx = self.get_period(a1,b1,c1)
        self.periody = self.get_period(a2,b2,c2)
        self.period_brute = self.get_period_brute()

        assert(np.abs(self.periodx-self.periody)<1e-10)
        self.period = self.periodx


        if self.run:
            
            
            self.run_full_sim()


    def get_freq(self,a,b,c):
        """
        get average fast  frequency given sx,sy
        """
        d = a+b*self.sbar-c*self.sbar
        if d < 0:
            d = 0
        return np.sqrt(d)
 
    def get_period(self,a,b,c):
        """
        get average fast period given sx, sy
        """
        f = self.get_freq(a,b,c)
        if f == 0:
            return np.inf
        else:
            return 1./f

    def x0(self,t,a,b,c,f=None):
        """
        analytic limit cycle (found using mathematica)
        """
        
        if f == None:
            f = self.sbar
        return 2*np.arctan(f*np.tan((t+1./(2.*f))*f*pi))

    def get_period_brute(self):
        """
        get brute force period
        """
        T = 500
        x = np.zeros(int(T/self.dt))
        spike_times = np.array([])
        
        for i in range(0,int(T/self.dt)-1):
            xprime = (1 - cos(x[i]) + (1 + cos(x[i]))*(self.a1+self.b1*self.sbar-self.c1*self.sbar))*pi
            
            x[i+1] = x[i] + self.dt*xprime
            if x[i+1] >= 2*pi:
                x[i+1] = 0.
                spike_times = np.append(spike_times,self.dt*i)
        return spike_times[-1] - spike_times[-2]

    def get_sbar(self,check=False,return_both=False):
        aa = self.a1
        bb = self.b1
        cc = self.c1
         
        # two roots
        #a1 = (-(cc-bb) + sqrt((cc-bb)**2 + 4*aa*pi**2))/(2.*pi**2)
        #a2 = (-(cc-bb) - sqrt((cc-bb)**2 + 4*aa*pi**2))/(2.*pi**2)

        a1 = (-(cc-bb) + sqrt((cc-bb)**2 + 4*aa))/2.
        a2 = (-(cc-bb) - sqrt((cc-bb)**2 + 4*aa))/2.
        
        if check == True:
            "check if other fixed point/freq is the same"
            aa = self.a2
            bb = self.b2
            cc = self.c2
            
            # two roots
            a1p = (-(cc-bb) + sqrt((cc-bb)**2 + 4*aa))/2.
            
            print "sxbar - sybar =",a1-a1p

        if not(return_both):
            if a1 > 0:
                return a1
            if a2 > 0:
                return a2
            if a1 <= 0 and a2 <= 0:
                raise ValueError('no positive fixed point found')
        else:
            return (-(self.c1-self.b1) + sqrt((self.c1-self.b1)**2 + 4*self.a1))/2.,
            (-(self.c2-self.b2) + sqrt((self.c2-self.b2)**2 + 4*self.a2))/2.
        
    def run_full_sim(self):
        
        # solution arrays
        self.phasex = np.zeros((1,self.N))
        self.phasey = np.zeros((1,self.N))
        
        self.x = np.zeros((1,self.N))
        self.y = np.zeros((1,self.N))
        self.sx = np.zeros(1)
        self.sy = np.zeros(1)
        
        # spike times



        # inits
        r1 = np.random.uniform(size=self.N)*2*pi-pi
        r2 = np.random.uniform(size=self.N)*2*pi-pi
        
        sx0 = self.sbar#1.017038158211262644
        sy0 = self.sbar#1.000222753692209698
        
        file_not_found = False
        while True:
            if self.use_last and not(file_not_found):
                if os.path.isfile(self.filename_x) and\
                   os.path.isfile(self.filename_y) and\
                   os.path.isfile(self.filename_s):
                    self.x[0,:] = np.loadtxt(self.filename_x)
                    self.y[0,:] = np.loadtxt(self.filename_y)
                    
                    sv = np.loadtxt(self.filename_s)
                    self.sx[0] = sv[0]
                    self.sy[0] = sv[1]
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
            elif self.use_init_option == '1':

                print 'init option 1: x1-x2 antiphase. all other in phase. Tested in xppcall, xppaut'

                if ((abs(self.a1-2)+abs(self.b1-5)+abs(self.c1-1)) > 1e-10) and\
                   ((abs(self.a2-2)+abs(self.b2-5)+abs(self.c2-1)) > 1e-10):
                    print 'warning: parameters are different from testing phase.'

                
                self.x[0,:] = [-2.6,2.282]
                self.y[0,:] = [-2.862,-2.862]


                self.sx[0] = 4.44#self.sbar#1.017038158211262644
                self.sy[0] = 4.44#self.sbar#1.000222753692209698


                break


            elif self.use_init_option == '2':

                print 'init option 2: x1-x2 antiphase. all other in phase. Tested in xppcall, xppaut'

                if ((abs(self.a1-2)+abs(self.b1-5)+abs(self.c1-1)) > 1e-10) and\
                   ((abs(self.a2-2)+abs(self.b2-5)+abs(self.c2-1)) > 1e-10):
                    print 'warning: parameters are different from testing phase.'

                
                self.x[0,:] = [-3.078,1.0946]
                self.y[0,:] = [-3.0783,-3.07835]


                self.sx[0] = 4.44#self.sbar#1.017038158211262644
                self.sy[0] = 4.44#self.sbar#1.000222753692209698


                break

            elif self.use_init_option == '3':

                if ((abs(self.a1-2)+abs(self.b1-5)+abs(self.c1-1)) > 1e-10) and\
                   ((abs(self.a2-2)+abs(self.b2-5)+abs(self.c2-1)) > 1e-10):
                    print 'warning: parameters are different from testing phase.'

                print 'init option 3: x1-x2 and x1-y1 antiphase. all other in phase. Tested in xppcall, xppaut'

                self.x[0,:] = [-2.54045,2.56837]
                self.y[0,:] = [1.811140,1.81114]


                self.sx[0] = 4.44#self.sbar#1.017038158211262644
                self.sy[0] = 4.44#self.sbar#1.000222753692209698


                break


            elif self.use_init_option == '4':

                if ((abs(self.a1-2)+abs(self.b1-5)+abs(self.c1-1)) > 1e-10) and\
                   ((abs(self.a2-2)+abs(self.b2-5)+abs(self.c2-1)) > 1e-10):
                    print 'warning: parameters are different from testing phase.'

                print 'init option 4: all in phase. Tested in xppcall, xppaut'

                self.x[0,:] = [-1.3668,1.26765]
                self.y[0,:] = [0,0]


                self.sx[0] = 4.44#self.sbar#1.017038158211262644
                self.sy[0] = 4.44#self.sbar#1.000222753692209698


                break



            else:
                file_not_found = True

            if file_not_found or self.use_random:
                print 'using random inits.', r1,r2
                
                self.x[0,:] = r1
                self.y[0,:] = r2
                
                self.sx[0] = sx0#self.sbar#np.random.randn(1)
                self.sy[0] = sy0#self.sbar#np.random.randn(1)
                break

        
        #print self.x[0,:],self.y[0,:]
        #print self.a1+self.b1*self.sx[0]-self.c1*self.sy[0]
        freqx = np.sqrt(self.a1+self.b1*self.sx[0]-self.c1*self.sy[0])
        freqy = np.sqrt(self.a2+self.b2*self.sx[0]-self.c2*self.sy[0])
        
        # phase from 0 to T
        self.phasex[0,:] = np.arctan(np.tan(self.x[0,:]/2.)/freqx)/(freqx*pi)
        self.phasey[0,:] = np.arctan(np.tan(self.y[0,:]/2.)/freqy)/(freqy*pi)
        
        # phase from -pi to pi
        self.phasex[0,:] *= 2*pi*freqx
        self.phasey[0,:] *= 2*pi*freqy

        print 'running full_sim'

        if self.N == 2:
            npa, vn = xpprun('theta2.ode',
                             xppname='xppaut8',
                             inits={'x1':self.x[0,0],'x2':self.x[0,1],
                                    'y1':self.y[0,0],'y2':self.y[0,1],
                                    'sx':self.sx[0], 'sy':self.sy[0]},
                             parameters={'eps':self.eps,'a':self.a1,'b':self.b1,'c':self.c1},
                             clean_after=True)
        elif self.N == 1:
            npa, vn = xpprun('theta1.ode',
                             xppname='xppaut8',
                             inits={'x1':self.x[0,0],'y1':self.y[0,0],
                                    'sx':self.sx[0], 'sy':self.sy[0]},
                             parameters={'eps':self.eps,'a':self.a1,'b':self.b1,'c':self.c1},
                             clean_after=True)



        t = npa[:,0]
        sv = npa[:,1:]

        num_opts = read_numerics('theta2.ode')

        self.t = t
        self.T = self.t[-1]
        self.dt = float(num_opts['dt'])

        # define solution arrays to fit xpp output
        self.phasex = np.zeros((len(self.t),self.N))
        self.phasey = np.zeros((len(self.t),self.N))
        
        self.x = np.zeros((len(self.t),self.N))
        self.y = np.zeros((len(self.t),self.N))
        self.sx = np.zeros(len(self.t))
        self.sy = np.zeros(len(self.t))

        # save phase values
        # phase from 0 to T
        self.phasex = np.arctan(np.tan(self.x/2.)/freqx)/(freqx*pi)
        self.phasey = np.arctan(np.tan(self.y/2.)/freqy)/(freqy*pi)
        
        # phase from -pi to pi
        self.phasex[:,:] *= 2*pi*freqx
        self.phasey[:,:] *= 2*pi*freqy


        # save solutions to compatible format
        if self.N == 2:
            self.x[:,0] = sv[:,vn.index('x1')]
            self.x[:,1] = sv[:,vn.index('x2')]
            
            self.y[:,0] = sv[:,vn.index('y1')]
            self.y[:,1] = sv[:,vn.index('y2')]
        elif self.N == 1:
            self.x[:,0] = sv[:,vn.index('x1')]
            self.y[:,0] = sv[:,vn.index('y1')]

        self.sx = sv[:,vn.index('sx')]
        self.sy = sv[:,vn.index('sy')]

        # get spike times
        if self.N == 2:
            self.spike_x1 = self.t[argrelextrema(self.x[:,0],np.greater)[0]]
            self.spike_x2 = self.t[argrelextrema(self.x[:,1],np.greater)[0]]
            self.spike_y1 = self.t[argrelextrema(self.y[:,0],np.greater)[0]]
            self.spike_y2 = self.t[argrelextrema(self.y[:,1],np.greater)[0]]
        elif self.N == 1:
            self.spike_x1 = self.t[argrelextrema(self.x[:,0],np.greater)[0]]
            self.spike_y1 = self.t[argrelextrema(self.y[:,0],np.greater)[0]]


        if self.save_last:
            np.savetxt(self.filename_x,self.x[-1,:])
            np.savetxt(self.filename_y,self.y[-1,:])
            np.savetxt(self.filename_s,np.array([self.sx[-1],self.sy[-1]]))
        
        if self.save_ss:
            np.savetxt(self.filename_x_ss,self.x[-1,:])
            np.savetxt(self.filename_y_ss,self.y[-1,:])
            np.savetxt(self.filename_s_ss,np.array([self.sx[-1],self.sy[-1]]))
            #return self.wba,trba,swb,strb



    def sim_avg(self):
        """
        averaged sim
        """
        sxa = np.zeros(self.TN)
        sya = np.zeros(self.TN)
        
        sxa[0] = sx[0]
        sya[0] = sy[0]
        
        for i in range(0,self.TN-1):
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

        return sxa,sya

    def get_full_ss(self):
        """
        get full numerical steady state
        """
        pass

    def plot(self,choice='full',cutoff=True):
        fig = plt.figure()
        if cutoff:
            tstart = self.T - self.periodx*10
        else:
            tstart = 0
        ts_idx = int(tstart/self.dt)

        z = np.linspace(0,1,len(self.t)+len(self.t)/5)#[:len(self.t)-ts_idx]

        color = plt.cm.Greys(z)

        if choice == 'full':
            
            ax = plt.subplot(121)
            ax.set_title("xi,yi")

            for i in range(self.N):
                ax.plot(self.t[ts_idx:],self.x[ts_idx:,i]+np.random.uniform(1)/10.,label='x'+str(i))
                ax.plot(self.t[ts_idx:],self.y[ts_idx:,i]+np.random.uniform(1)/10.,label='y'+str(i))

            ax.legend()

            ax2 = plt.subplot(122)
            ax2.set_title("sx,sy")
            ax2.plot(self.t[ts_idx:],self.sx[ts_idx:]+np.random.uniform(1)/10.,label='sx')
            ax2.plot(self.t[ts_idx:],self.sy[ts_idx:]+np.random.uniform(1)/10.,label='sy')
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

                plt.subplots_adjust(top=0.85)
                plt.suptitle("xj-yj"+self.paramstitle)
            elif self.N == 1:
                # plot all combinations of relative phases
                ax11 = plt.subplot(111)
                ax11.set_title("x1 vs y1")
                #ax11.scatter(self.x[ts_idx:,0],self.x[ts_idx:,1],color=color,edgecolor='none')
                ax11.scatter(self.y[ts_idx:,0],self.x[ts_idx:,0],color=color,edgecolor='none')

                #plt.subplots_adjust(top=0.85)
                #plt.suptitle("xj-yj"+self.paramstitle)
            

        elif choice == 'phase-diff':
            ax = plt.subplot(111)
            ax.set_title("numerical phase diff")

            
            for i in range(self.N-1):
                diff1 = self.phasex[:,i+1]-self.phasex[:,0]
                diff2 = self.phasey[:,i]-self.phasey[:,0]
                diff3 = self.phasey[:,i]-self.phasex[:,i]

                diff1 = np.mod(diff+pi,2*pi)-pi
                ax.scatter(self.t,diff,color='blue',s=1,edgecolor='none',label='y1-x1')
            
            #for slc in unlink_wrap(diff,[-self.period/2.,self.period/2.]):
            #    ax.plot(self.t[slc],diff[slc],color='blue')

            #ax.scatter()
            self.spike_x1 = np.array(self.spike_x1)
            self.spike_y1 = np.array(self.spike_y1)
            
            per = 1./self.sbar
            nlen = np.amin([len(self.spike_x1),len(self.spike_y1)])

            ndiff1 = 2*pi*(np.mod((self.spike_y1[:nlen] - self.spike_x1[:nlen])/per+.5,1)-.5)

            ax.scatter(self.spike_y1[:nlen],ndiff1)

            ax.plot([self.t[0],self.t[-1]],[pi,pi],color='gray')
            ax.plot([self.t[0],self.t[-1]],[-pi,-pi],color='gray')
            #ax.set_ylim(-self.period/2.-.1,self.period/2.+.1)

        elif choice == 'phase':
            ax = plt.subplot(111)
            ax.set_title("numerical phase")

            f = 1./self.period

            phasex = np.arctan(np.tan(self.x/2.)/f)/(f*pi)
            phasey = np.arctan(np.tan(self.y/2.)/f)/(f*pi)

            ax.plot(self.t,phasex[:,0],label='x1')
            ax.plot(self.t,phasex[:,1],label='x2')

            ax.plot(self.t,phasey[:,0],label='y1')
            ax.plot(self.t,phasey[:,1],label='y2')

            #for slc in unlink_wrap(diff,[-self.period/2.,self.period/2.]):
            #    ax.plot(self.t[slc],diff[slc],color='blue')
            
        elif choice == 'freq':
            ax = plt.subplot(111)
            ax.set_title("frequency")

            ax.plot(self.spike_x1[1:],1./np.diff(self.spike_x1),label='x1 freq')
            ax.plot(self.spike_y1[1:],1./np.diff(self.spike_y1),label='y1 freq')

            ax.plot(self.spike_x2[1:],1./np.diff(self.spike_x2),label='x2 freq')
            ax.plot(self.spike_y2[1:],1./np.diff(self.spike_y2),label='y2 freq')

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


class Phase(Theta):
    def __init__(self,run_full=False,
                 run_phase=False,
                 recompute_h=False,
                 recompute_beta=False,
                 recompute_slow_lc=False,
                 use_init_option=None,
                 T=100,
                 a1=0.,b1=0.,c1=0.,
                 a2=0.,b2=0.,c2=0.,
                 eps=.005,dt=.05,
                 mux=1.,muy=1.,
                 t0=0.,full_T=1000.,
                 N=2,integration_N=100):

        """
        old params
        a1=.5,b1=7.,c1=6.5,
        a2=1.1,b2=25.,c2=25.1,
        """

        self.recompute_beta = recompute_beta
        self.recompute_h = recompute_h
        self.run_full = run_full
        self.run_phase = run_phase
        self.dt = dt
        self.recompute_slow_lc = recompute_slow_lc
        self.use_init_option = use_init_option

        self.integration_N = integration_N


        Theta.__init__(self,run=self.run_full,
                       a1=a1,b1=b1,c1=c1,
                       a2=a2,b2=b2,c2=c2,
                       eps=eps,mux=mux,muy=muy,t0=t0,
                       T=full_T,dt=self.dt,N=N,
                       use_init_option=use_init_option)
        #call state variables with self.x,self.y,self.sx,self.sy

        self.phi = np.linspace(.001,2*pi,self.integration_N)


        #self.get_slowmod_lc()

        #self.load_h_static()
        self.load_h() # use this for everything
        #self.load_beta() # load beta functions

        #self.slowper = self.ta[-1]

        self.T = T

        if self.run_phase:

            self.t = np.linspace(0,self.T,int(self.T/self.dt))
            self.TN = int(self.T/self.dt)
            self.run_phase_sim()

        else:
            self.t = np.array([0,1])
            self.TN = 2
            self.thx=np.array([[0,1],[-69,-69]])
            self.thy=np.array([[0,1],[-69,-69]])


    def average(self,y,t):
        """
        rhs of mean field model 
        """
        sx = y[0]
        sy = y[1]
        rhs1 = self.eps*(-sx + sqrt(self.a1+self.b1*sx-self.c1*sy))/self.mux
        rhs2 = self.eps*(-sy + sqrt(self.a2+self.b2*sx-self.c2*sy))/self.muy

        return (rhs1,rhs2)
        

    def get_sin_coeffs(self,dat):
        """
        returns coefficients a+b*sin(x) for a function that appears to be sinusoidal
        dat: data array of y-vales
        """
        maxv = np.amax(dat)
        minv = np.amin(dat)
        
        amp = (maxv-minv)/2.
        
        yshift = minv+amp

        # WLOG(?) use shifted sin: A*sin(x-c)
        peak_idx = np.argmax(dat) # solution peak idx

        dom = np.linspace(0,2*pi,len(dat))

        xshift = (np.argmax(yshift+amp*sin(dom))-peak_idx)*2*pi/(1.*len(dat))

        return yshift,amp,xshift


    def lc(self,t,f,b=None,c=None,d=None):
        """
        analytic limit cycle (found using mathematica)
        """
        f2 = f**2.
        per = 1./f

        tt = t + per/2.
        tt2 = t*2*pi*f+pi
        tt3 = t*2*pi*f
            
        if d == 't':
            return (2.*f2*pi)/(cos(f*pi*tt)**2. + f2*sin(f*pi*tt)**2.)
        elif d == None:
            return 2.*np.arctan(f*np.tan(f*pi*tt))
        else:
            raise ValueError("invalid derivative choice for def lc, "+str(d))


    def z(self,t,a,b,c,f=None):
        """
        analytic iPRC (found using mathematica, simplified by hand)
        t: time
        a,b,c: coupling parameters
        f: frequency. If no frequency entered, use mean frequency
        """

        if f == None:
            f = self.sbar

        f2 = f**2
        per = 1./f
        tt = t + per/2.
        return (cos(tt*f*pi)**2 + f2*sin(tt*f*pi)**2)/(2*pi*f2)

    def adj(self,t,f):
        """
        analytic iPRC (found using mathematica, simplified by hand)
        t: time
        f: frequency. If no frequency entered, use mean frequency
        """

        f2=f**2.
        per = 1./f
        tt = t + per/2.
        return (cos(tt*f*pi)**2 + f2*sin(tt*f*pi)**2.)/(2.*pi*f2)

    def run_phase_sim(self):
        print 'Running phase_sim'
        self.thx = np.zeros((self.TN,self.N))
        self.thy = np.zeros((self.TN,self.N))
        
        #self.thx[0,:] = 0#np.random.uniform(size=self.N)*self.period

        if self.use_init_option == '1':
            
            print 'init option 1: x1-x2 antiphase. all other in phase. Tested in xppcall, xppaut'
            
            if ((abs(self.a1-2)+abs(self.b1-5)+abs(self.c1-1)) > 1e-10) and\
               ((abs(self.a2-2)+abs(self.b2-5)+abs(self.c2-1)) > 1e-10):
                print 'warning: parameters are different from testing phase.'

            self.xx = np.array([-2.6,2.282])
            self.yy = np.array([-2.862,-2.862])

            freqx = 4.44
            freqy = 4.44

            self.thx[0,:] = np.arctan(np.tan(self.xx/2.)/freqx)/(freqx*pi)
            self.thy[0,:] = np.arctan(np.tan(self.yy/2.)/freqy)/(freqy*pi)
        
            # phase from -pi to pi
            self.thx[0,:] *= 2*pi*freqx
            self.thy[0,:] *= 2*pi*freqy

        elif self.use_init_option == '2':
            
            print 'init option 1: x1-x2 antiphase. all other in phase. Tested in xppcall, xppaut'
            
            if ((abs(self.a1-2)+abs(self.b1-5)+abs(self.c1-1)) > 1e-10) and\
               ((abs(self.a2-2)+abs(self.b2-5)+abs(self.c2-1)) > 1e-10):
                print 'warning: parameters are different from testing phase.'

            self.xx = np.array([-3.078,1.0946])
            self.yy = np.array([-3.0783,-3.07835])

            freqx = 4.44
            freqy = 4.44

            self.thx[0,:] = np.arctan(np.tan(self.xx/2.)/freqx)/(freqx*pi)
            self.thy[0,:] = np.arctan(np.tan(self.yy/2.)/freqy)/(freqy*pi)
        
            # phase from -pi to pi
            self.thx[0,:] *= 2*pi*freqx
            self.thy[0,:] *= 2*pi*freqy
            


        elif self.use_init_option == '3':
            print 'init option 2: x1-x2 and x1-y1 antiphase. all other in phase. Tested in xppcall, xppaut'            
            
            if ((abs(self.a1-2)+abs(self.b1-5)+abs(self.c1-1)) > 1e-10) and\
               ((abs(self.a2-2)+abs(self.b2-5)+abs(self.c2-1)) > 1e-10):
                print 'warning: parameters are different from testing phase.'

            self.xx = np.array([-2.54045,2.56837])
            self.yy = np.array([1.811140,1.81114])

            freqx = 4.44
            freqy = 4.44

            self.thx[0,:] = np.arctan(np.tan(self.xx/2.)/freqx)/(freqx*pi)
            self.thy[0,:] = np.arctan(np.tan(self.yy/2.)/freqy)/(freqy*pi)
        
            # phase from -pi to pi
            self.thx[0,:] *= 2*pi*freqx
            self.thy[0,:] *= 2*pi*freqy
            

        elif self.use_init_option == '4':

            if ((abs(self.a1-2)+abs(self.b1-5)+abs(self.c1-1)) > 1e-10) and\
               ((abs(self.a2-2)+abs(self.b2-5)+abs(self.c2-1)) > 1e-10):
                print 'warning: parameters are different from testing phase.'

            print 'init option 4: all in phase. Tested in xppcall, xppaut'


            self.xx = np.array([-1.3668,1.26765])
            self.yy = np.array([0.,0.])

            freqx = 4.44
            freqy = 4.44

            self.thx[0,:] = np.arctan(np.tan(self.xx/2.)/freqx)/(freqx*pi)
            self.thy[0,:] = np.arctan(np.tan(self.yy/2.)/freqy)/(freqy*pi)
        
            # phase from -pi to pi
            self.thx[0,:] *= 2*pi*freqx
            self.thy[0,:] *= 2*pi*freqy


        else:
            self.thx[0,0] = 0
            self.thx[0,1] = pi
            self.thy[0,0] = pi
            self.thy[0,1] = 0#np.random.uniform(size=self.N)*self.period

        i = np.arange(0,self.N)

        for k in range(0,self.TN-1):
            stdout.write("\r  ... simulating phase... %d%%" % int((100.*(k+1)/(self.TN-1))))
            stdout.flush()

            sum11 = 0
            sum12 = 0
            sum21 = 0
            sum22 = 0

            for j in range(self.N):

                in11 = self.thx[k,j] - self.thx[k,i]
                in12 = self.thy[k,j] - self.thx[k,i]
                in21 = self.thx[k,j] - self.thy[k,i]
                in22 = self.thy[k,j] - self.thy[k,i]

                sum11 += self.h(in11,'11')
                sum12 += self.h(in12,'12')

                sum21 += self.h(in21,'21')
                sum22 += self.h(in22,'22')

            thxprime = (sum11 + sum12)/self.N
            thyprime = (sum21 + sum22)/self.N

            self.thx[k+1,i] = self.thx[k,i] + self.eps*self.dt*thxprime
            self.thy[k+1,i] = self.thy[k,i] + self.eps*self.dt*thyprime
        print
            
    def load_beta(self):
        """
        load or compute h dynamic function (two varying frequencies)
        """
        file_not_found = False

        print 'loading or recomputing beta functions'

        self.b1file = self.savedir+"b1"+self.paramsfile
        self.b2file = self.savedir+"b2"+self.paramsfile

        while True:

            if self.recompute_beta or file_not_found:

                # get lookup table for sx,sy given mux,muy, get period of oscillation
                # interpolate to be used as function of time

                #trange = np.linspace(0,self.ta[-1],self.ta[-1]/.1) # period of slowmod

                # generate h dynamic coefficients
                self.beta1 = self.generate_beta(self.ta,'1')
                self.beta2 = self.generate_beta(self.ta,'2')

                
                beta_dat1 = self.beta1(self.ta)
                beta_dat2 = self.beta2(self.ta)

                np.savetxt(self.b1file,beta_dat1)
                np.savetxt(self.b2file,beta_dat2)

                #self.beta1 = interp1d(self.ta,beta_dat1)
                #self.beta2 = interp1d(self.ta,beta_dat2)
                
                break

            elif os.path.isfile(self.b1file) and os.path.isfile(self.b2file):

                beta_dat1 = np.loadtxt(self.b1file)
                beta_dat2 = np.loadtxt(self.b2file)

                self.beta1 = interp1d(self.ta,beta_dat1)
                self.beta2 = interp1d(self.ta,beta_dat2)

                break

            else:
                file_not_found = True

    def generate_beta(self,trange,choice,N=None):
        """
        generate beta functions.
        trange: one period of slow limit cycle
        """

        tn = len(trange)
        trange_original = copy.deepcopy(trange)

        if tn > 100:
            # compute the interval required to get len down to 100
            interval = tn/100
            trange = trange[::interval]

            # append final time value so interpolation works on all of self.ta
            trange = np.append(trange,trange_original[-1])
            tn = len(trange)

        beta = np.zeros(tn)

        freq1 = sqrt(self.a1 + self.b1*self.sxa(trange) -self.c1*self.sya(trange))
        freq2 = sqrt(self.a2 + self.b2*self.sxa(trange) -self.c2*self.sya(trange))

        if N == None:
            N = self.integration_N

        dsxa = self.dsxa(trange)
        dsya = self.dsya(trange)

        for k in range(tn):

            stdout.write("\r  ... building beta"+str(choice)+"... %d%%" % int((100.*(k+1)/tn)))
            stdout.flush()

            v1 = np.linspace(0.001,1./freq1[k],N) # integration variable
            v2 = np.linspace(0.001,1./freq2[k],N) # integration variable
            #s = np.linspace(0,1./freq,N) # integration variable
            tot = 0


            for i in range(N):

                if choice == '1':
                    freq = freq1[k]
                    tot += self.adj(v1[i],freq) * self.lc(v1[i],freq,self.b1,self.c1,d='sx') * dsxa[k] +\
                           self.adj(v1[i],freq) * self.lc(v1[i],freq,self.b1,self.c1,d='sy') * dsya[k]

                elif choice == '2':
                    freq = freq2[k]
                    tot += self.adj(v2[i],freq) * self.lc(v2[i],freq,self.b2,self.c2,d='sx') * dsxa[k] +\
                           self.adj(v2[i],freq) * self.lc(v2[i],freq,self.b2,self.c2,d='sy') * dsya[k]

            tot *= freq*2*pi/(N*self.eps)
            #tot /= N*self.eps
            beta[k] = tot

        print
    
        return interp1d(trange,beta)


    def generate_dlc_sx(self):
        """
        brute-force generate dPhi/d(sx), compare to analytic in plots
        """
        TN = 200
        fN = 100

        frange = np.linspace(self.sbar-.05,self.sbar+.05,fN)

        trange = np.linspace(0,2*pi,TN)
        data_diff = np.zeros((fN-1,TN)) #freq,time
        data_lc = np.zeros((fN-1,TN)) #freq,time

        dx = (frange[-1]-frange[0])/fN


        for i in range(fN-1):
            temp1 = self.lc(trange/(2*pi*frange[i]),frange[i])
            temp2 = self.lc(trange/(2*pi*frange[i+1]),frange[i+1])

            data_lc[i,:] = temp1
            
            data_diff[i,:] = (temp2 - temp1)/(1.*dx)

        return data_diff,data_lc,trange


    def h(self,theta,choice):
        """
        if h are sin functions
        t: time (must be within one period of slow osc)
        theta: domain value
        """

        #dom,h = self.generate_h(self.sxa(t),self.sya(t),choice,return_domain=True)
        #fn = interp1d(dom,h)
        #return fn(theta)

        if (choice == '11') or (choice == 'xx'):
            return self.h11(np.mod(theta,self.interpdom[-1]))
        elif (choice == '12') or (choice == 'xy'):
            return self.h12(np.mod(theta,self.interpdom[-1]))
        elif (choice == '21') or (choice == 'yx'):
            return self.h21(np.mod(theta,self.interpdom[-1]))
        elif (choice == '22') or (choice == 'yy'):
            return self.h22(np.mod(theta,self.interpdom[-1]))


    def load_h(self):
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

                dom,h11 = self.generate_h(self.sbar,self.sbar,'11',return_domain=True)
                h12 = self.generate_h(self.sbar,self.sbar,'12',return_domain=False)
                h21 = self.generate_h(self.sbar,self.sbar,'21',return_domain=False)
                h22 = self.generate_h(self.sbar,self.sbar,'22',return_domain=False)

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


    def generate_h(self,sx,sy,choice,N=None,return_domain=False):
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

        s = np.linspace(0,1./freq,N) # integration variable

        tot = 0
        for i in range(N):
            tot += self.adj(s[i],freq) * (1.+cos(self.lc(s[i],freq))) * ff(s[i]+s,freq)

        tot *= p/N
        if return_domain:
            return np.linspace(0,2*pi,N),tot

        return tot

    def generate_h_coeffs(self,trange,choice,deg=2.,plot_data=False):
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


    def fx(self,t,f=None):
        if f == None:
            f = 1./self.period
        return (np.mod(1.-t*f,1.)-.5)/self.mux

    def fy(self,t,f=None):
        if f == None:
            f = 1./self.period
        return (np.mod(1.-t*f,1.)-.5)/self.muy

    def normalization(self):
        """
        numerical approximation of normalization condition.
        """
        #self.slow_osc_exist = slow_osc(self.sbar,
        #                               self.a1,self.b1,self.c1,
        #                               self.a2,self.b2,self.c2,
        #                               self.mux,self.muy)
        
        
        if self.slow_osc_exist:
            # skip some frequencies to save computation time
            freq1 = self.freqxa(self.ta)[::100]
            freq2 = self.freqya(self.ta)[::100]
        else:
            # if no oscillation, I have 2 identical values in frequency array.
            freq1 = self.freqxa(self.ta)
            freq2 = self.freqya(self.ta)

        tot1 = 0
        tot2 = 0
        
        normal1 = np.zeros(len(freq1))
        normal2 = np.zeros(len(freq2))

        
        for k in range(len(freq1)):
            s1 = np.linspace(0,1./freq1[k],self.integration_N) # integration variable
            s2 = np.linspace(0,1./freq2[k],self.integration_N) # integration variable

            for i in range(self.integration_N):

                tot1 += self.adj(s1[i],freq1[k]) * self.lc(s1[i],freq1[k],d='t')
                tot2 += self.adj(s2[i],freq2[k]) * self.lc(s2[i],freq2[k],d='t')

            tot1 *= 2*pi*freq1[k]/self.integration_N
            tot2 *= 2*pi*freq2[k]/self.integration_N

            #tot1 /= self.integration_N
            #tot2 /= self.integration_N

            normal1[k] = tot1
            normal2[k] = tot2

        return normal1, normal2

    def norm_no_int(self):
        """
        normalization without integrating. 
        functions should be constant in time.
        """
                
        if self.slow_osc_exist:
            # skip some frequencies to save computation time
            freq1 = self.freqxa(self.ta)[::100]
            freq2 = self.freqya(self.ta)[::100]
        else:
            # if no oscillation, I have 2 identical values in frequency array.
            freq1 = self.freqxa(self.ta)
            freq2 = self.freqya(self.ta)

        tot1 = 0
        tot2 = 0
        
        normal1 = np.zeros((len(freq1),self.integration_N))
        normal2 = np.zeros((len(freq2),self.integration_N))

        
        # for each frequency value, find the product of the functions on the domain
        # linspace(0,1./freqi[k],self.integration_N).
        for k in range(len(freq1)):
            s1 = np.linspace(0,1./freq1[k],self.integration_N) # domain
            s2 = np.linspace(0,1./freq2[k],self.integration_N) # domain

            normal1[k,:] = self.adj(s1,freq1[k]) * self.lc(s1,freq1[k],d='t')
            normal2[k,:] = self.adj(s2,freq2[k]) * self.lc(s2,freq2[k],d='t')

        return np.linspace(0,2*pi,self.integration_N),normal1, normal2



    def plot(self,choice='h_generated'):
        fig = plt.figure()
        ax = plt.subplot(111)
        
        #tt = np.linspace(0,self.ta[-1]/4.,10) # 1/4 period of slowmod

        #sx = self.sxa(tt)
        #sy = self.sya(tt)
        
        freq1 = [sqrt(self.a1 + self.b1*self.sbar - self.c1*self.sbar)]
        freq2 = [sqrt(self.a2 + self.b2*self.sbar - self.c2*self.sbar)]
        
        if choice == 'h_generated':
            ax1 = plt.subplot(221)
            
            ax1.set_title("h11")
            
            N = 10
            N2 = N/2
            ax1.plot(self.phi,self.h(self.phi,'11'))
            
            ax2 = plt.subplot(222)
            ax2.set_title("h12")
            ax2.plot(self.phi,self.h(self.phi,'12'))

            ax3 = plt.subplot(223)
            ax3.set_title("h21")
            ax3.plot(self.phi,self.h(self.phi,'21'))

            ax4 = plt.subplot(224)
            ax4.set_title("h22")
            ax4.plot(self.phi,self.h(self.phi,'22'))
            
            plt.suptitle("raw generated h funs"+self.paramstitle)
            plt.subplots_adjust(top=0.85)
            

        elif choice == 'z':

            s = np.linspace(0,1./self.sbar,100) # integration variable

            ax1 = plt.subplot(121)
            ax1.set_title("z1")

            ax1.plot(s,self.adj(s,freq1[0]))

            
            ax2 = plt.subplot(122)
            ax2.set_title("z2")

            ax2.plot(s,self.adj(s,freq2[0]))

        
            ax.legend()

        elif choice == 'lc':
            s = np.linspace(0,1./self.sbar,200) # integration variable

            ax1 = plt.subplot(121)
            ax1.set_title("lc1")

            ax1.plot(s,self.lc(s,freq1[0]))
            
            ax2 = plt.subplot(122)
            ax2.set_title("lc2")

            ax2.plot(s,self.lc(s,freq2[0]))


        elif choice == 'lc-deriv':
            s = np.linspace(0,2*pi,200) # integration variable

            z = np.linspace(0,1,len(sx))
            color = plt.cm.Greys(z)

            plt.suptitle("Limit Cycle"+self.paramstitle)

            ax1 = plt.subplot(121)
            ax1.set_title("lc1-deriv-sx(black),sy(red)")

            for i in range(len(freq1)):
                ivar1 = s/(2*pi*freq1[i])
                ax1.plot(s,self.lc(ivar1,freq1[i],self.b1,self.c1,d='sx'),color='black')
                ax1.plot(s,self.lc(ivar1,freq1[i],self.b1,self.c1,d='sy'),color='red')
            
            ax2 = plt.subplot(122)
            ax2.set_title("lc2-deriv-sx(black),sy(red)")

            for i in range(len(freq2)):
                ivar2 = s/(2*pi*freq2[i])
                ax2.plot(s,self.lc(ivar2,freq2[i],self.b2,self.c2,d='sx'),color='black')
                ax2.plot(s,self.lc(ivar2,freq2[i],self.b2,self.c2,d='sy'),color='red')

            #ax.legend()

        elif choice == 'dlc':
            s = np.linspace(0,2*pi,200) # integration variable

            z = np.linspace(0,1,len(sx))
            color = plt.cm.Greys(z)

            plt.suptitle("Limit Cycle Derivative"+self.paramstitle)

            ax1 = plt.subplot(121)
            ax1.set_title("lc1-deriv-sx(black),sy(red)")

            for i in range(len(freq1)):
                ivar1 = s/(2*pi*freq1[i])
                ax1.plot(s,self.lc(ivar1,freq1[i],self.b1,self.c1,d='sx'),color='black')
                ax1.plot(s,self.lc(ivar1,freq1[i],self.b1,self.c1,d='sy'),color='red')
            
            ax2 = plt.subplot(122)
            ax2.set_title("lc2-deriv-sx(black),sy(red)")

            for i in range(len(freq2)):
                ivar2 = s/(2*pi*freq2[i])
                ax2.plot(s,self.lc(ivar2,freq2[i],self.b2,self.c2,d='sx'),color='black')
                ax2.plot(s,self.lc(ivar2,freq2[i],self.b2,self.c2,d='sy'),color='red')
            

        elif choice == 'fx':
            freq = .95
            ax.set_title("fx"+self.paramstitle+"_freq="+str(freq))

            s = np.linspace(0,2*pi,100) # integration variable
            ivar = s/(2*pi*freq)

            ax.plot(ivar,self.fx(ivar,freq))

        elif choice == 'fy':
            freq = .95
            ax.set_title("fy"+self.paramstitle+"_freq="+str(freq))

            s = np.linspace(0,2*pi,100) # integration variable
            ivar = s/(2*pi*freq)

            ax.plot(ivar,self.fy(ivar,freq))

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

            ax2.set_xlim(0,self.period)
            ax2.set_ylim(0,self.period)

            plt.suptitle("th-plane"+self.paramstitle)
            plt.subplots_adjust(top=.85)


        elif choice == 'thdiff':
            ax = plt.subplot(111)#fig.add_subplot(121)
            ax.set_title('thy1 - thx1')

            ax.scatter(self.t,np.mod(self.thx[:,1]-self.thx[:,0]+pi,2*pi)-pi,label='thx2-thx1',color='blue')
            ax.scatter(self.t,np.mod(self.thy[:,1]-self.thy[:,0]+pi,2*pi)-pi,label='thy2-thy1',color='red')
            ax.scatter(self.t,np.mod(self.thy[:,0]-self.thx[:,0]+pi,2*pi)-pi,label='thy1-thx1',color='green')

            ax.set_xlim(self.t[0],self.t[-1])
            ax.set_ylim(-pi-.01,pi+.01)
            
            ax.legend()


        elif choice == 'n1rhs':
            ax = plt.subplot(111)#fig.add_subplot(121)
            ax.set_title('N=1, rhs')
            
            ax.plot(self.phi,
                    self.h(-self.phi,'21')+self.h(0,'22')
                    -self.h(0,'11')-self.h(self.phi,'12'))


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
            ax.plot(self.phi,1/self.adj(self.sbar*self.phi/(2*pi),self.sbar))
        
        elif choice == 'dlc/dt':
            ax = plt.subplot(111)

        else:
            raise ValueError('Invalid plot choice (in class phase): '+str(choice))

        #plt.tight_layout()

        return fig


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "lvserhpf", ["use-last","save-last","use-ss","save-ss","use-random","help","run-phase","run-full"])

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    use_last=False;save_last=False;use_ss=False;save_ss=False;use_random=False
    run_full=False;run_phase=False

    if opts == []:
        print "Please run using flags -p (phase model) and/or -f (full model)"
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):

            usage()
            sys.exit()
        else:
            if opt in ("-l","--use-last"):
                use_last = True
                print "use_last=True"
            elif opt in ('-v','--save-last'):
                save_last = True
                print "save_last=True"
            elif opt in ('-s','--use-ss'):
                use_ss = True
                print "use_ss=True"
            elif opt in ('-e','save-ss'):
                save_ss = True
                print "save_ss=True"
            elif opt in ('-r','use-random'):
                use_random = True
                print "use_random=True"
            elif opt in ('-p','run-phase'):
                run_phase = True
                print "run class phase=True"
            elif opt in ('-f','run-full'):
                run_full = True
                print "run class theta (full sim)=True"


    mux = 1.
    muy = 1.
    N = 2

    a1=2.;b1=5.;c1=1.
    a2=2.;b2=5.;c2=1.

    if run_full:

        sim = Theta(use_last=use_last,
                    save_last=save_last,
                    use_ss=use_ss,
                    save_ss=save_ss,
                    a1=a1,b1=b1,c1=c1,
                    a2=a2,b2=b2,c2=c2,
                    T=2000,eps=.01,dt=.005,muy=muy,N=N,
                    use_init_option='4')

        #sim = Theta(T=5000,a=a,b=b,c=c,mux=mux,muy=muy)
        #sim.plot(cutoff=False)
        sim.plot('xj-yj',cutoff=False)
        sim.plot('phase-diff')

        #sim.plot('phase')
        
        #sim.plot('freq')
        #sim.plot('period')
        #sim.get_sbar(check=True)

        #print sim.sbar
        #print sim.periodx
        #print sim.periody
        #print sim.period_brute

        #f = 1./np.abs(sim.spike_t1[-1]-sim.spike_t1[-2])
        #print freq (numerical)=',f,'; sbar=',sim.sbar,'; period=',sim.period,'; freq (analytic)=',1./sim.period
    

    if run_phase:
        phase = Phase(run_full=False,
                      recompute_h=True,
                      run_phase=True,
                      recompute_slow_lc=False,
                      a1=a1,b1=b1,c1=c1,
                      a2=a2,b2=b2,c2=c2,
                      T=5000,dt=.05,muy=muy,N=2,eps=.01,
                      use_init_option='4')

        #phase.plot('fx')
        #phase.plot('fy')
        #phase.plot('z')
        #phase.plot('lc')
        #phase.plot('lc-deriv')

        phase.plot('h_generated')

        #phase.plot('beta_generated') # takes too long. just recompute and plot beta_approx
        #phase.plot('beta_approx')
        phase.plot('n1rhs')


        #phase.plot('th-plane')
        #phase.plot('th-t')
        phase.plot('thdiff')
        #phase.plot('si')
        #phase.plot('dsi')

        #phase.plot('normalization')
        #phase.plot('norm-no-int')
        #phase.plot('dphi-dsx')
        #phase.plot('dphi-dsy')

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
