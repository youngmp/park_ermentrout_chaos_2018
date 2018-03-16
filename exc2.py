"""
a pair of mutually excitatory pulse coupled neurons.

"""

import thetaslowmod as thsm

import numpy as np
#from scipy.integrate import odeint
import matplotlib.pyplot as plt

np.random.seed(0)

cos = np.cos
sin = np.sin
pi = np.pi
sqrt = np.sqrt


class Full(object):
    """
    full sim. for simulating the original and averaged system.
    """
    def __init__(self,
                 mode='full',
                 run=True,
                 a=1.,b=.5,
                 eps=.01,
                 mu1=1.,mu2=1.,
                 t0=0.,T=1000.,dt=.01):

        self.mode = mode
        self.run = run

        self.a = a
        self.b = b
        self.eps = eps
        self.mu1 = mu1
        self.mu2 = mu2
        self.t0 = t0
        self.T = T
        self.dt = dt
        self.TN = int((self.T-self.t0)/self.dt)
        self.t = np.linspace(self.t0,self.T,self.TN)

        self.sbar = self.get_sbar()
        self.period = self.get_period()

        print 'sbar =',self.sbar

        self.x1_spikes = []

        #self.period = pi/np.sqrt(a+b*sbar)
        
        if self.run == True:
            self.x1,self.x2,self.s1,self.s2 = self.sim_full()



    def get_sbar(self):
        return (self.b+sqrt(self.b**2 + 4*self.a*pi**2))/(2*pi**2)
        #return (.5*(self.b+sqrt(4*self.a+self.b**2)),
        #        sqrt(self.a+.5*self.b*(self.b+sqrt(4*self.a+self.b**2))))
        
    def get_period(self):
        return pi/sqrt(self.a+self.b*self.sbar)


    def sim_full(self,x10=1.5,x20=0,s10=.34,s20=.34):
        
        x1 = np.zeros(self.TN)
        x2 = np.zeros(self.TN)
        s1 = np.zeros(self.TN)
        s2 = np.zeros(self.TN)
        
        x1[0] = x10
        x2[0] = x20
        s1[0] = s10
        s2[0] = s20
        


        for i in range(0,self.TN-1):
            x1prime = 1 - cos(x1[i]) + (1 + cos(x1[i]))*(self.a+self.b*s2[i])
            x2prime = 1 - cos(x2[i]) + (1 + cos(x2[i]))*(self.a+self.b*s1[i])
            
            # count delta function contributions
            #deltaxsum = np.sum((xj[i-1,:]<pi)*(xj[i,:]>pi))/(2.*taux*N)
            #deltaysum = np.sum((yj[i-1,:]<pi)*(yj[i,:]>pi))/(2.*tauy*N)
            
            if (x1[i-1]<pi)*(x1[i]>pi):
                self.x1_spikes.append(self.t[i])

            delta1 = np.sum((x1[i-1]<pi)*(x1[i]>pi))
            delta2 = np.sum((x2[i-1]<pi)*(x2[i]>pi))
            
            x1[i+1] = np.mod(x1[i] + self.dt*x1prime,2*pi)
            x2[i+1] = np.mod(x2[i] + self.dt*x2prime,2*pi)
            s1[i+1] = s1[i] + self.dt*(-self.eps*s1[i]/self.mu1) + self.eps*delta1/(1.*self.mu1)
            s2[i+1] = s2[i] + self.dt*(-self.eps*s2[i]/self.mu2) + self.eps*delta2/(1.*self.mu2)

        return x1,x2,s1,s2

    def sim_x_ana(self):
        """
        for debugging. simulation with analytic x
        """
        pass

    def sim_s_ana(self):
        """
        for debugging. simulation with analytic s_i
        """
        pass

    def sim_avg(self):
        """
        averaged sim
        """
        pass

    def get_full_ss(self):
        """
        get full numerical steady state
        """
        pass

    def plot(self,choice='main',cutoff=False):
        fig = plt.figure()

        if cutoff:
            tstart = 1000#self.T - self.periodx*10
        else:
            tstart = 0
        ts_idx = int(tstart/self.dt)

        z = np.linspace(0,1,self.TN-ts_idx)
        color = plt.cm.Greys(z)

        if choice == 'main':
            ax = plt.subplot(121)
            ax.set_title("th1,th2, full")
            ax.plot(self.t[ts_idx:],self.x1[ts_idx:])
            ax.plot(self.t[ts_idx:],self.x2[ts_idx:])

            ax = plt.subplot(122)
            ax.set_title("s1,s2, full")
            ax.plot(self.t[ts_idx:],self.s1[ts_idx:])
            ax.plot(self.t[ts_idx:],self.s2[ts_idx:])

        elif choice == 'x1-vs-x2':
            ax = plt.subplot(111)
            ax.scatter(self.x1[ts_idx:],self.x2[ts_idx:],color=color)

        elif choice == 'freq':
            ax = plt.subplot()
            ax.plot(self.x1_spikes[1:],1./np.diff(self.x1_spikes))
            ax.plot([self.t[0],self.t[-1]],[self.sbar,self.sbar])

        elif choice == 'phase-diff':
            ax = plt.subplot(111)
            ax.set_title("numerical phase diff. period="+str(self.period))
            
            phasex = np.arctan(np.tan(self.x1/2.)/sqrt(self.a+self.b*self.sbar))/sqrt(self.a+self.b*self.sbar)
            phasey = np.arctan(np.tan(self.x2/2.)/sqrt(self.a+self.b*self.sbar))/sqrt(self.a+self.b*self.sbar)
            diff = np.mod(phasey-phasex,self.period)#+self.period/2.,self.period)-self.period/2.
            for slc in thsm.unlink_wrap(diff,[-self.period/2.,self.period/2.]):
                ax.plot(self.t[slc],diff[slc],color='blue')
            
            ax.set_ylim(0-.1,self.period+.1)
            #ax.set_ylim(-self.period/2.-.1,self.period/2.+.1)

            ax.plot([self.t[0],self.t[-1]],[self.period/2.,self.period/2.],color='gray')

        return fig
        #plt.show()
        

class Phase(object):
    def __init__(self,
                 a=1.,b=.5,
                 eps=.01,
                 mu1=1.,mu2=1.,
                 t0=0.,T=1000.,dt=.01):
        #Full.__init__(self)

        self.a = a
        self.b = b
        self.eps = eps
        self.mu1 = mu1
        self.mu2 = mu2
        self.t0 = t0
        self.T = T
        self.dt = dt
        self.TN = int((self.T-self.t0)/self.dt)
        self.t = np.linspace(self.t0,self.T,self.TN)

        self.sbar = self.get_sbar()
        self.period = self.get_period()

    def get_sbar(self):
        return (self.b+sqrt(self.b**2 + 4*self.a*pi**2))/(2*pi**2)
        #return (.5*(self.b+sqrt(4*self.a+self.b**2)),
        #        sqrt(self.a+.5*self.b*(self.b+sqrt(4*self.a+self.b**2))))
        
    def get_period(self):
        return pi/sqrt(self.a+self.b*self.sbar)

    def x0(self,t):
        """
        limit cycle
        """
        c = self.a + self.b*self.sbar
        return 2*np.arctan(sqrt(c)*np.tan(sqrt(c)*(t+self.period/2.)))
    
    def z(self,t):
        c = self.a + self.b*self.sbar
        tt = t + self.period/2.
        return (cos(sqrt(c)*tt)**2 + c*sin(sqrt(c)*tt)**2)/(2*c)
        #return (cos(2.*sqrt(c)*t)*(1-c)+1+c)/(4.c)

    def sim_phase(self):
        
        pass

    def h1(self,phi):
        s = 0 # integration variable
        tot = 0
        while s < self.period:
            tot += self.b*self.z(s)*(1+cos(self.x0(s)))*self.q2(s+phi)*self.dt
            s += self.dt
        tot *= self.period
        return tot

    def h2(self,phi):
        s = 0 # integration variable
        tot = 0
        while s < self.period:
            tot += self.b*self.z(s)*(1+cos(self.x0(s)))*self.q1(s+phi)*self.dt
            s += self.dt
        tot *= self.period
        return tot

    def q1(self,t):
        return (np.mod(1-t/self.period,1.)-.5)/self.mu1

    def q2(self,t):
        return (np.mod(1-t/self.period,1.)-.5)/self.mu2

    def plot(self,choice='h'):
        fig = plt.figure()
        phi = np.linspace(0,self.period,100)
        
        if choice == 'h':
            ax = plt.subplot(121)
            ax.set_title("h1")
            ax.plot(phi,self.h1(phi))

            ax = plt.subplot(122)
            ax.set_title("h2")
            ax.plot(phi,self.h2(phi))

        elif choice == 'h2-h1':
            ax = plt.subplot(111)
            ax.set_title("-2hodd")
            ax.plot(phi,self.h2(-phi),label='h2(-x)')
            ax.plot(phi,-self.h1(phi),label='-h1(x)')
            ax.plot(phi,self.h2(-phi)-self.h1(phi),label='sum')
            ax.legend()

        elif choice == 'z':
            ax = plt.subplot(111)
            ax.set_title("z")
            ax.plot(phi,self.z(phi))

        elif choice == 'x0':
            ax = plt.subplot(111)
            ax.set_title("x0 (limit cycle)")
            ax.plot(phi,self.x0(phi))
        return fig

def main():
    mu1=1.;mu2=1.
    
    full = Full(mu1=mu1,mu2=mu2,T=5000)
    #full.plot(cutoff=True)
    #full.plot('x1-vs-x2',cutoff=True)
    full.plot('phase-diff')

    #full.plot('freq')
    

    if True:
        phase = Phase(mu1=mu1,mu2=mu2)
        #phase.plot(choice='h')
        phase.plot(choice='h2-h1')
        #phase.plot(choice='z')
        #phase.plot(choice='x0')
        phase.plot('z')

    plt.show()
        

if __name__ == "__main__":
    main()
