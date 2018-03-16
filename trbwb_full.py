"""
weak/slow coupling between WB and Traub Ca. 

Traub Ca is classically used as excitatory neurons,
WB is classically used as inhibitory neurons

Notes: We choose input parameters such that the frequencies are sufficiently identical. To ensure that the numerical integration of H functions is consistent, we use the period/frequency obtained by taking the average of the excitatory synaptic variable and forcing all domains to match this period/frequency. 

TODO:
-refine the smoothing function for export to phase model.
""" 

import matplotlib.pylab as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from scipy.integrate import odeint

import sys
import os
import getopt
from scipy.interpolate import interp1d
from xppcall import xpprun, read_numerics, read_pars
from thetaslowmod_lib import get_averaged

np.random.seed(10)

cos = np.cos
sin = np.sin
pi = np.pi
sqrt = np.sqrt
exp = np.exp


#plt.ion()

class Traub_Wb(object):
    def __init__(self,
                 use_last=False,
                 save_last=False,
                 use_ss=False,

                 save_ss=False,
                 check_sbar=False,
                 save_phase=False,

                 mux=1.,muy=5.,eps=.0025,
                 iwb=0.,itb=0.,
                 gee=1,gei=1,
                 gie=1,gii=1,

                 itb_mean=0,iwb_mean=0,
                 T=500,dt=.01,
                 phs_init_trb=None,

                 phs_init_wb=None,
                 population='both',
                 sbar=0.05,
                 sx0=0.,sy0=0.
                 ):

        """
        run: if True, run the full simulation
        use_last: if True, use final value of previous sim (use default init if no data)
        use_ss: use steady-state terms
        save_ss: save final value of sim as steady-state terms if desired
        
        phs_init_trb: initial phase values of traub systems with array length N
        phs_init_wb:initial phase values of wb systems with array length N
        
        population: 'e','i','both'. e for excitatory only, i for inhibitory only, and both for both populations. Used for debugging.
        """

        self.save_phase = save_phase
        self.use_last = use_last
        self.use_ss = use_ss
        #self.recompute_sbar = recompute_sbar
        self.check_sbar = check_sbar
        
        self.population = population

        assert(len(phs_init_trb) == len(phs_init_wb))
        self.N = len(phs_init_trb)

        # slow params
        self.mux=mux;self.muy=muy;self.eps=eps
        self.eps = eps

        # coupling params
        self.gee=gee;self.gei=gei
        self.gie=gie;self.gii=gii

        self.itb_mean=itb_mean;self.iwb_mean=iwb_mean
        self.itb = self.itb_mean - sbar*(self.gee - self.gei)
        self.iwb = self.iwb_mean - sbar*(self.gie - self.gii)

        self.dt = dt
        self.T = T
        self.t = np.linspace(0,self.T,int(self.T/self.dt))

        self.use_last=use_last
        self.save_last = save_last
        self.use_ss=use_ss
        self.save_ss = save_ss

        # GENERATE DIRECTORIES AND DEFINE FILENAMES
        self.generate_dirs()

        self.filename_wb = self.savedir + 'wb_init'+str(self.N)+'.dat'
        self.filename_trb = self.savedir + 'trb_init'+str(self.N)+'.dat'
        self.filename_s = self.savedir + 's_init'+str(self.N)+'.dat'

        # include a,b,c, 
        self.filename_wb_ss = self.savedir + 'wb_init'+str(self.N)+'.dat'
        self.filename_trb_ss = self.savedir + 'trb_init'+str(self.N)+'.dat'
        self.filename_s_ss = self.savedir + 's_init'+str(self.N)+'.dat'

        # MEAN/SBAR
        if self.check_sbar:
            # use this fn to check that sbar manual matches sbar from sims
            stbmean,swbmean = self.check_sbar() 
        
        self.sbarx = sbar
        self.sbary = sbar
        self.freqx_base = self.sbarx
        self.freqy_base = self.sbary
        self.perx_base = 1./self.freqx_base
        self.pery_base = 1./self.freqy_base

        # PHASE 
        self.load_phase_lookup_tables()

        self.phs_init_trb = phs_init_trb
        self.phs_init_wb = phs_init_wb

        if (self.phs_init_trb != None) and (self.phs_init_wb != None):
            #rint len(self.phs_init_trb),len(self.phs_init_wb)
            assert ((len(self.phs_init_trb) == self.N) and (len(self.phs_init_wb) == self.N))

            # convert phase from [0,1] to [0,T]
            self.phs_init_trb = np.array(self.phs_init_trb)*self.perx_base
            self.phs_init_wb = np.array(self.phs_init_wb)*self.pery_base

        self.sx0 = sx0
        self.sy0 = sy0

        # RUN SIMULATION
        if self.T == 0:

            self.t = np.linspace(0,1)
            self.T = self.t[-1]
            self.dt = 0.01

            # define solution arrays to fit xpp output
            self.phasex = np.zeros((len(self.t),self.N))
            self.phasey = np.zeros((len(self.t),self.N))

            # solution vectors
            self.trba= np.zeros((len(self.t),self.N,7))
            self.wba = np.zeros((len(self.t),self.N,3))

            self.stb = np.zeros(len(self.t))
            self.swb = np.zeros(len(self.t))

            self.sx_smooth = self.sx0
            self.sy_smooth = self.sy0

            self.freqx = 1./self.perx_base
            self.freqy = 1./self.pery_base

            self.perx = 1./self.freqx
            self.pery = 1./self.freqy

        else:
            self.run_full_sim()
            self.get_frequency_and_phase_data()

        if self.save_phase:
            nskip = 20
            # shortened solutions
            short_N = len(self.t[::nskip])
            t_short = self.t[::nskip]
            tb_short = self.trba[::nskip,:,:]
            wb_short = self.wba[::nskip,:,:]
            perx_short = self.perx[::nskip]
            pery_short = self.pery[::nskip]

            stb_short = self.stb[::nskip]
            swb_short = self.swb[::nskip]
            
            phs_t = np.zeros(short_N)
            phs_trb1 = np.zeros(short_N)
            phs_trb2 = np.zeros(short_N)

            phs_wb1 = np.zeros(short_N)
            phs_wb2 = np.zeros(short_N)
            
            for k in range(short_N):
                phs_t[k] = t_short[k]

                phs_trb1[k] = self.sv2phase(tb_short[k,0,:],'trb')
                phs_wb1[k] = self.sv2phase(wb_short[k,0,:],'wb')

                if self.N == 2:
                    phs_trb2[k] = self.sv2phase(tb_short[k,1,:],'trb')
                    phs_wb2[k] = self.sv2phase(wb_short[k,1,:],'wb')

            np.savetxt(self.savedir+'phase_t_tbwb.dat',phs_t)
            np.savetxt(self.savedir+'phase_tb1.dat',phs_trb1)
            np.savetxt(self.savedir+'phase_tb2.dat',phs_trb2)
            np.savetxt(self.savedir+'phase_wb1.dat',phs_wb1)
            np.savetxt(self.savedir+'phase_wb2.dat',phs_wb2)
            np.savetxt(self.savedir+'phase_stb.dat',stb_short)
            np.savetxt(self.savedir+'phase_swb.dat',swb_short)


    def get_frequency_and_phase_data(self):
        """
        given the simulation, smooth the synaptic variables to approximate the periods
        
        """

        # total number of entries in kernel
        div = 20
        kernelSize=len(self.t)/div

        # padding size. must be greater than kernelSize
        padN = kernelSize+div+1

        # load the frequency data
        self.tbfi_data = np.loadtxt('tbfi2.dat')
        self.wbfi_data = np.loadtxt('wbfi.dat')

        # convert to frequency functions
        self.tbfi = interp1d(self.tbfi_data[:,0],self.tbfi_data[:,1])
        self.wbfi = interp1d(self.wbfi_data[:,0],self.wbfi_data[:,1])

        self.tbfi_inv = interp1d(self.tbfi_data[:,1],self.tbfi_data[:,0])
        self.wbfi_inv = interp1d(self.wbfi_data[:,1],self.wbfi_data[:,0])
        
        self.sx_smooth = get_averaged(self.stb,dt=self.dt,
                                      padN=padN,kernelSize=kernelSize,
                                      time_pre=-self.t[padN],time_post=self.t[padN])
        self.sy_smooth = get_averaged(self.swb,dt=self.dt,
                                      padN=padN,kernelSize=kernelSize,
                                      time_pre=-self.t[padN],time_post=self.t[padN])
        
        self.freqx = self.tbfi(self.itb+self.gee*self.sx_smooth - self.gei*self.sy_smooth)
        self.freqy = self.wbfi(self.iwb+self.gie*self.sx_smooth - self.gii*self.sy_smooth)

        self.perx = 1./self.freqx
        self.pery = 1./self.freqy

    def load_phase_lookup_tables(self):
        """
        on call, by default, load data files for phase estimation.

        if on call, load_inhom_freq_tbls = True, then load remaining lookup tables.
        
        load_inhom_freq_tbls: by default we do not load the tables for inhomogeneous frequencies (i.e. frequencies differing from sx,sy)
        i: input current value

        The remaining tables were generated using XPP (verbatim from adj2.c in xppsrc):

        Step 1. Compute a singel orbit, adjoint, and H function (to load the 
        program with the correct right-hand sides for H function. Or just load in
        set file where it was done
        Step 2.  Set transient to some reasonable number to assure convergence 
        onto the limit cycle as you change parameters and total to be at least
        2 periods beyond the transient
        Step 3. Se up Poincare map - period - stop on section. This lets you
        get the period
        Step 4. In numerics averaging - click on adjrange
        Step 5. Initconds range over the parameter. It should find the periodic
        orbit, adjoint, and H function and save. Files are of the form
        orbit.parname_parvalue.dat etc
        
        """

        self.tb_var_names = ['vt','mt','nt','ht','wt','st','ca']
        self.wb_var_names = ['v','h','n']

        self.tb_valid_currents = np.array([ 5.41,5.4825,5.555,5.6275,5.7,5.7725,5.845,
                                            5.9175,5.99,    6.0416,    6.0625,6.135,6.2075,6.28,6.3525,
                                            6.425,6.4975,6.57,6.6425,6.715,6.7875])

        self.wb_valid_currents = np.array([0.69,0.7045,0.719,0.7335,0.748,0.7625,0.777,
                                           0.7915,0.806,    0.809079373711,    0.8205,0.835,0.8495,0.864,0.8785,
                                           0.893,0.9075,0.922,0.9365,0.951,0.9655])

        if (self.sbarx != 0.05) or (self.sbary != 0.05):
            print 'Warning: lookup tables by default assume sbarx,sbary == 0.05). Current sbarx,sbary=',self.sbarx,self.sbary

        self.tb_tab = {}
        self.wb_tab = {}
        
        for i in range(len(self.tb_valid_currents)):

            current = self.tb_valid_currents[i]
            if current == 6.0416:
                suffix = ''
            else:
                suffix = '_i'+str(current)
                    
            # trb_ca


            for name in self.tb_var_names:
                self.tb_tab[name+suffix] = np.loadtxt('tabular/'+name+suffix+'.tab')[3:]
                self.tb_tab[name+'_var'+suffix] = np.var(self.tb_tab[name+suffix])
            self.tb_tab['per'+suffix] = np.loadtxt('tabular/vt'+suffix+'.tab')[2]
            self.tb_tab['N'+suffix] = np.loadtxt('tabular/vt'+suffix+'.tab')[0]

            # WB

        for i in range(len(self.wb_valid_currents)):

            current = self.wb_valid_currents[i]
            if current == 0.809079373711:#6.0416:
                suffix = ''
            else:
                suffix = '_i0'+str(current)
            
            for name in self.wb_var_names:
                self.wb_tab[name+suffix] = np.loadtxt('tabular/'+name+suffix+'.tab')[3:]
                self.wb_tab[name+'_var'+suffix] = np.var(self.wb_tab[name+suffix])
            self.wb_tab['per'+suffix] = np.loadtxt('tabular/v'+suffix+'.tab')[2]
            self.wb_tab['N'+suffix] = np.loadtxt('tabular/v'+suffix+'.tab')[0]

            # load remaining lookup tables
            #print self.tb_tab.keys()
        #print self.tb_tab.keys()

    def sv2phase(self,a,choice,freq=0.05):
        """
        given state variables return phase value on [0,T]
        a: array of state variables, e.g. trba[i,0,:], wba[0,1,:]
        choice: 'trb' or 'wb'
        freq: frequency of oscillators. If the freq differs from baseline/default of 0.05 by some tolerance, then load remaining lookup tables.
        """
        
        if choice == 'trb':
            
            # determine mean input current value given freq.
            current = self.tbfi_inv(freq)

            # get index of corresponding lookup table file name suffix
            closest_idx_name = np.argmin(np.abs(self.tb_valid_currents-current))

            # get the suffix value as a string
            table_suffix_value = self.tb_valid_currents[closest_idx_name]
            if table_suffix_value == 6.0416:
                tbl_name = ''
            else:
                tbl_name = '_i'+str(table_suffix_value)




            minidx = np.argmin( (self.tb_tab['vt'+tbl_name]-a[0])**2./self.tb_tab['vt_var'+tbl_name] +\
                                (self.tb_tab['mt'+tbl_name]-a[1])**2./self.tb_tab['mt_var'+tbl_name] +\
                                (self.tb_tab['nt'+tbl_name]-a[2])**2./self.tb_tab['nt_var'+tbl_name] )
            return self.tb_tab['per'+tbl_name]*minidx/self.tb_tab['N'+tbl_name]


        if choice == 'wb':
            
            # determine mean input current value given freq.
            current = self.wbfi_inv(freq)

            # get index of corresponding lookup table file name suffix
            closest_idx_name = np.argmin(np.abs(self.wb_valid_currents-current))

            # get the suffix value as a string
            table_suffix_value = self.wb_valid_currents[closest_idx_name]
            
            if table_suffix_value == 0.809079373711:#6.0416:
                tbl_name = ''
            else:
                tbl_name = '_i0'+str(table_suffix_value)

            
            minidx = np.argmin( (self.wb_tab['v'+tbl_name]-a[0])**2./self.wb_tab['v_var'+tbl_name] + \
                                (self.wb_tab['h'+tbl_name]-a[1])**2./self.wb_tab['h_var'+tbl_name] + \
                                (self.wb_tab['n'+tbl_name]-a[2])**2./self.wb_tab['n_var'+tbl_name])
            return self.wb_tab['per'+tbl_name]*minidx/self.wb_tab['N'+tbl_name]

                    
    def phase2sv(self,phase,choice):
        """
        given phase value, return state variables
        phase: in 0 to T
        choice: 'trb' or 'wb'.
        """
        #print phase,choice,self.phs_init_wb
        #print self.trb_tab_N,phase,self.trb_tab_period
        #print self.tb_tab.keys()

        if choice == 'trb':
            m = int(self.tb_tab['N']*phase/self.tb_tab['per'])
            return (self.tb_tab['vt'][m],self.tb_tab['mt'][m],self.tb_tab['nt'][m],
                    self.tb_tab['ht'][m],self.tb_tab['wt'][m],self.tb_tab['st'][m],
                    self.tb_tab['ca'][m])

        if choice == 'wb':
            m = int(self.wb_tab['N']*phase/self.wb_tab['per'])
            return (self.wb_tab['v'][m],self.wb_tab['h'][m],self.wb_tab['n'][m])
            
    def generate_dirs(self):
        """
        generate top-level directories if needed
        define filenames
        """
        self.savedir = 'savedir_trbwb/'

        if not(os.path.isdir(self.savedir)):
            os.makedirs(self.savedir)

    def load_inits(self):
        """
        helper function for def run_full_sim. load inits.
        """
                    
        self.trba0 = np.zeros((self.N,7))
        self.wba0 = np.zeros((self.N,3))

        file_not_found = False
        while True:
            if self.use_last and not(file_not_found):
                if os.path.isfile(self.filename_trb) and\
                   os.path.isfile(self.filename_wb) and\
                   os.path.isfile(self.filename_s):
                    
                    self.trba0[:] = np.loadtxt(self.filename_trb)
                    self.wba0[:] = np.loadtxt(self.filename_wb)
                    self.stb0,self.swb0 = np.loadtxt(self.filename_s)
                    #print self.stb0,self.swb0, 'start'
                    #print self.trba0, 'all vars trb start'
                    break
                else:
                    file_not_found = True

            elif self.use_ss and not(file_not_found):
                if os.path.isfile(self.filename_trb_ss) and \
                   os.path.isfile(self.filename_wb_ss) and\
                   os.path.isfile(self.filename_s_ss):


                    self.trba0[:] = np.loadtxt(self.filename_trb_ss)
                    self.wba0[:] = np.loadtxt(self.filename_wb_ss)
                    self.stb0,self.swb0 = np.loadtxt(self.filename_s_ss)
                    break
                else:
                    file_not_found = True
            else:
                file_not_found = True

            if file_not_found:
                
                if (self.phs_init_trb == None) or (self.phs_init_wb == None):
                    print 'using random inits'

                    for i in range(self.N):
                        self.trba0[i,:] = self.phase2sv(np.random.rand(1),'trb')
                        self.wba0[i,:] = self.phase2sv(np.random.rand(1),'wb')

                else:

                    for i in range(self.N):
                        self.trba0[i,:] = self.phase2sv(self.phs_init_trb[i],'trb')
                        self.wba0[i,:] = self.phase2sv(self.phs_init_wb[i],'wb')

                self.stb0 = self.sx0
                self.swb0 = self.sy0
                
                break

        print 'inits trb =', self.trba0[0,:]
        print 'inits wb =',self.wba0[0,:]
        print 'inits sx,sy =',self.stb0,self.swb0

    def run_full_sim(self):

        # the initial conditions are structured as follows:
        # row i denotes neuron i, column j denotes state variable j.
        # for wb, [v,h,n]
        # for trb, [v,m,n,h,w,s,ca]

        # the solutions are structured as
        # wb[time,neuron#,state variable#]
        self.load_inits()

        print 'running full_sim'

        if self.N == 2:
            inits={'vt':self.trba0[0,0],
                   'mt':self.trba0[0,1],
                   'nt':self.trba0[0,2],
                   'ht':self.trba0[0,3],
                   'wt':self.trba0[0,4],
                   'st':self.trba0[0,5],
                   'ca':self.trba0[0,6],
                   'vtp':self.trba0[1,0],
                   'mtp':self.trba0[1,1],
                   'ntp':self.trba0[1,2],
                   'htp':self.trba0[1,3],
                   'wtp':self.trba0[1,4],
                   'stp':self.trba0[1,5],
                   'cap':self.trba0[1,6],
                   'v':self.wba0[0,0],
                   'h':self.wba0[0,1],
                   'n':self.wba0[0,2],
                   'vp':self.wba0[1,0],
                   'hp':self.wba0[1,1],
                   'np':self.wba0[1,2],
                   'stb':self.stb0, 'swb':self.swb0}

            if self.population == 'both':
                filename = 'tbwb2.ode'
            elif self.population == 'e':
                filename = 'tb2.ode'
            elif self.population == 'i':
                filename = 'wb2.ode'

        elif self.N == 1:
            filename = 'tbwb1.ode'

            inits={'vt':self.trba0[0,0],'mt':self.trba0[0,1],
                   'nt':self.trba0[0,2],'ht':self.trba0[0,3],
                   'wt':self.trba0[0,4],'st':self.trba0[0,5],
                   'ca':self.trba0[0,6],
                   'v':self.wba0[0,0],'h':self.wba0[0,1],'n':self.wba0[0,2],
                   'stb':self.stb0, 'swb':self.swb0}

        npa, vn = xpprun(filename,
                         xppname='xppaut',
                         inits=inits,
                         parameters={'total':self.T,'dt':self.dt,
                                     'eps':self.eps,'mux':self.mux,'muy':self.muy,
                                     'gee':self.gee,'gei':self.gei,'gie':self.gie,'gii':self.gii,
                                     'itb_mean':self.itb_mean,'iwb_mean':self.iwb_mean,'fFixed':self.sbarx},
                         clean_after=True)

        #print npa,vn
        print np.shape(npa)
        t = npa[:,0]
        sv = npa[:,1:]

        num_opts = read_numerics('tbwb'+str(self.N)+'.ode')

        self.t = t
        self.T = self.t[-1]
        self.dt = float(num_opts['dt'])

        # define solution arrays to fit xpp output
        self.phasex = np.zeros((len(self.t),self.N))
        self.phasey = np.zeros((len(self.t),self.N))

        # solution vectors
        self.trba= np.zeros((len(self.t),self.N,7))
        self.wba = np.zeros((len(self.t),self.N,3))
        
        print np.shape(self.trba)
        # spike arrays
        self.spiket_wb1 = np.zeros(len(self.t),dtype='int')
        self.spiket_wb2 = np.zeros(len(self.t),dtype='int')
        self.spiket_trb1 = np.zeros(len(self.t),dtype='int')
        self.spiket_trb2 = np.zeros(len(self.t),dtype='int')

        # save solutions to compatible format
        if self.N == 2:
            self.trba[:,0,0] = sv[:,vn.index('vt')]
            self.trba[:,0,1] = sv[:,vn.index('mt')]
            self.trba[:,0,2] = sv[:,vn.index('nt')]
            self.trba[:,0,3] = sv[:,vn.index('ht')]
            self.trba[:,0,4] = sv[:,vn.index('wt')]
            self.trba[:,0,5] = sv[:,vn.index('st')]
            self.trba[:,0,6] = sv[:,vn.index('ca')]

            self.trba[:,1,0] = sv[:,vn.index('vtp')]
            self.trba[:,1,1] = sv[:,vn.index('mtp')]
            self.trba[:,1,2] = sv[:,vn.index('ntp')]
            self.trba[:,1,3] = sv[:,vn.index('htp')]
            self.trba[:,1,4] = sv[:,vn.index('wtp')]
            self.trba[:,1,5] = sv[:,vn.index('stp')]
            self.trba[:,1,6] = sv[:,vn.index('cap')]

            self.wba[:,0,0] = sv[:,vn.index('v')]
            self.wba[:,0,1] = sv[:,vn.index('h')]
            self.wba[:,0,2] = sv[:,vn.index('n')]

            self.wba[:,1,0] = sv[:,vn.index('vp')]
            self.wba[:,1,1] = sv[:,vn.index('hp')]
            self.wba[:,1,2] = sv[:,vn.index('np')]

        elif self.N == 1:
            self.trba[:,0,0] = sv[:,vn.index('vt')]
            self.trba[:,0,1] = sv[:,vn.index('mt')]
            self.trba[:,0,2] = sv[:,vn.index('nt')]
            self.trba[:,0,3] = sv[:,vn.index('ht')]
            self.trba[:,0,4] = sv[:,vn.index('wt')]
            self.trba[:,0,5] = sv[:,vn.index('st')]
            self.trba[:,0,6] = sv[:,vn.index('ca')]

            self.wba[:,0,0] = sv[:,vn.index('v')]
            self.wba[:,0,1] = sv[:,vn.index('h')]
            self.wba[:,0,2] = sv[:,vn.index('n')]

        self.stb = sv[:,vn.index('stb')]
        self.swb = sv[:,vn.index('swb')]

        if self.save_last or self.save_ss:
            np.savetxt(self.filename_trb,self.trba[-1,:,:])
            np.savetxt(self.filename_wb,self.wba[-1,:,:])
            np.savetxt(self.filename_s,np.array([self.stb[-1],self.swb[-1]]))
            print self.stb[-1],self.swb[-1], 's vars end'
            print self.trba[-1,:,:], 'all vars trb1 end'
            print self.wba[-1,:,:], 'all wb end'
            #return self.wba,trba,swb,strb

    def plot(self,choice='wb1+trb1+si',nskip=20):
        fig = plt.figure()

        if choice=='wb1+trb1+si':
            ax = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            for i in range(self.N):
                ax.plot(self.t,self.trba[:,i,0],label='trb'+str(i)+' voltage')
                ax.plot(self.t,self.wba[:,i,0],label='wb1'+str(i)+'voltage')

            

            ax2.plot(self.t,self.stb,label='stb',alpha=.5,color='blue')
            ax2.plot(self.t,self.sx_smooth,label='stb_smoothed',color='blue')
            ax2.plot(self.t,self.swb,label='swb',alpha=.5,color='green')
            ax2.plot(self.t,self.sy_smooth,label='swb_smoothed',color='green')


            ax.set_ylim(-100,100)
            ax.set_xlim(0,self.t[-1])

            #ax.set_title('numerics')

            ax.legend()
            ax2.legend()


            return fig

        elif choice == 'wb':
            x0_wb = [-64,.78,.09]
            sol = odeint(self.wb_rhs,x0_wb,self.t)            
            ax.plot(self.t,sol[:,0])
            return fig
            
        elif choice == 'trb':
            x0_trb = [42.68904,.9935,.4645,.47785,.268,.2917,.294]
            sol_trb = odeint(self.trbca_rhs,x0_trb,self.t)
            ax.plot(self.t,sol_trb[:,0])
            return fig
        elif choice == 'diff':

            ax = fig.add_subplot(111)

            ax.set_title('Numerical Phase Difference')

            # shortened solutions
            short_N = len(self.t[::nskip])
            t_short = self.t[::nskip]
            tb_short = self.trba[::nskip,:,:]
            wb_short = self.wba[::nskip,:,:]
            perx_short = self.perx[::nskip]
            pery_short = self.pery[::nskip]
            
            phs_t = np.zeros(short_N)
            phs_trb1 = np.zeros(short_N)
            phs_trb2 = np.zeros(short_N)

            phs_wb1 = np.zeros(short_N)
            phs_wb2 = np.zeros(short_N)

            for k in range(short_N):
                phs_t[k] = t_short[k]

                phs_trb1[k] = self.sv2phase(tb_short[k,0,:],'trb')
                phs_wb1[k] = self.sv2phase(wb_short[k,0,:],'wb')

                if self.N == 2:
                    phs_trb2[k] = self.sv2phase(tb_short[k,1,:],'trb')
                    phs_wb2[k] = self.sv2phase(wb_short[k,1,:],'wb')


            if self.N == 2:
                diff1 = np.mod(phs_trb2-phs_trb1+perx_short/2.,perx_short)-perx_short/2.
                diff2 = np.mod(phs_wb2-phs_wb1+pery_short/2.,pery_short)-pery_short/2.
                ax.scatter(phs_t,diff1,label=r'$\theta_2^x-\theta_1^x$',s=10,color='blue',edgecolor='none')
                ax.scatter(phs_t,diff2,label=r'$\theta_2^y-\theta_1^y$',s=10,color='green',edgecolor='none')

            diff3 = np.mod(phs_wb1-phs_trb1+pery_short/2.,pery_short)-pery_short/2.
            ax.scatter(phs_t,diff3,label=r'$\theta_1^y-\theta_1^x$',s=10,color='red',edgecolor='none')

            
            # antiphase lines
            ax.plot(self.t,self.perx/2,ls='-',color='gray',label='perx')
            ax.plot(self.t,-self.perx/2,ls='-',color='gray')
            
            ax.plot(self.t,self.pery/2,ls='--',color='gray',label='pery')
            ax.plot(self.t,-self.pery/2,ls='--',color='gray')

            # display legend
            ax.legend()

            # lims
            ax.set_xlim(self.t[0],self.t[-1])
            ax.set_ylim(-np.amax(self.pery/2)-1,np.amax(self.pery/2)+1)

            return fig


def main():


    print 'use trbwb_master.py'


if __name__ == "__main__":
    main()
