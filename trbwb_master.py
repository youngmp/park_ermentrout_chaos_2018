"""
To get small amplitude oscillations, use the following parameters:

(mean field)
fFixed=.05 # desired mean frequency
itbFixed=6.03893379759 # total mean input current required for traub
iwbFixed=0.809079373711 # total mean input current required for wb

it=itbFixed-fFixed*(gee-gei)
i0=iwbFixed-fFixed*(gie-gii)

These equations yield

p it=6.793934,i0=0.3590794
p gee=102,gei=117.1
p gie=20,gii=11
p muy=23.5
p eps=whatever

(numerics)

p it=6.793934,i0=0.3590794
p gee=102,gei=117.1
p gie=20,gii=11
p muy=22
p eps=0.0025

these choices of muy give approximately the same small amplitude slow oscillation.

"""

# runs trbwb_full and trbwb_phase
import getopt
import sys
import matplotlib.pyplot as plt

from trbwb_full import Traub_Wb
from trbwb_phase import Phase
from trbwb_lib import *


def usage():
    print "-l, --use-last\t\t: use last data from last sim"
    print "-v, --save-last\t\t: save last data of current sim"
    print "-s, --use-ss\t\t: use last saved steady-state data"
    print "-e, --save-ss\t\t: save solution as steady-state data"
    print "-h, --help\t\t: help function"
    print "-p, --run-phase\t\t: run phase"
    print "-f, --run-full\t\t: run full"



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "lvsehpf", ["use-last","save-last","use-ss","save-ss","help","run-phase","run-full"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    use_last=False;save_last=False;use_ss=False;save_ss=False
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
            elif opt in ('-v','--save-last'):
                save_last = True
            elif opt in ('-s','--use-ss'):
                use_ss = True
            elif opt in ('-e','save-ss'):
                save_ss = True
            elif opt in ('-p','run-phase'):
                run_phase = True
                print "run class phase=True"
            elif opt in ('-f','run-full'):
                run_full = True
                print "run class Wb_traub=True"


    # coupling params

    # slow small osc params
    #gee=102;gei=117.1
    #gie=20;gii=11

    #gee_phase=102.;gei_phase=117.1
    #gie_phase=20.;gii_phase=11.

    #gee_full=102.;gei_full=117.1
    #gie_full=20.;gii_full=11.

    #gee_phase=1.;gei_phase=1.
    #gie_phase=1.;gii_phase=1.

    #gee_phase=5.;gei_phase=5.
    #gie_phase=5.;gii_phase=5.

    #gee_full=5.;gei_full=3.
    #gie_full=5.;gii_full=3.

    
    #gee_phase=101.5;gei_phase=104.
    #gie_phase=13.;gii_phase=10.5

    gee_phase=10.;gei_phase=24.
    gie_phase=13.;gii_phase=10.


    gee_full=gee_phase;gei_full=gei_phase
    gie_full=gie_phase;gii_full=gii_phase

    #gee_phase=gee_full;gei_phase=gei_full
    #gie_phase=gie_full;gii_phase=gii_full

    #gee_full=10.;gei_full=11.1
    #gie_full=10.;gii_full=11.


    # weak/slow params
    #mux_full=1.;muy_full=20.85;eps=.0025
    #mux_phase=1.;muy_phase=23.275#23.15
    
    #mux_full=1.;muy_full=24.79;eps=.00125
    #mux_phase=1.;muy_phase=24.79#23.15

    mux_full=1.;muy_full=2.4;eps=.0025
    mux_phase=1.;muy_phase=2.4#23.15

    #mux_full=1.;muy_full=1.;eps_full=.005
    #mux_phase=1.;muy_phase=1.;eps_phase=1

    #itb=6.793934;iwb=0.3590794
    #N = 2 # neuron number
    T_full=20000;dt_full=.01 # time params
    #T_phase=T_full*eps;dt_phase=.01
    T_phase=T_full*eps;dt_phase=.01

    sx0=.05+eps/4
    sy0=.05

    #phs_init_trb = [-.15]
    #phs_init_wb = [-.0]
    
    phs_init_trb = [0.,.1]
    phs_init_wb = [.3,-.2]

    #phs_init_trb = [.1,.1]
    #phs_init_wb = [.1,.1]

    #phs_init_trb = [0.1,0.2]
    #phs_init_wb = [0,0.5]

    # determine current data (good candidates: f=0.05,0.064)    
    fFixed=.05
    itb_mean,iwb_mean = get_mean_currents(fFixed)

    itb_full=itb_mean-fFixed*(gee_full-gei_full)
    iwb_full=iwb_mean-fFixed*(gie_full-gii_full)

    itb_phase=itb_mean-fFixed*(gee_phase-gei_phase)
    iwb_phase=iwb_mean-fFixed*(gie_phase-gii_phase)

    print 'itb_mean,iwb_mean',itb_mean,iwb_mean
    
    if run_full:
        print 'itb_full,iwb_full',itb_full,iwb_full
        dat = Traub_Wb(use_last=use_last,
                       save_last=save_last,
                       use_ss=use_ss,
                       save_ss=save_ss,
                       T=T_full,dt=dt_full,
                       itb=itb_full,iwb=iwb_full,
                       gee=gee_full,gei=gei_full,
                       gie=gie_full,gii=gii_full,
                       
                       mux=mux_full,muy=muy_full,eps=eps,
                       itb_mean=itb_mean,iwb_mean=iwb_mean,
                       phs_init_trb=phs_init_trb,
                       phs_init_wb=phs_init_wb,
                       sbar=fFixed,
                       sx0=sx0,sy0=sy0,
                       save_phase=True)

        #dat.plot()
        dat.plot('diff',nskip=400)

        # automate phase conversion
        # st_phs,end_phs for use in phase.
        st_phs_trb1 = dat.sv2phase(dat.trba[0,0,:],'trb')
        st_phs_wb1 = dat.sv2phase(dat.wba[0,0,:],'wb')

        if dat.N == 2:
            st_phs_trb2 = dat.sv2phase(dat.trba[0,1,:],'trb')
            st_phs_wb2 = dat.sv2phase(dat.wba[0,1,:],'wb')

    if run_phase:



        print 'itb_phase,iwb_phase',itb_phase,iwb_phase

        """
        phase = Phase(T=T_phase,dt=dt_phase,
                          itb_mean=itb_mean,iwb_mean=iwb_mean,
                          gee=gee_phase,gei=gei_phase,
                          gie=gie_phase,gii=gii_phase,
                          mux=mux_phase,muy=muy_phase,eps=eps,
                          phs_init_trb=phs_init_trb,
                          phs_init_wb=phs_init_wb,
                          sbar=fFixed,
                          sx0=.0498,sy0=.05,
                          verbose=True)
        """

        phase = Phase(use_last=use_last,
                      save_last=save_last,
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


        phase.plot('h',saveplot=False)
        #phase.plot('hodd')
        #phase.plot('z')
        #phase.plot('h_integrand')
        #phase.plot('f')
        #phase.plot('q')
        #phase.plot('sol')
        #phase.plot('slowmod-space')
        #phase.plot('slowmod-t')
        #phase.plot('diff')

        phase.plot('thdiff-unnormed')

        #phase.plot('freq')
        #phase.plot('currents')
        #phase.plot('1')

        #phase.plot('fi-curves')
    
    plt.show(block=True)

            


if __name__ == "__main__":
    main(sys.argv[1:])

