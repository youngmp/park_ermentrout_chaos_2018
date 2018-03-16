"""
run thetaslowmod phase or full or both.



"""

import getopt
import sys

import matplotlib.pyplot as plt

from thetaslowmod_full import Theta
from thetaslowmod_phase import Phase
from thetaslowmod_lib import *

def usage():
    print "-l, --use-last\t\t: use last data from last sim"
    print "-v, --save-last\t\t: save last data of current sim"
    print "-s, --use-ss\t\t: use last saved steady-state data"
    print "-e, --save-ss\t\t: save solution as steady-state data"
    print "-r, --use-random\t: use random inits"
    print "-h, --help\t\t: help function"
    print "-p, --run-phase\t\t: run phase"
    print "-f, --run-full\t\t: run full"

def inits(option='1'):
    """
    choose initial conditions
    if no option matches, use random inits.
    """
    if option == '1':
        #(x1,x2)=(1.2365755,1.2365755)
        #(y1,y2)=(-2.6654642,-2.6654642)
        #sx,sy=1.0430093,1.0241029
                
        xin = np.array([1.2365755,1.2365755])
        yin = np.array([-2.6654642,-2.6654642])
        
        sx0 = 1.0430093
        sy0 = 1.0241029

    elif option == '2':
        xin = np.array([-1.078,1.0946])
        yin = np.array([-.0783,-.47835])
        
        sx0 = 2.#1.017038158211262644
        sy0 = 2.#1.000222753692209698

    
    elif option == '3':
        xin = np.array([-2.54045,2.56837])
        yin = np.array([1.811140,1.81114])

        sx0 = 1.
        sy0 = 1.


    elif option =='4':
        xin = np.array([-1.3668,1.26765])
        yin = np.array([0,0])

        sx0 = .989
        sy0 = 1.


    elif option =='5':
        xin = np.array([-1.3668,1.26765])
        yin = np.array([.4,0])

        sx0 = .989
        sy0 = 1.

    elif option =='6':
        xin = np.array([-.1,1.26765])
        yin = np.array([-1,-.0])

        sx0 = .989
        sy0 = 1.


    elif option =='7':
        #in = np.array([-.1,1.26765])
        xin = np.array([-.1,0.])
        #yin = np.array([-1,-.0])
        yin = np.array([-1,.0])

        sx0 = 1.
        sy0 = 1.

        
        
    return xin,yin,sx0,sy0

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


    #print 'get lorenzian distributed numbers'
    #print 'get randomly distributed numbers on [-1,1]'
    #np.random.seed(0)
    #print np.random.uniform(-1,1,4)

    mux = 1.
    muy = 5.4
    #muy = 2.62

    eps = .01

    T = 30./eps#1000


    # supercrit values
    a1=.5;b1=7.;c1=6.5
    a2=1.1;b2=25.;c2=25.1

    # oscillations in full model, none in theory
    #a1=.5;b1=10.;c1=9.5
    #a2=.5;b2=5.0;c2=4.5


    #a1=1.;b1=2.;c1=1.1
    #a2=1.;b2=2.01;c2=1.08

    #a1=.1;b1=1.;c1=1.1
    #a2=a1;b2=b1;c2=c1

    #xin,yin,sx0,sy0 = inits('7')

    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
    sx0 += .005
    #sx0 += eps
    #sx0 = .2921372
    #sy0 = .3077621


    freqx0 = get_freq(a1,b1,c1,sx0,sy0)
    freqy0 = get_freq(a2,b2,c2,sx0,sy0)

    xin = np.array([0.,.5])
    yin = np.array([.2,.3])

    #xin = np.array([1.,-1.1])
    #yin = np.array([1.01,-2.])
    #xin = np.array([0.,0.])
    #yin = np.array([0.,0.])

    # overwrite sx0,sy0 below depending on existence of slow oscillations

    """
    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
    
    # check if slow oscillations exist.
    # if yes, use the starting mean sx0,sy0 as inits
    # if no, use sbar as inits.
    slow_osc_exist = slow_osc(sx0,sy0,
                              a1,b1,c1,
                              a2,b2,c2,
                              mux,muy)
    

    if slow_osc_exist:

        # starting frequency (part 1)
        freqx0 = get_freq(a1,b1,c1,sx0,sy0)
        freqy0 = get_freq(a2,b2,c2,sx0,sy0)

        thxin = sv2phase(xin,freqx0)
        thyin = sv2phase(yin,freqy0)

        p = Phase(T=0,
                  a1=a1,b1=b1,c1=c1,
                  a2=a2,b2=b2,c2=c2,
                  dt=.05,mux=mux,muy=muy,eps=eps,
                  thxin=thxin,thyin=thyin,sx0=sx0,sy0=sy0,
                  slowmod_lc_tol=1e-6,
                  recompute_slow_lc=False)

        # overwrite sx0,sy0 if slow osc exist
        sx0 = p.sxa_fn(0)
        sy0 = p.sya_fn(0)
    
        # starting frequency (part 2, more accurate)
        freqx0 = get_freq(a1,b1,c1,sx0,sy0)
        freqy0 = get_freq(a2,b2,c2,sx0,sy0)

        thxin = sv2phase(xin,freqx0)
        thyin = sv2phase(yin,freqy0)
    """

    if run_full:

        sim = Theta(use_last=use_last,
                    save_last=save_last,
                    use_ss=use_ss,
                    save_ss=save_ss,
                    with_mean_field=False,
                    a1=a1,b1=b1,c1=c1,
                    a2=a2,b2=b2,c2=c2,
                    T=T,eps=eps,dt=.001,mux=mux,muy=muy,
                    xin=xin,yin=yin,sx0=sx0,sy0=sy0,
                    heterogeneous_pop=True)

        #sim = Theta(T=5000,a=a,b=b,c=c,mux=mux,muy=muy)
        #sim.plot(cutoff=False)
        #sim.plot('xj-yj')

        sim.plot('phase-diff-raw',nskip=100)
        #sim.plot('phase-diffdiff-raw',nskip=100)
        #sim.plot('phase-diff-normed')

        #sim.plot('full-vars')
        
        #sim.plot('full-s-t')
        #sim.plot('full-s-space')
        #sim.plot('phx-phy')

        #sim.plot('phase')
        #sim.plot('freq')


    if run_phase:


        if run_full:
            sx0 = sim.sx_smooth[0]
            sy0 = sim.sy_smooth[0]

        else:
            sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
            #sx0 += .01

        # starting frequency
        freqx0 = get_freq(a1,b1,c1,sx0,sy0)
        freqy0 = get_freq(a2,b2,c2,sx0,sy0)

        Tx0 = 1./freqx0
        Ty0 = 1./freqy0
        
        print 
        print 'phase stuff'


        Tx_base = 1./sx0
        Ty_base = 1./sy0
        
        # put phase on [0,1)
        thxin = sv2phase(xin,freqx0)*Tx_base/Tx0
        thyin = sv2phase(yin,freqy0)*Ty_base/Ty0



        print 'thyin-thxin',thyin-thxin
        print '(thyin-thxin)*pery',(thyin-thxin)/Tx0
        
        phase = Phase(use_last=use_last,
                      save_last=save_last,
                      run_full=False,
                      recompute_h=True,
                      run_phase=True,
                      recompute_slow_lc=True,
                      a1=a1,b1=b1,c1=c1,
                      a2=a2,b2=b2,c2=c2,
                      T=T,dt=.05,mux=mux,muy=muy,
                      eps=eps,
                      integration_N=4096,
                      thxin=thxin,thyin=thyin,sx0=sx0,sy0=sy0,
                      heterogeneous_pop=True)

        #phase.plot('fx')
        #phase.plot('fy')
        #phase.plot('z')
        #phase.plot('lc')
        #phase.plot('lc-deriv')

        #phase.plot('h_hom',saveplot=False)
        #phase.plot('h_inhom',saveplot=False)

        #phase.plot('h')

        #phase.plot('n1rhs')

        #phase.plot('beta')
        #phase.plot('beta_full')

        #phase.plot('th-plane')
        #phase.plot('th-t')
        
        #phase.plot('thdiff-raw',nskip=1)
        phase.plot('thdiff-unnormed',nskip=1)
        #phase.plot('thdiffdiff-raw',nskip=1)
        
        #phase.plot('slowmod_lc-space')
        #phase.plot('slowmod_lc-t')
        #phase.plot('slowmod_lc-t-1per')

        #phase.plot('slowmod_lc-diff')

        #phase.plot('normalization')
        #phase.plot('norm-no-int')
        #phase.plot('dphi-dsx')
        #phase.plot('dphi-dsy')

    plt.show()




if __name__ == "__main__":
    main(sys.argv[1:])

