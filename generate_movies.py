"""

generate movies for supplementary files or presentations

Requires:
- TeX; may need to install texlive-extra-utils on linux
- XPPAut. Install via apt-get in linux, or install the precompiled binaries from http://www.math.pitt.edu/~bard/xpp/xpp.html
- Most tested XPP version for this script is version 8.
- xppcall.py. source code should be included. Latest version is online at https://github.com/iprokin/Py_XPPCALL


# last compiled using python 2.7.6
# numpy version 1.8.2
# scipy version 0.13.3
# matplotlib version 1.3.1
# last checked with xppaut veryion 6.11 and version 8.
# last run using xppcall.py from commit fa8c7b4


"""


from matplotlib import rc

matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm} \usepackage{xcolor} \setlength{\parindent}{0pt}']
matplotlib.rcParams.update({'figure.autolayout': True})

rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])


sizeOfFont = 20
fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}


import thetaslowmod_phase as tsmp

mod = np.mod
pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
sqrt = np.sqrt


color1 = '#CC79A7' # reddish purple
color2 = '#009E73' # blueish green
color3 = '#D55E00' # vermillion




def plot_surface_movie(simdata,skip,movie=False,
               file_prefix="mov/test",
               file_suffix=".png",
               title="",scene_number=1):
    """
    take fully computed surface solution and display the plot over time
    X: domain
    sol: full solution x1,x2,y1,y2 for all time
    TN: number of time steps
    skip: number of time steps to skip per plot display
    """
    sim = simdata # relabel for convenience

    TN = len(simdata.t)
    
    #N = sol[0,0,:,0]
    lo = -3#lo = np.amin(np.reshape(sol[:,0,:,:],(TN*N*N)))
    hi = 5#hi = np.amax(np.reshape(sol[:,0,:,:],(TN*N*N)))

    total_iter = int(TN/skip)-1
    start_iter = (scene_number-1)*total_iter
    end_iter = (scene_number)*total_iter

    label = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$",   r"$2\pi$"]
    j=0
    
    for i in range(total_iter):
        
        k = i*skip
        if (i <= 20) or ( (i >= 30) and (i <= 40)):

            c1='red'
            bg1 = 'yellow'
        else:

            c1 = 'black'
            bg1 = 'white'

        fig = plt.figure(figsize=(7,3.2))
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        plt.suptitle('g='+str(simdata.g)+'; q='+str(simdata.q),color=c1,backgroundcolor=bg1)

        ax.set_title('Solution u')
        ax.text(pi/2.,1.5,'T='+str(np.around(simdata.t[k],decimals=0)))
        ax2.set_title('Centroid')

        ax.plot(simdata.domain,simdata.u[k,:],lw=3)

        LL = np.argmin(np.abs(sim.domain - np.mod(sim.ph_angle[k],2*np.pi)))
        ax.scatter(sim.domain[LL],sim.sol[k,LL],color='red',s=100)

        ax2.scatter(cos(sim.ph_angle[k]),sin(sim.ph_angle[k]),s=100,color='red') # centroid
        xx = np.linspace(-pi,pi,100)
        ax2.plot(cos(xx),sin(xx), color='black', lw=3) # circle

        ax.set_xticks(np.arange(0,2+.5,.5)*pi)

        ax.set_xticklabels(label, fontsize=15)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        ax.tick_params(axis='x', which='major', pad=5)
        ax2.tick_params(axis='x', which='major', pad=5)

        ax.set_ylim(-2,2)
        ax.set_xlim(0,2*pi)

        ax2.set_ylim(-1.2,1.2)
        ax2.set_xlim(-1.2,1.2)

        j = start_iter+i

        fig.savefig(file_prefix+str(j)+file_suffix)
        plt.cla()
        plt.close()


        stdout.write("\r Simulation Recapping... %d%%" % int((100.*(k+1)/len(simdata.t))))
        stdout.flush()
    print


def mean_field_demo_theta(muy):
    """
    generate a video of the mean field model + phase differences
    """
    
    a1=.1;b1=1.;c1=1.1
    a2=a1;b2=b1;c2=c1

    #sx0 = .2921372
    #sy0 = .3077621

    mux = 1.

    eps = .01
    T = 1400
    #T = 20

    xin = np.array([0.,.5])
    yin = np.array([.2,.3])

    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
    Tx_base = 1./sx0
    Ty_base = 1./sy0

    print 'Tx_base',Tx_base,',Ty_base',Ty_base

    sx0 = sx0 + eps

    num1 = tsm.Theta(use_last=False,
                    save_last=False,
                    use_ss=False,
                    save_ss=False,
                    with_mean_field=False,
                    a1=a1,b1=b1,c1=c1,
                    a2=a2,b2=b2,c2=c2,
                    T=T,eps=eps,dt=.001,mux=mux,muy=muy,
                    xin=xin,yin=yin,sx0=sx0,sy0=sy0)
    
    # starting frequency (part 1)
    freqx0 = get_freq(a1,b1,c1,sx0,sy0)
    freqy0 = get_freq(a2,b2,c2,sx0,sy0)

    Tx0 = 1./freqx0
    Ty0 = 1./freqy0
    
    thxin = sv2phase(xin,freqx0)*Tx_base/Tx0
    thyin = sv2phase(yin,freqy0)*Ty_base/Ty0

    the1 = tsmp.Phase(use_last=False,
                       save_last=False,
                       run_full=False,
                       recompute_h=False,
                       run_phase=True,
                       recompute_slow_lc=False,
                       a1=a1,b1=b1,c1=c1,
                       a2=a2,b2=b2,c2=c2,
                       T=T,dt=.05,mux=mux,muy=1.,
                       eps=eps,
                       thxin=thxin,thyin=thyin,sx0=sx0,sy0=sy0)
    
    num2 = tsm.Theta(use_last=False,
                     save_last=False,
                     use_ss=False,
                     save_ss=False,
                     with_mean_field=False,
                     a1=a1,b1=b1,c1=c1,
                     a2=a2,b2=b2,c2=c2,
                     T=T,eps=eps,dt=.001,mux=mux,muy=1.4,
                     xin=xin,yin=yin,sx0=sx0,sy0=sy0)

    the2 = tsmp.Phase(use_last=False,
                      save_last=False,
                      run_full=False,
                      recompute_h=False,
                      run_phase=True,
                      recompute_slow_lc=False,
                      a1=a1,b1=b1,c1=c1,
                      a2=a2,b2=b2,c2=c2,
                      T=T,dt=.05,mux=mux,muy=1.4,
                      eps=eps,
                      thxin=thxin,thyin=thyin,sx0=sx0,sy0=sy0)
    
    # pass
    

def main():

    # create movie directory if it doesn't exist
    movdir = 'mov'
    if (not os.path.exists(movdir):
        os.makedirs(movdir)

        

    pass

if __name__ == "__main__":
    main()
