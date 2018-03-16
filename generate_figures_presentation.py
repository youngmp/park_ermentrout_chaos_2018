"""
figure generation code for presentations

""" 

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc

import thetaslowmod_full as tsm
import thetaslowmod_phase as tsmp

from thetaslowmod_full import Theta
from thetaslowmod_phase import Phase
from thetaslowmod_lib import *

matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm} \usepackage{xcolor} \setlength{\parindent}{0pt}']
matplotlib.rcParams.update({'figure.autolayout': True})

rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])

sizeOfFont = 20
fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}


color1 = '#CC79A7' # reddish purple
color2 = '#009E73' # blueish green
color3 = '#D55E00' # vermillion


def theta_spiking_mov():
    movdir = 'mov/' # save frames to this dir

    mux = 1.
    muy = 1.4
    #muy = 2.62

    eps = .01

    T = 1./eps#1000

    a1=.1;b1=1.;c1=1.1
    a2=a1;b2=b1;c2=c1

    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)

    freqx0 = get_freq(a1,b1,c1,sx0,sy0)
    freqy0 = get_freq(a2,b2,c2,sx0,sy0)

    xin = np.array([0.,.5])
    yin = np.array([.2,.3])



    sim = Theta(use_last=False,
                save_last=False,
                use_ss=False,
                save_ss=False,
                with_mean_field=False,
                a1=a1,b1=b1,c1=c1,
                a2=a2,b2=b2,c2=c2,
                T=T,eps=eps,dt=.001,mux=mux,muy=muy,
                xin=xin,yin=yin,sx0=sx0,sy0=sy0)


    time_interval_display = 10
    time_steps_display = int(time_interval_display/sim.dt)

    starttime = 0
    endtime = sim.T

    counter=1
    j=time_steps_display#int(starttime/sim.dt)
    nfinal = int(endtime/sim.dt)

    skipn = 50
    skipn2 = 1000

    while j < time_steps_display*2:

        fig = plt.figure(figsize=(7,5))

        ax11 = plt.subplot2grid((2,2),(0,0),colspan=2)
        #ax12 = plt.subplot2grid((2,2),(0,1))
        ax21 = plt.subplot2grid((2,2),(1,0),colspan=2)

        solt = sim.t[j-time_steps_display:j]
        solx1 = sim.x[j-time_steps_display:j,0]
        solx2 = sim.x[j-time_steps_display:j,1]

        soly1 = sim.y[j-time_steps_display:j,0]
        soly2 = sim.y[j-time_steps_display:j,1]

        solsx = sim.sx[j-time_steps_display:j]
        solsy = sim.sy[j-time_steps_display:j]

        ax11.plot(solt,solx1,color='#3232ff',lw=2,label=r'$x_1$')
        ax11.plot(solt,solx2,color='#9999ff',lw=2,label=r'$x_2$')

        ax11.plot(solt,soly1,color='#ff3232',lw=2,label=r'$y_1$',dashes=(5,2))
        ax11.plot(solt,soly2,color='#ff9999',lw=2,label=r'$y_2$',dashes=(5,2))

        ax21.plot(solt,solsx,color='blue',label=r'$s_x$')
        ax21.plot(solt,solsy,color='red',label=r'$s_y$')

        ax11.set_xlim(solt[0],solt[-1])
        ax21.set_xlim(solt[0],solt[-1])

        ax21.set_xlabel(r'$t$',fontsize=20)
        #ax11.set_xticks([])

        ax11.legend(loc='lower left')
        ax21.legend(loc='lower left')

        print j

        """
        ax11.set_title(r"\textbf{Oscillator 1}")
        ax12.set_title(r"\textbf{Oscillator 2}")
        ax21.set_title(r"\textbf{Phase Difference and Slow Parameter}")

        ax11.set_xlabel(r"\textbf{Voltage (mV)}",fontsize=15)
        ax12.set_xlabel(r"\textbf{Voltage (mV)}",fontsize=15)
        ax21.set_xlabel(r"\textbf{Time (ms)}",fontsize=15)

        ax11.set_ylabel(r"$\mathbf{n}$",fontsize=15)
        ax12.set_ylabel(r"$\mathbf{n}$",fontsize=15)
        ax21.set_ylabel(r"$\mathbf{\phi}$",fontsize=15)
        ax21b.set_ylabel(r'$\bm{q(t)}$',fontsize=15,color='red')

        ax11.set_xlim(vmin,vmax)
        ax12.set_xlim(vmin,vmax)
        ax21.set_xlim(t[0],t[-1])
        ax21b.set_xlim(t[0],t[-1])

        ax11.set_ylim(nmin,nmax)
        ax12.set_ylim(nmin,nmax)
        ax21.set_ylim(minval,maxval)
        ax21b.set_ylim(minvalp,maxvalp)

        ax21.set_yticks(np.arange(0,0.5+.125,.125)*2*np.pi)
        ax21.set_yticklabels(x_label, fontsize=15)


        plt.locator_params(nticks=4)
        ax11.xaxis.set_ticks(np.arange(-80,80,40)) # fix x label spacing
        ax12.xaxis.set_ticks(np.arange(-80,80,40))

        """
        j += skipn
        k = j
        """
        #g1.matshow(np.reshape(sol[k,:N],(rN,rN)))

        # oscillators 1,2
        ax11.scatter(vdat[k,1],ndat[k,1],color='red',s=50)
        ax12.scatter(vpdat[k,1],npdat[k,1],color='red',s=50)

        ax11.plot(vlo,nlo,lw=2) # lookup tables
        ax12.plot(vlo,nlo,lw=2)

        ax11.text(-80,0.35,r"\textbf{Approx. phase=}") # real time phase
        ax11.text(-80,0.3,r"$\quad$\textbf{"+str(theta1[j,1])+r"*2pi}")

        ax12.text(-80,0.35,r"\textbf{Approx. phase=}")
        ax12.text(-80,0.3,r"$\quad$\textbf{"+str(theta2[j,1])+r"*2pi}")


        # phase diff full + theory + param
        ax21.plot(t[:k][::skipn],dat[:k,1][::skipn]*2*np.pi,color='black',lw=2,label='Numerics')
        N = len(slow_phs_model)
        ax21.plot(np.linspace(0,dat[:,0][-1],N)[:k][::skipn2],slow_phs_model[:k][::skipn2]*2*np.pi,lw=5,color="#3399ff",label='Theory')

        ax21b.plot(t[:k][::skipn2],gm[:k][::skipn2],color='red',lw=2,label='Parameter')

        ax21.legend()
        """    
        fig.savefig(movdir+str(counter)+".png",dpi=80)

        #plt.pause(.01)
        #print t[k],k,counter


        plt.cla()
        plt.close()

        counter += 1


def theta_spiking():
    """
    generate example of spiking in theta model
    """

    mux = 1.
    muy = 1.4
    #muy = 2.62

    eps = .01

    T = 1./eps#1000

    a1=.1;b1=1.;c1=1.1
    a2=a1;b2=b1;c2=c1

    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)

    freqx0 = get_freq(a1,b1,c1,sx0,sy0)
    freqy0 = get_freq(a2,b2,c2,sx0,sy0)

    xin = np.array([0.,.5])
    yin = np.array([.2,.3])

    sim = Theta(use_last=False,
                save_last=False,
                use_ss=False,
                save_ss=False,
                with_mean_field=False,
                a1=a1,b1=b1,c1=c1,
                a2=a2,b2=b2,c2=c2,
                T=T,eps=eps,dt=.001,mux=mux,muy=muy,
                xin=xin,yin=yin,sx0=sx0,sy0=sy0)

    time_interval_display = 10
    time_steps_display = int(time_interval_display/sim.dt)

    starttime = 0
    endtime = sim.T

    counter=1
    j=time_steps_display#int(starttime/sim.dt)
    nfinal = int(endtime/sim.dt)

    skipn = 50
    skipn2 = 1000

    fig = plt.figure(figsize=(7,5))
    
    ax11 = plt.subplot2grid((2,2),(0,0),colspan=2)
    #ax12 = plt.subplot2grid((2,2),(0,1))
    ax21 = plt.subplot2grid((2,2),(1,0),colspan=2)
    
    solt = sim.t[j-time_steps_display:j]
    solx1 = sim.x[j-time_steps_display:j,0]
    solx2 = sim.x[j-time_steps_display:j,1]
    
    soly1 = sim.y[j-time_steps_display:j,0]
    soly2 = sim.y[j-time_steps_display:j,1]
    
    solsx = sim.sx[j-time_steps_display:j]
    solsy = sim.sy[j-time_steps_display:j]
    
    ax11.plot(solt,soly1,color='#ff3232',lw=2,label=r'$y_1$',dashes=(5,2))
    ax11.plot(solt,soly2,color='#ff9999',lw=2,label=r'$y_2$',dashes=(5,2))
        
    ax11.plot(solt,solx2,color='#9999ff',lw=2,label=r'$x_2$')
    ax11.plot(solt,solx1,color='#3232ff',lw=2,label=r'$x_1$')
    
    ax21.plot(solt,solsy,color='red',label=r'$s^y$',lw=2)
    ax21.plot(solt,solsx,color='blue',label=r'$s^x$',lw=2)
    
    ax11.set_xlim(solt[0],solt[-1])
    ax21.set_xlim(solt[0],solt[-1])
    
    ax11.set_ylabel(r'$x_i,y_i$')
    ax21.set_ylabel(r'$s^x,s^y$')


    ax21.set_xlabel(r'$t$',fontsize=20)
    #ax11.set_xticks([])
    
    ax11.legend(loc='lower left')
    ax21.legend(loc='lower left')
    
    #fig.savefig(movdir+str(counter)+".png",dpi=80)

    #plt.pause(.01)
    #print t[k],k,counter

    return fig


def micro_vs_macro_theta_num():
    """
    show changes in synchronization properties in slowly varying mean field.
    theta model
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
                    T=T,eps=eps,dt=.001,mux=mux,muy=1.,
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
    
    fig = plt.figure(figsize=(6,8))

    # 1st col
    ax11 = plt.subplot(521)
    ax21 = plt.subplot(523)

    # 2nd col
    ax12 = plt.subplot(522)
    ax22 = plt.subplot(524)

    # collect into lists for loop
    collist1 = [ax11,ax21]
    collist2 = [ax12,ax22]
    collist = [collist1,collist2]

    numlist = [num1,num2]
    thelist = [the1,the2]

    sxmin = np.amin([np.amin(num1.sx),np.amin(num2.sx),np.amin(the1.sxa),np.amin(the2.sxa)])
    symin = np.amin([np.amin(num1.sy),np.amin(num2.sy),np.amin(the1.sya),np.amin(the2.sya)])

    sxmax = np.amax([np.amax(num1.sx),np.amax(num2.sx),np.amax(the1.sxa),np.amax(the2.sxa)])
    symax = np.amax([np.amax(num1.sy),np.amax(num2.sy),np.amax(the1.sya),np.amax(the2.sya)])

    for k in range(len(collist)):
        ax1 = collist[k][0]
        ax2 = collist[k][1]
        #ax3 = collist[k][2]

        num = numlist[k]
        the = thelist[k]

        # plot sx vs sy
        sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
        #ax1.plot(num.sx,num.sy,color='gray',alpha=.4)
        ax1.plot(the.sxa,the.sya,color='black',lw=2)
        ax1.scatter([sx0],[sy0],marker='*',s=100,edgecolor='none',color='black')
        
        # make lims the same
        ax1.set_xlim(sxmin-.001,sxmax+.001)
        ax1.set_ylim(symin-.001,symax+.001)
        
        #ax1.set_xlim(sx0-2*eps,sx0+2*eps)
        #ax1.set_ylim(sy0-1.3*eps,sy0+1.3*eps)

        # draw arrow(s) in mean field
        arrowidx = len(the.sxa)/4
        ax1.annotate("",
                     xytext=(the.sxa[arrowidx-1],the.sya[arrowidx-1]),
                     xy=(the.sxa[arrowidx],the.sya[arrowidx]),
                     arrowprops=dict(arrowstyle="-|>",
                                     connectionstyle="arc3",color='black'))

        # reduce font size
        #ax1.tick_params(axis='y',direction='in',pad=-20)
        #ax1.tick_params(axis='x',direction='in',pad=-1)

        # ticks x
        startx, endx = ax1.get_xlim()
        startx = np.round(startx,2)
        endx = np.round(endx,2)
        ax1.xaxis.set_ticks([startx,endx])

        # ticks y
        starty, endy = ax1.get_ylim()
        starty = np.round(starty,2)
        endy = np.round(endy,2)
        ax1.yaxis.set_ticks([starty,endy])

        # change interval of labels between ax2 and ax3
        #ax2.tick_params(axis='x',pad=-100)
        #ax2.set_xlabel(' ',labelpad=-20)

        # antiphase lines (numerics)
        ax2.plot(num.t,-num.perx/2.,color='gray',zorder=0)
        ax2.plot(num.t,-num.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)

        ax2.plot(num.t,num.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
        ax2.plot(num.t,num.pery/2.,color='gray',ls='--',zorder=0)

        # antiphase lines (theory)
        #ax3.plot(the.t,-the.perx/2.,color='gray',zorder=0)
        #ax3.plot(the.t,-the.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)

        #ax3.plot(the.t,the.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
        #ax3.plot(the.t,the.pery/2.,color='gray',ls='--',zorder=0)

        nskip_num = 10000
        nskip_ph = 20
        for i in range(num.N-1):
            diff1 = num.phasex[:,i+1]-num.phasex[:,0]
            diff2 = num.phasey[:,i+1]-num.phasey[:,0]

            diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
            diff2 = np.mod(diff2+num.perx/2.,num.perx)-num.perx/2.

            #ax2.scatter(num.t[::nskip_num],diff1[::nskip_num],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
            #ax2.scatter(num.t[::nskip_num],diff2[::nskip_num],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

            ax2.plot(num.t[::nskip_num],diff1[::nskip_num],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)#,s=5,zorder=2)
            ax2.plot(num.t[::nskip_num],diff2[::nskip_num],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)#,s=5,zorder=2)

            #ax2.plot(num.t[::nskip_num],diff1[::nskip_num],color=color1,label=r'$\phi^x$',lw=2)#,s=5,zorder=2)
            #ax2.plot(num.t[::nskip_num],diff2[::nskip_num],color=color2,label=r'$\phi^y$',lw=2)#,s=5,zorder=2)

        diff3 = num.phasey[:,0]-num.phasex[:,0]
        diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
        #ax2.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
        #ax2.plot(num.t[::nskip_num],diff3[::nskip_num],color=color3,label=r'$\phi^z$',lw=2)#,s=5,zorder=2)
        ax2.plot(num.t[::nskip_num],diff3[::nskip_num],color=color3,label=r'$\theta^y_1-\theta^x_1$',lw=2)#,s=5,zorder=2)

        # plot theory (11)
        for i in range(num.N-1):
            #diff1 = the.thx[:,i+1]-the.thx[:,0]
            #diff2 = the.thy[:,i+1]-the.thy[:,0]

            diff1 = the.thx_unnormed[:,i+1]-the.thx_unnormed[:,0]
            diff2 = the.thy_unnormed[:,i+1]-the.thy_unnormed[:,0]

            #diff1 = np.mod(diff1+pi,2*pi)-pi
            #diff2 = np.mod(diff2+pi,2*pi)-pi

            diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
            diff2 = np.mod(diff2+the.perx/2.,the.perx)-the.perx/2.

            #t1,y1 = clean(the.t,diff1,tol=.1)
            #t2,y2 = clean(the.t,diff2,tol=.1)

            #ax3.scatter(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
            #ax3.scatter(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

            #ax3.plot(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)#,s=5,zorder=2)
            #ax3.plot(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)#,s=5,zorder=2)

            #ax3.plot(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,label=r'$\phi^x$',lw=2)#,s=5,zorder=2)
            #ax3.plot(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,label=r'$\phi^y$',lw=2)#,s=5,zorder=2)

            #ax11.plot(t1,diff1,color="#3399ff",ls='-',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)
            #ax11.plot(t2,diff2,color="#3399ff",ls='--',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)

        diff3 = the.thy_unnormed[:,0]-the.thx_unnormed[:,0]
        diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
        #ax3.scatter(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
        #ax3.plot(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,label=r'$\phi^z$',lw=2)#,s=5,zorder=2)

        # set limits
        ax2.set_xlim(num.t[0],num.t[-1])
        #ax3.set_xlim(num.t[0],num.t[-1])

        # fix xlabel in sims
        startx, endx = ax2.get_xlim()
        ax2.xaxis.set_ticks([startx,endx])
        ax2.set_xlabel(r'$t$',labelpad=-10)

        # fix xlabel in sims
        #startx, endx = ax3.get_xlim()
        #ax3.xaxis.set_ticks([startx,endx])


        if k == 0:
            fig.text(0.0, .675, r'\textbf{Phase Difference}', va='center', rotation='vertical')
            ax1.set_ylabel(r'$s^y$',labelpad=-10)

        ax1.set_xlabel(r'$s^x$',labelpad=-10)
        #ax3.set_xlabel(r'$\bm{t}$',labelpad=-10)

        if k == 0:
            ax1.set_title(r'\textbf{A $\quad$ Mean Field}',loc='left')
            ax2.set_title(r'\textbf{C $\quad$ Numerics}',loc='left')
            #ax3.set_title(r'\textbf{E $\quad$ Theory}',loc='left')

        else:
            ax1.set_title(r'\textbf{B $\quad$ Mean Field}',loc='left')
            ax2.set_title(r'\textbf{D $\quad$ Numerics}',loc='left')
            #ax3.set_title(r'\textbf{F $\quad$ Theory}',loc='left')

        #ax2.set_xticks([])

        if k == 0:
            #lgnd = ax3.legend(loc='lower center',bbox_to_anchor=(1.2,-1.),scatterpoints=1,ncol=3)
            lgnd = ax2.legend(loc='lower center',bbox_to_anchor=(1.2,-1.),scatterpoints=1,ncol=3)
            lgnd.legendHandles[2]._sizes = [30]
            lgnd.legendHandles[3]._sizes = [30]
            lgnd.legendHandles[4]._sizes = [30]


    per = 3.70156211872    

    ax21.set_yticks(np.arange(-per/2.,per/2.+per/2.,per/2.))
    ax22.set_yticks(np.arange(-per/2.,per/2.+per/2.,per/2.))
    #ax31.set_yticks(np.arange(-per/2.,per/2.+per/2.,per/2.))
    #ax32.set_yticks(np.arange(-per/2.,per/2.+per/2.,per/2.))

    ax21.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])
    ax22.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])
    #ax31.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])
    #ax32.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])


    plt.tight_layout()

    return fig


def micro_vs_macro_theta_num_ana(muy):
    """
    show changes in synchronization properties in slowly varying mean field.
    theta model
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
                    T=T,eps=eps,dt=.001,mux=mux,muy=1.,
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
    
    fig = plt.figure(figsize=(6,12))

    # 1st col
    ax11 = plt.subplot(621)
    ax21 = plt.subplot(623)
    ax31 = plt.subplot(625)

    # 2nd col
    ax12 = plt.subplot(622)
    ax22 = plt.subplot(624)
    ax32 = plt.subplot(626)

    # collect into lists for loop
    collist1 = [ax11,ax21,ax31]
    collist2 = [ax12,ax22,ax32]
    collist = [collist1,collist2]

    numlist = [num1,num2]
    thelist = [the1,the2]

    sxmin = np.amin([np.amin(num1.sx),np.amin(num2.sx),np.amin(the1.sxa),np.amin(the2.sxa)])
    symin = np.amin([np.amin(num1.sy),np.amin(num2.sy),np.amin(the1.sya),np.amin(the2.sya)])

    sxmax = np.amax([np.amax(num1.sx),np.amax(num2.sx),np.amax(the1.sxa),np.amax(the2.sxa)])
    symax = np.amax([np.amax(num1.sy),np.amax(num2.sy),np.amax(the1.sya),np.amax(the2.sya)])

    for k in range(len(collist)):
        ax1 = collist[k][0]
        ax2 = collist[k][1]
        ax3 = collist[k][2]

        num = numlist[k]
        the = thelist[k]

        # plot sx vs sy
        sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
        ax1.plot(num.sx,num.sy,color='gray',alpha=.4)
        ax1.plot(the.sxa,the.sya,color='black',lw=2)
        ax1.scatter([sx0],[sy0],marker='*',s=100,edgecolor='none',color='black')
        
        # make lims the same
        ax1.set_xlim(sxmin-.001,sxmax+.001)
        ax1.set_ylim(symin-.001,symax+.001)
        
        #ax1.set_xlim(sx0-2*eps,sx0+2*eps)
        #ax1.set_ylim(sy0-1.3*eps,sy0+1.3*eps)

        # draw arrow(s) in mean field
        arrowidx = len(the.sxa)/4
        ax1.annotate("",
                     xytext=(the.sxa[arrowidx-1],the.sya[arrowidx-1]),
                     xy=(the.sxa[arrowidx],the.sya[arrowidx]),
                     arrowprops=dict(arrowstyle="-|>",
                                     connectionstyle="arc3",color='black'))

        # reduce font size
        #ax1.tick_params(axis='y',direction='in',pad=-20)
        #ax1.tick_params(axis='x',direction='in',pad=-1)

        # ticks x
        startx, endx = ax1.get_xlim()
        startx = np.round(startx,2)
        endx = np.round(endx,2)
        ax1.xaxis.set_ticks([startx,endx])

        # ticks y
        starty, endy = ax1.get_ylim()
        starty = np.round(starty,2)
        endy = np.round(endy,2)
        ax1.yaxis.set_ticks([starty,endy])

        # change interval of labels between ax2 and ax3
        #ax2.tick_params(axis='x',pad=-100)
        #ax2.set_xlabel(' ',labelpad=-20)

        # antiphase lines (numerics)
        ax2.plot(num.t,-num.perx/2.,color='gray',zorder=0)
        ax2.plot(num.t,-num.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)

        ax2.plot(num.t,num.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
        ax2.plot(num.t,num.pery/2.,color='gray',ls='--',zorder=0)

        # antiphase lines (theory)
        ax3.plot(the.t,-the.perx/2.,color='gray',zorder=0)
        ax3.plot(the.t,-the.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)

        ax3.plot(the.t,the.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
        ax3.plot(the.t,the.pery/2.,color='gray',ls='--',zorder=0)

        nskip_num = 10000
        nskip_ph = 20
        for i in range(num.N-1):
            diff1 = num.phasex[:,i+1]-num.phasex[:,0]
            diff2 = num.phasey[:,i+1]-num.phasey[:,0]

            diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
            diff2 = np.mod(diff2+num.perx/2.,num.perx)-num.perx/2.

            #ax2.scatter(num.t[::nskip_num],diff1[::nskip_num],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
            #ax2.scatter(num.t[::nskip_num],diff2[::nskip_num],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

            #ax2.plot(num.t[::nskip_num],diff1[::nskip_num],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)#,s=5,zorder=2)
            #ax2.plot(num.t[::nskip_num],diff2[::nskip_num],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)#,s=5,zorder=2)

            ax2.plot(num.t[::nskip_num],diff1[::nskip_num],color=color1,label=r'$\phi^x$',lw=2)#,s=5,zorder=2)
            ax2.plot(num.t[::nskip_num],diff2[::nskip_num],color=color2,label=r'$\phi^y$',lw=2)#,s=5,zorder=2)

        diff3 = num.phasey[:,0]-num.phasex[:,0]
        diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
        #ax2.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
        ax2.plot(num.t[::nskip_num],diff3[::nskip_num],color=color3,label=r'$\phi^z$',lw=2)#,s=5,zorder=2)

        # plot theory (11)
        for i in range(num.N-1):
            #diff1 = the.thx[:,i+1]-the.thx[:,0]
            #diff2 = the.thy[:,i+1]-the.thy[:,0]

            diff1 = the.thx_unnormed[:,i+1]-the.thx_unnormed[:,0]
            diff2 = the.thy_unnormed[:,i+1]-the.thy_unnormed[:,0]

            #diff1 = np.mod(diff1+pi,2*pi)-pi
            #diff2 = np.mod(diff2+pi,2*pi)-pi

            diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
            diff2 = np.mod(diff2+the.perx/2.,the.perx)-the.perx/2.

            #t1,y1 = clean(the.t,diff1,tol=.1)
            #t2,y2 = clean(the.t,diff2,tol=.1)

            #ax3.scatter(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
            #ax3.scatter(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

            #ax3.plot(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)#,s=5,zorder=2)
            #ax3.plot(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)#,s=5,zorder=2)

            ax3.plot(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,label=r'$\phi^x$',lw=2)#,s=5,zorder=2)
            ax3.plot(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,label=r'$\phi^y$',lw=2)#,s=5,zorder=2)

            #ax11.plot(t1,diff1,color="#3399ff",ls='-',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)
            #ax11.plot(t2,diff2,color="#3399ff",ls='--',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)

        diff3 = the.thy_unnormed[:,0]-the.thx_unnormed[:,0]
        diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
        #ax3.scatter(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
        ax3.plot(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,label=r'$\phi^z$',lw=2)#,s=5,zorder=2)

        # set limits
        ax2.set_xlim(num.t[0],num.t[-1])
        ax3.set_xlim(num.t[0],num.t[-1])

        # fix xlabel in sims
        startx, endx = ax3.get_xlim()
        ax3.xaxis.set_ticks([startx,endx])


        if k == 0:
            fig.text(0.0, .675, r'\textbf{Phase Difference}', va='center', rotation='vertical')
            ax1.set_ylabel(r'$s^y$',labelpad=-10)

        ax1.set_xlabel(r'$s^x$',labelpad=-10)
        ax3.set_xlabel(r'$\bm{t}$',labelpad=-10)

        if k == 0:
            ax1.set_title(r'\textbf{A $\quad$ Mean Field}',loc='left')
            ax2.set_title(r'\textbf{C $\quad$ Numerics}',loc='left')
            ax3.set_title(r'\textbf{E $\quad$ Theory}',loc='left')

        else:
            ax1.set_title(r'\textbf{B $\quad$ Mean Field}',loc='left')
            ax2.set_title(r'\textbf{D $\quad$ Numerics}',loc='left')
            ax3.set_title(r'\textbf{F $\quad$ Theory}',loc='left')

        ax2.set_xticks([])

        if k == 0:
            lgnd = ax3.legend(loc='lower center',bbox_to_anchor=(1.2,-1.),scatterpoints=1,ncol=3)
            lgnd.legendHandles[2]._sizes = [30]
            lgnd.legendHandles[3]._sizes = [30]
            lgnd.legendHandles[4]._sizes = [30]


    per = 3.70156211872
    

    ax21.set_yticks(np.arange(-per/2.,per/2.+per/2.,per/2.))
    ax22.set_yticks(np.arange(-per/2.,per/2.+per/2.,per/2.))
    ax31.set_yticks(np.arange(-per/2.,per/2.+per/2.,per/2.))
    ax32.set_yticks(np.arange(-per/2.,per/2.+per/2.,per/2.))

    ax21.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])
    ax22.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])
    ax31.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])
    ax32.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])


    plt.tight_layout()

    return fig


def theta_mean_field_example():
    """
    show 2 examples of mean field vs full sim
    """
    
    fig = plt.figure()
    ax11 = fig.add_subplot(221)
    ax12 = fig.add_subplot(222)
    ax21 = fig.add_subplot(223)
    ax22 = fig.add_subplot(224)
    
    # 


    # supercrit values
    a1=.5;b1=7.;c1=6.5
    a2=1.1;b2=25.;c2=25.1

    total = 1150 #1500 default
    mux=1.;muy=5.45

    eps = .005

    xin = np.array([-.1,0.])
    #xin = np.array([-2.22,1.0946])
    
    yin = np.array([-1.1,.3])
    #yin = np.array([-2,-.47835])


    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
    Tx_base = 1./sx0
    Ty_base = 1./sy0

    # check if slow oscillations exist.
    # if yes, use the starting mean sx0,sy0 as inits
    # if no, use sbar as inits.
    slow_osc_exist = slow_osc(sx0,sy0,
                              a1,b1,c1,
                              a2,b2,c2,
                              mux,muy)
    
    if slow_osc_exist:

        # run to get initial slowmod LC
        p = tsmp.Phase(T=0,
                       a1=a1,b1=b1,c1=c1,
                       a2=a2,b2=b2,c2=c2,
                       dt=.05,mux=mux,muy=muy,eps=eps,
                       thxin=[0,0],thyin=[0,0],sx0=sx0,sy0=sy0,
                       slowmod_lc_tol=1e-6,
                       recompute_slow_lc=True,
                       use_slowmod_lc=True)

        sx0 = p.sxa_1per_fn(0)
        sy0 = p.sya_1per_fn(0)

        # starting frequency
        freqx0 = get_freq(a1,b1,c1,sx0,sy0)
        freqy0 = get_freq(a2,b2,c2,sx0,sy0)
        
        Tx0 = 1./freqx0
        Ty0 = 1./freqy0

        thxin = sv2phase(xin,freqx0)*Tx_base/Tx0
        thyin = sv2phase(yin,freqy0)*Ty_base/Ty0
        
        if False:
            p.plot('slowmod_lc-t')
            plt.plot(p.sxa_1per_fn(p.ta))
            plt.show()

    #sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
    #sx0 += eps


    num = tsm.Theta(a1=a1,b1=b1,c1=c1,
                    a2=a2,b2=b2,c2=c2,
                    T=total,eps=eps,dt=.001,
                    mux=mux,muy=muy,
                    xin=xin,yin=yin,
                    sx0=sx0,sy0=sy0
                 )

    the = tsmp.Phase(a1=a1,b1=b1,c1=c1,
                     a2=a2,b2=b2,c2=c2,
                     T=total,eps=eps,dt=.05,
                     thxin=thxin,thyin=thyin,
                     mux=mux,muy=muy,
                     sx0=sx0,sy0=sy0,
                     run_phase=True,
                     recompute_h=True
                 )

    num2 = tsm.Theta(a1=a1,b1=b1,c1=c1,
                    a2=a2,b2=b2,c2=c2,
                    T=total,eps=eps,dt=.001,
                    mux=mux,muy=1,
                    xin=xin,yin=yin,
                    sx0=sx0-.01,sy0=sy0
                 )

    the2 = tsmp.Phase(a1=a1,b1=b1,c1=c1,
                     a2=a2,b2=b2,c2=c2,
                     T=total,eps=eps,dt=.05,
                     thxin=thxin,thyin=thyin,
                     mux=mux,muy=1,
                     sx0=sx0-.01,sy0=sy0,
                     run_phase=True,
                     recompute_h=True
                 )
    
    
    
    # plot mean field + sim field
    ax11.plot(num2.sx,num2.sy,color='k',label=r'$s^x$',lw=1)
    ax12.plot(the2.sxa,the2.sya,color='k',label=r'$\bar s^x$',lw=1)

    ax21.plot(num.sx,num.sy,color='k',label=r'$s^x$',lw=1)
    ax22.plot(the.sxa,the.sya,color='k',label=r'$\bar s^x$',lw=1)

    # get largest/smallest limits
    num_sx_min = np.amin(num.sx);num_sx_max = np.amax(num.sx)
    num_sy_min = np.amin(num.sy);num_sy_max = np.amax(num.sy)

    the_sx_min = np.amin(the.sxa);the_sx_max = np.amax(the.sxa)
    the_sy_min = np.amin(the.sya);the_sy_max = np.amax(the.sya)

    num2_sx_min = np.amin(num2.sx);num2_sx_max = np.amax(num2.sx)
    num2_sy_min = np.amin(num2.sy);num2_sy_max = np.amax(num2.sy)

    the2_sx_min = np.amin(the2.sxa);the2_sx_max = np.amax(the2.sxa)
    the2_sy_min = np.amin(the2.sya);the2_sy_max = np.amax(the2.sya)

    inf_x = np.amin([num_sx_min,num2_sx_min,the_sx_min,the2_sx_min])
    sup_x = np.amax([num_sx_max,num2_sx_max,the_sx_max,the2_sx_max])

    inf_y = np.amin([num_sy_min,num2_sy_min,the_sy_min,the2_sy_min])
    sup_y = np.amax([num_sy_max,num2_sy_max,the_sy_max,the2_sy_max])

    marg = .002

    ax11.set_xlim(inf_x-marg,sup_x+marg)
    ax12.set_xlim(inf_x-marg,sup_x+marg)
    ax21.set_xlim(inf_x-marg,sup_x+marg)
    ax22.set_xlim(inf_x-marg,sup_x+marg)


    ax11.set_ylim(inf_y-marg,sup_y+marg)
    ax12.set_ylim(inf_y-marg,sup_y+marg)
    ax21.set_ylim(inf_y-marg,sup_y+marg)
    ax22.set_ylim(inf_y-marg,sup_y+marg)

    ax11.set_xticks([])

    ax12.set_xticks([])
    ax12.set_yticks([])

    ax22.set_yticks([])
    

    
    ax11.set_title(r'\textbf{Spiking Network}')
    ax12.set_title(r'\textbf{Mean Field}')

    ax11.set_ylabel(r'$s^y$',fontsize=15)
    ax21.set_ylabel(r'$s^y$',fontsize=15)

    ax21.set_xlabel(r'$s^x$',fontsize=15)
    ax22.set_xlabel(r'$s^x$',fontsize=15)


    ax11.text(.93,1,r'$\mu=1$',fontsize=15)
    ax21.text(.92,1,r'$\mu=5.45$',fontsize=15)


    
    return fig

def generate_figure(function, args, filenames, title="", title_pos=(0.5,0.95)):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;
    fig = function(*args)
    #fig.text(title_pos[0], title_pos[1], title, ha='center')
    if type(filenames) == list:
        for name in filenames:
            if name.split('.')[-1] == 'ps':
                fig.savefig(name, orientation='landscape')
            else:
                fig.savefig(name,bbox_inches='tight')
    else:
        if name.split('.')[-1] == 'ps':
            fig.savefig(filenames,orientation='landscape')
        else:
            fig.savefig(filenames)



def main():

    figures = [
        
        #(theta_spiking_mov,[],[])
        #(theta_spiking,[],['theta_spiking.pdf']),
        (micro_vs_macro_theta_num,[],['micro_vs_macro_theta_num.pdf']),
        #(theta_mean_field_example,[],['theta_mean_field_example.pdf','theta_mean_field_example.png'])
        ]


    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    main()
