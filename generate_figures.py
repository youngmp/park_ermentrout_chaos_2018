"""
Run to generate figures
Requires:
- TeX; may need to install texlive-extra-utils on linux
- XPPAut. Install via apt-get in linux, or install the precompiled binaries from http://www.math.pitt.edu/~bard/xpp/xpp.html
- Most tested XPP version for this script is version 8.
- xppcall.py. source code should be included. Latest version is online at https://github.com/iprokin/Py_XPPCALL


the main() function at the end calls the individual figure functions.

figures are saved as both png and pdf.

Copyright (c) 2017, Youngmin Park, Bard Ermentrout
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
 
# last compiled using python 2.7.6
# numpy version 1.8.2
# scipy version 1.0.0
# matplotlib version 2.1.2
# last checked with xppaut veryion 6.11 and version 8.
# last run using xppcall.py from commit fa8c7b4

import os
import matplotlib
import copy
import pylab

from scipy.signal import argrelextrema
from scipy.optimize import brentq
from matplotlib import pyplot as plt
from matplotlib import rc
from sys import stdout
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
import scipy as sp
import matplotlib.ticker as ticker

from matplotlib import cm

matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm} \usepackage{xcolor} \setlength{\parindent}{0pt}']
matplotlib.rcParams.update({'figure.autolayout': True})

rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])

sizeOfFont = 20
fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}


import trbwb_full as twf
import trbwb_phase as twp
import thetaslowmod_full as tsm
import thetaslowmod_phase as tsmp
from thetaslowmod_lib import *
from trbwb_lib import *

mod = np.mod
pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
sqrt = np.sqrt

color1 = '#CC79A7' # reddish purple
color2 = '#009E73' # blueish green
color3 = '#D55E00' # vermillion

greens = ['#00ff04','#0d9800','#144700','#6ff662','#449f29']
off_greens = []

blues = ['#0000FF','#0099FF','#0033FF','#00CCFF','#0066FF']
off_blues = []


labels = [r'$-T/2$',r'$0$',r'$T/2$']
labels2 = [r'$-T/2$',r'$0$']

# last tested on:
# python 2.7.6
# matplotlib 2.1.2
# numpy 1.14.0
# scipy 1.0.0
# Ubuntu 14.04 LTS, x64, i5-4460 CPU
# March 15th, 2018


# draw arrows in 3d https://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector/11156353#11156353
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
            

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def full_vs_phase_theta_varying_fig(nskip_ph=1,nskip_num=1):
    """
    full vs phase in the theta model
    """

    fig = plt.figure(figsize=(6,6))
    ax11 = plt.subplot(311)
    ax21 = plt.subplot(312)
    ax31 = plt.subplot(313)

    #a1=1.;b1=1.;c1=.5
    #a2=a1;b2=b1;c2=c1


    # supercrit values
    a1=.5;b1=7.;c1=6.5
    a2=1.1;b2=25.;c2=25.1

    total = 1500 #1500 default
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

    # plot mean field + sim field
    ax11.plot(num.t,num.sx,color='blue',alpha=.35,label=r'$s^x$',lw=1)
    ax11.plot(num.t,num.sy,color='red',alpha=.35,label=r'$s^y$',lw=1)
    ax11.plot(the.t,the.sxa,color='blue',alpha=.85,label=r'$\bar s^x$',lw=1)
    ax11.plot(the.t,the.sya,color='red',alpha=.85,label=r'$\bar s^y$',lw=1)
    ax11.set_xlim(the.t[0],the.t[-1])

    # inset showing order eps magnitude changes in syn vars
    ax11ins = inset_axes(ax11,width="20%",height="50%",loc=8)
    ax11ins.plot(num.t,num.sy,color='red',alpha=.35,label=r'$s^y$',lw=2)
    ax11ins.plot(the.t,the.sya,color='red',alpha=.85,label=r'$\bar s^y$')

    ax11ins.plot(num.t,num.sx,color='blue',alpha=.35,label=r'$s^x$',lw=2)
    ax11ins.plot(the.t,the.sxa,color='blue',alpha=.85,label=r'$\bar s^x$')

    ax11ins.set_xlim(300,305)
    ax11ins.set_ylim(.975,.98)

    #ax11ins.set_xlim(865,885)
    #ax11ins.set_ylim(.9875,.992)
    ax11ins.set_xticks([])
    ax11ins.set_yticks([])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax11, ax11ins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax11.set_xticks([])
    ax11.legend()

    # antiphase lines
    ax21.plot(num.t,-num.perx/2.,color='gray',zorder=0)
    ax21.plot(num.t,-num.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax21.plot(num.t,num.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax21.plot(num.t,num.pery/2.,color='gray',ls='--',zorder=0)
    ax21.set_xticks([])

    ax31.plot(the.t,-the.perx/2.,color='gray',zorder=0)
    ax31.plot(the.t,-the.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax31.plot(the.t,the.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax31.plot(the.t,the.pery/2.,color='gray',ls='--',zorder=0)

    t1_num = copy.deepcopy(num.t[::nskip_num])
    t2_num = copy.deepcopy(num.t[::nskip_num])
    t3_num = copy.deepcopy(num.t[::nskip_num])

    # plot numerics (21)
    for i in range(num.N-1):
        diff1 = num.phasex[:,i+1]-num.phasex[:,0]
        diff2 = num.phasey[:,i+1]-num.phasey[:,0]

        diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
        diff2 = np.mod(diff2+num.pery/2.,num.pery)-num.pery/2.

        x1,y1 = clean(t1_num,diff1[::nskip_num],tol=.1)
        x2,y2 = clean(t2_num,diff2[::nskip_num],tol=.1)

        #ax21.scatter(num.t[::nskip_num],diff1[::nskip_num],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        #ax21.scatter(num.t[::nskip_num],diff2[::nskip_num],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

        #ax21.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2,lw=2)
        #ax21.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2,lw=2)

        ax21.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax21.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)

    diff3 = num.phasey[:,0]-num.phasex[:,0]
    diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
    x3,y3 = clean(t3_num,diff3[::nskip_num],tol=.5)
    #pos = np.where(np.abs(np.diff(y)) >= tol)[0]
    
    #x[pos] = np.nan
    #y[pos] = np.nan

    #ax21.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax21.plot(x3,y3,color=color3,label=r'$\phi^z$',zorder=1)
    ax21.set_ylim(-.8,.8)

    """
    hatch = ''
    color = 'yellow'
    alpha = .2
    ec = 'black'

    p = patches.Rectangle((0,.4),700,.2,fill=True,color=color,hatch=hatch,alpha=alpha,zorder=-1,ec=ec)
    ax21.add_patch(p)
    #ax21.text(200,.6,'Zoom Below')
    """

    """
    # inset showing small antiphase wiggles in numerics
    ax21ins = inset_axes(ax21,width="20%",height=.5,loc=9)#'lower center',bbox_to_anchor=(.4,.4))
    ax21ins.plot(x1,y1,color=color1,label=r'$s^y$',ls=2)
    #ax21ins.plot(the.t,the.sya,color='red',alpha=.75,label=r'$\bar s^y$')

    #ax21ins.plot(num.t,num.sx,color='blue',alpha=.25,label=r'$s^x$')
    #ax21ins.plot(the.t,the.sxa,color='blue',alpha=.75,label=r'$\bar s^x$')

    ax21ins.set_xlim(800,1000)
    ax21ins.set_ylim(-.6,-.4)
    ax21ins.set_xticks([])
    ax21ins.set_yticks([])



    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax21, ax21ins, loc1=2, loc2=4, fc="none", ec="0.5")
    """

    t1_the = copy.deepcopy(the.t[::nskip_ph])
    t2_the = copy.deepcopy(the.t[::nskip_ph])
    t3_the = copy.deepcopy(the.t[::nskip_ph])

    # plot theory (31)
    for i in range(num.N-1):
        diff1 = the.thx_unnormed[:,i+1]-the.thx_unnormed[:,0]
        diff2 = the.thy_unnormed[:,i+1]-the.thy_unnormed[:,0]
        
        diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
        diff2 = np.mod(diff2+the.perx/2.,the.perx)-the.perx/2.

        x1,y1 = clean(t1_the,diff1[::nskip_ph],tol=.5)
        x2,y2 = clean(t2_the,diff2[::nskip_ph],tol=.5)        
        
        #ax31.scatter(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        #ax31.scatter(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

        #ax31.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2,lw=2)
        #ax31.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2,lw=2)

        ax31.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax31.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)


    diff3 = the.thy_unnormed[:,0]-the.thx_unnormed[:,0]
    diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
    
    x3,y3 = clean(t3_the,diff3[::nskip_ph],tol=.5)

    #ax31.scatter(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax31.plot(x3,y3,color=color3,label=r'$\phi^z$',zorder=1)

    """
    # show region of zoom
    #ax11.set_ylim(.4,.6)
    #ax21.plot()
    p = patches.Rectangle((0,.4),700,.2,fill=True,color=color,hatch=hatch,alpha=alpha,zorder=-1,ec=ec)
    ax31.add_patch(p)
    #ax31.text(200,.6,'Zoom Below')
    """

    ax31.set_xlim(num.t[0],num.t[-1])
    ax21.set_xlim(num.t[0],num.t[-1])

    ax31.set_ylim(-.8,.8)

    
    ax31.set_xlabel(r'$\bm{t}$')
    fig.text(0.0, 0.35, r'\textbf{Phase Difference}', va='center', rotation='vertical')


    lgnd = ax31.legend(loc='lower center',bbox_to_anchor=(.5,-.6),scatterpoints=1,ncol=3)
    lgnd.legendHandles[2]._sizes = [30]
    lgnd.legendHandles[3]._sizes = [30]
    lgnd.legendHandles[4]._sizes = [30]

    # subplot labels
    ax11.set_title(r'\textbf{A} $\quad$ \textbf{Mean Field}',loc='left')
    ax21.set_title(r'\textbf{B} $\quad$ \textbf{Numerics}',loc='left')
    ax31.set_title(r'\textbf{C} $\quad$ \textbf{Theory}',loc='left')

    return fig


def full_vs_phase_theta_varying_fig_zoomed(nskip_ph=1,nskip_num=1):
    """
    same as above: full vs phase in the theta model, but zoomed in
    """

    fig = plt.figure(figsize=(6,4))
    ax11 = plt.subplot(211)
    ax21 = plt.subplot(212)

    # supercrit values
    a1=.5;b1=7.;c1=6.5
    a2=1.1;b2=25.;c2=25.1

    total = 700 #1500 default
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

    # antiphase lines
    ax11.plot(num.t,-num.perx/2.,color='gray',zorder=0)
    ax11.plot(num.t,-num.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax11.plot(num.t,num.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax11.plot(num.t,num.pery/2.,color='gray',ls='--',zorder=-1)

    ax21.plot(the.t,-the.perx/2.,color='gray',zorder=0)
    ax21.plot(the.t,-the.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax21.plot(the.t,the.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax21.plot(the.t,the.pery/2.,color='gray',ls='--',zorder=-1)

    t1_num = copy.deepcopy(num.t[::nskip_num])
    t2_num = copy.deepcopy(num.t[::nskip_num])
    t3_num = copy.deepcopy(num.t[::nskip_num])

    # plot numerics (21)
    for i in range(num.N-1):
        diff1 = num.phasex[:,i+1]-num.phasex[:,0]
        diff2 = num.phasey[:,i+1]-num.phasey[:,0]

        #diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
        diff2 = np.mod(diff2+num.pery/2.,num.pery)-num.pery/2.

        x1,y1 = clean(t1_num,diff1[::nskip_num],tol=.1)
        x2,y2 = clean(t2_num,diff2[::nskip_num],tol=.1)

        ax11.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax11.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)


    diff3 = num.phasey[:,0]-num.phasex[:,0]
    diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
    x3,y3 = clean(t3_num,diff3[::nskip_num],tol=.5)
    #pos = np.where(np.abs(np.diff(y)) >= tol)[0]
    
    #x[pos] = np.nan
    #y[pos] = np.nan

    #ax21.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax11.plot(x3,y3,color=color3,label=r'$\phi^z$',zorder=1)
    #ax21.set_ylim(-.8,.8)

    t1_the = copy.deepcopy(the.t[::nskip_ph])
    t2_the = copy.deepcopy(the.t[::nskip_ph])
    t3_the = copy.deepcopy(the.t[::nskip_ph])

    # plot theory (31)
    for i in range(the.N-1):
        diff1 = the.thx_unnormed[:,i+1]-the.thx_unnormed[:,0]
        diff2 = the.thy_unnormed[:,i+1]-the.thy_unnormed[:,0]
        
        #diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
        diff2 = np.mod(diff2+the.perx/2.,the.perx)-the.perx/2.

        x1,y1 = clean(t1_the,diff1[::nskip_ph],tol=.5)
        x2,y2 = clean(t2_the,diff2[::nskip_ph],tol=.5)        

        ax21.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax21.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)

    diff3 = the.thy_unnormed[:,0]-the.thx_unnormed[:,0]
    diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
    
    x3,y3 = clean(t3_the,diff3[::nskip_ph],tol=.5)

    ax21.plot(x3,y3,color=color3,label=r'$\phi^z$',zorder=1)

    """
    color = 'yellow'
    hatch = ''
    alpha = .1

    p = patches.Rectangle((0,.4),700,.2,fill=True,color=color,hatch=hatch,alpha=alpha,zorder=-1)
    ax11.add_patch(p)
    #ax21.text(200,.6,'Zoom Below')
    """
    
    """
    # inset showing order eps magnitude changes in syn vars
    ax11ins = inset_axes(ax11,width="20%",height="50%",loc=8)
    ax11ins.plot(num.t,num.sy,color='red',alpha=.35,label=r'$s^y$',lw=2)
    ax11ins.plot(the.t,the.sya,color='red',alpha=.85,label=r'$\bar s^y$')

    ax11ins.plot(num.t,num.sx,color='blue',alpha=.35,label=r'$s^x$',lw=2)
    ax11ins.plot(the.t,the.sxa,color='blue',alpha=.85,label=r'$\bar s^x$')

    ax11ins.set_xlim(300,305)
    ax11ins.set_ylim(.975,.98)

    #ax11ins.set_xlim(865,885)
    #ax11ins.set_ylim(.9875,.992)
    ax11ins.set_xticks([])
    ax11ins.set_yticks([])
    """
    
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    #mark_inset(ax11, ax11ins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    #ax11.set_xticks([])
    #ax11.legend()


    p = patches.Rectangle((0,.4),700,.2,fill=True,color=color,hatch=hatch,alpha=alpha,zorder=-1)
    ax21.add_patch(p)
    #ax21.text(200,.6,'Zoom Below')


    lgnd = ax21.legend(loc='lower center',bbox_to_anchor=(.5,-1),scatterpoints=1,ncol=3)

    ax11.set_xticks([])

    ax11.set_xlim(the.t[0],the.t[-1])
    ax21.set_xlim(the.t[0],the.t[-1])

    ax11.set_ylim(.4,.6)
    ax21.set_ylim(.4,.6)


    ax11.set_title(r'\textbf{A} $\quad$ \textbf{Numerics}',loc='left')
    ax21.set_title(r'\textbf{B} $\quad$ \textbf{Theory}',loc='left')

    ax21.set_xlabel(r'$\bm{t}$')
    fig.text(0.0, 0.5, r'\textbf{Phase Difference}', va='center', rotation='vertical')




    return fig

def full_vs_phase_theta_const_fig(nskip_ph=1,nskip_num=1):
    """
    full vs phase in the theta model
    """

    fig = plt.figure(figsize=(6,6))
    ax11 = plt.subplot(311)
    ax21 = plt.subplot(312)
    ax31 = plt.subplot(313)

    #a1=1.;b1=1.;c1=.5
    #a2=a1;b2=b1;c2=c1


    # supercrit values
    a1=.5;b1=7.;c1=6.5
    a2=1.1;b2=25.;c2=25.1

    total = 1500 #1500 default
    mux=1.;muy=5.45

    eps = .005

    xin = np.array([-.1,0.])
    #xin = np.array([-2.22,1.0946])
    
    yin = np.array([-1.1,.3])
    #yin = np.array([-2,-.47835])


    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)

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
                       recompute_slow_lc=False,
                       use_slowmod_lc=True)

        sx0 = p.sxa_fn(0)
        sy0 = p.sya_fn(0)

        # starting frequency
        freqx0 = get_freq(a1,b1,c1,sx0,sy0)
        freqy0 = get_freq(a2,b2,c2,sx0,sy0)

        thxin = sv2phase(xin,freqx0)
        thyin = sv2phase(yin,freqy0)

    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
    sx0 += eps


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

    # plot mean field + sim field
    ax11.plot(num.t,num.sx,color='blue',alpha=.25,label=r'$s^x$')
    ax11.plot(num.t,num.sy,color='red',alpha=.25,label=r'$s^y$')
    ax11.plot(the.t,the.sxa,color='blue',alpha=.75,label=r'$\bar s^x$')
    ax11.plot(the.t,the.sya,color='red',alpha=.75,label=r'$\bar s^y$')
    ax11.set_xlim(the.t[0],the.t[-1])

    # inset showing order eps magnitude changes in syn vars
    ax11ins = inset_axes(ax11,width="20%",height="50%",loc=9)
    ax11ins.plot(num.t,num.sy,color='red',alpha=.25,label=r'$s^y$')
    ax11ins.plot(the.t,the.sya,color='red',alpha=.75,label=r'$\bar s^y$')

    ax11ins.plot(num.t,num.sx,color='blue',alpha=.25,label=r'$s^x$')
    ax11ins.plot(the.t,the.sxa,color='blue',alpha=.75,label=r'$\bar s^x$')

    ax11ins.set_xlim(865,885)
    ax11ins.set_ylim(.9875,.992)
    ax11ins.set_xticks([])
    ax11ins.set_yticks([])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax11, ax11ins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax11.set_xticks([])
    ax11.legend()

    # antiphase lines
    ax21.plot(num.t,-num.perx/2.,color='gray',zorder=0)
    ax21.plot(num.t,-num.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax21.plot(num.t,num.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax21.plot(num.t,num.pery/2.,color='gray',ls='--',zorder=0)
    ax21.set_xticks([])

    ax31.plot(the.t,-the.perx/2.,color='gray',zorder=0)
    ax31.plot(the.t,-the.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax31.plot(the.t,the.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax31.plot(the.t,the.pery/2.,color='gray',ls='--',zorder=0)

    t1_num = copy.deepcopy(num.t[::nskip_num])
    t2_num = copy.deepcopy(num.t[::nskip_num])
    t3_num = copy.deepcopy(num.t[::nskip_num])

    # plot numerics (21)
    for i in range(num.N-1):
        diff1 = num.phasex[:,i+1]-num.phasex[:,0]
        diff2 = num.phasey[:,i+1]-num.phasey[:,0]

        diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
        diff2 = np.mod(diff2+num.pery/2.,num.pery)-num.pery/2.

        x1,y1 = clean(t1_num,diff1[::nskip_num],tol=.1)
        x2,y2 = clean(t2_num,diff2[::nskip_num],tol=.1)

        #ax21.scatter(num.t[::nskip_num],diff1[::nskip_num],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        #ax21.scatter(num.t[::nskip_num],diff2[::nskip_num],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

        ax21.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2)
        ax21.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2)

    diff3 = num.phasey[:,0]-num.phasex[:,0]
    diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
    x3,y3 = clean(t3_num,diff3[::nskip_num],tol=.5)
    #pos = np.where(np.abs(np.diff(y)) >= tol)[0]
    
    #x[pos] = np.nan
    #y[pos] = np.nan

    #ax21.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax21.plot(x3,y3,color=color3,label=r'$\theta^y_1-\theta^x_1$',zorder=2)

    ax21.set_ylim(-.8,.8)

    """
    # inset showing small antiphase wiggles in numerics
    ax21ins = inset_axes(ax21,width="20%",height=.5,loc=9)#'lower center',bbox_to_anchor=(.4,.4))
    ax21ins.plot(x1,y1,color=color1,label=r'$s^y$',ls=2)
    #ax21ins.plot(the.t,the.sya,color='red',alpha=.75,label=r'$\bar s^y$')

    #ax21ins.plot(num.t,num.sx,color='blue',alpha=.25,label=r'$s^x$')
    #ax21ins.plot(the.t,the.sxa,color='blue',alpha=.75,label=r'$\bar s^x$')

    ax21ins.set_xlim(800,1000)
    ax21ins.set_ylim(-.6,-.4)
    ax21ins.set_xticks([])
    ax21ins.set_yticks([])



    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax21, ax21ins, loc1=2, loc2=4, fc="none", ec="0.5")
    """

    t1_the = copy.deepcopy(the.t[::nskip_ph])
    t2_the = copy.deepcopy(the.t[::nskip_ph])
    t3_the = copy.deepcopy(the.t[::nskip_ph])

    # plot theory (31)
    for i in range(num.N-1):
        diff1 = the.thx[:,i+1]-the.thx[:,0]
        diff2 = the.thy[:,i+1]-the.thy[:,0]
        
        diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
        diff2 = np.mod(diff2+the.perx/2.,the.perx)-the.perx/2.

        x1,y1 = clean(t1_the,diff1[::nskip_ph],tol=.5)
        x2,y2 = clean(t2_the,diff2[::nskip_ph],tol=.5)        
        
        #ax31.scatter(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        #ax31.scatter(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

        ax31.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2)
        ax31.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2)


    diff3 = the.thy[:,0]-the.thx[:,0]
    diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
    
    x3,y3 = clean(t3_the,diff3[::nskip_ph],tol=.5)

    #ax31.scatter(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax31.plot(x3,y3,color=color3,label=r'$\theta^y_1-\theta^x_1$',zorder=2)

    ax31.set_xlim(num.t[0],num.t[-1])
    ax21.set_xlim(num.t[0],num.t[-1])

    ax31.set_ylim(-.8,.8)

    
    ax31.set_xlabel(r'$\bm{t}$')
    fig.text(0.0, 0.35, r'\textbf{Phase Difference}', va='center', rotation='vertical')

    ax21.set_title(r'\textbf{Numerics}')
    ax31.set_title(r'\textbf{Theory}')

    lgnd = ax31.legend(loc='lower center',bbox_to_anchor=(.5,-.9),scatterpoints=1,ncol=3)
    lgnd.legendHandles[2]._sizes = [30]
    lgnd.legendHandles[3]._sizes = [30]
    lgnd.legendHandles[4]._sizes = [30]

    return fig



def full_vs_phase_trbwb():
    """
    plot full phase vs theory phase in trb-wb
    for now a temporary function.
    """

    fig = plt.figure(figsize=(6,6))
    ax11 = plt.subplot(311)
    ax21 = plt.subplot(312)
    ax31 = plt.subplot(313)

    #fig = plt.figure(figsize=(6,5))
    #ax1 = fig.add_subplot(111)
    
    # sim model params
    mux=1.
    #muy_full=20.85
    #muy_full=23.275
    #muy_phase=23.275
    muy_full=24.79
    muy_phase=24.79
    eps = 0.00125

    # coupling params
    #gee_phase=102.;gei_phase=117.1
    #gie_phase=20.;gii_phase=11.

    #gee_full=102.;gei_full=117.1
    #gie_full=20.;gii_full=11.

    #gee_phase=90.;gei_phase=89.
    #gie_phase=15.;gii_phase=14.

    gee_phase=101.5;gei_phase=104.
    gie_phase=13.;gii_phase=10.5    

    #gee_full=10.;gei_full=11.1
    #gie_full=10.;gii_full=11.

    gee_full=gee_phase;gei_full=gei_phase
    gie_full=gie_phase;gii_full=gii_phase

    # sim time params
    T_full=20000;dt_full=.005 # time params
    #T_full=5000;dt_full=.01 # time params
    #T_full=1000;dt_full=.01 # time params
    T_phase=T_full*eps;dt_phase=.01

    phs_init_trb = [0.,.1]
    phs_init_wb = [-.1,.3]

    #sx0=.0485;sy0=.049981
    sx0=.0502;sy0=.0505


    # determine current data (good candidates: f=0.05,0.064)    
    fFixed=.05
    #fFixed=.064 # need new lookup tables if you want to use this

    itb_mean,iwb_mean = get_mean_currents(fFixed)

    itb_full=itb_mean-fFixed*(gee_full-gei_full)
    iwb_full=iwb_mean-fFixed*(gie_full-gii_full)

    itb_phase=itb_mean-fFixed*(gee_phase-gei_phase)
    iwb_phase=iwb_mean-fFixed*(gie_phase-gii_phase)

    print 'itb_full,itb_phase',itb_full,itb_phase
    print 'iwb_full,iwb_phase',iwb_full,iwb_phase
    print 'itb_mean,iwb_mean',itb_mean,iwb_mean


    dat = twf.Traub_Wb(T=T_full,dt=dt_full,
                       itb=itb_full,iwb=iwb_full,
                       gee=gee_full,gei=gei_full,
                       gie=gie_full,gii=gii_full,
                       mux=mux,muy=muy_full,eps=eps,
                       itb_mean=itb_mean,iwb_mean=iwb_mean,
                       phs_init_trb=phs_init_trb,
                       phs_init_wb=phs_init_wb,
                       sbar=fFixed,
                       sx0=sx0,sy0=sy0)

    the = twp.Phase(T=T_phase,dt=dt_phase,
                    itb_mean=itb_mean,iwb_mean=iwb_mean,
                    gee=gee_phase,gei=gei_phase,
                    gie=gie_phase,gii=gii_phase,
                    mux=mux,muy=muy_phase,eps=eps,
                    phs_init_trb=phs_init_trb,
                    phs_init_wb=phs_init_wb,
                    sbar=fFixed,
                    sx0=dat.sx_smooth[0],sy0=dat.sy_smooth[0])


    # plot mean field + sim field
    ax11.plot(dat.t,dat.stb,color='blue',alpha=.25,label=r'$s^x$')
    ax11.plot(dat.t,dat.swb,color='red',alpha=.25,label=r'$s^y$')
    ax11.plot(the.t,the.sxa_data,color='blue',alpha=.75,label=r'$\bar s^x$')
    ax11.plot(the.t,the.sya_data,color='red',alpha=.75,label=r'$\bar s^y$')
    ax11.set_xlim(the.t[0],the.t[-1])

    # inset showing order eps magnitude changes in syn vars
    ax11ins = inset_axes(ax11,width="20%",height=.5,loc=9)
    ax11ins.plot(dat.t,dat.swb,color='red',alpha=.25,label=r'$s^y$')
    ax11ins.plot(the.t,the.sya_data,color='red',alpha=.75,label=r'$\bar s^y$')

    ax11ins.plot(dat.t,dat.stb,color='blue',alpha=.25,label=r'$s^x$')
    ax11ins.plot(the.t,the.sxa_data,color='blue',alpha=.75,label=r'$\bar s^x$')

    #ax11ins.set_xlim(8500,9500)
    #ax11ins.set_ylim(.0498,.0506)
    ax11ins.set_xlim(14310,14540)
    ax11ins.set_ylim(.049,.0501)
    ax11ins.set_xticks([])
    ax11ins.set_yticks([])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax11, ax11ins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax11.set_xticks([])
    ax11.legend()            

    ## PLOT NUMERICS 

    nskip = 500
    # shortened solutions
    short_N = len(dat.t[::nskip])
    t_short = dat.t[::nskip]
    tb_short = dat.trba[::nskip,:,:]
    wb_short = dat.wba[::nskip,:,:]
    perx_short = dat.perx[::nskip]
    pery_short = dat.pery[::nskip]

    sx_short = dat.stb[::nskip]
    sy_short = dat.swb[::nskip]

    phs_t = np.zeros(short_N)
    phs_trb1 = np.zeros(short_N)
    phs_trb2 = np.zeros(short_N)
    
    phs_wb1 = np.zeros(short_N)
    phs_wb2 = np.zeros(short_N)


    for k in range(short_N):
        i = k
        phs_t[k] = t_short[i]
        
        phs_trb1[k] = dat.sv2phase(tb_short[i,0,:],'trb',1./perx_short[k])
        phs_wb1[k] = dat.sv2phase(wb_short[i,0,:],'wb',1./pery_short[k])

        if dat.N == 2:
            phs_trb2[k] = dat.sv2phase(tb_short[i,1,:],'trb',1./perx_short[k])
            phs_wb2[k] = dat.sv2phase(wb_short[i,1,:],'wb',1./pery_short[k])


    t1_num = copy.deepcopy(t_short)
    t2_num = copy.deepcopy(t_short)
    t3_num = copy.deepcopy(t_short)


    for i in range(dat.N-1):
    #if dat.N == 2:

        diff1 = (phs_trb2-phs_trb1)
        diff2 = (phs_wb2-phs_wb1)

        diff1 = np.mod(diff1+perx_short/2.,perx_short)-perx_short/2.
        diff2 = np.mod(diff2+pery_short/2.,pery_short)-pery_short/2.

        x1,y1 = clean(t1_num,diff1,tol=2)
        x2,y2 = clean(t2_num,diff2,tol=2)

        ax21.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2,lw=2)
        ax21.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2,lw=2)

        #ax21.scatter(t_short,diff1,label=r'$\theta_2^x-\theta_1^x$',s=10,color=color1,edgecolor='none')
        #ax21.scatter(t_short,diff2,label=r'$\theta_2^y-\theta_1^y$',s=10,color=color2,edgecolor='none')

    diff3 = phs_wb1-phs_trb1
    diff3 = np.mod(diff3+pery_short/2.,pery_short)-pery_short/2.
    x3,y3 = clean(t3_num,diff3,tol=2)
    ax21.plot(x3,y3,color=color3,label=r'$\theta^y_1-\theta^x_1$',zorder=2,lw=1)
    #ax21.scatter(t_short,diff3,label=r'$\theta_1^y-\theta_1^x$',s=10,color=color3,edgecolor='none')


    ax21.set_xlim(phs_t[0],phs_t[-1])
    ax21.set_ylim(-np.amax(pery_short)/2-1,np.amax(pery_short)/2+1)

    # antiphase lines
    ax21.plot(t_short,perx_short/2.,ls='-',color='gray')
    ax21.plot(t_short,-perx_short/2.,ls='-',color='gray',label=r'$T^x/2$')
    ax21.plot(t_short,pery_short/2.,ls='--',color='gray')
    ax21.plot(t_short,-pery_short/2.,ls='--',color='gray',label=r'$T^y/2$')


    ## PLOT THEORY



    t1_the = copy.deepcopy(the.t)
    t2_the = copy.deepcopy(the.t)
    t3_the = copy.deepcopy(the.t)
    

    #if the.N == 2:
    for i in range(dat.N-1):
        diff1 = the.thx_unnormed[:,i+1]-the.thx_unnormed[:,0]
        diff2 = the.thy_unnormed[:,i+1]-the.thy_unnormed[:,0]

        diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
        diff2 = np.mod(diff2+the.pery/2.,the.pery)-the.pery/2.

        x1,y1 = clean(t1_the,diff1)
        x2,y2 = clean(t2_the,diff2)

        #ax31.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2,lw=2)
        #ax31.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2,lw=2)

        ax31.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax31.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)

        #ax31.scatter(the.t,the.diff1,label=r'$\theta_2^x-\theta_1^x$',s=10,color=color1)
        #ax31.scatter(the.t,the.diff2,label=r'$\theta_2^y-\theta_1^y$',s=10,color=color2)
    
    diff3 = the.thy_unnormed[:,0]-the.thx_unnormed[:,0]
    diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.

    x3,y3 = clean(t3_the,diff3,tol=1)    
    #ax31.plot(x3,y3,label=r'$\theta_1^y-\theta_1^x$',lw=1,color=color3)
    ax31.plot(x3,y3,label=r'$\phi^z$',lw=1,color=color3)
    #ax31.scatter(the.t,the.diff3,label=r'$\theta_1^y-\theta_1^x$',s=10,color=color3)

    
    #ax31.set_ylim(-the.per/2.,the.per/2.)
    ax31.plot(the.t,the.perx_data/2.,ls='-',color='gray')
    ax31.plot(the.t,-the.perx_data/2.,ls='-',color='gray',label=r'$T^x/2$')
    ax31.plot(the.t,the.pery_data/2.,ls='--',color='gray')
    ax31.plot(the.t,-the.pery_data/2.,ls='--',color='gray',label=r'$T^y/2$')
    
    #ax31.legend()    
    ax31.set_xlim(the.t[0],the.t[-1])
    ax31.set_ylim(-np.amax(the.pery_data)/2-1,np.amax(the.pery_data)/2+1)

    
    #print -np.amax(the.pery_data)/2-1,np.amax(the.pery_data)/2+1
    #print fig.get_size_inches()*fig.dpi

    #ndiff3 = np.mod(-phs_wb1 + phs_trb1+pery_short/2.,pery_short)-pery_short/2.
        
    
    #ax21.set_xlabel(r'$t$')
    ax21.set_xticks([])
    fig.text(0.0, 0.35, r'\textbf{Phase Difference}', va='center', rotation='vertical')
    #ax21.set_ylabel(r'\textbf{Phase diff}')

    ax31.set_xlabel(r'$t$')

    #ax1.set_xlim(dat.t[0],dat.t[-1])
    #ax1.set_ylim(-pi-.1,pi+.1)

    #box = ax21.get_position()
    #ax21.set_position([box.x0, box.y0, box.width*0.7, box.height])
    #ax21.legend(loc='center left', bbox_to_anchor=(1,.5))
    #ax11.set_ylim(-pi-.1,pi+.1)

    # plot titles/labels
    ax11.set_title(r'\textbf{A} $\quad$ \textbf{Mean Field}',loc='left')
    ax21.set_title(r'\textbf{B} $\quad$ \textbf{Numerics}',loc='left')
    ax31.set_title(r'\textbf{C} $\quad$ \textbf{Theory}',loc='left')



    lgnd = ax31.legend(loc='lower center',bbox_to_anchor=(.5,-.9),scatterpoints=1,ncol=3)
    lgnd.legendHandles[2]._sizes = [30]
    lgnd.legendHandles[3]._sizes = [30]
    lgnd.legendHandles[4]._sizes = [30]



    plt.tight_layout()
    #plt.show()

    return fig



periodization_lower = -6
periodization_upper = 6


def K_diff(x,se,si):
    """
    difference of gaussians
    """
    A=1./(sqrt(pi)*se);B=1./(sqrt(pi)*si)
    return (A*exp(-(x/se)**2) -
            B*exp(-(x/si)**2))
    

def K_diff_p(x,se,si):
    """
    (periodic version)
    ricker wavelet https://en.wikipedia.org/wiki/Mexican_hat_wavelet
    """
    tot = 0
    for n in range(periodization_lower,periodization_upper+1):
        tot = tot + K_diff(x+n*2*pi,se,si)
    return tot


def K_gauss(x,s):
    A=1./(sqrt(pi)*s)
    return A*exp(-(x/s)**2)


def K_gauss_p(x,s):
    tot = 0
    for n in range(periodization_lower,periodization_upper+1):
        tot = tot + K_gauss(x+n*2*pi,s)
    return tot
    

## the following two functions override the default behavior or twiny()
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)

def make_spine_invisible(ax, direction):
    if direction in ["right", "left"]:
        ax.yaxis.set_ticks_position(direction)
        ax.yaxis.set_label_position(direction)
    elif direction in ["top", "bottom"]:
        ax.xaxis.set_ticks_position(direction)
        ax.xaxis.set_label_position(direction)
    else:
        raise ValueError("Unknown Direction : %s" % (direction,))

    ax.spines[direction].set_visible(True)


    
def gaussian_connections(cell=1,intervals=15):
    """
    draw gaussian type connections for w^{xx}_{1i}, w^{xy}_{1i}
    """

    fig = plt.figure(figsize=(5,5))
    sb = fig.add_subplot(111)
    par2 = sb.twiny() # create a second axes
    par2.spines["bottom"].set_position(("axes", -.2)) # move it down

    #sb.xaxis.set_major_locator(ticker.FixedLocator([0,1,2,3]))

    domain = np.linspace(0,2*pi,100)
    domainN = len(domain)

    # smooth functions
    exval = K_gauss_p(domain,.5)
    inval = -K_gauss_p(domain,1)

    # cell index values
    domchoice = np.linspace(0,2*pi*(1-1./intervals),intervals)#domain[::intervals]
    exchoice = K_gauss_p(domchoice,.5)#exval[::intervals]
    inchoice = -K_gauss_p(domchoice,1)#inval[::intervals]

    idxN = len(domchoice)

    center = ((cell-1)/idxN)*2*pi

    neuron_idx = np.arange(1,idxN+1,1)
    #poslabels = [r'$\frac{'+str(i-1-len(neuron_idx)/2)+r'}{'+str(len(neuron_idx))+r'}\pi$' for i in neuron_idx]
    poslabels = [r'$\left(\frac{'+str(i-1)+r'}{'+str(len(neuron_idx))+r'}\right)2\pi$' for i in neuron_idx]
    idxlabels = [str(i) for i in neuron_idx]

    poslabels[0] = '0'
    
    # annotate
    for i in range(idxN):
        # annotate excitatory
        sb.annotate(r'$w^{xx}_{'+str(cell)+str(i+1)+r'}$',
                    xy=(domchoice[i],exchoice[i]),
                    xytext=(domchoice[i],exchoice[i]+.05))
        
        # annotate inhibitory
        sb.annotate(r'$w^{xy}_{'+str(cell)+str(i+1)+r'}$',
                    xy=(domchoice[i],inchoice[i]),
                    xytext=(domchoice[i],inchoice[i]-.15))


    #sb.plot(domain[:domainN/2],exval[:domainN/2])    # excitatory gaussian
    #sb.plot(domain[domainN/2:]-2*pi,exval[domainN/2:])    # excitatory gaussian
    #sb.plot(domain[:domainN/2],inval[:domainN/2])    # inhibitory gaussian
    #sb.plot(domain[domainN/2:]-2*pi,inval[domainN/2:])    # inhibitory gaussian

    sb.plot(domain,exval,lw=2,color='black',label='Excitatory Gaussian')    # excitatory gaussian
    sb.plot(domain,inval,lw=2,ls='--',dashes=(5,2),color='red',label='Inhibitory Gaussian')    # inhibitory gaussian


    sb.set_xlabel("Position")
    sb.set_xlim(0,2*pi)
    #sb.set_xticks(sb.get_xticks()[::5])
    sb.set_xticks([])
    sb.set_xticks(domchoice)
    sb.set_xticklabels([])
    sb.set_xticklabels(poslabels)

    sb.spines['top'].set_color('none')
    sb.spines['right'].set_color('none')

    sb.yaxis.set_ticks_position('none')
    

    plt.subplots_adjust(bottom=0.17) # make room on bottom


    ## override the default behavior for a twiny axis
    make_patch_spines_invisible(par2) 
    make_spine_invisible(par2, "bottom")
    par2.set_xlabel("Cell Index")




    assert(len(neuron_idx)<len(domain))
    
    par2.scatter(domchoice,exchoice,color='black',s=50)
    par2.scatter(domchoice,inchoice,color='red',s=50)
    par2.set_xlim(0,2*pi)

    # set index number
    par2.set_xticks([])
    par2.set_xticks(domchoice)
    par2.set_xticklabels([])
    par2.set_xticklabels(idxlabels)
    

    sb.legend()


    return fig


def micro_vs_macro1_theta():
    """
    show changes in synchronization properties despite no changes in mean field.
    theta model
    """


def micro_vs_macro2_theta(plot_label='phi',show_numerical_synapse=True):
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
        
        if show_numerical_synapse:
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
        nskip_ph = 50
        for i in range(num.N-1):
            diff1 = num.phasex[:,i+1]-num.phasex[:,0]
            diff2 = num.phasey[:,i+1]-num.phasey[:,0]

            diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
            diff2 = np.mod(diff2+num.perx/2.,num.perx)-num.perx/2.

            #ax2.scatter(num.t[::nskip_num],diff1[::nskip_num],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
            #ax2.scatter(num.t[::nskip_num],diff2[::nskip_num],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

            #ax2.plot(num.t[::nskip_num],diff1[::nskip_num],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)#,s=5,zorder=2)
            #ax2.plot(num.t[::nskip_num],diff2[::nskip_num],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)#,s=5,zorder=2)

            if plot_label == 'phi':
                ax2.plot(num.t[::nskip_num],diff1[::nskip_num],color=color1,label=r'$\phi^x$',lw=2)#,s=5,zorder=2)
                ax2.plot(num.t[::nskip_num],diff2[::nskip_num],color=color2,label=r'$\phi^y$',lw=2)#,s=5,zorder=2)
            else:
                ax2.plot(num.t[::nskip_num],diff1[::nskip_num],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)#,s=5,zorder=2)
                ax2.plot(num.t[::nskip_num],diff2[::nskip_num],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)#,s=5,zorder=2)                

        diff3 = num.phasey[:,0]-num.phasex[:,0]
        diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
        #ax2.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
        
        if plot_label == 'phi':
            ax2.plot(num.t[::nskip_num],diff3[::nskip_num],color=color3,label=r'$\phi^z$',lw=2)#,s=5,zorder=2)
        else:
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



            if plot_label == 'phi':
                ax3.plot(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,label=r'$\phi^x$',lw=2)#,s=5,zorder=2)
                ax3.plot(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,label=r'$\phi^y$',lw=2)#,s=5,zorder=2)

            else:
                ax3.plot(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)#,s=5,zorder=2)
                ax3.plot(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)#,s=5,zorder=2)

            #ax11.plot(t1,diff1,color="#3399ff",ls='-',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)
            #ax11.plot(t2,diff2,color="#3399ff",ls='--',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)

        diff3 = the.thy_unnormed[:,0]-the.thx_unnormed[:,0]
        diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
        #ax3.scatter(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)

        if plot_label == 'phi':
            ax3.plot(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,label=r'$\phi^z$',lw=2)#,s=5,zorder=2)
        else:
            ax3.plot(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,label=r'$\theta^y_1-\theta^x_1$',lw=2)#,s=5,zorder=2)

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
            lgnd = ax3.legend(loc='lower center',bbox_to_anchor=(1.2,-.6),scatterpoints=1,ncol=3)
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


def micro_vs_macro2_tbwb():
    """
    mean field + phase locking stuff
    """

    nskip = 10

    #gee_phase=101.5;gei_phase=104.
    #gie_phase=13.;gii_phase=10.5

    gee_phase=10.;gei_phase=24.
    gie_phase=13.;gii_phase=10.

    gee_full=gee_phase;gei_full=gei_phase
    gie_full=gie_phase;gii_full=gii_phase
    
    mux_phase1=1.;muy_phase1=1.#23.15
    mux_phase2=1.;muy_phase2=2.4#23.15

    mux_full1=mux_phase1;muy_full1=muy_phase1
    mux_full2=mux_phase2;muy_full2=muy_phase2

    eps = .0025
    T_full=20000;dt_full=.01 # time params
    T_phase=T_full*eps;dt_phase=.01
    #T_phase=0.;dt_phase=.01

    sx0=.05+eps/4.
    sy0=.05
    
    phs_init_trb = [.0,.1]
    phs_init_wb = [.3,-.2]


    # determine current data (good candidates: f=0.05,0.064)    
    fFixed=.05
    itb_mean,iwb_mean = get_mean_currents(fFixed)
    
    itb_full=itb_mean-fFixed*(gee_full-gei_full)
    iwb_full=iwb_mean-fFixed*(gie_full-gii_full)

    itb_phase=itb_mean-fFixed*(gee_phase-gei_phase)
    iwb_phase=iwb_mean-fFixed*(gie_phase-gii_phase)


    print 'itb_full',itb_full,'iwb_full',iwb_full
    print 'itb_phase',itb_phase,'iwb_pase',iwb_phase

    #mp.figure()
    #mp.show()
    
    phase1 = twp.Phase(use_last=False,
                   save_last=False,
                   recompute_beta=True,
                    recompute_h=True,
                   sbar=fFixed,
                   T=T_phase,dt=dt_phase,
                   itb_mean=itb_mean,iwb_mean=iwb_mean,
                   gee=gee_phase,gei=gei_phase,
                   gie=gie_phase,gii=gii_phase,
                   mux=mux_phase1,muy=muy_phase1,eps=eps,
                   phs_init_trb=phs_init_trb,
                   phs_init_wb=phs_init_wb,
                   sx0=sx0,sy0=sy0,
                   verbose=True)

    phase2 = twp.Phase(use_last=False,
                   save_last=False,
                   recompute_beta=True,
                   sbar=fFixed,
                   T=T_phase,dt=dt_phase,
                   itb_mean=itb_mean,iwb_mean=iwb_mean,
                   gee=gee_phase,gei=gei_phase,
                   gie=gie_phase,gii=gii_phase,
                   mux=mux_phase2,muy=muy_phase2,eps=eps,
                   phs_init_trb=phs_init_trb,
                   phs_init_wb=phs_init_wb,
                   sx0=sx0,sy0=sy0,
                   verbose=True)

    # if recompute true, then re-run full model (TAKES FOREVER)

    f = twf.Traub_Wb(T=0,phs_init_trb=[0],phs_init_wb=[0])# get savedir

    """
    recompute_full = False
    if recompute_full:

        ph_tb1a = np.loadtxt(f.savedir+'phase_tb1_muy=6.dat')
        ph_tb2a = np.loadtxt(f.savedir+'phase_tb2_muy=6.dat')
        ph_tbwb_t_a = np.loadtxt(f.savedir+'phase_t_tbwb_muy=6.dat')
        ph_wb1a = np.loadtxt(f.savedir+'phase_wb1_muy=6.dat')
        ph_wb2a = np.loadtxt(f.savedir+'phase_wb2_muy=6.dat')

    else:
        pass
    """

    fig = plt.figure(figsize=(6,12))
    ax11 = fig.add_subplot(621)
    ax21 = fig.add_subplot(623)
    ax31 = fig.add_subplot(625)

    ax12 = fig.add_subplot(622)
    ax22 = fig.add_subplot(624)
    ax32 = fig.add_subplot(626)

    #### PLOT NUMERICS LEFT COL
    #ax11.set_title('Slowmod lc space (mean field, parset 1)')
    ax11.plot(phase1.sxa_data,phase1.sya_data,lw=2,color='black')
    ax11.scatter(phase1.sbarx,phase1.sbary,facecolor='k',marker='*',edgecolor='none',s=200,zorder=5)

    # plot direction
    halfidx = len(phase1.sxa_data)/50
    ax11.annotate('', xy=(phase1.sxa_data[halfidx], phase1.sya_data[halfidx]), xycoords='data',
                  xytext=(phase1.sxa_data[halfidx-1], phase1.sya_data[halfidx-1]), textcoords='data',
                  arrowprops=dict(arrowstyle="-|>",
                                  ec="black",
                                  fc='black',
                                  lw=2))
    
    # get min/max for mean field
    sim1_mean_xmin = np.amin(phase1.sxa_data)
    sim1_mean_xmax = np.amax(phase1.sxa_data)

    sim1_mean_ymin = np.amin(phase1.sya_data)
    sim1_mean_ymax = np.amax(phase1.sya_data)

    # text margin
    marg = eps/400
    # draw arrow
    ann = ax11.annotate('', xy=(sx0-marg, sy0), xycoords='data',
                        xytext=(.05+marg, sy0), textcoords='data',
                        arrowprops=dict(arrowstyle="<|-|>",
                                        ec="red",
                                        fc='red',alpha=.5))

    # label arrow
    ann = ax11.text(sx0-eps/6.,sy0+marg,r'$\varepsilon/4$')
    
    """
    ph_tb1a = np.loadtxt(f.savedir+'phase_tb1_muy=6_eps_pert.dat')
    ph_tb2a = np.loadtxt(f.savedir+'phase_tb2_muy=6_eps_pert.dat')
    ph_tbwb_t_a = np.loadtxt(f.savedir+'phase_t_tbwb_muy=6_eps_pert.dat')
    ph_wb1a = np.loadtxt(f.savedir+'phase_wb1_muy=6_eps_pert.dat')
    ph_wb2a = np.loadtxt(f.savedir+'phase_wb2_muy=6_eps_pert.dat')
    """

    # pert sx0+eps/5, sy0 default.
    ph_tb1a = np.loadtxt(f.savedir+'phase_tb1_muy=1.dat')
    ph_tb2a = np.loadtxt(f.savedir+'phase_tb2_muy=1.dat')
    ph_tbwb_t_a = np.loadtxt(f.savedir+'phase_t_tbwb_muy=1.dat')
    ph_wb1a = np.loadtxt(f.savedir+'phase_wb1_muy=1.dat')
    ph_wb2a = np.loadtxt(f.savedir+'phase_wb2_muy=1.dat')


    T_base = 1./fFixed
    diff1 = ph_tb2a - ph_tb1a
    diff1 = np.mod(diff1 + T_base/2.,T_base) - T_base/2.

    diff2 = ph_wb2a - ph_wb1a
    diff2 = np.mod(diff2 + T_base/2.,T_base) - T_base/2.
    
    diff3 = ph_wb1a - ph_tb1a
    diff3 = np.mod(diff3 + T_base/2.,T_base) - T_base/2.
    
    # cut discontinuities
    t1_num = copy.deepcopy(ph_tbwb_t_a)
    t2_num = copy.deepcopy(ph_tbwb_t_a)
    t3_num = copy.deepcopy(ph_tbwb_t_a)

    x1,y1 = clean(t1_num,diff1,tol=5)
    x2,y2 = clean(t2_num,diff2,tol=5)
    x3,y3 = clean(t3_num,diff3,tol=5)
    
    ax21.plot(x1,y1,color=color1,lw=2)
    ax21.plot(x2,y2,color=color2,lw=2)
    ax21.plot(x3,y3,color=color3,lw=2)

    #### PLOT NUMERICS RIGHT COL
    # draw arrow
    ann = ax12.annotate('', xy=(sx0-marg, sy0), xycoords='data',
                        xytext=(.05+marg, sy0), textcoords='data',
                        arrowprops=dict(arrowstyle="<|-|>",
                                        ec="red",
                                        fc='red',alpha=.5))
    # label arrow
    ann = ax12.text(sx0-eps/6.,sy0+marg,r'$\varepsilon/4$')

    #ax12.set_title('Slowmod lc space (mean field, parset 2)')
    ph_tb1a = np.loadtxt(f.savedir+'phase_tb1_muy=2.5.dat')
    ph_tb2a = np.loadtxt(f.savedir+'phase_tb2_muy=2.5.dat')
    ph_tbwb_t_a = np.loadtxt(f.savedir+'phase_t_tbwb_muy=2.5.dat')
    ph_wb1a = np.loadtxt(f.savedir+'phase_wb1_muy=2.5.dat')

    ph_wb2a = np.loadtxt(f.savedir+'phase_wb2_muy=2.5.dat')
    ph_stba = np.loadtxt(f.savedir+'phase_stb_muy=2.5.dat')
    ph_swba = np.loadtxt(f.savedir+'phase_swb_muy=2.5.dat')

    diff1 = ph_tb2a - ph_tb1a
    diff1 = np.mod(diff1 + T_base/2.,T_base) - T_base/2.

    diff2 = ph_wb2a - ph_wb1a
    diff2 = np.mod(diff2 + T_base/2.,T_base) - T_base/2.
    
    diff3 = ph_wb1a - ph_tb1a
    diff3 = np.mod(diff3 + T_base/2.,T_base) - T_base/2.

    #ax12.plot(ph_stba,ph_swba,color='gray',alpha=.4)
    ax12.plot(phase2.sxa_data,phase2.sya_data,color='black',lw=2)
    ax12.scatter(phase2.sbarx,phase2.sbary,facecolor='.1',s=200,marker='*',edgecolor='none',zorder=5)

    # plot direction
    halfidx = len(phase2.sxa_data)/50
    ax12.annotate('', xy=(phase2.sxa_data[halfidx], phase2.sya_data[halfidx]), xycoords='data',
                  xytext=(phase2.sxa_data[halfidx-1], phase2.sya_data[halfidx-1]), textcoords='data',
                  arrowprops=dict(arrowstyle="-|>",
                                  ec="black",
                                  fc='black',
                                  lw=2))

    #ax12.legend()

    t1_num = copy.deepcopy(ph_tbwb_t_a)
    t2_num = copy.deepcopy(ph_tbwb_t_a)
    t3_num = copy.deepcopy(ph_tbwb_t_a)

    x1,y1 = clean(t1_num,diff1,tol=5)
    x2,y2 = clean(t2_num,diff2,tol=5)
    x3,y3 = clean(t3_num,diff3,tol=5)
    

    i=0
    ax22.plot(x1,y1,color=color1,lw=2,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$')
    ax22.plot(x2,y2,color=color2,lw=2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$')
    ax22.plot(x3,y3,color=color3,lw=2,label=r'$\theta^y_1-\theta^x_1$')





    ########### LEFT COL THEORY

    #ax31.set_title('thji - thjk, raw (theory)')
    #per = 1.
    sxvals = phase1.sxa_fn(phase1.t)
    syvals = phase1.sya_fn(phase1.t)
    
    freqx_arr = phase1.freqx_fn(phase1.t)
    freqy_arr = phase1.freqy_fn(phase1.t)
    
    perx_arr = 1./freqx_arr
    pery_arr = 1./freqy_arr

    t1_num = copy.deepcopy(phase1.t[::nskip])
    t2_num = copy.deepcopy(phase1.t[::nskip])
    t3_num = copy.deepcopy(phase1.t[::nskip])

    
    for i in range(phase1.N-1):
        diff1 = (phase1.thx[:,i+1]-phase1.thx[:,0])*perx_arr/phase1.Tx_base
        diff2 = (phase1.thy[:,i+1]-phase1.thy[:,0])*pery_arr/phase1.Ty_base
        
        diff1 = np.mod(diff1+perx_arr/2.,perx_arr)-perx_arr/2.
        diff2 = np.mod(diff2+pery_arr/2.,pery_arr)-pery_arr/2.
        
        diff1 = diff1[::nskip]
        diff2 = diff2[::nskip]

        x1,y1 = clean(t1_num,diff1,tol=5)
        x2,y2 = clean(t2_num,diff2,tol=5)

        #ax31.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',lw=2)
        #ax31.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',lw=2)

        ax31.plot(x1,y1,color=color1,label=r'$\phi^x$',lw=2)
        ax31.plot(x2,y2,color=color2,label=r'$\phi^y$',lw=2)

    diff3 = (phase1.thy[:,0]-phase1.thx[:,0])*(pery_arr/phase1.Ty_base)
    diff3 = np.mod(diff3+pery_arr/2.,pery_arr)-pery_arr/2.
    diff3 = diff3[::nskip]
    
    #ax31.scatter(phase1.t[::nskip],diff3,edgecolor='none',color='red',label='thy1-thx1')
    x3,y3 = clean(t3_num,diff3,tol=5)
    #ax31.plot(x3,y3,color=color3,label=r'$\theta^y_1-\theta^x_1$',lw=2)
    ax31.plot(x3,y3,color=color3,label=r'$\phi^z$',lw=2)




    # antiphase lines
    ax21.plot(phase1.t,-perx_arr/2.,color='gray')
    ax21.plot(phase1.t,-pery_arr/2.,color='gray',ls='--')
    ax21.plot(phase1.t,perx_arr/2.,color='gray')
    ax21.plot(phase1.t,pery_arr/2.,color='gray',ls='--')

    ax31.plot(phase1.t,-perx_arr/2.,color='gray',label=r'$T^x/2$')
    ax31.plot(phase1.t,-pery_arr/2.,color='gray',label=r'$T^y/2$',ls='--')
    ax31.plot(phase1.t,perx_arr/2.,color='gray')
    ax31.plot(phase1.t,pery_arr/2.,color='gray',ls='--')





    ######## RIGHT COL THEORY


    #per = 1.
    sxvals = phase2.sxa_fn(phase2.t)
    syvals = phase2.sya_fn(phase2.t)
    
    freqx_arr = phase2.freqx_fn(phase2.t)
    freqy_arr = phase2.freqy_fn(phase2.t)
    
    perx_arr = 1./freqx_arr
    pery_arr = 1./freqy_arr


    ax22.plot(phase1.t,-perx_arr/2.,color='gray')
    ax22.plot(phase1.t,-pery_arr/2.,color='gray',ls='--')
    ax22.plot(phase1.t,perx_arr/2.,color='gray')
    ax22.plot(phase1.t,pery_arr/2.,color='gray',ls='--')
        
    ax32.plot(phase2.t,-perx_arr/2.,color='gray')
    ax32.plot(phase2.t,-pery_arr/2.,color='gray',ls='--')
    ax32.plot(phase2.t,perx_arr/2.,color='gray')
    ax32.plot(phase2.t,pery_arr/2.,color='gray',ls='--')


    t1_num = copy.deepcopy(phase2.t[::nskip])
    t2_num = copy.deepcopy(phase2.t[::nskip])
    t3_num = copy.deepcopy(phase2.t[::nskip])
    
    for i in range(phase2.N-1):
        diff1 = (phase2.thx[:,i+1]-phase2.thx[:,0])*perx_arr/phase2.Tx_base
        diff2 = (phase2.thy[:,i+1]-phase2.thy[:,0])*pery_arr/phase2.Ty_base
        
        diff1 = np.mod(diff1+perx_arr/2.,perx_arr)-perx_arr/2.
        diff2 = np.mod(diff2+pery_arr/2.,pery_arr)-pery_arr/2.
        
        diff1 = diff1[::nskip]
        diff2 = diff2[::nskip]

        x1,y1 = clean(t1_num,diff1,tol=5)
        x2,y2 = clean(t2_num,diff2,tol=5)

        ax32.plot(x1,y1,color=color1,lw=2)
        ax32.plot(x2,y2,color=color2,lw=2)
        
    
    diff3 = (phase2.thy[:,0]-phase2.thx[:,0])*(pery_arr/phase2.Ty_base)
    diff3 = np.mod(diff3+pery_arr/2.,pery_arr)-pery_arr/2.
    diff3 = diff3[::nskip]

    x3,y3 = clean(t3_num,diff3,tol=5)    
    ax32.plot(x3,y3,color=color3,lw=2)

    # plot lims

    ax11.set_xlim(sim1_mean_xmin-4*marg,sim1_mean_xmax+3*marg)
    ax11.set_ylim(sim1_mean_ymin-2*marg,sim1_mean_ymax+marg)

    ax12.set_xlim(sim1_mean_xmin-4*marg,sim1_mean_xmax+3*marg)
    ax12.set_ylim(sim1_mean_ymin-2*marg,sim1_mean_ymax+marg)

    ax31.set_xlim(phase1.t[0],phase1.t[-1])
    ax32.set_xlim(phase2.t[0],phase2.t[-1])

    ax21.set_ylim(-11,11)
    ax22.set_ylim(-11,11)

    ax31.set_ylim(-11,11)
    ax32.set_ylim(-11,11)


    # labels and legends

    #ax32.set_title('thji - thjk, raw (theory)')

    # x labels
    ax11.set_xlabel(r'$s^x$')
    ax12.set_xlabel(r'$s^x$')
    ax31.set_xlabel(r'$\bm{t}$',labelpad=-10)
    ax32.set_xlabel(r'$\bm{t}$',labelpad=-10)

    # y labels
    ax11.set_ylabel(r'$s^y$')
    #ax12.set_ylabel(r'$s^y$')

    fig.text(0.0, .675, r'\textbf{Phase Difference}', va='center', rotation='vertical')

    # xticks
    ax11.set_xticks([])
    ax12.set_xticks([])

    ax21.set_xticks([])
    ax22.set_xticks([])
    
    startx, endx = ax31.get_xlim()
    ax31.xaxis.set_ticks([startx,endx])
    ax32.xaxis.set_ticks([startx,endx])

    # yticks
    ax11.set_yticks([])
    ax12.set_yticks([])
    
    # subplot titles
    ax11.set_title(r'\textbf{A} $\quad$ \textbf{Mean Field}',loc='left')
    ax12.set_title(r'\textbf{B} $\quad$ \textbf{Mean Field}',loc='left')
    
    ax21.set_title(r'\textbf{C} $\quad$ \textbf{Numerics}',loc='left')
    ax22.set_title(r'\textbf{D} $\quad$ \textbf{Numerics}',loc='left')
    
    ax31.set_title(r'\textbf{E} $\quad$ \textbf{Theory}',loc='left')
    ax32.set_title(r'\textbf{F} $\quad$ \textbf{Theory}',loc='left')

    lgnd = ax31.legend(loc='lower center',bbox_to_anchor=(1.2,-1.),scatterpoints=1,ncol=3)

    return fig

def get_antiphase(fx,xlo,xhi,N=100,tau=None):
    """
    helper function for micro_vs_macro2_theta_stability()
    get approx. antiphase value given sinusoidal function + zero at approximately antiphase
    
    xlo,xhi: domain limits
    fx: interpolated function
    """
    # move away from in-phase solution
    approx_half = (xhi-xlo)/2.
    margin = (xhi-approx_half)/10. # starting point to left and right of approximate antiphase


    if tau == None:
        argin = ()
        def f(x):
            return fx(x)

    else:
        argin = (tau,)
        def f(x,tau):
            return fx(x,tau)


    # get antiphase using root convergence
    antip = brentq(f,approx_half-margin,approx_half+margin,args=argin)

    return antip


def get_antiphase_fulltheta(rhs,xlo,xhi,choice='asa',tau=1.):
    """
    helper function for micro_vs_macro2_theta_stability()
    
    get approx. antiphase value by restricting solutions to single degree of freedom
    
    
    fx: interpolated function
    """
    np.random.seed(0)
    # move away from in-phase solution
    # look for exact antiphase solution
    approx_half = (xhi-xlo)/2.
    margin = (xhi-approx_half)/4. # starting point to left and right of approximate antiphase

    #print 'brentq',xlo,xhi

    # list all possible in-phase, antiphase solutions
    #phaselist = ['sss','ssa','sas','saa','ass','asa','aas','aaa']

    def f(x,tau):
        #print x,'brentq'
        if choice == 'ssa':
            if False:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                xx = np.linspace(0,per,1000)
                ax.plot([0,per],[0,0],color='black')
                
                c = np.linspace(1.,1.1,100)
                
                for tt in c:
                    
                    ax.plot(xx,rhs(0.,0,xx,tt)[-1],color=str((tt-1.)/.2))
                #ax.scatter(approx_half-margin,0)
                #ax.scatter(approx_half+margin,0)
                
                
                px,py,pz = rhs(0.,0.,approx_half-margin,tau)
                px2,py2,pz2 = rhs(0.,0.,approx_half+margin,tau)
                print 'lo',px+py+pz,'hi',px2+py2+pz2,'tau',tau
                
                #print px+py+pz
                plt.show()
                
            px,py,pz = rhs(0.,0.,x,tau)
        elif choice == 'sas':
            px,py,pz = rhs(0.,x,0.,tau)
        elif choice == 'saa':
            px,py,pz = rhs(0.,x,x,tau)
        elif choice == 'ass':
            px,py,pz = rhs(x,0.,0.,tau)
        elif choice == 'asa':
            px,py,pz = rhs(x,0.,x,tau)
        elif choice == 'aas':
            px,py,pz = rhs(x,x,0.,tau)
        elif choice == 'aaa':
            if False:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                xx = np.linspace(0,per,100)
                for tt in np.linspace(1,2,100):
                    ax.plot(xx,rhs(xx,xx,xx,tt)[0])
                    ax.plot(xx,rhs(xx,xx,xx,tt)[1])
                    ax.plot(xx,rhs(xx,xx,xx,tt)[2])
                #ax.scatter(approx_half-margin,0)
                #ax.scatter(approx_half+margin,0)
                
                
                px,py,pz = rhs(0.,0.,approx_half-margin,tau)
                px2,py2,pz2 = rhs(0.,0.,approx_half+margin,tau)
                print 'lo',px+py+pz,'hi',px2+py2+pz2,'tau',tau
                
                #print px+py+pz
                plt.show()

            px,py,pz = rhs(x,x,x,tau)
        else:
            raise ValueError('Invalid choice '+str(choice))

        #print px+py-pz,x,choice
        return px*np.random.rand(1)/10000 + py*np.random.rand(1)/10000 + pz*np.random.rand(1)/10000
        #if (np.abs(px) > tol) or (np.abs(py) > tol) or (np.abs(pz) > tol):
        #    err = np.abs(px)+np.abs(py)+np.abs(pz)
        #    return px+py/err+pz*err
        #else:
        #    return px+py+pz

    # get antiphase using root convergence
    try:
        antip = brentq(f,approx_half-margin,approx_half+margin,args=(tau,))
    except ValueError:
        return np.nan

    return antip


"""
# load dat file
namexx = 'hxx_fixed_a1=0.1_b1=1.0_c1=1.1_a2=0.1_b2=1.0_c2=1.1_eps=0.01_mux=1.0_muy=1.0.dat'
namexy = 'hxy_fixed_a1=0.1_b1=1.0_c1=1.1_a2=0.1_b2=1.0_c2=1.1_eps=0.01_mux=1.0_muy=1.0.dat'
nameyx = 'hyx_fixed_a1=0.1_b1=1.0_c1=1.1_a2=0.1_b2=1.0_c2=1.1_eps=0.01_mux=1.0_muy=1.0.dat'
nameyy = 'hyy_fixed_a1=0.1_b1=1.0_c1=1.1_a2=0.1_b2=1.0_c2=1.1_eps=0.01_mux=1.0_muy=1.0.dat'

#namexx = 'hxx_a1=0.5_b1=7.0_c1=6.5_a2=1.1_b2=25.0_c2=25.1_eps=0.01_mux=1.0_muy=1.0.dat'
#namexy = 'hxy_a1=0.5_b1=7.0_c1=6.5_a2=1.1_b2=25.0_c2=25.1_eps=0.01_mux=1.0_muy=1.0.dat'
#nameyx = 'hyx_a1=0.5_b1=7.0_c1=6.5_a2=1.1_b2=25.0_c2=25.1_eps=0.01_mux=1.0_muy=1.0.dat'
#nameyy = 'hyy_a1=0.5_b1=7.0_c1=6.5_a2=1.1_b2=25.0_c2=25.1_eps=0.01_mux=1.0_muy=1.0.dat'

datxx=np.loadtxt(namexx);datxy=np.loadtxt(namexy);datyx=np.loadtxt(nameyx);datyy=np.loadtxt(nameyy)
per = datxx[-1,0]

xlo = datxx[0,0]
xhi = datxx[-1,0]

hxx_interp = interp1d(datxx[:,0],datxx[:,1])
hyx_interp = interp1d(datyx[:,0],datyx[:,1])
"""
per=3.70156211872

def hxx(x):
    return 2*0.00334672*cos(2*pi*x/per) - 2*0.54595706*sin(2*pi*x/per)    

def dhxx(x):
    return -2*0.00334672*sin(2*pi*x/per)*2*pi/per - 2*0.54595706*cos(2*pi*x/per)*2*pi/per

def hxy(x,tau):
    return (-2*0.00368139*cos(2*pi*x/per) + 2*0.60055277*sin(2*pi*x/per))/tau

def dhxy(x,tau):
    return (2*0.00368139*sin(2*pi*x/per)*2*pi/per + 2*0.60055277*cos(2*pi*x/per)*2*pi/per)/tau

def hyx(x):
    return 2*0.00334672*cos(2*pi*x/per) - 2*0.54595706*sin(2*pi*x/per)

def dhyx(x):
    return -2*0.00334672*sin(2*pi*x/per)*2*pi/per - 2*0.54595706*cos(2*pi*x/per)*2*pi/per

def hyy(x,tau):
    return (-2*0.00368139*cos(2*pi*x/per) + 2*0.60055277*sin(2*pi*x/per))/tau

def dhyy(x,tau):
    return (2*0.00368139*sin(2*pi*x/per)*2*pi/per + 2*0.60055277*cos(2*pi*x/per)*2*pi/per)/tau


def rhs(px,py,pz,tau):
    """
    RHS of theta model phase equations
    """
    
    rhs_x=hxx(-px)-hxx(px)+hxy(-px+pz,tau)-hxy(pz,tau)+hxy(py-px+pz,tau)-hxy(py+pz,tau)
    rhs_y=hyx(-py-pz)-hyx(-pz)+hyy(-py,tau)+hyx(px-py-pz)-hyx(px-pz)-hyy(py,tau)
    rhs_z=hyx(-pz)+hyx(px-pz)+hyy(py,tau)-hxy(pz,tau)-hxx(px)-hxy(py+pz,tau)
    
    return np.array([rhs_x,rhs_y,rhs_z])


"""
dhxx_interp = interp1d(datxx[:,0],np.gradient(datxx[:,1],np.diff(datxx[:,0])[0]))
dhyx_interp = interp1d(datyx[:,0],np.gradient(datyx[:,1],np.diff(datyx[:,0])[0]))


def dhxx(x):
    x = mod(x,per)

    return dhxx_interp(x)

def dhxy(x,tau):
    x = mod(x,per)
    dhxy_interp = interp1d(datxy[:,0],np.gradient(datxy[:,1]/tau,np.diff(datxy[:,0])[0]))
    return dhxy_interp(x)

def dhyx(x):
    x = mod(x,per)
    return dhyx_interp(x)

def dhyy(x,tau):
    x = mod(x,per)
    dhyy_interp = interp1d(datyy[:,0],np.gradient(datyy[:,1]/tau,np.diff(datyy[:,0])[0]))
    return dhyy_interp(x)
"""

def jac(px,py,pz,tau):
    """
    Jacobian matrix
    """
    
    j11 = -dhxx(-px)-dhxx(px)-dhxy(-px+pz,tau)-dhxy(py-px+pz,tau)
    j12 = dhxy(py-px+pz,tau)-dhxy(py+pz,tau)
    j13 = dhxy(-px+pz,tau)-dhxy(pz,tau)+dhxy(py-px+pz,tau)-dhxy(py+pz,tau)
    
    j21 = dhyx(px-py+pz)-dhyx(px-pz)
    j22 = -dhyx(-py-pz)-dhyx(px-py-pz)-dhyy(-py,tau)-dhyy(py,tau)
    j23 = -dhyx(-py-pz)+dhyx(-pz)-dhyx(px-py-pz)+dhyx(px-pz)
    
    j31 = dhyx(px-pz)-dhxx(px)
    j32 = dhyy(py,tau)-dhxy(py+pz,tau)
    j33 = -dhyx(-pz)-dhyx(px-pz)-dhxy(pz,tau)-dhxy(py+pz,tau)
    
    return np.array([[j11,j12,j13],
                     [j21,j22,j23],
                     [j31,j32,j33]])

def get_antiphase_tuple(a,choice):
    """
    helper function. get the shape of tuple given phase solution type
    we need this since antiphase values may be calculated on the fly.

    previously we hard coded

    sollist = [(0.,0.,0.),
              (0.,0.,a),
              (0.,a,0.),
              
              (0.,a,a),
              (a,0.,0.),
              (a,0.,a),
              
              (a,a,0,),
              (a,a,a)
    ]

    or 

    solsss = [0,0,0]
    solssa = [0,0,a]
    solsas = [0,a,0]
    
    solsaa = [0,a,a]
    solass = [a,0,0]
    solasa = [a,0,a]
    
    solaas = [a,a,0]
    solaaa = [a,a,a]

    but this is obsolete.
    """

    if choice == 'ssa':
        return (0.,0.,a)
    elif choice == 'sas':
        return (0.,a,0.)
    elif choice == 'saa':
        return (0.,a,a)
    elif choice == 'ass':
        return (a,0.,0.)
    elif choice == 'asa':
        return (a,0.,a)
    elif choice == 'aas':
        return (a,a,0,)
    elif choice == 'aaa':
        return (a,a,a)
    else:
        raise ValueError('Invalid choice '+str(choice))


def micro_vs_macro2_theta_existence():
    """
    show existence of some antiphase solutions
    A00, 0A0, and 00A

    
    """
    
    fig = plt.figure(figsize=(6,6))
    taulist = np.linspace(1,3,100)


def micro_vs_macro2_theta_existence2():
    """
    show existence of some antiphase solutions
    AA0, 0AA, and A0A
    
    """
    
    fig = plt.figure(figsize=(6,6))
    #taulist = np.linspace(1,3,100)
    
    
    

def one_nonsync_existence_stability():
    """
    create figure of two solutions =0.
    """
    
    fig = plt.figure(figsize=(3,2))
    
    ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(132)
    #ax3 = fig.add_subplot(133)

    #### FIRST ANTIPHASE SOLUTION A00
    pxa00 = np.loadtxt('dat/pxa00_bifurcation.dat')
    #ax = axlist[i].twinx()
    
    val,ty = collect_disjoint_branches(pxa00,remove_isolated=True,isolated_number=5,remove_redundant=False,redundant_threshold=.2,N=2,fix_reverse=False)

    for key in val.keys():
        
        x = val[key][:,0]
        y = val[key][:,3]
        
        # given python branch, just determine stability along that one
        stbl = np.zeros(len(x))
                
        for i in range(len(x)):
            tau = val[key][i,0]
            px = y[i]
            py = 0
            pz = 0
            
            #print px,py,pz,tau
            eigs = np.linalg.eig(jac(px,py,pz,tau))[0]
            # if real part positive, mark unstable
            
            if np.sum(np.real(eigs)>0):
                stbl[i] = 1
            else:
                stbl[i] = -1
            #print stbl1[i]

        # plot stable
        ax1.plot(x[stbl<0],(mod(y+per/2.,per)-per/2.)[stbl<0],label=key,color='black',lw=2,zorder=2)
        # plot unstable
        ax1.plot(x[stbl>0],(mod(y+per/2.,per)-per/2.)[stbl>0],label=key,color='black',lw=2,zorder=2,ls='--',dashes=(4,2))


    #### SECOND ANTIPHASE SOLUTION 0A0
    py0a0 = np.loadtxt('dat/py0a0_bifurcation.dat')

    val,ty = collect_disjoint_branches(py0a0,remove_isolated=True,isolated_number=3,remove_redundant=False,redundant_threshold=.2,N=2,fix_reverse=False)
    for key in val.keys():
        
        x = val[key][:,0]
        y = val[key][:,3]
        
        # given python branch, just determine stability along that one
        stbl = np.zeros(len(x))
        
        for i in range(len(x)):
            tau = val[key][i,0]
            px = 0
            py = y[i]
            pz = 0
            
            #print px,py,pz,tau
            eigs = np.linalg.eig(jac(px,py,pz,tau))[0]
            # if real part positive, mark unstable
            
            if np.sum(np.real(eigs)>0):
                stbl[i] = 1
            else:
                stbl[i] = -1
            #print stbl1[i]

        # plot stable
        #ax2.plot(x[stbl<0],(mod(y+per/2.,per)-per/2.)[stbl<0],label=key,color='black',lw=2,zorder=2)
        # plot unstable
        #ax2.plot(x[stbl>0],(mod(y+per/2.,per)-per/2.)[stbl>0],label=key,color='black',lw=2,zorder=2,ls='--',dashes=(4,2))
    
    ##### THIRD ANTIPHASE SOLUTION 00A
    pz00a = np.loadtxt('dat/pz00a_bifurcation.dat')
    val,ty = collect_disjoint_branches(pz00a,remove_isolated=True,isolated_number=3,remove_redundant=False,redundant_threshold=.2,N=2,fix_reverse=False)
    for key in val.keys():

        x = val[key][:,0]
        y = val[key][:,3]
        
        # given python branch, just determine stability along that one
        stbl = np.zeros(len(x))
        
        for i in range(len(x)):
            tau = val[key][i,0]
            px = 0
            py = 0
            pz = y[i]
            
            #print px,py,pz,tau
            eigs = np.linalg.eig(jac(px,py,pz,tau))[0]
            # if real part positive, mark unstable
            
            if np.sum(np.real(eigs)>0):
                stbl[i] = 1
            else:
                stbl[i] = -1
            #print stbl1[i]

        # plot stable
        #ax3.plot(x[stbl<0],(mod(y+per/2.,per)-per/2.)[stbl<0],label=key,color='black',lw=2,zorder=2)
        # plot unstable
        #ax3.plot(x[stbl>0],(mod(y+per/2.,per)-per/2.)[stbl>0],label=key,color='black',lw=2,zorder=2,ls='--',dashes=(4,2))
        #ax3.plot(val[key][:,0],mod(val[key][:,3]+per/2.,per)-per/2.,label=key,color='black',lw=1,zorder=-2)
        
        
        #ax3.set_yticks(np.arange(-1,1+1,1)*per/2.)
        #ax3.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])


    # vertical lines showing sample solutions in mean field failure fig
    ax1.plot([1.005,1.005],[-10,10],color='red')
    ax1.plot([1.4,1.4],[-10,10],color='red')

    # horizontal line for in-phase
    #ax1.plot([1,1.5],[0,0],color='gray',zorder=-1)
    
    ax1.set_yticks(np.arange(-1,1+1,1)*per/2.)
    ax1.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])

    ax1.set_xlabel(r'$\mu^y$')
    ax1.locator_params(axis='x',nbins=3)
    
    ax1.set_xlim(1,1.5)
    ax1.set_ylim(-per/2.,per/2.)


    """
    ax2.set_yticks(np.arange(-1,1+1,1)*per/2.)
    ax2.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])
    
    #ax.locator_params(axis='y',nbins=4)
    ax2.set_xlim(1,1.5)
    ax2.set_ylim(-per/2.,per/2.)

    ax2.set_xlabel(r'$\mu^y$')
    ax2.locator_params(axis='x',nbins=3)

    # horizontal line for in-phase
    #ax2.plot([1,1.5],[0,0],color='gray',zorder=-1)


    # vertical lines showing sample solutions in mean field failure fig
    ax2.plot([1.005,1.005],[-10,10],color='red')
    ax2.plot([1.4,1.4],[-10,10],color='red')


    # horizontal line for in-phase
    #ax3.plot([1,1.5],[0,0],color='gray',zorder=-1)

    # vertical lines showing sample solutions in mean field failure fig
    ax3.plot([1.005,1.005],[-10,10],color='red')
    ax3.plot([1.4,1.4],[-10,10],color='red')

        
    #ax.locator_params(axis='y',nbins=4)
    ax3.set_xlim(1,1.5)
    ax3.set_ylim(-per/2.,per/2.)
        
    ax3.set_xlabel(r'$\mu^y$')
    ax3.locator_params(axis='x',nbins=3)
    """
    
    #ax1.set_title(r'\textbf{A}',loc='left')
    #ax2.set_title(r'\textbf{B}',loc='left')
    #ax3.set_title(r'\textbf{C}',loc='left')

    # label degrees of freedom
    ax1.set_ylabel(r'$\phi^x$',labelpad=-15)
    #ax2.set_ylabel(r'$\phi^y$',labelpad=-15)
    #ax3.set_ylabel(r'$\phi^z$',labelpad=-15)

    return fig



def two_nonsync_existence_stability():
    """
    create existence and stability figure for 2 variables = 0
    """

    dashes = (3,2)

    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(121, projection='3d')
    #ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(122, projection='3d')

    # AA0
    pxyaa0 = np.loadtxt('dat/pxyaa0_bifurcation.dat')
    
    val,ty = collect_disjoint_branches(pxyaa0,remove_isolated=True,
                                       isolated_number=5,
                                       remove_redundant=False,
                                       redundant_threshold=.2,
                                       N=2,fix_reverse=False,
                                       zero_column_exist=True)

    per=3.70156211872
    #per = 

    ax1_xlo = 10
    ax1_xhi = -10

    ax1_ylo = 10
    ax1_yhi = -10

    #print val.keys()

    for key in val.keys() :
        muy = val[key][:,0]
        pxa = val[key][:,2]#np.mod(val[key][:,2]+per/2.,per)-per/2.
        pya = np.mod(val[key][:,3]+per/2.,per)-per/2.#val[key][:,3]#

        #print pxa,pya,muy
        #print val[key][0,:]
        #print muy

        muy2 = muy[(muy<1.5)*(muy>1)]
        pxa2 = pxa[(muy<1.5)*(muy>1)]
        pya2 = pya[(muy<1.5)*(muy>1)]

        # given python branch, just determine stability along that one
        stbl = np.zeros(len(muy2))

        for i in range(len(muy2)):
            tau = muy2[i]
            px = pxa2[i]
            py = pya2[i]
            pz = 0
            
            #print px,py,pz,tau
            eigs = np.linalg.eig(jac(px,py,pz,tau))[0]
            # if real part positive, mark unstable
            
            if np.sum(np.real(eigs)>0):
                stbl[i] = 1
            else:
                stbl[i] = -1
            #print stbl1[i]

            if i%2 == 0:
                if (key != 'br0'):
                    # plot a series of vertical lines to show projection
                    ax1.plot([px,px],[py,py],[1,tau],color='gray',alpha=.2,zorder=-2)




        if (key != 'br0') and (key != 'br7') and (key != 'br6'):
            # plot stable
            ax1.plot(pxa2[stbl<0],
                     pya2[stbl<0],
                     muy2[stbl<0],
                     label=key,color='black',lw=2,zorder=2)

            if False:
                print key
                fig = plt.figure()
                ax = fig.add_subplot(111,projection='3d')
                #ax.plot(pxa2[stbl<0],pya2[stbl<0],muy2[stbl<0])
                ax.plot(pxa2[stbl>0],pya2[stbl>0],muy2[stbl>0])
                plt.show()

            
            # plot unstable
            ax1.plot(pxa2[stbl>0],
                     pya2[stbl>0],
                     muy2[stbl>0],
                     label=key,color='black',lw=2,zorder=2,ls='--',dashes=dashes)

            # plot projection
            ax1.plot(pxa,pya,1,color='gray')


        #ax1.plot(py,pz,muy,color='black',lw=2)
        #proj_fig = collect(pya2,pza2,use_nonan=False,lwstart=1,lwend=1,
        #                   cmapmin=0.5,cmapmax=0.5,cmap='gray')

        #ax1.add_collection3d(proj_fig,zdir='z',zs=1)
        #ax1.add_collection3d(plotobj)

        # get x bounds


        if np.nanmin(pxa) < ax1_xlo:
            ax1_xlo = np.nanmin(pxa)
        if np.nanmax(pxa) > ax1_xhi:
            ax1_xhi = np.nanmax(pxa)

        # get y bounds
        if np.nanmin(pya) < ax1_ylo:
            ax1_ylo = np.nanmin(pya)
        if np.nanmax(pya) > ax1_yhi:
            ax1_yhi = np.nanmax(pya)

    #plt.show()
    ###  0AA
    pyz0aa = np.loadtxt('dat/pyz0aa_bifurcation.dat')

    val,ty = collect_disjoint_branches(pyz0aa,remove_isolated=True,
                                       isolated_number=10,
                                       remove_redundant=False,
                                       redundant_threshold=.2,
                                       N=2,fix_reverse=True)




    #ax2 = fig.add_subplot(311, projection='3d')

    per=3.70156211872
    #per = 

    xlo = 10
    xhi = -10

    ylo = 10
    yhi = -10

    print val.keys()

    for key in val.keys():
        muy = val[key][:,0]
        pya = np.mod(val[key][:,2]+per/2.,per)-per/2.
        pza = np.mod(val[key][:,3]+per/2.,per)-per/2.

        muy2 = muy[muy<1.5]
        pya2 = pya[muy<1.5]
        pza2 = pza[muy<1.5]


        # given python branch, just determine stability along that one
        stbl = np.zeros(len(muy2))

        for i in range(len(muy2)):
            tau = muy2[i]
            px = 0
            py = pya2[i]
            pz = pza2[i]
            
            #print px,py,pz,tau
            eigs = np.linalg.eig(jac(px,py,pz,tau))[0]
            # if real part positive, mark unstable
            
            if np.sum(np.real(eigs)>0):
                stbl[i] = 1
            else:
                stbl[i] = -1
            #print stbl1[i]

            if i%2 == 0:
                # plot a series of vertical lines to show projection
                pass
                #ax2.plot([py,py],[pz,pz],[1,tau],color='gray',alpha=.2,zorder=-2)



        # plot stable
        """
        ax2.plot(pya2[stbl<0],
                 pza2[stbl<0],
                 muy2[stbl<0],
                 label=key,color='black',lw=2,zorder=2)
        
        # plot unstable
        ax2.plot(pya2[stbl>0],
                 pza2[stbl>0],
                 muy2[stbl>0],
                 label=key,color='black',lw=2,zorder=2,ls='--',dashes=dashes)

        #ax2.plot(py,pz,muy,color='black',lw=2)
        #proj_fig = collect(pya2,pza2,use_nonan=False,lwstart=1,lwend=1,
        #                   cmapmin=0.5,cmapmax=0.5,cmap='gray')

        #ax2.add_collection3d(proj_fig,zdir='z',zs=1)
        #ax2.add_collection3d(plotobj)
        ax2.plot(pya2,pza2,1,color='gray')

        # get x bounds
        if np.nanmin(pya) < xlo:
            xlo = np.nanmin(pya)
        if np.nanmax(pya) > xhi:
            xhi = np.nanmax(pya)

        # get y bounds
        if np.nanmin(pza) < ylo:
            ylo = np.nanmin(pza)
        if np.nanmax(pza) > yhi:
            yhi = np.nanmax(pza)
        """



    ###  A0A
    pxza0a = np.loadtxt('dat/pxza0a_bifurcation.dat')

    val,ty = collect_disjoint_branches(pxza0a,remove_isolated=True,
                                       isolated_number=3,
                                       remove_redundant=False,
                                       redundant_threshold=.2,
                                       N=2,fix_reverse=True)

    #ax2 = fig.add_subplot(311, projection='3d')

    per=3.70156211872
    #per = 

    xlo = 10
    xhi = -10

    ylo = 10
    yhi = -10


    keys = val.keys()

    #print val['br3']

    #print keys[:-3]

    for key in keys:
        muy = val[key][:,0]
        pxa_raw = val[key][:,2]
        pza_raw = val[key][:,3]

        pxa = np.mod(val[key][:,2]+per/2.,per)-per/2.
        pza = np.mod(val[key][:,3]+per/2.,per)-per/2.

        muy2 = muy[(muy<1.5)*(muy>1.)]
        pxa2 = pxa[(muy<1.5)*(muy>1.)]
        pza2 = pza[(muy<1.5)*(muy>1.)]


        # given python branch, just determine stability along that one
        stbl = np.zeros(len(muy2))

        for i in range(len(muy2)):
            tau = muy2[i]
            px = pxa2[i]
            py = 0
            pz = pza2[i]
            
            #print px,py,pz,tau
            eigs = np.linalg.eig(jac(px,py,pz,tau))[0]

            #print tau,px,py,pz,eigs
            
            # if real part positive, mark unstable
            if np.sum(np.real(eigs)>0):
                stbl[i] = 1
            else:
                stbl[i] = -1
            #print stbl1[i]

            if i%1 == 0:
                # plot a series of vertical lines to show projection
                ax3.plot([px,px],[pz,pz],[1,tau],color='gray',alpha=.2,zorder=-10)


        if (key != 'br6') and (key != 'br5') and (key != 'br4'):
            # plot stable
            ax3.plot(pxa2[stbl<0],
                     pza2[stbl<0],
                     muy2[stbl<0],
                     label=key,color='black',lw=2,zorder=10)

            # plot unstable
            ax3.plot(pxa2[stbl>0],
                     pza2[stbl>0],
                     muy2[stbl>0],
                     label=key,color='black',lw=2,zorder=10,ls='--',dashes=dashes)

            if False:
                fig2 = plt.figure()
                ax33 = fig2.add_subplot(111,projection='3d')
                ax33.plot(pxa2[stbl>0],
                          pza2[stbl>0],
                          muy2[stbl>0],label=key)

                ax33.plot(pxa2[stbl<0],
                          pza2[stbl<0],
                          muy2[stbl<0],label=key)

                plt.legend()
                print key,pxa_raw
                plt.show()


        """
        h = 1.
        v = []
        for k in range(0, len(pxa2) - 1):
            x = [pxa2[k], pxa2[k+1], pxa2[k+1], pxa2[k]]
            y = [pza2[k], pza2[k+1], pza2[k+1], pza2[k]]
            z = [muy2[k], muy2[k+1],       h,     h]
            v.append(zip(x, y, z))
        poly3dCollection = Poly3DCollection(v[::-1])
        poly3dCollection.set_alpha(.001)
        #poly3dCollection.set_edgecolor('blue')
        poly3dCollection.set_facecolor('blue')
        poly3dCollection.set_sort_zpos(-2)

        ax3.add_collection3d(poly3dCollection)
        """
        
        #ax2.plot(py,pz,muy,color='black',lw=2)
        #proj_fig = collect(pya2,pza2,use_nonan=False,lwstart=1,lwend=1,
        #                   cmapmin=0.5,cmapmax=0.5,cmap='gray')

        #ax2.add_collection3d(proj_fig,zdir='z',zs=1)
        #ax2.add_collection3d(plotobj)
        ax3.plot(pxa2,pza2,1,color='gray')

        # get x bounds
        if np.nanmin(pxa) < xlo:
            xlo = np.nanmin(pxa)
        if np.nanmax(pxa) > xhi:
            xhi = np.nanmax(pxa)

        # get y bounds
        if np.nanmin(pxa) < ylo:
            ylo = np.nanmin(pxa)
        if np.nanmax(pza) > yhi:
            yhi = np.nanmax(pza)

    

    labels = [r'$-T/2$',r'$0$',r'$T/2$']
    labels2 = [r'$-T/2$',r'$0$']


    """
    labels = [r'$-T/2$',r'$0$',r'$T/2$']
    labels2 = [r'$-T/2$',r'$0$']

    ax1.set_xticks(np.arange(-1,1+1,1)*per/2.)
    ax2.set_xticks(np.arange(-1,1+1,1)*per/2.)

    ax1.set_xticklabels(labels)
    ax2.set_xticklabels(labels)

    ax1.set_yticks(np.arange(-1,1+1,1)*per/2.)
    ax2.set_yticks(np.arange(-1,1,1)*per/2.)

    ax1.set_yticklabels(labels)
    ax2.set_yticklabels(labels2)

    ax1.set_xlim(ax1_xlo-.1,ax1_xhi+1)
    ax1.set_ylim(ax1_ylo-.1,ax1_yhi+.1)
    ax1.set_zlim(1,1.5)

    ax1.set_xlabel(r'$\phi^x$')
    ax1.set_ylabel(r'$\phi^y$')
    #ax1.set_zlabel(r'muy')


    ax1.set_zlabel(r'$\mu^y$')
    #ax1.locator_params(axis='x',nbins=5)
    

    ax2.set_xlim(xlo-.1,xhi+1)
    ax2.set_ylim(ylo-.1,yhi+.1)
    ax2.set_zlim(1,1.5)

    ax2.set_xlabel(r'$\phi^y$')
    ax2.set_ylabel(r'$\phi^z$')
    #ax2.set_zlabel(r'muy')

    ax2.set_zlabel(r'$\mu^y$')
    #ax2.locator_params(axis='x',nbins=5)


    # fix tick positions
    #mpl.rcParams['xtick.major.pad']=0
    ax1.tick_params(axis='x',pad=-20)
    """

    ax1.set_xticks(np.arange(-1,1+1,1)*per/2.)
    #ax2.set_xticks(np.arange(-1,1+1,1)*per/2.)
    ax3.set_xticks(np.arange(-1,1+1,1)*per/2.)

    ax1.set_xticklabels(labels,size=8)
    #ax2.set_xticklabels(labels,size=8)
    ax3.set_xticklabels(labels,size=8)

    ax1.set_yticks(np.arange(-1,1+1,1)*per/2.)
    #ax2.set_yticks(np.arange(-1,1,1)*per/2.)
    ax3.set_yticks(np.arange(-1,1+1,1)*per/2.)

    ax1.set_yticklabels(labels,size=8)
    #ax2.set_yticklabels(labels2,size=8)
    ax3.set_yticklabels(labels,size=8)


    """
    #ax1.tick_params(axis='z',size=8)
    ax1_zlabels = [item.get_text() for item in ax1.get_zticklabels()]
    print ax1_zlabels
    ax1.set_zticklabels(ax1_zlabels,size=15)
    """

    ax1.tick_params(axis='x',labelsize=7,pad=-2.5)
    #ax2.tick_params(axis='x',labelsize=7,pad=-2.5)
    ax3.tick_params(axis='x',labelsize=7,pad=-2.5)

    ax1.tick_params(axis='y',labelsize=7,pad=-2.5)
    #ax2.tick_params(axis='y',labelsize=7,pad=-2.5)
    ax3.tick_params(axis='y',labelsize=7,pad=-2.5)
    
    ax1.tick_params(axis='z',labelsize=7,pad=-2.5)
    #ax2.tick_params(axis='z',labelsize=7,pad=-2.5)
    ax3.tick_params(axis='z',labelsize=7,pad=-2.5)

    ax1.set_xlim(ax1_xlo-.1,ax1_xhi+.2)
    ax1.set_ylim(ax1_ylo-.2,ax1_yhi+.1)
    ax1.set_zlim(1,1.5)

    ax3.set_xlim(xlo-.1,xhi+.2)
    ax3.set_ylim(ylo-.2,yhi+.1)
    ax3.set_zlim(1,1.5)

    ax1.set_xlabel(r'$\phi^x$',labelpad=-8)
    ax1.set_ylabel(r'$\phi^y$',labelpad=-2.5)
    ax1.set_zlabel(r'$\mu^y$',labelpad=-3)

    #ax2.set_xlabel(r'$\phi^y$',labelpad=-8)
    #ax2.set_ylabel(r'$\phi^z$',labelpad=-8)
    #ax2.set_zlabel(r'$\mu^y$',labelpad=-2.5)

    ax3.set_xlabel(r'$\phi^x$',labelpad=-8)
    ax3.set_ylabel(r'$\phi^z$',labelpad=-2.5)
    ax3.set_zlabel(r'$\mu^y$')


    # title
    ax1.set_title(r'\textbf{A}',loc='left')
    ax3.set_title(r'\textbf{B}',loc='left')

    # fix tick positions
    #mpl.rcParams['xtick.major.pad']=0
    #ax1.tick_params(axis='x',pad=0)

    # fix label positions
    """
    ax1.xaxis._axinfo['label']['space_factor'] = 0.
    ax1.yaxis._axinfo['label']['space_factor'] = 0.
    ax1.zaxis._axinfo['label']['space_factor'] = 0.

    ax2.xaxis._axinfo['label']['space_factor'] = 0.
    ax2.yaxis._axinfo['label']['space_factor'] = 0.
    ax2.zaxis._axinfo['label']['space_factor'] = 0.

    ax3.xaxis._axinfo['label']['space_factor'] = 0.
    ax3.yaxis._axinfo['label']['space_factor'] = 0.
    ax3.zaxis._axinfo['label']['space_factor'] = 0.
    """
    
    # generate viewing angles
    ax1.view_init(30, -60)
    #ax2.view_init(30, -50)
    ax3.view_init(30, -60)

    return fig


def two_nonsync_existence_stability_v2():
    """
    create existence and stability figure for 2 variables = 0
    """

    dashes = (3,2)

    fig = plt.figure(figsize=(6,2))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ###  A0A
    pxza0a = np.loadtxt('dat/pxza0a_bifurcation2.dat')

    # get stable idxs
    stbl = (pxza0a[:,0]==1)
    unstbl = np.logical_not(stbl)

    # get idx where py near zero
    py_zero_idx = (np.abs(pxza0a[:,7])<.001)

    # get vars
    mu = pxza0a[:,3]
    px = pxza0a[:,6]
    pz = pxza0a[:,8]

    # plot stable
    ax1.scatter(mu[stbl*py_zero_idx],
                px[stbl*py_zero_idx],
                color='black',s=1)
        
    ax2.scatter(mu[stbl*py_zero_idx],
                pz[stbl*py_zero_idx],
                color='black',s=1)
        
    # plot unstable
    ax1.scatter(mu[unstbl*py_zero_idx],
                px[unstbl*py_zero_idx],
                color='red',s=1)
        
    ax2.scatter(mu[unstbl*py_zero_idx],
                pz[unstbl*py_zero_idx],
                color='red',s=1)


    xlo = -10
    xhi = 10

    ylo = -10
    yhi = 10

    # get x bounds
    if np.nanmin(px) < xlo:
        xlo = np.nanmin(px)
    if np.nanmax(px) > xhi:
        xhi = np.nanmax(px)

    # get y bounds
    if np.nanmin(pz) < ylo:
        ylo = np.nanmin(pz)
    if np.nanmax(pz) > yhi:
        yhi = np.nanmax(pz)

            
    #ax2.legend()

    labels = [r'$-T/2$',r'$0$',r'$T/2$']
    labels2 = [r'$-T/2$',r'$0$']

    #ax1.set_xticks(np.arange(-1,1+1,1)*per/2.)
    #ax2.set_xticks(np.arange(-1,1+1,1)*per/2.)
    #ax2.set_xticks(np.arange(-1,1+1,1)*per/2.)

    #ax1.set_xticklabels(labels,size=8)
    #ax2.set_xticklabels(labels,size=8)
    #ax2.set_xticklabels(labels,size=8)

    ax1.set_yticks(np.arange(-1,1+1,1)*per/2.)
    #ax2.set_yticks(np.arange(-1,1,1)*per/2.)
    ax2.set_yticks(np.arange(-1,1+1,1)*per/2.)

    ax1.set_yticklabels(labels,size=8)
    #ax2.set_yticklabels(labels2,size=8)
    ax2.set_yticklabels(labels,size=8)


    ax1.tick_params(axis='x',labelsize=7)
    #ax2.tick_params(axis='x',labelsize=7,pad=-2.5)
    ax2.tick_params(axis='x',labelsize=7)

    ax1.tick_params(axis='y',labelsize=7)
    #ax2.tick_params(axis='y',labelsize=7,pad=-2.5)
    ax2.tick_params(axis='y',labelsize=7)
    
    #ax1.tick_params(axis='z',labelsize=7,pad=-2.5)
    #ax2.tick_params(axis='z',labelsize=7,pad=-2.5)
    #ax2.tick_params(axis='z',labelsize=7,pad=-2.5)

    ax1.set_xlim(1,1.5)
    #ax1.set_ylim(ylo-.2,yhi+.1)
    #ax1.set_zlim(1,1.5)

    ax2.set_xlim(1,1.5)
    #ax2.set_ylim(ylo-.2,yhi+.1)
    #ax2.set_zlim(1,1.5)

    ax1.set_xlabel(r'$\mu^y$')
    ax1.set_ylabel(r'$\phi^x$',labelpad=-2.5)
    #ax1.set_zlabel(r'$\mu^y$',labelpad=-3)

    ax2.set_xlabel(r'$\mu^y$')
    ax2.set_ylabel(r'$\phi^z$',labelpad=-2.5)
    #ax2.set_zlabel(r'$\mu^y$')


    # title
    ax1.set_title(r'\textbf{A}',loc='left')
    ax2.set_title(r'\textbf{B}',loc='left')

    # brute force draw the synchrony lines
    ax1.plot([1,1.1],[0,0],color='black')
    ax1.plot([1.1,1.5],[0,0],color='red')

    ax2.plot([1.,1.1],[0,0],color='black')
    ax2.plot([1.1,1.5],[0,0],color='red')


    # fix tick positions
    #mpl.rcParams['xtick.major.pad']=0
    #ax1.tick_params(axis='x',pad=0)

    # fix label positions
    """
    ax1.xaxis._axinfo['label']['space_factor'] = 0.
    ax1.yaxis._axinfo['label']['space_factor'] = 0.
    ax1.zaxis._axinfo['label']['space_factor'] = 0.

    ax2.xaxis._axinfo['label']['space_factor'] = 0.
    ax2.yaxis._axinfo['label']['space_factor'] = 0.
    ax2.zaxis._axinfo['label']['space_factor'] = 0.

    ax2.xaxis._axinfo['label']['space_factor'] = 0.
    ax2.yaxis._axinfo['label']['space_factor'] = 0.
    ax2.zaxis._axinfo['label']['space_factor'] = 0.
    """
    
    # generate viewing angles
    #ax1.view_init(30, -60)
    #ax2.view_init(30, -50)
    #ax2.view_init(30, -60)

    #plt.tight_layout()
    #plt.show()
    return fig


def two_nonsync_existence_stability_1guy():
    """
    create existence and stability figure for 2 variables = 0
    """

    dashes = (3,2)

    fig = plt.figure(figsize=(6,6))
    ax2 = fig.add_subplot(111, projection='3d')


    #plt.show()
    ###  0AA
    pyz0aa = np.loadtxt('dat/pyz0aa_bifurcation.dat')

    val,ty = collect_disjoint_branches(pyz0aa,remove_isolated=True,
                                       isolated_number=10,
                                       remove_redundant=False,
                                       redundant_threshold=.2,
                                       N=2,fix_reverse=True)




    #ax2 = fig.add_subplot(311, projection='3d')

    per=3.70156211872
    #per = 

    xlo = 10
    xhi = -10

    ylo = 10
    yhi = -10

    print val.keys()

    for key in val.keys():
        muy = val[key][:,0]
        pya = np.mod(val[key][:,2]+per/2.,per)-per/2.
        pza = np.mod(val[key][:,3]+per/2.,per)-per/2.

        muy2 = muy[(muy<1.5)*(muy>1)]
        pya2 = pya[(muy<1.5)*(muy>1)]
        pza2 = pza[(muy<1.5)*(muy>1)]


        # given python branch, just determine stability along that one
        stbl = np.zeros(len(muy2))

        for i in range(len(muy2)):
            tau = muy2[i]
            px = 0
            py = pya2[i]
            pz = pza2[i]
            
            #print px,py,pz,tau
            eigs = np.linalg.eig(jac(px,py,pz,tau))[0]
            # if real part positive, mark unstable
            
            if np.sum(np.real(eigs)>0):
                stbl[i] = 1
            else:
                stbl[i] = -1
            #print stbl1[i]

            if i%2 == 0:
                # plot a series of vertical lines to show projection

                ax2.plot([py,py],[pz,pz],[1,tau],color='gray',alpha=.2,zorder=-2)



        # plot stable

        ax2.plot(pya2[stbl<0],
                 pza2[stbl<0],
                 muy2[stbl<0],
                 label=key,color='black',lw=2,zorder=2)
        
        # plot unstable
        ax2.plot(pya2[stbl>0],
                 pza2[stbl>0],
                 muy2[stbl>0],
                 label=key,color='black',lw=2,zorder=2,ls='--',dashes=dashes)

        #ax2.plot(py,pz,muy,color='black',lw=2)
        #proj_fig = collect(pya2,pza2,use_nonan=False,lwstart=1,lwend=1,
        #                   cmapmin=0.5,cmapmax=0.5,cmap='gray')

        #ax2.add_collection3d(proj_fig,zdir='z',zs=1)
        #ax2.add_collection3d(plotobj)
        ax2.plot(pya2,pza2,1,color='gray')

        # get x bounds
        if np.nanmin(pya) < xlo:
            xlo = np.nanmin(pya)
        if np.nanmax(pya) > xhi:
            xhi = np.nanmax(pya)

        # get y bounds
        if np.nanmin(pza) < ylo:
            ylo = np.nanmin(pza)
        if np.nanmax(pza) > yhi:
            yhi = np.nanmax(pza)


        #ax2.plot(py,pz,muy,color='black',lw=2)
        #proj_fig = collect(pya2,pza2,use_nonan=False,lwstart=1,lwend=1,
        #                   cmapmin=0.5,cmapmax=0.5,cmap='gray')

        #ax2.add_collection3d(proj_fig,zdir='z',zs=1)
        #ax2.add_collection3d(plotobj)
    



    """
    labels = [r'$-T/2$',r'$0$',r'$T/2$']
    labels2 = [r'$-T/2$',r'$0$']

    ax1.set_xticks(np.arange(-1,1+1,1)*per/2.)
    ax2.set_xticks(np.arange(-1,1+1,1)*per/2.)

    ax1.set_xticklabels(labels)
    ax2.set_xticklabels(labels)

    ax1.set_yticks(np.arange(-1,1+1,1)*per/2.)
    ax2.set_yticks(np.arange(-1,1,1)*per/2.)

    ax1.set_yticklabels(labels)
    ax2.set_yticklabels(labels2)

    ax1.set_xlim(ax1_xlo-.1,ax1_xhi+1)
    ax1.set_ylim(ax1_ylo-.1,ax1_yhi+.1)
    ax1.set_zlim(1,1.5)

    ax1.set_xlabel(r'$\phi^x$')
    ax1.set_ylabel(r'$\phi^y$')
    #ax1.set_zlabel(r'muy')


    ax1.set_zlabel(r'$\mu^y$')
    #ax1.locator_params(axis='x',nbins=5)
    

    ax2.set_xlim(xlo-.1,xhi+1)
    ax2.set_ylim(ylo-.1,yhi+.1)
    ax2.set_zlim(1,1.5)

    ax2.set_xlabel(r'$\phi^y$')
    ax2.set_ylabel(r'$\phi^z$')
    #ax2.set_zlabel(r'muy')

    ax2.set_zlabel(r'$\mu^y$')
    #ax2.locator_params(axis='x',nbins=5)


    # fix tick positions
    #mpl.rcParams['xtick.major.pad']=0
    ax1.tick_params(axis='x',pad=-20)
    """

    #ax1.set_xticks(np.arange(-1,1+1,1)*per/2.)
    ax2.set_xticks(np.arange(-1,1+1,1)*per/2.)
    #ax3.set_xticks(np.arange(-1,1+1,1)*per/2.)

    #ax1.set_xticklabels(labels,size=8)
    ax2.set_xticklabels(labels,size=8)
    #ax3.set_xticklabels(labels,size=8)

    #ax1.set_yticks(np.arange(-1,1+1,1)*per/2.)
    ax2.set_yticks(np.arange(-1,1,1)*per/2.)
    #ax3.set_yticks(np.arange(-1,1+1,1)*per/2.)

    #ax1.set_yticklabels(labels,size=8)
    ax2.set_yticklabels(labels2,size=8)
    #ax3.set_yticklabels(labels,size=8)


    """
    #ax1.tick_params(axis='z',size=8)
    ax1_zlabels = [item.get_text() for item in ax1.get_zticklabels()]
    print ax1_zlabels
    ax1.set_zticklabels(ax1_zlabels,size=15)
    """

    #ax1.tick_params(axis='x',labelsize=7,pad=-2.5)
    ax2.tick_params(axis='x',labelsize=7,pad=-2.5)
    #ax3.tick_params(axis='x',labelsize=7,pad=-2.5)

    #ax1.tick_params(axis='y',labelsize=7,pad=-2.5)
    ax2.tick_params(axis='y',labelsize=7,pad=-2.5)
    #ax3.tick_params(axis='y',labelsize=7,pad=-2.5)
    
    #ax1.tick_params(axis='z',labelsize=7,pad=-2.5)
    ax2.tick_params(axis='z',labelsize=7,pad=-2.5)
    #ax3.tick_params(axis='z',labelsize=7,pad=-2.5)

    """
    ax1.set_xlim(ax1_xlo-.1,ax1_xhi+.2)
    ax1.set_ylim(ax1_ylo-.2,ax1_yhi+.1)
    ax1.set_zlim(1,1.5)
    """

    ax2.set_xlabel(r'$\phi^y$',labelpad=-8)
    ax2.set_ylabel(r'$\phi^z$',labelpad=-8)
    ax2.set_zlabel(r'$\mu^y$',labelpad=-2.5)

    #ax3.set_xlabel(r'$\phi^x$',labelpad=-8)
    #ax3.set_ylabel(r'$\phi^z$',labelpad=-2.5)
    #ax3.set_zlabel(r'$\mu^y$',labelpad=-5)

    # fix tick positions
    #mpl.rcParams['xtick.major.pad']=0
    #ax1.tick_params(axis='x',pad=0)

    # fix label positions
    """
    ax1.xaxis._axinfo['label']['space_factor'] = 0.
    ax1.yaxis._axinfo['label']['space_factor'] = 0.
    ax1.zaxis._axinfo['label']['space_factor'] = 0.

    ax2.xaxis._axinfo['label']['space_factor'] = 0.
    ax2.yaxis._axinfo['label']['space_factor'] = 0.
    ax2.zaxis._axinfo['label']['space_factor'] = 0.

    ax3.xaxis._axinfo['label']['space_factor'] = 0.
    ax3.yaxis._axinfo['label']['space_factor'] = 0.
    ax3.zaxis._axinfo['label']['space_factor'] = 0.
    """
    
    # generate viewing angles
    #ax1.view_init(30, -60)
    ax2.view_init(30, -50)
    #ax3.view_init(30, -60)

    #plt.tight_layout()
    #plt.show()
    return fig


def micro_vs_macro2_theta_stability():
    """
    show loss of stability using phase model fixed point analysis
    """
    
    taulist = np.linspace(1,1.5,100)
        
    # approx antiphase
    antipxx = get_antiphase(hxx,datxx[0,0],datxx[-1,0])
    antipxy = get_antiphase(hxy,datxy[0,0],datxy[-1,0],tau=1)
    antipyx = get_antiphase(hyx,datyx[0,0],datyx[-1,0])
    antipyy = get_antiphase(hyy,datyy[0,0],datyy[-1,0],tau=1)

    # pick one value as antiphase (generially not the case, but the theta model H functions are simple enough)
    a = antipxx
    
    # list all possible in-phase, antiphase solutions
    phaselist = ['sss', #1
                 'ssa', #2
                 'saa', #3
                 'sas', #4
                 'asa', #5
                 'ass', #6
                 'aas', #7
                 'aaa' #8
    ]

    phaselist2 = ['000', #1
                  '00A', #2
                  '0AA', #3
                  '0A0', #4
                  'A0A', #5
                  'A00', #6
                  'AA0', #7
                  'AAA' #8
    ]

    
    evxlist = np.zeros((len(phaselist),len(taulist)),dtype='complex')
    evylist = np.zeros((len(phaselist),len(taulist)),dtype='complex')
    evzlist = np.zeros((len(phaselist),len(taulist)),dtype='complex')
    
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(datxx[:,0],hxx(datxx[:,0]))
        ax.plot(datxx[:,0],hxx(-datxx[:,0]))
        ax.scatter(a,0)
        ax.plot([0,4],[0,0])
        plt.show()
        
    for i in range(len(taulist)):
        tau = taulist[i]

        for j in range(len(phaselist)):
            
            if phaselist[j] != 'sss':
                # if we are looking for non-synchronous solutions, look for antiphase.
                a = get_antiphase_fulltheta(rhs,xlo,xhi,choice=phaselist[j],tau=tau)
                if np.isnan(a):
                    print phaselist[j], 'DNE for tau =',tau
                    eigs = (np.nan,np.nan,np.nan)
                else:
                    px,py,pz = get_antiphase_tuple(a,choice=phaselist[j])
                    
                    err = np.sum(rhs(px,py,pz,tau))
                    #if (err >= 1e-10) or (phaselist[j] == 'aaa'):
                    #    print rhs(px,py,pz,tau),phaselist[j],"(px,py,pz) =",(px,py,pz)
                    eigs = np.linalg.eig(jac(px,py,pz,tau))[0]
            else:
                eigs = np.linalg.eig(jac(0.,0.,0.,tau))[0]
            
            
            evxlist[j,i],evylist[j,i],evzlist[j,i] = eigs
            
            if phaselist[j] == 'asa':
                print tau,px,py,pz,eigs
            
    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(4, 2)

    # populate list with subplots
    axlist = []
    for i in range(4):
        for j in range(2):
            axlist.append(plt.subplot(gs[i,j]))

    for i in range(len(axlist)):

        # if single antiphase solution, plot existence curves
        if phaselist[i] == 'sas':
            py0a0 = np.loadtxt('dat/py0a0_bifurcation.dat')
            
            val,ty = collect_disjoint_branches(py0a0,remove_isolated=True,isolated_number=5,remove_redundant=False,redundant_threshold=.2,N=2,fix_reverse=False)

            ax = axlist[i].twinx()
            for key in val.keys():
                ax.plot(val[key][:,0],mod(val[key][:,3]+per/2.,per)-per/2.,label=key,color='black',lw=1,zorder=-2)

                ax.set_yticks(np.arange(-1,1+1,1)*per/2.)
                ax.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])

                #ax.locator_params(axis='y',nbins=4)
                ax.set_xlim(taulist[0],taulist[-1])
                ax.set_ylim(-per/2.,per/2.)
            #plt.figure()

            #for key in val.keys():
            #    ax.plot(val[key][:,0],val[key][:,3],label=key)


            pass
            
        if phaselist[i] == 'ass':
            pxa00 = np.loadtxt('dat/pxa00_bifurcation.dat')
            ax = axlist[i].twinx()
            ax.plot(pxa00[:43,0],mod(pxa00[:43,1]+per/2.,per)-per/2.,color='black',zorder=-2)
            ax.plot(pxa00[43:,0],mod(pxa00[43:,1]+per/2.,per)-per/2.,color='black',zorder=-2)

            ax.set_yticks(np.arange(-1,1+1,1)*per/2.)
            ax.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])
            
            ax.set_xlim(taulist[0],taulist[-1])
            ax.set_ylim(-per/2.,per/2.)

        if phaselist[i] == 'ssa':
            pz00a = np.loadtxt('dat/pz00a_bifurcation.dat')
            val,ty = collect_disjoint_branches(pz00a,remove_isolated=True,isolated_number=5,remove_redundant=False,redundant_threshold=.2,N=2,fix_reverse=False)
            ax = axlist[i].twinx()
            for key in val.keys():
                ax.plot(val[key][:,0],mod(val[key][:,3]+per/2.,per)-per/2.,label=key,color='black',lw=1,zorder=-2)


                ax.set_yticks(np.arange(-1,1+1,1)*per/2.)
                ax.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])

                #ax.locator_params(axis='y',nbins=4)
                ax.set_xlim(taulist[0],taulist[-1])
                ax.set_ylim(-per/2.,per/2.)
        
        # sync line
        axlist[i].plot([1,3],[0,0],color='gray')

        #print i
        axlist[i].set_title(phaselist2[i])
        
        axlist[i].plot(taulist,np.real(evxlist[i,:]),lw=3,color='blue',label=r'$\lambda^x$',ls='-',alpha=.7)
        #axlist[i].plot(taulist,np.imag(evxlist[i,:]),lw=3,color='green',label=r'$\lambda^x$',ls='-')
        
        axlist[i].plot(taulist,evylist[i,:],lw=3,color='green',label=r'$\lambda^y$',ls='--',dashes=(5,1),alpha=.7)
        axlist[i].plot(taulist[::5],evzlist[i,:][::5],lw=3,color='black',label=r'$\lambda^z$',ls='',marker='+',markeredgewidth=1.5,alpha=.75)



        axlist[i].set_xlim(taulist[0],taulist[-1])
        #axlist[i].legend()

        # force y axis to have same intervals above and below x-axis.
        mindata = np.nanmin([np.nanmin(evxlist[i,:]),np.nanmin(evylist[i,:]),np.nanmin(evzlist[i,:])])
        maxdata = np.nanmax([np.nanmax(evxlist[i,:]),np.nanmax(evylist[i,:]),np.nanmax(evzlist[i,:])])
        bound = np.nanmax([np.abs(mindata),np.abs(maxdata)])
        print mindata,maxdata,bound,phaselist2[i]
        axlist[i].set_ylim(-bound-bound/5,bound+bound/5)

        # limit the number of y ticks
        axlist[i].locator_params(axis='y',nbins=6)

        if i >= len(axlist)-2:
            axlist[i].set_xlabel(r'$\tau$')
        else:
            axlist[i].set_xticks([])

    lgnd = axlist[-1].legend(loc='lower center',bbox_to_anchor=(-.15,-1.1),ncol=3)

    #plt.tight_layout()
    #fig.subplots_adjust(bottom=-.5)
    #plt.show()
    
    return fig


def jac_theta(px,py,pz):
    """
    Jacobian matrix of theta model.
    """
    
    

def thetan(nskip_ph=10,nskip_num=500):
    """
    full vs phase in the theta model
    n theta exc vs n theta inh.
    """

    fig = plt.figure(figsize=(6,6))
    ax11 = plt.subplot(311)
    ax21 = plt.subplot(312)
    ax31 = plt.subplot(313)

    #a1=1.;b1=1.;c1=.5
    #a2=a1;b2=b1;c2=c1


    # supercrit values
    a1=.5;b1=7.;c1=6.5
    a2=1.1;b2=25.;c2=25.1

    total = 1500 #1500 default
    mux=1.;muy=5.5

    eps = .005


    xin = np.array([.1,.5,2.])
    yin = np.array([.7,.3,-.24])


    #xin = np.array([-.1,0.])
    #xin = np.array([-2.22,1.0946])
    
    #yin = np.array([-1.,.3])
    #yin = np.array([-2,-.47835])


    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)

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
                       recompute_slow_lc=False,
                       use_slowmod_lc=True)

        sx0 = p.sxa_fn(0)
        sy0 = p.sya_fn(0)

        # starting frequency
        freqx0 = get_freq(a1,b1,c1,sx0,sy0)
        freqy0 = get_freq(a2,b2,c2,sx0,sy0)

        thxin = sv2phase(xin,freqx0)
        thyin = sv2phase(yin,freqy0)

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

    # plot mean field + sim field
    ax11.plot(num.t,num.sx,color='blue',alpha=.25,label=r'$s^x$')
    ax11.plot(num.t,num.sy,color='red',alpha=.25,label=r'$s^y$')
    ax11.plot(the.t,the.sxa,color='blue',alpha=.75,label=r'$\bar s^x$')
    ax11.plot(the.t,the.sya,color='red',alpha=.75,label=r'$\bar s^y$')
    ax11.set_xlim(the.t[0],the.t[-1])

    # inset showing order eps magnitude changes in syn vars
    ax11ins = inset_axes(ax11,width="20%",height=.5,loc=8)
    ax11ins.plot(num.t,num.sy,color='red',alpha=.25,label=r'$s^y$')
    ax11ins.plot(the.t,the.sya,color='red',alpha=.75,label=r'$\bar s^y$')

    ax11ins.plot(num.t,num.sx,color='blue',alpha=.25,label=r'$s^x$')
    ax11ins.plot(the.t,the.sxa,color='blue',alpha=.75,label=r'$\bar s^x$')

    ax11ins.set_xlim(285,290)
    ax11ins.set_ylim(.9625,.9675)
    ax11ins.set_xticks([])
    ax11ins.set_yticks([])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax11, ax11ins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax11.set_xticks([])
    ax11.legend()

    # antiphase lines
    ax21.plot(num.t,-num.perx/2.,color='gray',zorder=0)
    ax21.plot(num.t,-num.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax21.plot(num.t,num.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax21.plot(num.t,num.pery/2.,color='gray',ls='--',zorder=0)
    ax21.set_xticks([])

    ax31.plot(the.t,-the.perx/2.,color='gray',zorder=0)
    ax31.plot(the.t,-the.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax31.plot(the.t,the.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax31.plot(the.t,the.pery/2.,color='gray',ls='--',zorder=0)

    # plot numerics (21)
    for i in range(num.N-1):
        diff1 = num.phasex[:,i+1]-num.phasex[:,0]
        diff2 = num.phasey[:,i+1]-num.phasey[:,0]

        diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
        diff2 = np.mod(diff2+num.perx/2.,num.perx)-num.perx/2.

        ax21.scatter(num.t[::nskip_num],diff1[::nskip_num],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        ax21.scatter(num.t[::nskip_num],diff2[::nskip_num],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

    diff3 = num.phasey[:,0]-num.phasex[:,0]
    diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
    ax21.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)


    # plot theory (31)
    for i in range(num.N-1):
        diff1 = the.thx[:,i+1]-the.thx[:,0]
        diff2 = the.thy[:,i+1]-the.thy[:,0]
        
        diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
        diff2 = np.mod(diff2+the.perx/2.,the.perx)-the.perx/2.
        
        ax31.scatter(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        ax31.scatter(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)


    diff3 = the.thy[:,0]-the.thx[:,0]
    diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
    ax31.scatter(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)

    ax31.set_xlim(num.t[0],num.t[-1])
    ax21.set_xlim(num.t[0],num.t[-1])

    
    ax31.set_xlabel(r'$\bm{t}$')
    fig.text(0.0, 0.35, r'\textbf{Phase Difference}', va='center', rotation='vertical')

    ax21.set_title(r'\textbf{Numerics}')
    ax31.set_title(r'\textbf{Theory}')

    lgnd = ax31.legend(loc='lower center',bbox_to_anchor=(.5,-.9),scatterpoints=1,ncol=3)
    lgnd.legendHandles[2]._sizes = [30]
    lgnd.legendHandles[3]._sizes = [30]
    lgnd.legendHandles[4]._sizes = [30]

    return fig



    """
    3 theta models
    """



def h_fun_theta():
    """
    show h functions of the theta model
    """
    
    mux = 1.
    muy = 1.
    #muy = 2.62

    eps = .01

    a1=.1;b1=1.;c1=1.1
    a2=a1;b2=b1;c2=c1


    sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
    

    phase = tsmp.Phase(use_last=False,
                       save_last=False,
                       run_full=False,
                       recompute_h=True,
                       run_phase=True,
                       recompute_slow_lc=False,
                       a1=a1,b1=b1,c1=c1,
                       a2=a2,b2=b2,c2=c2,
                       T=0,dt=.05,mux=mux,muy=muy,
                       eps=eps,
                       thxin=[0,0],thyin=[0,0],sx0=sx0,sy0=sy0)

    sx = phase.sxa_fn(0)
    sy = phase.sya_fn(0)
    
    domxx,totxx = phase.generate_h(sx,sy,choice='xx',return_domain=True)
    domxy,totxy = phase.generate_h_inhom(sx,sy,choice='xy')
    domyx,totyx = phase.generate_h_inhom(sx,sy,choice='yx')
    domyy,totyy = phase.generate_h(sx,sy,choice='yy',return_domain=True)


    # create fig object
    fig = plt.figure(figsize=(6,4))

    gs = gridspec.GridSpec(2, 2)
    ax11 = plt.subplot(gs[0, 0])
    ax12 = plt.subplot(gs[0, 1])
    ax21 = plt.subplot(gs[1, 0])
    ax22 = plt.subplot(gs[1, 1])

    #mp.figure()
    #mp.plot(domxx,totxx)
    #mp.show()

    ax11.plot(domxx,totxx,lw=2,color='black')
    ax12.plot(domxy,totxy,lw=2,color='black')
    ax21.plot(domyx,totyx,lw=2,color='black')
    ax22.plot(domyy,totyy,lw=2,color='black')

    
    #ax11.set_title(r'$H^{xx}$')
    #ax12.set_title(r'$H^{xy}$')
    #ax21.set_title(r'$H^{yx}$')
    #ax22.set_title(r'$H^{yy}$')

    ax11.set_title(r'\textbf{A}',loc='left')
    ax12.set_title(r'\textbf{B}',loc='left')
    ax21.set_title(r'\textbf{C}',loc='left')
    ax22.set_title(r'\textbf{D}',loc='left')

    ax11.text(0.1,.9,r'$H^{xx}$')
    ax12.text(0.1,1.,r'$H^{xy}$')
    ax21.text(0.1,.9,r'$H^{yx}$')
    ax22.text(0.1,1.,r'$H^{yy}$')

    ax21.set_xlabel(r'$\phi$')
    ax22.set_xlabel(r'$\phi$')

    ax11.set_xlim(domxx[0],domxx[-1])
    ax12.set_xlim(domxy[0],domxy[-1])
    ax21.set_xlim(domyx[0],domyx[-1])
    ax22.set_xlim(domyy[0],domyy[-1])


    ax11.locator_params(axis='x',nbins=5)
    ax12.locator_params(axis='x',nbins=5)
    ax21.locator_params(axis='x',nbins=5)
    ax22.locator_params(axis='x',nbins=5)

    
    ax11.locator_params(axis='y',nbins=3)
    ax12.locator_params(axis='y',nbins=3)
    ax21.locator_params(axis='y',nbins=3)
    ax22.locator_params(axis='y',nbins=3)
    
    return fig
    

    #ax = fig.add_subplot(111)

def slowmod_stability():
    """
    show stability in the theta model network N=2 when the synapses are slowly varying
    """
    import thetaslowmod_phase
    #import numpy as np
    #import matplotlib.pylab as mp

    #from scipy.interpolate import interp1d
    from scipy.optimize import brentq


    a1=.5;b1=7.;c1=6.5
    a2=1.1;b2=25.;c2=25.1

    p = thetaslowmod_phase.Phase(T=0,
                                 thxin=[0,0],
                                 thyin=[0,0],
                                 sx0=1,
                                 sy0=1,
                                 a1=a1,b1=b1,c1=c1,
                                 a2=a2,b2=b2,c2=c2,
                                 muy=5.43,
                                 eps=0.01
                                )

    T = 5
    dt = .0025
    TN = int(T/dt)
    t = np.linspace(0,T,TN)

    mean_sol = p.run_mean_field(1.02,1.02,T,dt,no_eps=True)

    if False:
        mp.figure()
        mp.plot(t,mean_sol[:,0])
        mp.plot(t,mean_sol[:,1])

    bxx = p.beta_xx_coeff
    bxy = p.beta_xy_coeff
    byx = p.beta_yx_coeff
    byy = p.beta_yy_coeff

    sbar = p.sbarx
    pz = 0.0

    zero_vals = np.zeros(TN)
    antiphase_vals = np.zeros(TN)
    antiphase_y_vals = np.zeros(TN)
    px_vals = np.zeros(TN)
    pz_vals = np.zeros(TN)

    px_vals[0] = .05
    pz_vals[0] = pz

    collect_times = np.linspace(434,439,4)
    collect_idx = []
    collected_fns = []
    collected_fn_doms = []
    collected_zeros = []

    for i in range(len(collect_times)):        
        idx = np.argmin(np.abs(t/p.eps-collect_times[i]))
        collect_idx.append(idx)
        print idx

    for i in range(TN):
        sx = p.sxa_fn(t[i])
        sy = p.sya_fn(t[i])

        freqx = np.sqrt(p.a1+p.b1*sx-p.c1*sy)
        freqy = np.sqrt(p.a2+p.b2*sx-p.c2*sy)

        Tx = 1./freqx
        Ty = 1./freqy

        #dom11,tot11 = p.generate_h(sx,sy,choice='xx',return_domain=True)
        #dom22,tot22 = p.generate_h(sx,sy,choice='yy',return_domain=True)

        #dom12,tot12 = p.generate_h_inhom(sx,sy,choice='xy',return_domain=True)
        #dom21,tot21 = p.generate_h_inhom(sx,sy,choice='yx',return_domain=True)

        # create linear interp
        #hxx = interp1d(dom11,tot11)
        #hxy = interp1d(dom12,tot12)
        #hyx = interp1d(dom21,tot21)
        #hyy = interp1d(dom22,tot22)

        pz = pz + dt*( (sx-p.sbarx)*(byx-bxx)/p.eps + (sy-p.sbary)*(byy-bxy)/p.eps)
        #px_rhs = .5*(hxx(np.mod(-dom11,Tx))-hxx(np.mod(dom11,Tx))+hxy(mp.mod(-dom11+pz,Tx))-hxy(np.mod(pz,Tx)))
        px_rhs = .5*(p.h11(np.mod(-p.interpdom,p.Tx_base))-\
                     p.h11(np.mod(p.interpdom,p.Tx_base))+\
                     p.h12(mp.mod(-p.interpdom+pz,p.Tx_base))-\
                     p.h12(np.mod(pz,p.Tx_base)))

        # create interpolation of right hand side
        px_rhs_fn1 = interp1d(p.interpdom,px_rhs)

        # create modulus interpolation of right hand side
        def px_rhs_fn2(x):
            x = np.mod(x,p.Tx_base)
            return px_rhs_fn1(x)

        if i < TN-1:

            px_vals[i+1] = px_vals[i] + dt*px_rhs_fn2(px_vals[i])

        pz_unnormed = pz*Ty
        pz_vals[i] = np.mod(pz_unnormed+Ty/2.,Ty)-Ty/2.
        # find zero
        maxper = p.Tx_base
        minper = 0

        halfper = p.Tx_base/2.
        margin = halfper*.9
        xlo = halfper - margin
        xhi = halfper + margin

        zero_vals[i] = brentq(px_rhs_fn2,xlo,xhi)*Tx
        antiphase_vals[i] = Tx/2.
        antiphase_y_vals[i] = Ty/2.
        
        # collect example right hand sides
        if i in collect_idx:
            collected_fns.append(px_rhs)
            collected_fn_doms.append(p.interpdom*Tx)
            collected_zeros.append(zero_vals[i])

    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)


    # clean the discontinuities in pz
    t1_num = copy.deepcopy(t)
    diff1 = np.mod(pz_vals+antiphase_y_vals,2*antiphase_y_vals)-antiphase_y_vals
    x1,y1 = clean(t1_num,diff1,tol=.1)

    ax.plot(t/p.eps,antiphase_vals,color='gray')
    ax.plot(t/p.eps,antiphase_y_vals,color='gray',ls='--')
    #ax.plot(t/p.eps,zero_vals,color='green',ls=':')
    ax.scatter((t/p.eps)[::2],zero_vals[::2],color='green',s=1)

    ax.plot(t/p.eps,px_vals*antiphase_vals*2.,color=color1,lw=2)
    ax.plot(x1/p.eps,y1,color=color3,lw=2,alpha=.2)

    # inset showing fast oscillations in roots
    axins = inset_axes(ax,width="30%",height="60%",loc=4)
    axins.plot(t/p.eps,antiphase_vals,color='gray')
    axins.plot(t/p.eps,antiphase_y_vals,color='gray',ls='--')

    #axins.plot(t/p.eps,zero_vals,color='black')
    axins.scatter((t/p.eps)[::1],zero_vals[::1],color='green',s=1)
    #axins.plot(t/p.eps,zero_vals,color='black')
    axins.plot(t/p.eps,px_vals*antiphase_vals*2.,color=color1,lw=2)
    axins.plot(x1/p.eps,y1,color=color3,lw=2,alpha=.2)

    # second inset showing h functions
    axins2 = plt.axes([.38, .3, .25, .3])


    #axins2 = inset_axes(ax,width="30%",height="50%",loc=8)
    # plot dots + corresponding H functions

    # define bounds variables to get max/mins below
    miny = 10
    maxy = -10
    minx = 10
    maxx = -10

    for i in range(len(collect_idx)):
        tt = t[collect_idx[i]]/p.eps
        yy = zero_vals[collect_idx[i]]
        dom = collected_fn_doms[i]
        fn = collected_fns[i]
        z = collected_zeros[i]

        color = str(1.*i/len(collect_idx))
        axins.scatter(tt,yy,color=color,edgecolor='none',zorder=3)
        axins2.plot(dom,fn,color=color)
        if (i == 0) or (i == len(collect_idx)-1):
            axins2.scatter(z,0,color=color)

        # get min/max y val and x vals
        if np.amin(dom) < minx:
            minx = np.amin(dom)
        if np.amax(dom) > maxx:
            maxx = np.amax(dom)
            
        if np.amin(fn) < miny:
            miny = np.amin(fn)
        if np.amax(fn) > maxy:
            maxy = np.amax(fn)
    #print minx,maxx,miny,maxy
    # mark inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # beautify phase
    ax.set_ylim(0,.6)
    ax.set_xlim(0,t[-1]/p.eps)

    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xlim(430,450)
    axins.set_ylim(.45,.53)

    axins2.set_xticks([])
    axins2.locator_params(axis='y',nbins=3)
    axins2.set_ylim(miny,maxy)
    axins2.set_xlim(minx,maxx)

    axins2.set_title('Right Hand Sides',fontsize=15)
    #axins2.set_xlabel(r'$\phi^x$',labelpad=-12)
    axins2.set_xlabel(r'$\phi^x$')

    ax.set_xlabel(r'$t$ (ms)')

    return fig


def h_fun_tbwb():
    """
    show H functions for traub wb.
    """

    gee_phase=10.;gei_phase=24.
    gie_phase=13.;gii_phase=10.

    gee_full=gee_phase;gei_full=gei_phase
    gie_full=gie_phase;gii_full=gii_phase
        
    mux_phase=1.;muy_phase=1.#23.15

    mux_full=mux_phase;muy_full=muy_phase

    eps = .0025

    fFixed=0.05
    sx0 = fFixed
    sy0 = fFixed

    itb_mean,iwb_mean = get_mean_currents(fFixed)

    itb_full=itb_mean-fFixed*(gee_full-gei_full)
    iwb_full=iwb_mean-fFixed*(gie_full-gii_full)

    itb_phase=itb_mean-fFixed*(gee_phase-gei_phase)
    iwb_phase=iwb_mean-fFixed*(gie_phase-gii_phase)

    tw_phase = twp.Phase(use_last=False,
                         save_last=False,
                         sbar=fFixed,
                         T=0,dt=0.01,
                         itb_mean=itb_mean,iwb_mean=iwb_mean,
                         gee=gee_phase,gei=gei_phase,
                         gie=gie_phase,gii=gii_phase,
                         mux=mux_phase,muy=muy_phase,eps=eps,
                         phs_init_trb=[0,0],
                         phs_init_wb=[0,0],
                         sx0=.05,sy0=.05,
                         #use_mean_field_data=[],
                         verbose=True)

    
    sx = tw_phase.sxa_fn(0)
    sy = tw_phase.sya_fn(0)
    
    domxx,totxx = tw_phase.generate_h(sx,sy,choice='xx',return_domain=True)
    domxy,totxy = tw_phase.generate_h(sx,sy,choice='xy',return_domain=True)
    domyx,totyx = tw_phase.generate_h(sx,sy,choice='yx',return_domain=True)
    domyy,totyy = tw_phase.generate_h(sx,sy,choice='yy',return_domain=True)

    # create fig object
    fig = plt.figure(figsize=(6,4))

    gs = gridspec.GridSpec(2, 2)
    ax11 = plt.subplot(gs[0, 0])
    ax12 = plt.subplot(gs[0, 1])
    ax21 = plt.subplot(gs[1, 0])
    ax22 = plt.subplot(gs[1, 1])

    ax11.plot(domxx,totxx,lw=2,color='black')
    ax12.plot(domxy,totxy,lw=2,color='black')
    ax21.plot(domyx,totyx,lw=2,color='black')
    ax22.plot(domyy,totyy,lw=2,color='black')

    ax11.set_title(r'\textbf{A}',loc='left')
    ax12.set_title(r'\textbf{B}',loc='left')
    ax21.set_title(r'\textbf{C}',loc='left')
    ax22.set_title(r'\textbf{D}',loc='left')

    ax11.text(1,.2,r'$H^{xx}$')
    ax12.text(10.,1.,r'$H^{xy}$')
    ax21.text(1,.75,r'$H^{yx}$')
    ax22.text(15,.5,r'$H^{yy}$')

    ax21.set_xlabel(r'$\phi$')
    ax22.set_xlabel(r'$\phi$')

    ax11.set_xlim(domxx[0],domxx[-1])
    ax12.set_xlim(domxy[0],domxy[-1])
    ax21.set_xlim(domyx[0],domyx[-1])
    ax22.set_xlim(domyy[0],domyy[-1])


    # remove xlabel ticks in top 2 plots
    ax11.set_xticks([])
    ax12.set_xticks([])

    # fix axis lims
    ax11.set_ylim(np.amin(totxx)-np.abs(np.amin(totxx))/10,np.amax(totxx)+np.amax(totxx)/10)
    ax12.set_ylim(np.amin(totxy)-np.abs(np.amin(totxy))/10,np.amax(totxy)+np.amax(totxy)/10)
    ax21.set_ylim(np.amin(totyx)-np.abs(np.amin(totyx))/10,np.amax(totyx)+np.amax(totyx)/10)
    ax22.set_ylim(np.amin(totyy)-np.abs(np.amin(totyy))/10,np.amax(totyy)+np.amax(totyy)/10)

    # fix number of tick marks
    ax11.locator_params(axis='y',nbins=6)
    ax12.locator_params(axis='y',nbins=7)

    return fig
    

    #ax = fig.add_subplot(111)

def tbwb_fi_old():
    """
    FI curve for the traub + ca model and the wb model
    """
    fig = plt.figure(figsize=(6,2))
    
    #f,(ax,ax2) = fig.add_subplots(1,2,sharey=True, facecolor='w')
    
    # code to break x axis
    # from https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122,sharey=ax)

    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()

    d = .02 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)

    # load data files
    tbfi = np.loadtxt('dat/tbfi2.dat')
    wbfi = np.loadtxt('dat/wbfi2.dat')
    
    ax.plot([0,7],[.05,.05],color='gray')
    ax.plot(tbfi[:,0],1000*tbfi[:,1],color='black',lw=2,label=r'Traub+Ca')
    ax.plot(wbfi[:,0],1000*wbfi[:,1],color='black',lw=2,ls='--',dashes=(5,2),label=r'Wang-Buzs{\'a}ki')

    ax2.plot([0,7],[.05,.05],color='gray')
    ax2.plot(tbfi[:,0],1000*tbfi[:,1],color='black',lw=2,label=r'Traub+Ca')
    ax2.plot(wbfi[:,0],1000*wbfi[:,1],color='black',lw=2,ls='--',dashes=(5,2),label=r'Wang-Buzs{\'a}ki')
    
    ax.set_xlim(0,2)
    ax.set_ylim(0,.06*1000)

    ax2.set_xlim(4.5,6.5)
    ax2.set_ylim(0,.06*1000)

    ax.text(1.6,-.015,'Input Current (mA)')
    ax.set_ylabel('Frequency (Hz)')
    #ax.locator_params(axis='y',nbins=5)
    
    ax2.legend(loc='lower right')

    return fig

def tbwb_fi():
    """
    FI curve for the traub + ca model and the wb model
    """
    fig = plt.figure(figsize=(6,2))
    
    #f,(ax,ax2) = fig.add_subplots(1,2,sharey=True, facecolor='w')
    
    # code to break x axis
    # from https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # load data files
    tbfi = np.loadtxt('dat/tbfi2.dat')
    wbfi = np.loadtxt('dat/wbfi2.dat')
    
    ax.plot([0,7],[.05*1000,.05*1000],color='gray')
    ax.plot(tbfi[:,0],1000*tbfi[:,1],color='black',lw=2,label=r'Traub+Ca')
    #ax.plot(wbfi[:,0],wbfi[:,1],color='black',lw=2,ls='--',dashes=(5,2),label=r'Wang-Buzs{\'a}ki')

    ax2.plot([0,7],[.05*1000,.05*1000],color='gray')
    #ax2.plot(tbfi[:,0],tbfi[:,1],color='black',lw=2,label=r'Traub+Ca')
    ax2.plot(wbfi[:,0],1000*wbfi[:,1],color='black',lw=2,ls='--',dashes=(5,2),label=r'Wang-Buzs{\'a}ki')

    # clean up
    
    ax.set_xlim(0,7)
    ax.set_ylim(0,.06*1000)

    ax2.set_xlim(0,1.5)
    ax2.set_ylim(0,.06*1000)

    ax.text(5.3,-20,'Input Current ($\mu$A/cm$^2$)')
    ax.set_ylabel('Frequency (Hz)')
    #ax.locator_params(axis='y',nbins=5)
    
    #ax2.legend(loc='lower right')

    # title
    ax.set_title(r'\textbf{A} $\quad$ Traub+Ca',loc='left')
    ax2.set_title(r'\textbf{B} $\quad$ Wang-Buzs{\'a}ki',loc='left')

    # remove y label on ax2
    ax2.set_yticks([])
    
    return fig


def tbwb_slowmod_stability():
    """
    show stability in the theta model network N=2 when the synapses are slowly varying
    """

    #import matplotlib.pylab as mp

    #from scipy.interpolate import interp1d
    from scipy.optimize import brentq


    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111,projection='3d')

    # load data
    bifdata = np.loadtxt('dat/tbwb_slowmod_fixed_points.ode.bif.dat')

    pval = bifdata[:,3]
    px = bifdata[:,5]
    py = bifdata[:,6]
    
    
    """
    # beautify phase
    ax.set_ylim(0,.6)
    ax.set_xlim(0,t[-1]/p.eps)

    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xlim(430,450)
    axins.set_ylim(.45,.53)

    axins2.set_xticks([])
    axins2.locator_params(axis='y',nbins=3)
    axins2.set_ylim(miny,maxy)
    axins2.set_xlim(minx,maxx)

    axins2.set_title('Right Hand Sides',fontsize=15)
    #axins2.set_xlabel(r'$\phi^x$',labelpad=-12)
    axins2.set_xlabel(r'$\phi^x$')

    ax.set_xlabel(r'$t$')
    """

    return fig




tbwb_per = 20.

def hxx_tbwb(x):
    x = 2*pi*x/tbwb_per
    return -7.08955747047e-05*cos(0*x)+0.0*sin(0*x)-0.202795667523*cos(1*x)-0.0725604693117*sin(1*x)-0.00148292073642*cos(2*x)-0.0366398313165*sin(2*x)+0.00409745449641*cos(3*x)-0.00780934709169*sin(3*x)+0.00197546736019*cos(4*x)-0.00224877555534*sin(4*x)+0.000935337707629*cos(5*x)-0.000842820549328*sin(5*x)+0.000470847004031*cos(6*x)-0.000369605241819*sin(6*x)+0.000248907947718*cos(7*x)-0.000173765104158*sin(7*x)+0.000135765175724*cos(8*x)-8.00588710022e-05*sin(8*x)+7.5547553658e-05*cos(9*x)-3.05724966816e-05*sin(9*x)+4.28038215367e-05*cos(10*x)-2.79640273226e-06*sin(10*x)+4.28038215367e-05*cos(-10*x)+2.79640273224e-06*sin(-10*x)+7.55475536581e-05*cos(-9*x)+3.05724966816e-05*sin(-9*x)+0.000135765175724*cos(-8*x)+8.00588710022e-05*sin(-8*x)+0.000248907947718*cos(-7*x)+0.000173765104158*sin(-7*x)+0.000470847004031*cos(-6*x)+0.000369605241819*sin(-6*x)+0.00093533770763*cos(-5*x)+0.000842820549328*sin(-5*x)+0.00197546736019*cos(-4*x)+0.00224877555534*sin(-4*x)+0.00409745449641*cos(-3*x)+0.00780934709169*sin(-3*x)-0.00148292073642*cos(-2*x)+0.0366398313165*sin(-2*x)-0.202795667523*cos(-1*x)+0.0725604693117*sin(-1*x)


def dhxx_tbwb(x):
    x = 2*pi*x/tbwb_per

    return -0*7.08955747047e-05*sin(0*x)+0*0.0*cos(0*x)+1*0.202795667523*sin(1*x)-1*0.0725604693117*cos(1*x)+2*0.00148292073642*sin(2*x)-2*0.0366398313165*cos(2*x)-3*0.00409745449641*sin(3*x)-3*0.00780934709169*cos(3*x)-4*0.00197546736019*sin(4*x)-4*0.00224877555534*cos(4*x)-5*0.000935337707629*sin(5*x)-5*0.000842820549328*cos(5*x)-6*0.000470847004031*sin(6*x)-6*0.000369605241819*cos(6*x)-7*0.000248907947718*sin(7*x)-7*0.000173765104158*cos(7*x)-8*0.000135765175724*sin(8*x)-8*8.00588710022e-05*cos(8*x)-9*7.5547553658e-05*sin(9*x)-9*3.05724966816e-05*cos(9*x)-10*4.28038215367e-05*sin(10*x)-10*2.79640273226e-06*cos(10*x)+10*4.28038215367e-05*sin(-10*x)-10*2.79640273224e-06*cos(-10*x)+9*7.55475536581e-05*sin(-9*x)-9*3.05724966816e-05*cos(-9*x)+8*0.000135765175724*sin(-8*x)-8*8.00588710022e-05*cos(-8*x)+7*0.000248907947718*sin(-7*x)-7*0.000173765104158*cos(-7*x)+6*0.000470847004031*sin(-6*x)-6*0.000369605241819*cos(-6*x)+5*0.00093533770763*sin(-5*x)-5*0.000842820549328*cos(-5*x)+4*0.00197546736019*sin(-4*x)-4*0.00224877555534*cos(-4*x)+3*0.00409745449641*sin(-3*x)-3*0.00780934709169*cos(-3*x)-2*0.00148292073642*sin(-2*x)-2*0.0366398313165*cos(-2*x)-1*0.202795667523*sin(-1*x)-1*0.0725604693117*cos(-1*x)


def hxy_tbwb(x,tau):
    x = 2*pi*x/tbwb_per
    return (+0.000170149379291*cos(0*x)+0.0*sin(0*x)+0.486709602054*cos(1*x)+0.174145126348*sin(1*x)+0.0035590097674*cos(2*x)+0.0879355951597*sin(2*x)-0.00983389079137*cos(3*x)+0.0187424330201*sin(3*x)-0.00474112166445*cos(4*x)+0.00539706133281*sin(4*x)-0.00224481049831*cos(5*x)+0.00202276931839*sin(5*x)-0.00113003280967*cos(6*x)+0.000887052580365*sin(6*x)-0.000597379074524*cos(7*x)+0.000417036249978*sin(7*x)-0.000325836421737*cos(8*x)+0.000192141290405*sin(8*x)-0.000181314128779*cos(9*x)+7.33739920359e-05*sin(9*x)-0.000102729171688*cos(10*x)+6.71136655742e-06*sin(10*x)-0.000102729171688*cos(-10*x)-6.71136655737e-06*sin(-10*x)-0.000181314128779*cos(-9*x)-7.33739920358e-05*sin(-9*x)-0.000325836421737*cos(-8*x)-0.000192141290405*sin(-8*x)-0.000597379074524*cos(-7*x)-0.000417036249978*sin(-7*x)-0.00113003280967*cos(-6*x)-0.000887052580365*sin(-6*x)-0.00224481049831*cos(-5*x)-0.00202276931839*sin(-5*x)-0.00474112166445*cos(-4*x)-0.00539706133281*sin(-4*x)-0.00983389079137*cos(-3*x)-0.0187424330201*sin(-3*x)+0.0035590097674*cos(-2*x)-0.0879355951597*sin(-2*x)+0.486709602054*cos(-1*x)-0.174145126348*sin(-1*x))/tau


def dhxy_tbwb(x,tau):
    x = 2*pi*x/tbwb_per
    return (-0*0.000170149379291*sin(0*x)+0*0.0*cos(0*x)-1*0.486709602054*sin(1*x)+1*0.174145126348*cos(1*x)-2*0.0035590097674*sin(2*x)+2*0.0879355951597*cos(2*x)+3*0.00983389079137*sin(3*x)+3*0.0187424330201*cos(3*x)+4*0.00474112166445*sin(4*x)+4*0.00539706133281*cos(4*x)+5*0.00224481049831*sin(5*x)+5*0.00202276931839*cos(5*x)+6*0.00113003280967*sin(6*x)+6*0.000887052580365*cos(6*x)+7*0.000597379074524*sin(7*x)+7*0.000417036249978*cos(7*x)+8*0.000325836421737*sin(8*x)+8*0.000192141290405*cos(8*x)+9*0.000181314128779*sin(9*x)+9*7.33739920359e-05*cos(9*x)+10*0.000102729171688*sin(10*x)+10*6.71136655742e-06*cos(10*x)-10*0.000102729171688*sin(-10*x)+10*6.71136655737e-06*cos(-10*x)-9*0.000181314128779*sin(-9*x)+9*7.33739920358e-05*cos(-9*x)-8*0.000325836421737*sin(-8*x)+8*0.000192141290405*cos(-8*x)-7*0.000597379074524*sin(-7*x)+7*0.000417036249978*cos(-7*x)-6*0.00113003280967*sin(-6*x)+6*0.000887052580365*cos(-6*x)-5*0.00224481049831*sin(-5*x)+5*0.00202276931839*cos(-5*x)-4*0.00474112166445*sin(-4*x)+4*0.00539706133281*cos(-4*x)-3*0.00983389079137*sin(-3*x)+3*0.0187424330201*cos(-3*x)+2*0.0035590097674*sin(-2*x)+2*0.0879355951597*cos(-2*x)+1*0.486709602054*sin(-1*x)+1*0.174145126348*cos(-1*x))/tau


def hyx_tbwb(x):
    x = 2*pi*x/tbwb_per
    return +0.00739984015196*cos(1*x)-0.645702252381*sin(1*x)+0.0601573274705*cos(2*x)-0.110560558974*sin(2*x)+0.0229388247268*cos(3*x)-0.0396982260339*sin(3*x)+0.00832508094211*cos(4*x)-0.0192105796094*sin(4*x)+0.00234539031317*cos(5*x)-0.0105383503546*sin(5*x)-0.000129446281085*cos(6*x)-0.00604216845148*sin(6*x)-0.00107155689917*cos(7*x)-0.0034580055653*sin(7*x)-0.00132212642833*cos(8*x)-0.00190208677426*sin(8*x)-0.00126817262939*cos(9*x)-0.000952621220457*sin(9*x)-0.00109490819503*cos(10*x)-0.000378364867116*sin(10*x)-0.00109490819503*cos(-10*x)+0.000378364867116*sin(-10*x)-0.00126817262939*cos(-9*x)+0.000952621220457*sin(-9*x)-0.00132212642833*cos(-8*x)+0.00190208677426*sin(-8*x)-0.00107155689917*cos(-7*x)+0.0034580055653*sin(-7*x)-0.000129446281085*cos(-6*x)+0.00604216845148*sin(-6*x)+0.00234539031317*cos(-5*x)+0.0105383503546*sin(-5*x)+0.00832508094211*cos(-4*x)+0.0192105796094*sin(-4*x)+0.0229388247268*cos(-3*x)+0.0396982260339*sin(-3*x)+0.0601573274705*cos(-2*x)+0.110560558974*sin(-2*x)+0.00739984015195*cos(-1*x)+0.645702252381*sin(-1*x)

def dhyx_tbwb(x):
    x = 2*pi*x/tbwb_per
    return -1*0.00739984015196*sin(1*x)-1*0.645702252381*cos(1*x)-2*0.0601573274705*sin(2*x)-2*0.110560558974*cos(2*x)-3*0.0229388247268*sin(3*x)-3*0.0396982260339*cos(3*x)-4*0.00832508094211*sin(4*x)-4*0.0192105796094*cos(4*x)-5*0.00234539031317*sin(5*x)-5*0.0105383503546*cos(5*x)+6*0.000129446281085*sin(6*x)-6*0.00604216845148*cos(6*x)+7*0.00107155689917*sin(7*x)-7*0.0034580055653*cos(7*x)+8*0.00132212642833*sin(8*x)-8*0.00190208677426*cos(8*x)+9*0.00126817262939*sin(9*x)-9*0.000952621220457*cos(9*x)+10*0.00109490819503*sin(10*x)-10*0.000378364867116*cos(10*x)-10*0.00109490819503*sin(-10*x)-10*0.000378364867116*cos(-10*x)-9*0.00126817262939*sin(-9*x)-9*0.000952621220457*cos(-9*x)-8*0.00132212642833*sin(-8*x)-8*0.00190208677426*cos(-8*x)-7*0.00107155689917*sin(-7*x)-7*0.0034580055653*cos(-7*x)-6*0.000129446281085*sin(-6*x)-6*0.00604216845148*cos(-6*x)+5*0.00234539031317*sin(-5*x)-5*0.0105383503546*cos(-5*x)+4*0.00832508094211*sin(-4*x)-4*0.0192105796094*cos(-4*x)+3*0.0229388247268*sin(-3*x)-3*0.0396982260339*cos(-3*x)+2*0.0601573274705*sin(-2*x)-2*0.110560558974*cos(-2*x)+1*0.00739984015195*sin(-1*x)-1*0.645702252381*cos(-1*x)

    
def hyy_tbwb(x,tau):
    x = 2*pi*x/tbwb_per
    
    return (-0.00569218473228*cos(1*x)+0.496694040293*sin(1*x)-0.046274867285*cos(2*x)+0.0850465838264*sin(2*x)-0.0176452497899*cos(3*x)+0.0305370969491*sin(3*x)-0.006403908417*cos(4*x)+0.0147773689303*sin(4*x)-0.00180414639475*cos(5*x)+0.00810642334966*sin(5*x)+9.95740623731e-05*cos(6*x)+0.00464782188575*sin(6*x)+0.000824274537822*cos(7*x)+0.002660004281*sin(7*x)+0.00101702032949*cos(8*x)+0.0014631436725*sin(8*x)+0.000975517407223*cos(9*x)+0.000732785554198*sin(9*x)+0.000842237073099*cos(10*x)+0.000291049897782*sin(10*x)+0.000842237073099*cos(-10*x)-0.000291049897782*sin(-10*x)+0.000975517407223*cos(-9*x)-0.000732785554198*sin(-9*x)+0.00101702032949*cos(-8*x)-0.0014631436725*sin(-8*x)+0.000824274537822*cos(-7*x)-0.002660004281*sin(-7*x)+9.95740623731e-05*cos(-6*x)-0.00464782188575*sin(-6*x)-0.00180414639475*cos(-5*x)-0.00810642334966*sin(-5*x)-0.006403908417*cos(-4*x)-0.0147773689303*sin(-4*x)-0.0176452497899*cos(-3*x)-0.0305370969491*sin(-3*x)-0.046274867285*cos(-2*x)-0.0850465838264*sin(-2*x)-0.00569218473227*cos(-1*x)-0.496694040293*sin(-1*x))/tau

def dhyy_tbwb(x,tau):
    x = 2*pi*x/tbwb_per
    
    return (1*0.00569218473228*sin(1*x)+1*0.496694040293*cos(1*x)+2*0.046274867285*sin(2*x)+2*0.0850465838264*cos(2*x)+3*0.0176452497899*sin(3*x)+3*0.0305370969491*cos(3*x)+4*0.006403908417*sin(4*x)+4*0.0147773689303*cos(4*x)+5*0.00180414639475*sin(5*x)+5*0.00810642334966*cos(5*x)-6*9.95740623731e-05*sin(6*x)+6*0.00464782188575*cos(6*x)-7*0.000824274537822*sin(7*x)+7*0.002660004281*cos(7*x)-8*0.00101702032949*sin(8*x)+8*0.0014631436725*cos(8*x)-9*0.000975517407223*sin(9*x)+9*0.000732785554198*cos(9*x)-10*0.000842237073099*sin(10*x)+10*0.000291049897782*cos(10*x)+10*0.000842237073099*sin(-10*x)+10*0.000291049897782*cos(-10*x)+9*0.000975517407223*sin(-9*x)+9*0.000732785554198*cos(-9*x)+8*0.00101702032949*sin(-8*x)+8*0.0014631436725*cos(-8*x)+7*0.000824274537822*sin(-7*x)+7*0.002660004281*cos(-7*x)+6*9.95740623731e-05*sin(-6*x)+6*0.00464782188575*cos(-6*x)-5*0.00180414639475*sin(-5*x)+5*0.00810642334966*cos(-5*x)-4*0.006403908417*sin(-4*x)+4*0.0147773689303*cos(-4*x)-3*0.0176452497899*sin(-3*x)+3*0.0305370969491*cos(-3*x)-2*0.046274867285*sin(-2*x)+2*0.0850465838264*cos(-2*x)-1*0.00569218473227*sin(-1*x)+1*0.496694040293*cos(-1*x))/tau
    
#tau = 2.62


if False:
    mp.figure()
    mp.title('h derivattives (ana)')
    x = np.linspace(0,20,200)
    mp.plot(x,dhxx_tbwb(x))
    mp.plot(x,dhxy_tbwb(x,1))
    mp.plot(x,dhyx_tbwb(x))
    mp.plot(x,dhyy_tbwb(x,1))

    mp.figure()
    mp.title('h derivattives (num)')
    x = np.linspace(0,20,200)
    mp.plot(x,np.gradient(hxx_tbwb(x),20./200))
    mp.plot(x,np.gradient(hxy_tbwb(x,1),20./200))
    mp.plot(x,np.gradient(hyx_tbwb(x),20./200))
    mp.plot(x,np.gradient(hyy_tbwb(x,1),20./200))


    mp.figure()
    mp.title('h fns')
    mp.plot(x,hxx_tbwb(x))
    mp.plot(x,hxy_tbwb(x,1))
    mp.plot(x,hyx_tbwb(x))
    mp.plot(x,hyy_tbwb(x,1))
    
    mp.show()


def rhs_tbwb(px,py,pz,tau):
    """
    RHS_TBWB of theta model phase equations
    """
    
    rhs_tbwb_x=hxx_tbwb(-px)-hxx_tbwb(px)+hxy_tbwb(-px+pz,tau)-hxy_tbwb(pz,tau)+hxy_tbwb(py-px+pz,tau)-hxy_tbwb(py+pz,tau)
    rhs_tbwb_y=hyx_tbwb(-py-pz)-hyx_tbwb(-pz)+hyy_tbwb(-py,tau)+hyx_tbwb(px-py-pz)-hyx_tbwb(px-pz)-hyy_tbwb(py,tau)
    rhs_tbwb_z=hyx_tbwb(-pz)+hyy_tbwb(0,tau)+hyx_tbwb(px-pz)+hyy_tbwb(py,tau)-hxx_tbwb(0)-hxy_tbwb(pz,tau)-hxx_tbwb(px)-hxy_tbwb(py+pz,tau)
    
    return np.array([rhs_tbwb_x,rhs_tbwb_y,rhs_tbwb_z])
        
    
def jac_tbwb(px,py,pz,tau):
    """
    Jac_Tbwbobian matrix
    """
    j11 = -dhxx_tbwb((-px))-dhxx_tbwb((px))-dhxy_tbwb((-px+pz),tau)-dhxy_tbwb((py-px+pz),tau)
    j12 = dhxy_tbwb((py-px+pz),tau)-dhxy_tbwb((py+pz),tau)
    j13 = dhxy_tbwb((-px+pz),tau)-dhxy_tbwb((pz),tau)+dhxy_tbwb((py-px+pz),tau)-dhxy_tbwb((py+pz),tau)
        
    j21 = dhyx_tbwb((px-py+pz))-dhyx_tbwb((px-pz))
    j22 = -dhyx_tbwb((-py-pz))-dhyx_tbwb((px-py-pz))-dhyy_tbwb((-py),tau)-dhyy_tbwb((py),tau)
    j23 = -dhyx_tbwb((-py-pz))+dhyx_tbwb((-pz))-dhyx_tbwb((px-py-pz))+dhyx_tbwb((px-pz))
        
    j31 = dhyx_tbwb((px-pz))-dhxx_tbwb((px))
    j32 = dhyy_tbwb((py),tau)-dhxy_tbwb((py+pz),tau)
    j33 = -dhyx_tbwb((-pz))-dhyx_tbwb((px-pz))-dhxy_tbwb((pz),tau)-dhxy_tbwb((py+pz),tau)
        
    return np.array([[j11,j12,j13],
                     [j21,j22,j23],
                     [j31,j32,j33]])


if False:
    mp.figure()
    x = np.linspace(-10,10,200)

    tau = 3.5

    """
    (-5.1514348342607263e-14, 1.1546319456101628e-13, -7.9541000000000004, 3.5476299999999998)
    (6.2172489379008766e-14, -2.6645352591003757e-14, -7.9618000000000002, 3.5970300000000002)
    (3.1974423109204508e-14, -6.7501559897209518e-14, -7.9694000000000003, 3.6464500000000002)
    (-7.9936057773011271e-14, -2.4868995751603507e-14, -7.9768000000000008, 3.69591)
    (1.4210854715202004e-14, -5.5067062021407764e-14, -7.9840000000000018, 3.74539)
    (8.8817841970012523e-15, -8.8817841970012523e-15, -7.9909999999999997, 3.7948900000000001)
    (4.6185277824406512e-14, 7.1054273576010019e-15, -7.9979000000000013, 3.8444199999999999)
    (1.9539925233402755e-14, 2.8421709430404007e-14, -8.0045999999999999, 3.8939699999999999)
    (-3.5527136788005009e-15, 2.8421709430404007e-14, -8.011099999999999, 3.94354)
    """
    
    
    print np.linalg.eig(jac_tbwb(6.76898,0,14.3486,2.5))

    evals = np.zeros(len(x))
    
    for i in range(len(x)):
        evals[i] = sum(np.linalg.eig(jac_tbwb(0,0,x[i],tau))[0])

    mp.plot([-10,10],[0,0],color='black')
    
    mp.plot(x,rhs_tbwb(0,0,x,tau)[-1])
    mp.plot(x,evals)
    mp.show()

def tbwb_stability_2d():


    ax1_xlo = 10
    ax1_xhi = -10

    ax1_ylo = 10
    ax1_yhi = -10


    dashes = (3,2)

    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(132, projection='3d')
    ax2 = fig.add_subplot(122)

    # AA0
    #pxyaa0 = np.loadtxt('pxyaa0_bifurcation.dat')
    #filelist = ['tbwb_fixed_pts_pxza0a.ode.bif.dat','tbwb_fixed_pts_pxza0a.ode.bif2.dat']
    filelist = ['dat/tbwb_fixed_pts.ode.bif.dat']
    #bifdat = np.loadtxt()
    #bifdat2 = np.loadtxt()

    for name in filelist:
        print name
        bifdat = np.loadtxt(name)

        tauvals_old = bifdat[:,3]
        px = bifdat[:,6]
        pz = bifdat[:,7]


        tauvals = tauvals_old[tauvals_old<3]
        px = np.mod(px[tauvals_old<3]+10,20)-10
        pz = np.mod(pz[tauvals_old<3]+10,20)-10

        #mp.figure()
        #mp.plot(tauvals,px)
        #mp.plot(tauvals,pz)
        #mp.show()


        for i in range(1,len(tauvals)):

            # plot shadows
            #ax1.plot([px[i],px[i]],[pz[i],pz[i]],[1,tauvals[i]],color='gray',alpha=.5)

            # plot projection
            #ax1.scatter(px[i],tauvals[i],color='gray',alpha=.5)

            
            if sum(eig_sgn)>0:

                # make sure adjac_tbwbent lines are within some threshold
                startpos_px = np.array([tauvals[i-1],px[i-1],pz[i-1]])
                endpos_px = np.array([tauvals[i],px[i],pz[i]])


                diff1 = np.linalg.norm(startpos_px - endpos_px)

                tol = 1

                alpha = 1

                if diff1 < tol:

                    #ax1.plot([px[i-1],px[i]],[pz[i-1],pz[i-1]],[tauvals[i-1],tauvals[i]],color='red',lw=3,zorder=-1,alpha=alpha)
                    ax1.plot([tauvals[i-1],tauvals[i]],[px[i-1],px[i]],color='red',ls=':',lw=2)
                    ax2.plot([tauvals[i-1],tauvals[i]],[pz[i-1],pz[i]],color='red',ls=':',lw=2)

                #ax.scatter(px[i],pz[i],tauvals[i],s=5,facecolor='none',edgecolor='red')
                #scatCollection2 = ax.scatter(px[i],pz[i],tauvals[i], s=5,
                #                            c='red',
                #                            edgecolor='none'                                        
                #)

            else:

                # make sure adjac_tbwbent lines are within some threshold
                startpos_px = np.array([tauvals[i-1],px[i-1],pz[i-1]])
                endpos_px = np.array([tauvals[i],px[i],pz[i]])

                diff1 = np.linalg.norm(startpos_px - endpos_px)

                tol = 1

                if diff1 < tol:   
                    #ax1.plot([px[i-1],px[i]],[pz[i-1],pz[i-1]],[tauvals[i-1],tauvals[i]],color='black',lw=3,zorder=-1)

                    #if i%3 == 0:
                        # plot a series of vertical lines to show projection
                        #ax1.plot([px[i],px[i]],[pz[i],pz[i]],[1,tauvals[i]],color='gray',alpha=.2,zorder=2)

                    ax1.plot([tauvals[i-1],tauvals[i]],[px[i-1],px[i]],color='black',lw=1,zorder=-1)
                    ax2.plot([tauvals[i-1],tauvals[i]],[pz[i-1],pz[i]],color='black',lw=1,zorder=-1)

                    #ax1.scatter([tauvals[i-1],tauvals[i]],[px[i-1],px[i]],color='black',lw=3,zorder=-1)
                    #ax2.scatter([tauvals[i-1],tauvals[i]],[pz[i-1],pz[i]],color='black',lw=3,zorder=-1)

                #ax1.scatter(tauvals[i],px[i],color='green',edgecolor='none',s=5)
                #ax2.scatter(tauvals[i],pz[i],color='green',edgecolor='none',s=5)

                #ax.scatter(px[i],pz[i],tauvals[i],s=5,facecolor='none',edgecolor='green')
                #scatCollection3 = ax.scatter(px[i],pz[i],tauvals[i], s=5,
                #                             c='green',
                #                             edgecolor='none'                            
                #)

            
            
    xlo = -10
    xhi = 10
    
    ylo = -10
    yhi = 10
    
    # get x bounds
    if np.nanmin(px) < ax1_xlo:
        ax1_xlo = np.nanmin(px)
    if np.nanmax(px) > ax1_xhi:
        ax1_xhi = np.nanmax(px)
        
    # get y bounds
    if np.nanmin(pz) < ax1_ylo:
        ax1_ylo = np.nanmin(pz)
    if np.nanmax(pz) > ax1_yhi:
        ax1_yhi = np.nanmax(pz)
        
    # get z bounds:
    if np.nanmax(tauvals) < yhi:
        ax1_zhi = np.nanmax(tauvals)


    #val,ty = collect_disjoint_branches(bifdat,remove_isolated=True,
    #                                   isolated_number=5,
    #                                   remove_redundant=False,
    #                                   redundant_threshold=.2,
    #                                   N=2,fix_reverse=False,
    #                                   zero_column_exist=True)


    """
    fig = plt.figure(figsize=(6,2))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    dashes = (5,2)
    

    bifdat = np.loadtxt('dat/tbwb_fixed_pts_pxza0a.ode.bif.dat')

    tauvals_old = bifdat[:,3]
    px = bifdat[:,6]
    pz = bifdat[:,7]

    tauvals = tauvals_old[tauvals_old<3]
    px = np.mod(px[tauvals_old<3]+10,20)-10
    pz = np.mod(pz[tauvals_old<3]+10,20)-10

    
    
    for i in range(1,len(tauvals)):

        # plot shadows
        #ax1.plot([px[i],px[i]],[pz[i],pz[i]],[1,tauvals[i]],color='gray',alpha=.5)

        # plot projection
        #ax1.scatter(px[i],tauvals[i],color='gray',alpha=.5)

        
        if sum(np.linalg.eig(jac_tbwb(px[i],0,pz[i],tauvals[i]))[0])>0:

            # make sure adjac_tbwbent lines are within some threshold
            startpos_px = np.array([tauvals[i-1],px[i-1],pz[i-1]])
            endpos_px = np.array([tauvals[i],px[i],pz[i-1]])

            diff1 = np.linalg.norm(startpos_px - endpos_px)

            tol = .25

            if diff1 < tol:
                ax1.plot([tauvals[i-1],tauvals[i]],[px[i-1],px[i]],color='red',ls=':',lw=2)
                ax2.plot([tauvals[i-1],tauvals[i]],[pz[i-1],pz[i]],color='red',ls=':',lw=2)
            
            #ax.scatter(px[i],pz[i],tauvals[i],s=5,facecolor='none',edgecolor='red')
            #scatCollection2 = ax.scatter(px[i],pz[i],tauvals[i], s=5,
            #                            c='red',
            #                            edgecolor='none'                                        
            #)

        else:

            # make sure adjac_tbwbent lines are within some threshold
            startpos_px = np.array([tauvals[i-1],px[i-1],pz[i-1]])
            endpos_px = np.array([tauvals[i],px[i],pz[i]])

            diff1 = np.linalg.norm(startpos_px - endpos_px)

            tol = .25

            if diff1 < tol:   
                ax1.plot([tauvals[i-1],tauvals[i]],[px[i-1],px[i]],color='black',lw=3,zorder=-1)
                ax2.plot([tauvals[i-1],tauvals[i]],[pz[i-1],pz[i]],color='black',lw=3,zorder=-1)

            #ax1.scatter(tauvals[i],px[i],color='green',edgecolor='none',s=5)
            #ax2.scatter(tauvals[i],pz[i],color='green',edgecolor='none',s=5)

            #ax.scatter(px[i],pz[i],tauvals[i],s=5,facecolor='none',edgecolor='green')
            #scatCollection3 = ax.scatter(px[i],pz[i],tauvals[i], s=5,
            #                             c='green',
            #                             edgecolor='none'                            
            #)


    # plot antiphase lines
    ax1.plot([1,3],[-10,-10],color='gray',ls='--')
    ax1.plot([1,3],[10,10],color='gray',ls='--')

    ax2.plot([1,3],[-10,-10],color='gray',ls='--')
    ax2.plot([1,3],[10,10],color='gray',ls='--')



    # vertical lines showing sample solutions in mean field failure fig
    ax1.plot([1.005,1.005],[-10,10],color='red')
    ax1.plot([2.5,2.5],[-10,10],color='red')

    ax2.plot([1.005,1.005],[-10,10],color='red')
    ax2.plot([2.5,2.5],[-10,10],color='red')

    
    ax1.set_xlim(.9,3)
    ax2.set_xlim(.9,3)

    ax1.set_ylim(-11,11)
    ax2.set_ylim(-11,11)
            
    ax1.set_xlabel(r'$\mu^y$')
    ax2.set_xlabel(r'$\mu^y$')

    #ax2.set_yticks([])

    ax1.set_ylabel(r'$\phi^x$',labelpad=-.2)
    ax2.set_ylabel(r'$\phi^z$',labelpad=-.2)

    #plt.show()
    """


    return fig



def tbwb_stability_2d_v2():
    """
    2d is a misnomer. i just mean that i am plotting each variabe separately as a function of the bifurction parameter muy (tau in the ode files). I used tbwb_fixed_points.ode.
    """

    ax1_xlo = 10
    ax1_xhi = -10

    ax1_ylo = 10
    ax1_yhi = -10


    dashes = (3,2)

    fig = plt.figure(figsize=(6,2))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #ax3 = fig.add_subplot(133)

    # AA0
    #pxyaa0 = np.loadtxt('dat/pxyaa0_bifurcation.dat')
    #filelist = ['dat/tbwb_fixed_pts_pxza0a.ode.bif.dat','dat/tbwb_fixed_pts_pxza0a.ode.bif2.dat']
    filelist = ['dat/tbwb_fixed_pts.ode.bif.dat']
    #bifdat = np.loadtxt()
    #bifdat2 = np.loadtxt()

    for name in filelist:
        print name
        bifdat = np.loadtxt(name)

        tauvals_old = bifdat[:,3]
        px = bifdat[:,6]
        py = bifdat[:,7]
        pz = bifdat[:,8]

        maxtau = 3.


        bif_type_vals = bifdat[:,0][tauvals_old<maxtau]
        tauvals = tauvals_old[tauvals_old<maxtau]
        px = np.mod(px[tauvals_old<maxtau]+10.,20.)-10.
        py = np.mod(py[tauvals_old<maxtau]+10.,20.)-10.
        #pz = np.mod(pz[tauvals_old<maxtau]+10.,20.)-10.
        pz = np.mod(pz[tauvals_old<maxtau],20.)

        for i in range(1,len(tauvals)):

            # make sure adjac_tbwbent lines are within some threshold
            startpos_px = np.array([tauvals[i-1],px[i-1],py[i-1],pz[i-1]])
            endpos_px = np.array([tauvals[i],px[i],py[i],pz[i]])
            diff1 = np.linalg.norm(startpos_px - endpos_px)
            tol = 1

            # just get stability from xpp
            eig_sgn = -1
            bif_type = bif_type_vals[i]
            if bif_type == 1:
                eig_sgn = -1
            if bif_type == 2:
                eig_sgn = 1

            
            if eig_sgn>0:

                if diff1 < tol:

                    #ax1.plot([px[i-1],px[i]],[pz[i-1],pz[i-1]],[tauvals[i-1],tauvals[i]],color='red',lw=3,zorder=-1,alpha=alpha)
                    ax1.plot([tauvals[i-1],tauvals[i]],[px[i-1],px[i]],color='red',ls='-',lw=1)
                    #ax2.plot([tauvals[i-1],tauvals[i]],[py[i-1],py[i]],color='red',ls=':',lw=1)
                    ax2.plot([tauvals[i-1],tauvals[i]],[pz[i-1],pz[i]],color='red',ls='-',lw=1)

            else:
                tol = 1

                if diff1 < tol:   

                    ax1.plot([tauvals[i-1],tauvals[i]],[px[i-1],px[i]],color='black',lw=2,zorder=-1)
                    #ax2.plot([tauvals[i-1],tauvals[i]],[py[i-1],py[i]],color='black',lw=2,zorder=-1)
                    ax2.plot([tauvals[i-1],tauvals[i]],[pz[i-1],pz[i]],color='black',lw=2,zorder=-1)
            
    xlo = -10
    xhi = 10
    
    ylo = -10
    yhi = 10
    
    # get x bounds
    if np.nanmin(px) < ax1_xlo:
        ax1_xlo = np.nanmin(px)
    if np.nanmax(px) > ax1_xhi:
        ax1_xhi = np.nanmax(px)
        
    # get y bounds
    if np.nanmin(pz) < ax1_ylo:
        ax1_ylo = np.nanmin(pz)
    if np.nanmax(pz) > ax1_yhi:
        ax1_yhi = np.nanmax(pz)
        
    # get z bounds:
    if np.nanmax(tauvals) < yhi:
        ax1_zhi = np.nanmax(tauvals)

    # plot antiphase lines
    ax1.plot([-1,3],[-10,-10],color='gray',ls='--',zorder=-3)
    ax1.plot([-1,3],[10,10],color='gray',ls='--',zorder=-3)

    ax2.plot([-1,3],[-10,-10],color='gray',ls='--',zorder=-3)
    ax2.plot([-1,3],[10,10],color='gray',ls='--',zorder=-3)

    # plot in-phase lines
    ax2.plot([-1,3],[20,20],color='gray',ls='-',zorder=-3)
    ax2.plot([-1,3],[0,0],color='gray',ls='-',zorder=-3)

    # vertical lines showing sample solutions in mean field failure fig
    ax1.plot([1.005,1.005],[-10,10],color='red',zorder=-3,ls=':')
    ax1.plot([2.4,2.4],[-10,10],color='red',zorder=-3,ls=':')

    ax2.plot([1.005,1.005],[0,20],color='red',zorder=-3,ls=':')
    ax2.plot([2.4,2.4],[0,20],color='red',zorder=-3,ls=':')

    ax1.set_yticks(np.arange(-1,1+1,1)*tbwb_per/2.)
    ax1.set_yticklabels([r'$-T/2$',r'$0$',r'$T/2$'])

    ax2.set_yticks(np.arange(0,2+1,1)*tbwb_per/2.)
    ax2.set_yticklabels([r'$0$',r'$T/2$',r'$T$'])

    ax1.set_xlim(.9,3)
    ax2.set_xlim(.9,3)

    ax1.set_ylim(-11,11)
    ax2.set_ylim(-1,21)
            
    ax1.set_xlabel(r'$\mu^y$')
    ax2.set_xlabel(r'$\mu^y$')

    ax1.set_ylabel(r'$\phi^x$',labelpad=-20)
    ax2.set_ylabel(r'$\phi^z$',labelpad=4)

    ax1.set_title(r'\textbf{A}',loc='left')
    ax2.set_title(r'\textbf{B}',loc='left')
    
    return fig



def tbwb_stability_3d():

    """
    # load dat files
    namexx = 'h11_tbwb_eps=0.0025_mux=1.0_muy=1.0_gee=10.0_gei=24.0_gie=13.0_gii=10.0_N=2.dat'
    namexy = 'h12_tbwb_eps=0.0025_mux=1.0_muy=1.0_gee=10.0_gei=24.0_gie=13.0_gii=10.0_N=2.dat'
    nameyx = 'h21_tbwb_eps=0.0025_mux=1.0_muy=1.0_gee=10.0_gei=24.0_gie=13.0_gii=10.0_N=2.dat'
    nameyy = 'h22_tbwb_eps=0.0025_mux=1.0_muy=1.0_gee=10.0_gei=24.0_gie=13.0_gii=10.0_N=2.dat'

    datxx=np.loadtxt(namexx);datxy=np.loadtxt(namexy);datyx=np.loadtxt(nameyx);datyy=np.loadtxt(nameyy)
    tbwb_per = datxx[-1,0]

    hxx_interp = interp1d(datxx[:,0],datxx[:,1])
    hyx_interp = interp1d(datyx[:,0],datyx[:,1])
    """    

    taumax = 3.

    ax1_xlo = 10
    ax1_xhi = -10

    ax1_ylo = 10
    ax1_yhi = -10

    dashes = (3,2)

    fig = plt.figure(figsize=(3,3))
    ax1 = fig.add_subplot(111, projection='3d')
    #ax2 = fig.add_subplot(132, projection='3d')
    #ax2 = fig.add_subplot(122, projection='3d')

    # AA0
    #pxyaa0 = np.loadtxt('pxyaa0_bifurcation.dat')
    #bifdat = np.loadtxt('tbwb_fixed_pts_pxza0a.ode.bif.dat')
    #filelist = ['tbwb_fixed_pts.ode.bif.dat']
    bifdat = np.loadtxt('dat/tbwb_fixed_pts.ode.bif.dat')

    tauvals_old = bifdat[:,3]
    px = bifdat[:,6]
    py = bifdat[:,7]
    pz = bifdat[:,8]

    tauvals = tauvals_old[tauvals_old<taumax]
    px = np.mod(px[tauvals_old<taumax]+10,20)-10
    py = np.mod(py[tauvals_old<taumax]+10,20)-10
    pz = np.mod(pz[tauvals_old<taumax],20)


    for i in range(1,len(tauvals)):

        # plot shadows
        #ax1.plot([px[i],px[i]],[pz[i],pz[i]],[1,tauvals[i]],color='gray',alpha=.5)

        # plot projection
        #ax1.scatter(px[i],tauvals[i],color='gray',alpha=.5)

        
        eigs = np.linalg.eig(jac_tbwb(px[i],py[i],pz[i],tauvals[i]))[0]
        eig_sgn = eigs>0
        
        #print eig_sgn
        # make sure adjac_tbwbent lines are within some threshold
        startpos_px = np.array([tauvals[i-1],px[i-1],pz[i-1]])
        endpos_px = np.array([tauvals[i],px[i],pz[i]])
        
        diff1 = np.linalg.norm(startpos_px - endpos_px)
        
        # make alpha less for distance less than max norm (distance farther from viewer)

        alpha = 1-tauvals[i]/taumax

        # Reds colormap index based on percentage distance from top:
        cmap_idx = int((259/1.5)*(1-alpha))
        

        
        #print alpha

        tol = .25
        
        if sum(eig_sgn)>0:

            if diff1 < tol:
                #ax1.plot([px[i-1],px[i]],[pz[i-1],pz[i-1]],[tauvals[i-1],tauvals[i]],color='red',lw=3,zorder=-1)
                #ax1.plot([px[i-1],px[i]],[pz[i-1],pz[i]],[tauvals[i-1],tauvals[i]],color='red',lw=3,zorder=-1,alpha=alpha)

                cm_tup = cm.Reds(cmap_idx)
                
                ax1.scatter(px[i],pz[i],tauvals[i],c=cm_tup,s=2,zorder=0,alpha=1,edgecolors=cm_tup)

                # vertical line projection
                if i%5 == 0:
                    #ax1.plot([px[i],px[i]],[pz[i],pz[i]],[1,tauvals[i]],color='gray',alpha=.5,zorder=10)

                    # shadow
                    ax1.scatter(px[i],pz[i],1,c=cm_tup,alpha=.5,s=.1,edgecolors=cm_tup)

                
                #ax1.plot([px[i-1],px[i]],[pz[i-1],pz[i]],[tauvals[i-1],tauvals[i]],color='red',lw=3,zorder=-1)
                
                #ax1.plot([tauvals[i-1],tauvals[i]],[px[i-1],px[i]],color='red',ls=':',lw=2)
                #ax2.plot([tauvals[i-1],tauvals[i]],[pz[i-1],pz[i]],color='red',ls=':',lw=2)
            
            #ax.scatter(px[i],pz[i],tauvals[i],s=5,facecolor='none',edgecolor='red')
            #scatCollection2 = ax.scatter(px[i],pz[i],tauvals[i], s=5,
            #                            c='red',
            #                            edgecolor='none'                                        
            #)

        else:

            # make sure adjac_tbwbent lines are within some threshold


            if diff1 < tol:   
                #ax1.plot([px[i-1],px[i]],[pz[i-1],pz[i-1]],[tauvals[i-1],tauvals[i]],color='black',lw=3,zorder=-1)
                #ax1.plot([px[i-1],px[i]],[pz[i-1],pz[i]],[tauvals[i-1],tauvals[i]],color='black',lw=3,zorder=-1,alpha=alpha)
                ax1.scatter(px[i],pz[i],tauvals[i],c=str(alpha),s=2,zorder=0,edgecolors=str(alpha))


                # vertical line projection
                if i%5 == 0:
                    #ax1.plot([px[i],px[i]],[pz[i],pz[i]],[1,tauvals[i]],color='gray',alpha=.5)

                    # shadow
                    ax1.scatter(px[i],pz[i],1,c=str(alpha),edgecolors=str(alpha),alpha=.5,s=.1)

                
            
    xlo = -10
    xhi = 10
    
    ylo = -10
    yhi = 10
    
    # get x bounds
    if np.nanmin(px) < ax1_xlo:
        ax1_xlo = np.nanmin(px)
    if np.nanmax(px) > ax1_xhi:
        ax1_xhi = np.nanmax(px)
        
    # get y bounds
    if np.nanmin(pz) < ax1_ylo:
        ax1_ylo = np.nanmin(pz)
    if np.nanmax(pz) > ax1_yhi:
        ax1_yhi = np.nanmax(pz)
        
    # get z bounds:
    if np.nanmax(tauvals) < yhi:
        ax1_zhi = np.nanmax(tauvals)




    #val,ty = collect_disjoint_branches(bifdat,remove_isolated=True,
    #                                   isolated_number=5,
    #                                   remove_redundant=False,
    #                                   redundant_threshold=.2,
    #                                   N=2,fix_reverse=False,
    #                                   zero_column_exist=True)



    #tbwb_per = 20


    ax1.tick_params(axis='x',labelsize=7,pad=-2.5)
    ax1.tick_params(axis='y',labelsize=7,pad=-2.5)
    ax1.tick_params(axis='z',labelsize=7,pad=-2.5)


    #ax1.set_xlim(ax1_xlo,ax1_xhi)
    ax1.set_ylim(0,20)
    ax1.set_zlim(1,taumax)

    #ax1.set_xlabel(r'$\phi^x$',labelpad=-8)
    #ax1.set_ylabel(r'$\phi^z$',labelpad=-2.5)
    #ax1.set_zlabel(r'$\mu^y$',labelpad=-3)

    ax1.set_xlabel(r'$\phi^x$',labelpad=-8)

    ax1.set_ylabel(r'$\phi^z$',labelpad=0)
    ax1.yaxis._axinfo['label']['space_factor'] = 2.0

    ax1.set_zlabel(r'$\mu^y$',labelpad=-8)

    #ax2.set_xlabel(r'$\phi^y$',labelpad=-8)
    #ax2.set_ylabel(r'$\phi^z$',labelpad=-8)
    #ax2.set_zlabel(r'$\mu^y$',labelpad=-2.5)

    #ax2.set_xlabel(r'$\phi^x$',labelpad=-8)
    #ax2.set_ylabel(r'$\phi^Z$',labelpad=-2.5)
    #ax2.set_zlabel(r'$\mu^y$')


    # title
    #ax1.set_title(r'\textbf{A}',loc='left')
    #ax2.set_title(r'\textbf{B}',loc='left')

    ax1.view_init(50, -35)
    #ax2.view_init(30, -50)
    #ax2.view_init(30, -60)

    #plt.show()

    # point to fixed points + corresponding label
    #p1 = [0., 12.2959, 2.4] # fixed point
    #p1s = [p1[0]-6,p1[1]+10,p1[2]] # arrow starting point. same as text.
    #a1 = Arrow3D([p1s[0], p1[0]], [p1s[1], p1[1]], [p1s[2], p1[2]], mutation_scale=10,
    #             lw=1, arrowstyle="-|>", color="r")
    #ax1.text(p1s[0],p1s[1],p1s[2],'('+str(p1[0])+','+str(p1[1])+')',fontsize=6,color='r')

    
    p2 = [4.19533, 13.9421, 2.4]
    p2s = [p2[0]-3,p2[1]+8,p2[2]]
    a2 = Arrow3D([p2s[0], p2[0]], [p2s[1], p2[1]], [p2s[2], p2[2]], mutation_scale=10,
                 lw=1, arrowstyle="-|>", color="k")
    ax1.text(p2s[0],p2s[1],p2s[2],'('+str(p2[0])+','+str(p2[1])+')*',fontsize=6)

    #ax1.scatter(p2[0],p2[1],p2[2],marker='*',color='black',s=100)

    
    #p3 = [6.84678, 14.3707, 2.4]
    #p3s = [13,2,p3[2]]
    #a3 = Arrow3D([p3s[0], p3[0]], [p3s[1], p3[1]], [p3s[2], p3[2]], mutation_scale=10,
    #             lw=1, arrowstyle="-|>", color="r",zorder=1)
    #ax1.text(p3s[0]-2,p3s[1]-5,p3s[2],'('+str(p3[0])+','+str(p3[1])+')',fontsize=6,color='r')
    

    p4 = [-4.1622, 9.77581, 2.4]
    p4s = [-10,20,p4[2]]
    a4 = Arrow3D([p4s[0], p4[0]], [p4s[1], p4[1]], [p4s[2], p4[2]], mutation_scale=10,
                 lw=1, arrowstyle="-|>", color="k")
    ax1.text(p4s[0],p4s[1],p4s[2],'('+str(p4[0])+','+str(p4[1])+')',fontsize=6)

    p5 = [-6.83394, 7.53307,2.4]
    p5s = [-10,0,2.4]
    a5 = Arrow3D([p5s[0], p5[0]], [p5s[1], p5[1]], [p5s[2], p5[2]], mutation_scale=10,
                 lw=1, arrowstyle="-|>", color="k")
    ax1.text(p5s[0]-1,p5s[1]-5,p5s[2],'('+str(p5[0])+','+str(p5[1])+')',fontsize=6)


    # muy = 1
    p6 = [0,13,1]
    p6s = [3,10,1]
    a6 = Arrow3D([p6s[0], p6[0]], [p6s[1], p6[1]], [p6s[2], p6[2]], mutation_scale=10,
                 lw=1, arrowstyle="wedge", color="k")
    ax1.text(p6s[0]+1,p6s[1]-1,p6s[2],'('+str(p6[0])+','+str(p6[1])+')',fontsize=6)
    
    #ax1.add_artist(a1)
    ax1.add_artist(a2)
    #ax1.add_artist(a3)
    ax1.add_artist(a4)
    ax1.add_artist(a5)
    ax1.add_artist(a6)

    
    #plt.tight_layout()

    ax1.set_xticks(np.arange(-1,1+1,1)*tbwb_per/2.)
    #ax1.set_yticks(np.arange(-1,1+1,1)*tbwb_per/2.)
    ax1.set_yticks(np.arange(0,1+.5,.5)*tbwb_per)

    #print np.arange(0,1+1,1)*tbwb_per

    #labels = [r'$-T/2$',r'$0$',r'$T/2$']
    
    ax1.set_xticklabels(labels,size=8)
    ax1.set_yticklabels([r'$0$',r'$T/2$',r'$T$'],size=8)


    return fig


def theta_hopf():
    """
    show hopf bifurcation + criticality in theta model
    """
    data = np.loadtxt('dat/theta_supercrit.dat')

    val,ty = collect_disjoint_branches(data,remove_isolated=True,isolated_number=1,remove_redundant=False,redundant_threshold=.2,N=2,fix_reverse=False)

    fig = plt.figure(figsize=(6,2))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for key in val.keys():
        
        x = val[key][:,0]
        y1 = val[key][:,2]
        y2 = val[key][:,3]

        if (key != 'br2') and (key != 'br4'):
            if key == 'br3':
                ax1.scatter(x,y1,s=20,color='green',alpha=.6,edgecolor='none')
                ax1.scatter(x,2-y1,s=20,color='green',alpha=.6,edgecolor='none')

            elif key == 'br0':
                ax1.plot(x,y1,lw=2,color='black')

            else:
                ax1.plot(x,y1,lw=1,color='red',ls='-')

    ax1.scatter(5.41,1,color='black',s=50,zorder=10)
    ax1.text(5.25,1.01,r'HB')


    data2 = np.loadtxt('dat/averaged_tbwb.ode.bif.dat')

    val,ty = collect_disjoint_branches(data2,remove_isolated=True,isolated_number=1,remove_redundant=False,redundant_threshold=.2,N=2,fix_reverse=False)

    for key in val.keys():
        
        x = val[key][:,0]
        y1 = val[key][:,2]
        y2 = val[key][:,3]

        if (key != 'br16') and (key != 'br13') and\
           (key != 'br11') and (key != 'br15') and\
           (key != 'br18') and (key != 'br9') and\
           (key != 'br7') and (key != 'br5') and\
           (key != 'br3') and (key != 'br1') and\
           (key != 'br23') and (key != 'br20'):
            if (key == 'br12') or (key == 'br10'):
                #ax2.plot(x,y1,lw=1,color='black',ls='-',label=key)
                pass

            if (key == 'br14') or (key == 'br19'):
                #ax2.plot(x,y1,lw=1,color='red',ls='-',label=key)
                pass


            if (key == 'br21'):
                nskip = 100
                ax2.scatter(x[::nskip],y1[::nskip],lw=1,color='green',s=20,alpha=.6,edgecolor='none')
                ax2.scatter(x[::nskip],.1-y1[::nskip],lw=1,color='green',s=20,alpha=.6,edgecolor='none')

    ax2.plot([0,22.9],[.05,.05],color='black',lw=2)
    ax2.plot([22.9,50],[.05,.05],color='red')


    ax2.scatter(22.9,.050,color='black',s=50,zorder=10)
    ax2.text(22.5,.0505,r'HB')

    ax1.locator_params(ax1is='y',nbins=5)

    ax1.set_xlim(5,6)
    ax2.set_xlim(22,24)

    ax1.set_ylim(0.8,1.2)
    ax2.set_ylim(.04,.06)

    ax1.set_title(r"\textbf{A $\quad$ Theta Model}",loc='left')
    ax2.set_title(r"\textbf{B $\quad$ Traub+Ca, WB}",loc='left')

    ax1.set_xlabel(r'$\mu^y$')
    ax2.set_xlabel(r'$\mu^y$')
    ax1.set_ylabel(r'$\bar s^x$')
    ax1.legend()

    return fig


def theta_input_het(nskip_ph=1,nskip_num=1):
    """
    thetaslowmod and nonslowmod with input heterogeneities
    """

    fig = plt.figure(figsize=(6,6))
    ax11 = plt.subplot(311)
    ax21 = plt.subplot(312)
    ax31 = plt.subplot(313)

    #a1=1.;b1=1.;c1=.5
    #a2=a1;b2=b1;c2=c1


    # supercrit values
    a1=.5;b1=7.;c1=6.5
    a2=1.1;b2=25.;c2=25.1

    total = 1500 #1500 default
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
                    sx0=sx0,sy0=sy0,
                    heterogeneous_pop=True
                 )

    the = tsmp.Phase(a1=a1,b1=b1,c1=c1,
                     a2=a2,b2=b2,c2=c2,
                     T=total,eps=eps,dt=.05,
                     thxin=thxin,thyin=thyin,
                     mux=mux,muy=muy,
                     sx0=sx0,sy0=sy0,
                     run_phase=True,
                     recompute_h=True,
                     heterogeneous_pop=True
                 )

    # plot mean field + sim field
    ax11.plot(num.t,num.sx,color='blue',alpha=.35,label=r'$s^x$',lw=1)
    ax11.plot(num.t,num.sy,color='red',alpha=.35,label=r'$s^y$',lw=1)
    ax11.plot(the.t,the.sxa,color='blue',alpha=.85,label=r'$\bar s^x$',lw=1)
    ax11.plot(the.t,the.sya,color='red',alpha=.85,label=r'$\bar s^y$',lw=1)
    ax11.set_xlim(the.t[0],the.t[-1])

    # inset showing order eps magnitude changes in syn vars
    ax11ins = inset_axes(ax11,width="20%",height="50%",loc=8)
    ax11ins.plot(num.t,num.sy,color='red',alpha=.35,label=r'$s^y$',lw=2)
    ax11ins.plot(the.t,the.sya,color='red',alpha=.85,label=r'$\bar s^y$')

    ax11ins.plot(num.t,num.sx,color='blue',alpha=.35,label=r'$s^x$',lw=2)
    ax11ins.plot(the.t,the.sxa,color='blue',alpha=.85,label=r'$\bar s^x$')

    ax11ins.set_xlim(300,305)
    ax11ins.set_ylim(.975,.98)

    #ax11ins.set_xlim(865,885)
    #ax11ins.set_ylim(.9875,.992)
    ax11ins.set_xticks([])
    ax11ins.set_yticks([])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax11, ax11ins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax11.set_xticks([])
    ax11.legend()

    # antiphase lines
    ax21.plot(num.t,-num.perx/2.,color='gray',zorder=0)
    ax21.plot(num.t,-num.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax21.plot(num.t,num.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax21.plot(num.t,num.pery/2.,color='gray',ls='--',zorder=0)
    ax21.set_xticks([])

    ax31.plot(the.t,-the.perx/2.,color='gray',zorder=0)
    ax31.plot(the.t,-the.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax31.plot(the.t,the.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax31.plot(the.t,the.pery/2.,color='gray',ls='--',zorder=0)

    t1_num = copy.deepcopy(num.t[::nskip_num])
    t2_num = copy.deepcopy(num.t[::nskip_num])
    t3_num = copy.deepcopy(num.t[::nskip_num])

    # plot numerics (21)
    for i in range(num.N-1):
        diff1 = num.phasex[:,i+1]-num.phasex[:,0]
        diff2 = num.phasey[:,i+1]-num.phasey[:,0]

        diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
        diff2 = np.mod(diff2+num.pery/2.,num.pery)-num.pery/2.

        x1,y1 = clean(t1_num,diff1[::nskip_num],tol=.1)
        x2,y2 = clean(t2_num,diff2[::nskip_num],tol=.1)

        #ax21.scatter(num.t[::nskip_num],diff1[::nskip_num],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        #ax21.scatter(num.t[::nskip_num],diff2[::nskip_num],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

        #ax21.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2,lw=2)
        #ax21.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2,lw=2)

        ax21.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax21.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)

    diff3 = num.phasey[:,0]-num.phasex[:,0]
    diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
    x3,y3 = clean(t3_num,diff3[::nskip_num],tol=.5)
    #pos = np.where(np.abs(np.diff(y)) >= tol)[0]
    
    #x[pos] = np.nan
    #y[pos] = np.nan

    #ax21.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax21.plot(x3,y3,color=color3,label=r'$\phi^z$',zorder=1)
    ax21.set_ylim(-.8,.8)

    """
    hatch = ''
    color = 'yellow'
    alpha = .2
    ec = 'black'

    p = patches.Rectangle((0,.4),700,.2,fill=True,color=color,hatch=hatch,alpha=alpha,zorder=-1,ec=ec)
    ax21.add_patch(p)
    #ax21.text(200,.6,'Zoom Below')
    """

    """
    # inset showing small antiphase wiggles in numerics
    ax21ins = inset_axes(ax21,width="20%",height=.5,loc=9)#'lower center',bbox_to_anchor=(.4,.4))
    ax21ins.plot(x1,y1,color=color1,label=r'$s^y$',ls=2)
    #ax21ins.plot(the.t,the.sya,color='red',alpha=.75,label=r'$\bar s^y$')

    #ax21ins.plot(num.t,num.sx,color='blue',alpha=.25,label=r'$s^x$')
    #ax21ins.plot(the.t,the.sxa,color='blue',alpha=.75,label=r'$\bar s^x$')

    ax21ins.set_xlim(800,1000)
    ax21ins.set_ylim(-.6,-.4)
    ax21ins.set_xticks([])
    ax21ins.set_yticks([])



    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax21, ax21ins, loc1=2, loc2=4, fc="none", ec="0.5")
    """

    t1_the = copy.deepcopy(the.t[::nskip_ph])
    t2_the = copy.deepcopy(the.t[::nskip_ph])
    t3_the = copy.deepcopy(the.t[::nskip_ph])

    # plot theory (31)
    for i in range(num.N-1):
        diff1 = the.thx_unnormed[:,i+1]-the.thx_unnormed[:,0]
        diff2 = the.thy_unnormed[:,i+1]-the.thy_unnormed[:,0]
        
        diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
        diff2 = np.mod(diff2+the.perx/2.,the.perx)-the.perx/2.

        x1,y1 = clean(t1_the,diff1[::nskip_ph],tol=.5)
        x2,y2 = clean(t2_the,diff2[::nskip_ph],tol=.5)        
        
        #ax31.scatter(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        #ax31.scatter(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

        #ax31.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2,lw=2)
        #ax31.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2,lw=2)

        ax31.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax31.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)


    diff3 = the.thy_unnormed[:,0]-the.thx_unnormed[:,0]
    diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
    
    x3,y3 = clean(t3_the,diff3[::nskip_ph],tol=.5)

    #ax31.scatter(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax31.plot(x3,y3,color=color3,label=r'$\phi^z$',zorder=1)

    """
    # show region of zoom
    #ax11.set_ylim(.4,.6)
    #ax21.plot()
    p = patches.Rectangle((0,.4),700,.2,fill=True,color=color,hatch=hatch,alpha=alpha,zorder=-1,ec=ec)
    ax31.add_patch(p)
    #ax31.text(200,.6,'Zoom Below')
    """

    ax31.set_xlim(num.t[0],num.t[-1])
    ax21.set_xlim(num.t[0],num.t[-1])

    ax31.set_ylim(-.8,.8)

    
    ax31.set_xlabel(r'$\bm{t}$')
    fig.text(0.0, 0.35, r'\textbf{Phase Difference}', va='center', rotation='vertical')


    lgnd = ax31.legend(loc='lower center',bbox_to_anchor=(.5,-.9),scatterpoints=1,ncol=3)
    lgnd.legendHandles[2]._sizes = [30]
    lgnd.legendHandles[3]._sizes = [30]
    lgnd.legendHandles[4]._sizes = [30]

    # subplot labels
    ax11.set_title(r'\textbf{A} $\quad$ \textbf{Mean Field}',loc='left')
    ax21.set_title(r'\textbf{B} $\quad$ \textbf{Numerics}',loc='left')
    ax31.set_title(r'\textbf{C} $\quad$ \textbf{Theory}',loc='left')

    return fig



def theta_input_het2(nskip_ph=1,nskip_num=1):
    """
    thetaslowmod and nonslowmod with input heterogeneities
    """

    fig = plt.figure(figsize=(6,6))
    ax11 = plt.subplot(311)
    ax21 = plt.subplot(312)
    ax31 = plt.subplot(313)

    #a1=1.;b1=1.;c1=.5
    #a2=a1;b2=b1;c2=c1



    a1=.1;b1=1.;c1=1.1
    a2=a1;b2=b1;c2=c1

    #sx0 = .2921372
    #sy0 = .3077621

    mux = 1.

    eps = .01
    T = 1600
    total = T
    #T = 20

    xin = np.array([0.,.5])
    yin = np.array([.2,.3])

    mux=1.;muy=1.5

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
    else:
        sx0,sy0 = get_sbar(a1,b1,c1,a2,b2,c2)
        sx0 += eps

        # starting frequency
        freqx0 = get_freq(a1,b1,c1,sx0,sy0)
        freqy0 = get_freq(a2,b2,c2,sx0,sy0)
        
        Tx0 = 1./freqx0
        Ty0 = 1./freqy0

        thxin = sv2phase(xin,freqx0)*Tx_base/Tx0
        thyin = sv2phase(yin,freqy0)*Ty_base/Ty0



    num = tsm.Theta(a1=a1,b1=b1,c1=c1,
                    a2=a2,b2=b2,c2=c2,
                    T=total,eps=eps,dt=.001,
                    mux=mux,muy=muy,
                    xin=xin,yin=yin,
                    sx0=sx0,sy0=sy0,
                    heterogeneous_pop=True
                 )

    the = tsmp.Phase(a1=a1,b1=b1,c1=c1,
                     a2=a2,b2=b2,c2=c2,
                     T=total,eps=eps,dt=.05,
                     thxin=thxin,thyin=thyin,
                     mux=mux,muy=muy,
                     sx0=sx0,sy0=sy0,
                     run_phase=True,
                     recompute_h=True,
                     heterogeneous_pop=True
                 )

    # plot mean field + sim field
    ax11.plot(num.t,num.sx,color='blue',alpha=.35,label=r'$s^x$',lw=1)
    ax11.plot(num.t,num.sy,color='red',alpha=.35,label=r'$s^y$',lw=1)
    ax11.plot(the.t,the.sxa,color='blue',alpha=.85,label=r'$\bar s^x$',lw=1)
    ax11.plot(the.t,the.sya,color='red',alpha=.85,label=r'$\bar s^y$',lw=1)

    ax11.set_xlim(the.t[0],the.t[-1])

    # get bounds
    lower_bound = np.amin([np.amin(num.sx),np.amin(num.sy),np.amin(the.sxa),np.amin(the.sya)])
    upper_bound = np.amax([np.amax(num.sx),np.amax(num.sy),np.amax(the.sxa),np.amax(the.sya)])
    ax11.set_ylim(lower_bound-.025,upper_bound+.025)

    ax11.set_xticks([])
    ax11.legend(loc='lower right')

    # antiphase lines
    ax21.plot(num.t,-num.perx/2.,color='gray',zorder=0)
    ax21.plot(num.t,-num.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax21.plot(num.t,num.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax21.plot(num.t,num.pery/2.,color='gray',ls='--',zorder=0)
    ax21.set_xticks([])

    ax31.plot(the.t,-the.perx/2.,color='gray',zorder=0)
    ax31.plot(the.t,-the.pery/2.,color='gray',label=r'$T^y/2$',ls='--',zorder=0)
    ax31.plot(the.t,the.perx/2.,color='gray',label=r'$T^x/2$',zorder=0)
    ax31.plot(the.t,the.pery/2.,color='gray',ls='--',zorder=0)

    

    t1_num = copy.deepcopy(num.t[::nskip_num])
    t2_num = copy.deepcopy(num.t[::nskip_num])
    t3_num = copy.deepcopy(num.t[::nskip_num])

    # plot numerics (21)
    for i in range(num.N-1):
        diff1 = num.phasex[:,i+1]-num.phasex[:,0]
        diff2 = num.phasey[:,i+1]-num.phasey[:,0]

        diff1 = np.mod(diff1+num.perx/2.,num.perx)-num.perx/2.
        diff2 = np.mod(diff2+num.pery/2.,num.pery)-num.pery/2.

        x1,y1 = clean(t1_num,diff1[::nskip_num],tol=.1)
        x2,y2 = clean(t2_num,diff2[::nskip_num],tol=.1)

        #ax21.scatter(num.t[::nskip_num],diff1[::nskip_num],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        #ax21.scatter(num.t[::nskip_num],diff2[::nskip_num],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

        #ax21.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2,lw=2)
        #ax21.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2,lw=2)

        ax21.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax21.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)

    diff3 = num.phasey[:,0]-num.phasex[:,0]
    diff3 = np.mod(diff3+num.pery/2.,num.pery)-num.pery/2.
    x3,y3 = clean(t3_num,diff3[::nskip_num],tol=.5)
    #pos = np.where(np.abs(np.diff(y)) >= tol)[0]
    
    #x[pos] = np.nan
    #y[pos] = np.nan

    #ax21.scatter(num.t[::nskip_num],diff3[::nskip_num],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax21.plot(x3,y3,color=color3,label=r'$\phi^z$',zorder=1)
    ax21.set_ylim(np.nanmin(y3)-.1,np.nanmax(y3)+.1)

    """
    hatch = ''
    color = 'yellow'
    alpha = .2
    ec = 'black'

    p = patches.Rectangle((0,.4),700,.2,fill=True,color=color,hatch=hatch,alpha=alpha,zorder=-1,ec=ec)
    ax21.add_patch(p)
    #ax21.text(200,.6,'Zoom Below')
    """

    """
    # inset showing small antiphase wiggles in numerics
    ax21ins = inset_axes(ax21,width="20%",height=.5,loc=9)#'lower center',bbox_to_anchor=(.4,.4))
    ax21ins.plot(x1,y1,color=color1,label=r'$s^y$',ls=2)
    #ax21ins.plot(the.t,the.sya,color='red',alpha=.75,label=r'$\bar s^y$')

    #ax21ins.plot(num.t,num.sx,color='blue',alpha=.25,label=r'$s^x$')
    #ax21ins.plot(the.t,the.sxa,color='blue',alpha=.75,label=r'$\bar s^x$')

    ax21ins.set_xlim(800,1000)
    ax21ins.set_ylim(-.6,-.4)
    ax21ins.set_xticks([])
    ax21ins.set_yticks([])



    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax21, ax21ins, loc1=2, loc2=4, fc="none", ec="0.5")
    """

    t1_the = copy.deepcopy(the.t[::nskip_ph])
    t2_the = copy.deepcopy(the.t[::nskip_ph])
    t3_the = copy.deepcopy(the.t[::nskip_ph])

    # plot theory (31)
    for i in range(num.N-1):
        diff1 = the.thx_unnormed[:,i+1]-the.thx_unnormed[:,0]
        diff2 = the.thy_unnormed[:,i+1]-the.thy_unnormed[:,0]
        
        diff1 = np.mod(diff1+the.perx/2.,the.perx)-the.perx/2.
        diff2 = np.mod(diff2+the.perx/2.,the.perx)-the.perx/2.

        x1,y1 = clean(t1_the,diff1[::nskip_ph],tol=.5)
        x2,y2 = clean(t2_the,diff2[::nskip_ph],tol=.5)        
        
        #ax31.scatter(the.t[::nskip_ph],diff1[::nskip_ph],color=color1,edgecolor='none',label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',s=5,zorder=2)
        #ax31.scatter(the.t[::nskip_ph],diff2[::nskip_ph],color=color2,edgecolor='none',label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',s=5,zorder=2)

        #ax31.plot(x1,y1,color=color1,label=r'$\theta^x_'+str(i+2)+r'-\theta^x_1$',zorder=2,lw=2)
        #ax31.plot(x2,y2,color=color2,label=r'$\theta^y_'+str(i+2)+r'-\theta^y_1$',zorder=2,lw=2)

        ax31.plot(x1,y1,color=color1,label=r'$\phi^x$',zorder=2,lw=2)
        ax31.plot(x2,y2,color=color2,label=r'$\phi^y$',zorder=2,lw=2)


    diff3 = the.thy_unnormed[:,0]-the.thx_unnormed[:,0]
    diff3 = np.mod(diff3+the.pery/2.,the.pery)-the.pery/2.
    
    x3,y3 = clean(t3_the,diff3[::nskip_ph],tol=.5)

    #ax31.scatter(the.t[::nskip_ph],diff3[::nskip_ph],color=color3,edgecolor='none',label=r'$\theta^y_1-\theta^x_1$',s=5,zorder=2)
    ax31.plot(x3,y3,color=color3,label=r'$\phi^z$',zorder=1)

    """
    # show region of zoom
    #ax11.set_ylim(.4,.6)
    #ax21.plot()
    p = patches.Rectangle((0,.4),700,.2,fill=True,color=color,hatch=hatch,alpha=alpha,zorder=-1,ec=ec)
    ax31.add_patch(p)
    #ax31.text(200,.6,'Zoom Below')
    """

    ax31.set_xlim(num.t[0],num.t[-1])
    ax21.set_xlim(num.t[0],num.t[-1])

    #ax31.set_ylim(-1.5,1.5)
    ax31.set_ylim(np.nanmin(y3)-.1,np.nanmax(y3)+.1)
    
    ax31.set_xlabel(r'$\bm{t}$')
    fig.text(0.0, 0.35, r'\textbf{Phase Difference}', va='center', rotation='vertical')


    lgnd = ax31.legend(loc='lower center',bbox_to_anchor=(.5,-.8),scatterpoints=1,ncol=3)
    lgnd.legendHandles[2]._sizes = [30]
    lgnd.legendHandles[3]._sizes = [30]
    lgnd.legendHandles[4]._sizes = [30]

    # subplot labels
    ax11.set_title(r'\textbf{A} $\quad$ \textbf{Mean Field}',loc='left')
    ax21.set_title(r'\textbf{B} $\quad$ \textbf{Numerics}',loc='left')
    ax31.set_title(r'\textbf{C} $\quad$ \textbf{Theory}',loc='left')

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

    #print matplotlib.rcParams
    #print matplotlib.matplotlib_fname()
    #print ss.plot("u0b").__dict__.keys()
    #print ss.plot("u0b").axes
    #ss = twod.SteadyState()
    #ss.plot()
    #plt.show()
    #ker = twod.Kernel()
    #sim1 = twod.SimDat(g=5,zshift1=.1,zshift2=.3)
    #sim2 = twod.SimDat(q=1.9,g=4.8,zshift1=.1,zshift2=.12,T=10000)

    figures = [

        #(h_fun_theta,[],['h_fun_theta.pdf','h_fun_theta.eps']), # fig 1


        #(micro_vs_macro2_theta,['phi',True],['micro_vs_macro2_theta.pdf','micro_vs_macro2_theta.eps']), # fig 2

        #(one_nonsync_existence_stability,[],['one_nonsync_existence_stability.pdf','one_nonsync_existence_stability.eps']), # fig 3
        #(two_nonsync_existence_stability_v2,[],['two_nonsync_existence_stability.pdf','two_nonsync_existence_stability.eps']), # fig 4
        #(theta_hopf,[],['theta_hopf.pdf','theta_hopf.eps']), # fig 5

        (full_vs_phase_theta_varying_fig,[],['thetaslowmod_full_vs_theory.pdf','thetaslowmod_full_vs_theory.eps']), # fig 6


        #(theta_input_het2,[],['thetaslowmod_input_het.pdf','thetaslowmod_input_het.eps']), # noslowmod, fig 7
        
        #(tbwb_fi,[],['tbwb_fi.pdf','tbwb_fi.eps']), # fig 8
        #(h_fun_tbwb,[],['h_fun_tbwb.pdf','h_fun_tbwb.eps']), # fig 9
        #(micro_vs_macro2_tbwb,[],['micro_vs_macro_tbwb.pdf','micro_vs_macro_tbwb.eps']), # fig 10

        #(tbwb_stability_2d_v2,[],['tbwb_stability_2d.pdf','tbwb_stability_2d.eps']), # figure 11
        #(full_vs_phase_trbwb,[],['wbtrb_full_vs_theory.pdf','wbtrb_full_vs_theory.eps']), # figure 12



        # junk below here

        #(theta_input_het,[],['thetaslowmod_input_het.pdf'])

        #(full_vs_phase_theta_const_fig,[],['thetaslowmod_full_vs_theory.png','thetaslowmod_full_vs_theory.pdf']),  

        #(gaussian_connections,[1,7],['gaussian_connections.pdf']),

        #(micro_vs_macro2_theta_existence,[],['micro_vs_macro2_theta_existence.pdf']),

        #(micro_vs_macro2_theta_stability,[],['micro_vs_macro2_theta_stability.pdf']),
        #(micro_vs_macro2_theta_existence,[],['micro_vs_macro2_theta_existence.pdf']),
                
        #(theta3,[],['theta3_test.pdf']),
        #(slowmod_stability,[],['slowmod_stability.pdf']),        

        #(iprc_theta,[],['iprc_theta.pdf']),
        #(iprc_tbwb,[],['iprc_tbwb.pdf']),

        
        #(tbwb_slowmod_stability,[],['tbwb_slowmod_stability.pdf']),

        #(full_vs_phase_theta_varying_fig_zoomed,[],['thetaslowmod_full_vs_theory_zoomed.pdf']), # fig 7
        #(tbwb_stability_2d,[],['tbwb_stability_2d.pdf']), # figure 12
        #(tbwb_stability_3d,[],['tbwb_stability_3d.pdf']), # figure 12        
        #(two_nonsync_existence_stability_1guy,[],['temp.pdf']), # checking fixed point stability of phix=0,y,z.



        ]


    for fig in figures:
        print fig
        generate_figure(*fig)


if __name__ == "__main__":
    main()
