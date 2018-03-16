# generate long movie of weakly coupled theta models
# by default, the data files correspond to figure 8 of the paper.
# starttime sets the time at which you start saving frames.

# avconv -r 90 -start_number 1 -i test%d.png -b:v 1000k test.mp4


import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
from sys import stdout

#from matplotlib import rcParams

#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm} \usepackage{xcolor} \definecolor{blue1}{HTML}{3399FF}']
matplotlib.rcParams.update({'figure.autolayout': True})

sizeOfFont = 20
fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}
lamomfsize=15 #lambda omega figure size


#import euler
import thetaslowmod as tsm
import generate_figures as gf

cos = np.cos
sin = np.sin
pi = np.pi

movdir = 'mov/' # save frames to this dir

### DATA

def plot_movie(num,skip,file_prefix="mov/test",
               file_suffix=".png",
               scene_number=1):

    for i in range(num.N-1):
        diff1 = num.phasex[:,i+1]-num.phasex[:,0]
        diff2 = num.phasey[:,i+1]-num.phasey[:,0]

        #diff1 = np.mod(diff1+pi,2*pi)-pi
        #diff2 = np.mod(diff2+pi,2*pi)-pi

        diff1 = np.mod(diff1+num.per/2.,num.per)-num.per/2.
        diff2 = np.mod(diff2+num.per/2.,num.per)-num.per/2.

    diff3 = num.phasey[:,0]-num.phasex[:,0]
    diff3 = np.mod(diff3+num.per/2.,num.per)-num.per/2.
    #ax11.scatter(num.t,diff3,color='red',edgecolor='none',label=r'$y_1-x_1$')

    TN = len(num.t)
    total_iter = int(TN/skip)-1
    start_iter = (scene_number-1)*total_iter
    end_iter = (scene_number)*total_iter

    for i in range(total_iter):
        k = i*skip

        fig = plt.figure(figsize=(5,7))

        #plt.suptitle(r"Exc. $\quad\quad\quad$ Inh.")

        ax11 = plt.subplot2grid((3,2),(0,0))
        ax12 = plt.subplot2grid((3,2),(0,1))
        ax21 = plt.subplot2grid((3,2),(1,0))
        ax22 = plt.subplot2grid((3,2),(1,1))

        ax31 = plt.subplot2grid((3,2),(2,0),colspan=2)

        ax11.set_title(r"$x_1$")
        ax12.set_title(r"$y_1$")
        ax21.set_title(r"$x_2$")
        ax22.set_title(r"$y_2$")

        #ax11.set_xlabel(r"\textbf{Voltage (mV)}",fontsize=15)
        #ax12.set_xlabel(r"\textbf{Voltage (mV)}",fontsize=15)

        ax31.set_xlabel(r"\textbf{Time (ms)}",fontsize=15)
        ax31.set_ylabel(r"$\textbf{Phase Difference}$",fontsize=15)

        ax11.set_xlim(-1.1,1.1)
        ax12.set_xlim(-1.1,1.1)
        ax21.set_xlim(-1.1,1.1)
        ax22.set_xlim(-1.1,1.1)

        ax11.set_ylim(-1.1,1.1)
        ax12.set_ylim(-1.1,1.1)
        ax21.set_ylim(-1.1,1.1)
        ax22.set_ylim(-1.1,1.1)

        ax11.set_xticks([])
        ax12.set_xticks([])
        ax21.set_xticks([])
        ax22.set_xticks([])

        ax11.set_yticks([])
        ax12.set_yticks([])
        ax21.set_yticks([])
        ax22.set_yticks([])


        thvals = np.linspace(0,2*pi,100)
        ax11.plot(cos(thvals),sin(thvals),color='black',lw=2)
        ax21.plot(cos(thvals),sin(thvals),color='black',lw=2)
        ax12.plot(cos(thvals),sin(thvals),color='black',lw=2)
        ax22.plot(cos(thvals),sin(thvals),color='black',lw=2)


        # oscillators x
        ax11.scatter(cos(2*pi*num.phasex[k,0]/num.per),
                     sin(2*pi*num.phasex[k,0]/num.per),
                     color='red',s=50)
        ax21.scatter(cos(2*pi*num.phasex[k,1]/num.per),
                     sin(2*pi*num.phasex[k,1]/num.per),
                     color='red',s=50)

        # oscillators y
        ax12.scatter(cos(2*pi*num.phasey[k,0]/num.per),
                     sin(2*pi*num.phasey[k,0]/num.per),
                     color='red',s=50)
        ax22.scatter(cos(2*pi*num.phasey[k,1]/num.per),
                     sin(2*pi*num.phasey[k,1]/num.per),
                     color='red',s=50)


        # phase diff
        ax31.plot(num.t[:k],diff1[:k],
                  ls='-',color='black',lw=3,
                  label=r'$x_2-x_1$')
        ax31.plot(num.t[:k],diff2[:k],
                  ls='--',color='black',lw=3,
                  label=r'$y_2-x_1$')
        ax31.plot(num.t[:k],diff3[:k],
                  ls='-.',color='black',lw=3,
                  label=r'$y_1-x_1$')

        # antiphase lines
        ax31.plot([0,num.t[-1]],[num.per/2.,num.per/2.],color='gray')
        ax31.plot([0,num.t[-1]],[-num.per/2.,-num.per/2.],color='gray')

        # lims
        ax31.set_xlim(0,num.t[-1])
        ax31.set_ylim(-num.per/2.-.01,num.per/2.+.01)

        #box = ax31.get_position()
        #ax31.set_position([box.x0, box.y0, box.width*0.7, box.height])
        ax31.legend()



        j = start_iter+i
        fig.savefig(file_prefix+str(j)+file_suffix,dpi=80)

        plt.cla()
        plt.close()

        stdout.write("\r Simulation Recapping... %d%%" % int((100.*(k+1)/len(num.t))))
        stdout.flush()

    print




def main():

    a1=1.;b1=1.;c1=.5
    a2=a1;b2=b1;c2=c1
    
    total = 100
    
    num = tsm.Theta(use_init_option='manual',N=2,
                    a1=a1,b1=b1,c1=c1,
                    a2=a2,b2=b2,c2=c2,
                    T=total,eps=.01,dt=.0025,
                    xin=np.array([-2,0]),yin=np.array([1,2])
                )


    skip = 10
    plot_movie(num,skip,scene_number=1,starttime=1)

if __name__ == "__main__":
    main()
