"""
functions used by both thetaslowmod_phase and thetaslowmod_full
"""

import numpy as np
import scipy as sp
import matplotlib.pylab as mp
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection


from scipy.interpolate import interp1d

cos = np.cos
sin = np.sin
pi = np.pi
sqrt = np.sqrt

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def collect(x,y,use_nonan=True,lwstart=1.,lwend=5.,zorder=1.,cmapmax=1.,cmapmin=0.,cmap='copper'):
    """
    add desired line properties
    """
    x = np.real(x)
    y = np.real(y)
    
    x_nonan = x[(~np.isnan(x))*(~np.isnan(y))]
    y_nonan = y[(~np.isnan(x))*(~np.isnan(y))]
    
    if use_nonan:
        points = np.array([x_nonan, y_nonan]).T.reshape(-1, 1, 2)
    else:
        points = np.array([x, y]).T.reshape(-1, 1, 2)


    lwidths = np.linspace(lwstart,lwend,len(x_nonan))

    cmap = plt.get_cmap(cmap)
    #my_cmap = truncate_colormap(cmap,gshift/ga[-1],cmapmax)
    my_cmap = truncate_colormap(cmap,cmapmin,cmapmax)

    
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=lwidths,cmap=my_cmap, norm=plt.Normalize(0.0, 1.0),zorder=zorder)
    
    #points = np.array([x, y]).T.reshape(-1, 1, 2)
    #segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    #lc = LineCollection(segments, cmap=plt.get_cmap('copper'),
    #                    linewidths=1+np.linspace(0,1,len(x)-1)
    #                    #norm=plt.Normalize(0, 1)
    #)
    
    lc.set_array(np.sqrt(x**2+y**2))
    #lc.set_array(y)
    
    return lc


def collect3d(v1a,ga,v2a,use_nonan=True):
    """
    set desired line properties
    """
    
    v1a = np.real(v1a)
    ga = np.real(ga)
    v2a = np.real(v2a)
    
    # remove nans for linewidth stuff later.
    ga_nonan = ga[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v1a_nonan = v1a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v2a_nonan = v2a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    
    
    if use_nonan:
        sol = np.zeros((len(ga_nonan),3))
        sol[:,0] = v1a_nonan
        sol[:,1] = ga_nonan
        sol[:,2] = v2a_nonan
    else:
        sol = np.zeros((len(ga),3))
        sol[:,0] = v1a
        sol[:,1] = ga
        sol[:,2] = v2a
        
    
    sol = np.transpose(sol)
    
    points = np.array([sol[0,:],sol[1,:],sol[2,:]]).T.reshape(-1,1,3)
    segs = np.concatenate([points[:-1],points[1:]],axis = 1)
    line3d = Line3DCollection(segs,linewidths=(1.+(v1a_nonan)/(.001+np.amax(v1a_nonan))*6.),colors='k')
    
    return line3d



def collect3d_colorgrad(v1a,ga,v2a,use_nonan=True,
                        lwstart=1.,
                        lwend=5.,
                        zorder=1.,
                        cmapmin=0.,
                        cmapmax=1.,
                        paraxis='z',
                        cmap='copper',
                        return3d=True):
    """
    set desired line properties. with color gradient. and width denotes g value
    
    paraxis: choose where the line width starts, i.e. the axis where the parameter is plotted. 'z' = let z=0 be the starting point. 'x': let x=0 be the starting point.
    """

    v1a = np.real(v1a)
    ga = np.real(ga)
    v2a = np.real(v2a)
    
    # remove nans for linewidth stuff later.
    ga_nonan = ga[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v1a_nonan = v1a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v2a_nonan = v2a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    
    lwidths = np.linspace(lwstart,lwend,len(ga_nonan))

    assert(len(lwidths) > 0)

    cmap = plt.get_cmap(cmap)
    #my_cmap = truncate_colormap(cmap,gshift/ga[-1],cmapmax)
    my_cmap = truncate_colormap(cmap,cmapmin,cmapmax)

    if use_nonan:
        sol = np.zeros((len(ga_nonan),3))
        sol[:,0] = v1a_nonan
        sol[:,1] = ga_nonan
        sol[:,2] = v2a_nonan
    else:
        sol = np.zeros((len(ga),3))
        sol[:,0] = v1a
        sol[:,1] = ga
        sol[:,2] = v2a

    
    # shift width and colormap
    #lwidths = (1.+(ga_nonan-gshift)/(.001+np.amax(ga_nonan-gshift))*lwfactor)

    if return3d:

        sol = np.transpose(sol)

        points = np.array([sol[0,:],sol[1,:],sol[2,:]]).T.reshape(-1,1,3)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        line3d = Line3DCollection(segs,linewidths=lwidths,
                                  cmap=my_cmap,zorder=zorder)
        if paraxis == 'x':
            line3d.set_array(v1a_nonan)
        elif paraxis == 'y':
            line3d.set_array(ga_nonan)
        elif paraxis == 'z':
            line3d.set_array(v2a_nonan)
        else:
            raise ValueError('Invalid paraxis choice '+str(paraxis))

        return line3d

    else:
        if use_nonan:
            points = np.array([sol[:,0], sol[:,2]]).T.reshape(-1, 1, 2)
        else:
            points = np.array([sol[:,0], sol[:,2]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, linewidths=lwidths,
                            cmap=my_cmap,
                            zorder=zorder)
        lc.set_array(ga)
        return lc



def collect_disjoint_branches(diagram,all_sv=True,return_eval=False,sv_tol=.1,remove_isolated=True,isolated_number=2,remove_redundant=True,redundant_threshold=.01,N=20,fix_reverse=True,zero_column_exist=True):
    """
    collect all disjoint branches into disjoint arrays in a dict.



    diagram.dat: all_info.dat from xppauto version 8. currently not compatible with info.dat. also need multiple branches. will not collect single branch with multiple types.
    recall org for xpp version 8:
    type, branch, 0, par1, par2, period, uhigh[1..n], ulow[1..n], evr[1] evm[1] ... evr[n] evm[n]
    yes there is a zero there as of xpp version 8. I don't know why.

    for more information on how diagram is organized, see tree.pdf in the xpp source home directory.

    all_sv: True or False. in each branch, return all state variables (to be implemented)
    return_eval: return eigenvalues (to be implemented)
    sv_tol: difference in consecutive state variables. If above this value, break branch.
    remove_isolated: True/False. If a branch has fewer than isolated_number of points, do not include.
    isolated_number: see previous line

    remove_redundant: if branches overlap, remove. we require the max diff to be above redundant_threshold
    by default, we keep branches with a longer arc length.
    N: number of points to check for redundancy 

    fix_reverse: True/False. some branches are computed backwards as a function of the parameter. If so, reverse.


    
    """


    

    # get number of state variables (both hi and lo values, hence the 2*)
    varnum = 2*len(diagram[0,6:])/4

    # numer of preceding entries (tbparper stands for xpp type xpp branch parameter period)
    # diagram[:,6] is the first state variable over all parameter values
    # diagram[:,:6] are all the xpp types, xpp branches, parameters, periods for all parameter values
    tbparper = 6

    # column index of xpp branch type
    typeidx = 0
    
    # column index of xpp branch number
    bridx = 1
    

    if zero_column_exist:
        # column index of 0 guy
        zeroidx = 2
        
        # column index of bifurcation parameters
        
        par1idx = 3
        par2idx = 4
    else:
        zeroidx = 2
        par1idx = 2
        par2idx = 3

    # set up array values for retreival
    c1 = []
    c2 = []

    a = np.array([1])
    
    
    c1.append(typeidx)
    c1.append(bridx)
    
    c2.append(par1idx)
    c2.append(par2idx)
    
    for i in range(varnum):
        c2.append(tbparper+i)
    
    c1 = np.array(c1,dtype=int)
    c2 = np.array(c2,dtype=int)
    
    # store various branches to dictionary
    # this dict is for actual plotting values
    val_dict = {}

    # this dict is for type and xpp branch values
    type_dict = {}
    
    # loop over each coordinate. begin new branch if type, branch change values
    # or if parval, period, sv1, sv2, .. svn change discontinuously.
    # first set of comparisons is called c1
    # second set of comparisons is called c2
    
    brnum = 0
    

    val_dict['br'+str(brnum)] = np.zeros((1,2+varnum)) # branches are named in order they are created
    type_dict['br'+str(brnum)] = np.zeros((1,2))


    # initialize
    c1v_prev = np.array([list(diagram[0,c1])])
    c1v = np.array([list(diagram[1,c1])])
    
    c2v_prev = np.array([list(diagram[0,c2])])
    c2v = np.array([list(diagram[1,c2])])


    # val_dict has entries [par1, par2, sv1hi, sv1lo, ..., svnhi, svnlo]
    # type_dict has entries [type, br]
    # for a given xpp branch, consecutive terms are appended as a new row
    val_dict['br'+str(brnum)] = c2v_prev
    type_dict['br'+str(brnum)] = c1v_prev


    for i in range(2,len(diagram[:,0])):
        
        # get values for type and branch
        c1v_prev = np.array([list(diagram[i-1,c1])])
        c1v = np.array([list(diagram[i,c1])])

        # get values for svs and parameters
        c2v_prev = np.array([list(diagram[i-1,c2])])
        c2v = np.array([list(diagram[i,c2])])

        # append above values to current branch
        val_dict['br'+str(brnum)] = np.append(val_dict['br'+str(brnum)],c2v_prev,axis=0)
        type_dict['br'+str(brnum)] = np.append(type_dict['br'+str(brnum)],c1v_prev,axis=0)

        #print type_dict['br'+str(brnum)]

        # if either above threshold, start new key.
        if np.any( np.abs((c1v - c1v_prev))>=1):
            brnum += 1
            
            val_dict['br'+str(brnum)] = c2v
            type_dict['br'+str(brnum)] = c1v
            
        elif np.any( np.abs((c2v - c2v_prev))>=sv_tol):
            brnum += 1
            val_dict['br'+str(brnum)] = c2v
            type_dict['br'+str(brnum)] = c1v


    print val_dict.keys()

    # remove isolated points
    if remove_isolated:
        keyvals = val_dict.keys()
        
        for i in range(len(keyvals)):
            if len(val_dict[keyvals[i]]) <= isolated_number:
                val_dict.pop(keyvals[i])
                type_dict.pop(keyvals[i])


    
    # remove redundant branches
    # a python branch is removed if it shares multiple points (N) with another xpp branch.
    if remove_redundant:


        val_dict_final = {}
        type_dict_final = {}


        # get all xpp branch numbers
        brlist = np.unique(diagram[:,1])

        # collect all branches for each xpp branch number

        keyvals = val_dict.keys()


        keyignorelist = []
        keysavelist = []

        # loop over keys of python branches
        for i in range(len(keyvals)):

            key = keyvals[i]

            if not(key in keyignorelist):

                # get xpp branch number
                xppbrnum = type_dict[key][0,1]

                # loop over remaining python branches
                for j in range(i,len(keyvals)):
                    key2 = keyvals[j]

                    # make sure branches to be compared are distict
                    if not(key2 in keyignorelist) and (key2 != key):

                        #print xppbrnum,key2,type_dict[key2][0,1]

                        # if only 1 xpp branch...
                        

                        # if more than 1 xpp branch
                        # make sure xpp branches are different

                        #if xppbrnum != type_dict[key2][0,1]:
                        if True:



                            # loop over N different values
                            #N = 20
                            belowthresholdcount = 0
                            
                            dN = int(1.*len(val_dict[key][:,0])/N)
                            for i in range(N):
                                # check if N points in val_dict[key] are in val_dict[key2]

                                # first point
                                par1diff = np.amin(np.abs(val_dict[key][dN*i,0]-val_dict[key2][:,0]))
                                par2diff = np.amin(np.abs(val_dict[key][dN*i,1]-val_dict[key2][:,1]))
                                sv1diff = np.amin(np.abs(val_dict[key][dN*i,2]-val_dict[key2][:,2]))
                                sv2diff = np.amin(np.abs(val_dict[key][dN*i,3]-val_dict[key2][:,3]))

                                diff1 = par1diff + par2diff + sv1diff + sv2diff


                                if key == 'br12' and key2 == 'br51':
                                    print par1diff,par2diff,sv1diff,sv2diff
                                    print key,key2,diff1

                                #if (par1diff <= redundant_threshold) or\
                                #   (par2diff <= redundant_threshold) or\
                                #   (sv1diff <= redundant_threshold) or\
                                #   (sv2diff <= redundant_threshold):
                                if diff1 <= redundant_threshold:
                                    #print diff1,key,key2,belowthresholdcount,keyignorelist
                                    #print 'delete', key2
                                    belowthresholdcount += 1
                                    
                            if belowthresholdcount >= 3:

                                keyignorelist.append(key2)
                                #print 'del', key2
                            else:
                                
                                if not(key2 in keysavelist):
                                    #print 'keep', key2
                                    val_dict_final[key2] = val_dict[key2]
                                    type_dict_final[key2] = type_dict[key2]
                                    keysavelist.append(key2)

        for key in keyignorelist:
            if key in keysavelist:

                val_dict_final.pop(key)
                type_dict_final.pop(key)



    else:
        val_dict_final = val_dict
        type_dict_final = type_dict




    if fix_reverse and remove_isolated:
        for key in val_dict_final.keys():
            if val_dict_final[key][2,0] - val_dict_final[key][1,0] < 0:
                for i in range(varnum):
                    val_dict_final[key][:,i] = val_dict_final[key][:,i][::-1]



    return val_dict_final, type_dict_final



def phase2sv(phase,freq):
    """
    convert phase variable(s) to state variable(s)
    """
    pass

def sv2phase(sv,freq):
    """
    convert state variable(s) to phase variable(s).
    """
    return np.arctan(np.tan(sv/2.)/freq)/(freq*pi)


    
def linear_interp(xdata,ydata,x,return_data=False):
    """
    linearly interpolate the y value of an array pair xdata, ydata, given x coord
    xdata is a monotonically increasing array.
    """
    
    # find idx of intersection

    # if x is close to the edge of data, concatenate on that side
    minidx = np.argmin(np.abs(x-xdata))
    diff = x-xdata[minidx]
    diffsgn = np.sign(diff)

    # fraction between next and last idx
    ratio = diff/(diffsgn*(xdata[minidx+diffsgn]-xdata[minidx]))

    # linear interpolation
    #xapprox = xvalBefore+ratio*(xvalAfter-xvalBefore)
    yapprox = ydata[minidx]*(1-ratio)+ratio*(ydata[minidx+diffsgn])
    

    if return_data:
        return x,yapprox,(xdata,ydata,x)
    else:
        return yapprox

def average(y,t,a1,b1,c1,a2,b2,c2,mux,muy,eps):
    """
    rhs of mean field model 
    """
    sx = y[0]
    sy = y[1]
    rhs1 = eps*(-sx + sqrt(a1+b1*sx-c1*sy))/mux
    rhs2 = eps*(-sy + sqrt(a2+b2*sx-c2*sy))/muy
    
    return (rhs1,rhs2)


def average_no_eps(y,t,a1,b1,c1,a2,b2,c2,mux,muy):
    """
    rhs of mean field model  without epsilon term
    """
    sx = y[0]
    sy = y[1]
    rhs1 = (-sx + sqrt(a1+b1*sx-c1*sy))/mux
    rhs2 = (-sy + sqrt(a2+b2*sx-c2*sy))/muy
    
    return (rhs1,rhs2)

def average_jac(sbarx,sbary,a1,b1,c1,a2,b2,c2,mux,muy):
    """
    Jacobian of averaged system at the fixed point sbar
    """
    # WLOG use a1,b1,c1, since a2,b2,c2 are chosen to have the same value
    
    sx = sbarx
    sy = sbary
    
    w1 = 0.5/np.sqrt(a1 + b1*sx - c1*sy)
    w2 = 0.5/np.sqrt(a2 + b2*sx - c2*sy)
    
    return np.array([[(-1+b1*w1)/mux,-c1*w1/mux],
                     [b2*w2/muy,(-1-c2*w2)/muy]])


def slow_osc(sbarx,sbary,a1,b1,c1,a2,b2,c2,mux,muy):
    """
    return true or false
    mux,muy: parameters of slow/averaged system
    sbar: fixed point (assume (sbar,sbar))
    """
    w,v = np.linalg.eig(average_jac(sbarx,sbary,
                                    a1,b1,c1,
                                    a2,b2,c2,mux,muy))
    
    # get real parts
    rew1 = np.real(w[0])
    rew2 = np.real(w[1])

    if (rew1*rew2 > 0) and (rew1+rew2 > 0):
        return True
    else:
        return False

def clean(x,y,smallscale=False,tol=.5):

    pos = np.where(np.abs(np.diff(y)) >= tol)[0]
    
    x[pos] = np.nan
    y[pos] = np.nan
    return x,y



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


def get_averaged(data,dt,kernelSize=30,padN=10,time_pre=-15,time_post=15):
    """
    get moving average value of data.

    smooth first using valid to remove edge effects from zero padding.
    use the first smoothed data to linearly extrapolate data points beyond the valid range.
    i.e. we pad the data with linear extrapolation instead of zero padding.
    re-run the convolution with the linearly padded data.

    Edges are padded using a linear interpolation of size padN
    time_pre,time_post: time values before and after data over which to extend
    kernelSize: moving average window size
    """
        
    #print 'interpolating padded values for convolution...'
    # save phase values
    # phase from 0 to T
    
    if kernelSize > padN:
        print 'Warning: kernelSize > padN, you will get weird edge effects'

    # generate moving window
    #x = np.linspace(-10,10,kernelSize)
    #y = np.exp(-x**2.)
    #data_kernel = y/(np.sum(y))
    data_kernel = np.ones(kernelSize)/kernelSize
    
    # linearly interpolate the preceeding and proceeding padN values
    data_smooth_v1 = sp.signal.fftconvolve(data,data_kernel,'valid')
    
    # slope at start and end of data
    slope_0 = (data_smooth_v1[padN] - data_smooth_v1[0])/(dt*padN)
    slope_end = (data_smooth_v1[-1] - data_smooth_v1[-padN])/(dt*padN)

    # data values before and after data, using linear extrapolation
    prev_fx = np.linspace(time_pre,0,padN)*slope_0 + data_smooth_v1[0]
    post_fx = np.linspace(0,time_post,padN)*slope_end + data_smooth_v1[-1]

    # use first values of data to the extra left padN number of values
    # use last values of data to the extra right padN number of values
    # repeat for sy
    data_padded = np.zeros(len(data)+2*padN)
    
    data_padded[:padN] = prev_fx
    data_padded[padN:padN+len(data)] = data
    data_padded[padN+len(data):] = post_fx
    
    #print 'convolving...'        
    # convolve to same size and cut off ends. the previous lines should take care of most edge effects
    data_smooth = sp.signal.fftconvolve(data_padded,data_kernel,'same')[padN:padN+len(data)]
    
    return data_smooth


def get_freq(a,b,c,sx,sy):
    """
    get mean frequency on fast timescale
    """
    d = a+b*sx-c*sy
    if d < 0:
        d = 0
    return np.sqrt(d)

def get_period(a,b,c,sx,sy):
    """
    get mean period on fast timescale
    """
    f = get_freq(a,b,c,sx,sy)
    if f == 0:
        return np.inf
    else:
        return 1./f


def x0(t,f,d=None):
    """
    analytic limit cycle (found using mathematica)
    """

    f2 = f**2.
    per = 1./f
    
    tt = t + per/2.
    #tt2 = t*2*pi*f+pi
    #tt3 = t*2*pi*f


    if d == 't':
        return (2.*f2*pi)/(cos(f*pi*tt)**2. + f2*sin(f*pi*tt)**2.)
    elif d == None:
        return 2.*np.arctan(f*np.tan(f*pi*tt))
    else:
        raise ValueError("invalid derivative choice for def lc, "+str(d))

    #return 2*np.arctan(f*np.tan((t+1./(2.*f))*f*pi))


def z(t,f):
    """
    analytic iPRC (found using mathematica, simplified by hand)
    t: time
    a,b,c: coupling parameters
    f: coupling term. If f=None, use mean frequency.
    generally, the mean frequency will not be self.sbar when the parameters
    are slowly varying.
    """

    f2 = f**2
    
    per = 1./f
    tt = t + per/2.
    return (cos(tt*f*pi)**2 + f2*sin(tt*f*pi)**2)/(2*pi*f2)


def get_sbar(a1,b1,c1,a2,b2,c2,verbose=False):
    """
    get fixed point of mean field.

    a1,b1,c1: constants for exitatory system
    a2,b2,c2: constants for inhibitory system
    
    """
    # two roots
    #a1 = (-(cc-bb) + sqrt((cc-bb)**2 + 4*aa*pi**2))/(2.*pi**2)
    #a2 = (-(cc-bb) - sqrt((cc-bb)**2 + 4*aa*pi**2))/(2.*pi**2)
    
    
    sbarx = (-(c1-b1) + sqrt((c1-b1)**2. + 4.*a1))/2.
    #r2 = (-(c1-b1) - sqrt((c1-b1)**2. + 4.*a1))/2.
        
    sbary = (-(c2-b2) + sqrt((c2-b2)**2. + 4.*a2))/2.
    
    #print "sbarx - sbary =",sbarx-sbary

    if verbose:
        print 'verbose enabled for get_sbar'
        print 'sbarx =', sbarx
        print 'sbary =', sbary
        print '|sbary - sbarx| =',np.abs(sbary-sbarx)

    if sbarx <= 0 or sbary <= 0:
        raise ValueError('no strictly positive fixed point found')

    else:
        return sbarx,sbary



def dat2tab(dat,savename='out.tab'):
    """
    dat: data file. x-coordinates in first column, y-coordinates in second column
    
    """

    # assert that there are no more or less than 2 columns
    # maybe work around this later, but good enough for now
    assert(np.shape(dat)[-1]==2)
    
    # construct interpolated function
    f = interp1d(dat[:,0],dat[:,1])

    # construct uniform domain data
    dom = np.linspace(dat[0,0],dat[-1,0],len(dat[:,0]))

    # get function values at uniform domain points
    fval = f(dom)

    # generate the TAB file

    # convert all to string
    fvalstr = str(int(len(dom))) + '\n'\
              + str(dom[0]) + '\n'\
              + str(dom[-1]) + '\n'
    for i in range(len(fval)):
        fvalstr += str(fval[i])+'\n'

    # strip final \n
    fvalstr = fvalstr[:-2]

    """
    # append to beinning in the following order: xhi,xlo,npts
    fval=np.insert(fval,0,dom[-1]);fval=np.insert(fval,0,dom[0]);fval=np.insert(fval,0,len(dom))
    """

    # save TAB file
    f = open(savename,'w')
    f.write(fvalstr)
    f.close()
    #np.savetxt(savename,fval)

    
