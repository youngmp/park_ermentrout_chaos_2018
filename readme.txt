(depricated) th_inits: directory for trb_init.dat, wb_init.dat, and s_init.dat which are initial/final simulation solution values and are disposable.

savedir: directory for initial/final simulation solution values for thetaslowmod.py, h functions, h function coefficients, slow limit cycles. basically anything that takes a while to compute is saved here for later recall. it is okay to clear this file periodically.

savedir_trbwb: directory for initial/final simulation solution values for trbwb2.py, h functions, h function coefficients, slow limit cycles. basically anything that takes a while to compute is saved here for later recall. it is okay to clear this file periodically.

(depricated) sx_sy.dat: sx and sy values over one period of the small oscillation limit cycle. The time values are divided by epsilon so the actual period of the oscillation should be multiplied by eps. To compute this lookup table, I used the parameters

vt.tab, mt.tab, nt.tab, ht.tab, wt.tab, st.tab, ca.tab: XPP tab files for a limit cycle in the traub model. These are used as lookup tables for phase approximation and for phase-to-state-variable conversion. filenames correspond to name of state variables in trbwb2.ode. Use trbwb2.ode.decoupled.set to reproduce data.

v.tab, h.tab n.tab: XPP tab files. These are lookup tables for wb neurons. filenames correspond to name of state variables in trbwb2.ode. trbwb2.ode.decoupled.set

*_phs_lookup.tab: tab files for on-the-fly phase estimation. These guys have approximately 300 elements, as opposed to the 1000 in the other tab files, making these tables ideal for fast lookup.

p tau=5.5
p a1=.5,b1=7,c1=6.5
p a2=1.1,b2=25,c2=25.1

(depricated) sx.tab, sy.tab, tab file versions of sx_sy.dat above. I saved these in case I need to run them using XPP.

(depricated) fourier_approx.py: fourier stuff to approximate the sx,sy functions of time.

tbfi.dat,wbfi.dat: traub, wb frequency current curves produced using xpp/auto.

fi.py: plot trbfi.dat,wbfi.dat

trbwb_phase.py,trbwb_full.py,trbwb_master.py: simulation of full wb-trab coupled system with the phase model.

wb.ode: wb model in XPP

traub_ca.ode: traub model with calcium in XPP

thetaslowmod_phase.py,thetaslowmod_full.py,thetaslowmod_master.py: simulation of full exc/inh theta networks with the phase model.

choose_g.nb: choose conductance values so that the frequencies are the same between wb and traub.

lc.nb: compute limit cycle and adjoint of theta model

hopf.nb: defunct: compute coefficients to determine criticality of hopf bifurcation in the averaged theta network.

sbar.nb: choose parameters so that we get a supercritical hopf in the theta network + determine coefficients to fix the frequency/period/fixed point at 1 WLOG.

averaged.ode: mean field version of theta network.

swbss.dat, strss.dat: steady-state time series of slow variables for traub_ca and wang-buszaki.
p mux=1.,muy=1.
p eps=.01
p it=5.95
p i0=.791454
p gee=0,gei=0,gie=0,gii=0
Use these data files to verify that \bar s = 1/T.

swbss2.dat, strss2.dat: steady-state time series of slow variables for traub_ca and wang-buszaki.
p mux=1.,muy=1.
p eps=.01
p it=5.905,i0=.8
p gee=1,gei=2.019
p gie=1,gii=1.17417
Use these data files to verify that \bar s = 1/T.

averaged_tbwb.ode: to generate bifurcation diagram try: Ntst=100, Nmax=5000, Ds=.0005, Dsmin=1e-5, Dsmax=.0005, par min=0, par max=50


trb_adj.dat: traub adjoint using traub_ca.ode with traub_ca.base.set, and i=5.854996145726557356995. period approx 20.81

trb_adj2.dat: traub adjoint using traub_ca.ode with traub_ca.base.set, and i=5.95. period approx 20.38. Since this adjoint is closer in period to wb_adj.dat, it may be beneficial to use this data instead. Remember that once coupled, the period between the two trb and two wb become the same for the mean i value used to find trb_adj.dat.

wb_adj.dat: wb adjoint using wb.ode with wb.base2.set, i0=0.79145321756741363578785 period approx 20.38

wb_adj3.dat: wb adjoint using trbwb2.ode with trbwb2.ode.decoupled.set. zero phase is chosen to be when the AP crosses 0mV. This is the correct choice. period 20.38
trb_adj3.dat: trb adjoint using trbwb2.ode with trbwb2.ode.decoupled.set. zero phase chosen as in wb_adj3.dat. period 20.39


v_wb.dat, vp_wb.dat, vt.dat, vtrp.dat: voltage coordinate data files of coupled 2 traub and 2 WB neurons. Parameters as in wb.base2.set, traub_ca.base.set. mux=muy=1. eps=.01. These voltage traces are used to determine spike times, which are in turn used to determine the approximate phase value of the full numerics. This phase value is then compared to the theory. The data files are for 5000 time units at steady-state (which took about 15000 time units to reach).

v_wb2.dat, vp_wb2.dat, vt2.dat, vtrp2.dat: voltage coordinate data files of coupled 2 traub and 2 WB neurons. Parameters as in wb.base2.set, traub_ca.base.set. mux=muy=1. eps=.0025. The data files are for 5000 time units at steady-state (which took about 15000 time units to reach).

pxyaa0_bifurcation*.dat: bifurcation diagram (allinfo) for two degrees of freedom (in phi_x,phi_y and phi_z=0). *bifurcation2.dat shows the degeneracy at tau=1.1 and fixed points at the corners, but it is messy. *bifurcation.dat is the cleaner version excluding the degeneracy and the fixed points at the corners.

p[xyz]+[a0]+.dat: allinfo bifurcation diagram for the various degrees of freedom from xppauto. pxa00 denotes one degree of freedom in x, with the other two variables set to 0. pxza0a denotes two degrees of freedom (in phix,phiz) and phiy=0.
