#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
from scipy.optimize import fsolve, bisect, minimize_scalar
from time import time
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpl_patches
import pickle
plt.style.use('seaborn-whitegrid')
# plt.style.use('default')

#%%
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['legend.fontsize'] = 18
#mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.minor.size'] = 3
# %%

# %%



# %%
def shock_jump(lamsh, V1, D1, M1):
    Vb1 = V1 - de*lamsh
    fac = (gam-1)/(gam+1)
    D2 = D1/fac
    V2 = Vb1*fac + de*lamsh
    P2 = D1 * Vb1**2 * (2/(gam+1))
    M2 = M1
    return np.array([V2,D2,M2,P2])

def preshock(tht):
    eps = 1/s
    Mta = (3*np.pi/4)**2
    # tht = np.linspace(0.01, 2*np.pi,300)
    Mn = ((tht-np.sin(tht))/np.pi)**(-2/3*s)
    M = Mta * Mn
    lam = (1-np.cos(tht))/2 * Mn**(1/3+eps)
    # D = ( 3*np.pi/8 * (1-np.cos(tht)) * Mn**(3*eps/2-1) * np.sin(tht) * Mta**-2 )**-1
    D = 9*np.pi**2/ ( (2+6*eps) *(1-np.cos(tht))**3 - 9*eps*np.sin(tht) *(tht-np.sin(tht)) *(1-np.cos(tht)) ) * Mn**(-3*eps)
    # D_num = np.gradient(M,lam)/lam**2
    V = np.pi/2 * np.sin(tht)/(1-np.cos(tht)) * Mn**(1/3-eps/2)
    return lam,V,D,M


#%%
def odefunc_prof_init_Pless(lam, depvars):
    # lam = 1/laminv
    # lam = np.exp(l)
    V,D,M = depvars
    Vb = V - de*lam
    # linb = -np.array([2*D*V-2*D*lam, (de-1)*V*lam+2/9*M/lam, -3*lam**3*D])/lam
    # # der_num = np.transpose(np.linalg.solve(linMat1,linb1), (1,0))
    # linMat_det1 = Vb**2
    # # if linMat_det1 == 0: print(depvars)
    # linMat_cofac1 = np.array([[0,Vb,0],[Vb,-D,0],[0,0,linMat_det1]])
    # linMat_inv = linMat_cofac1/ linMat_det1
    # der = np.matmul(linMat_inv,linb)
    Fterm = (de-1)*V + 2/9*M/lam**2 # - 10*(-V/(lam/lamdi)**10)/lam
    der = np.array([-Fterm/Vb, (2*Vb*(lam-V)/lam + Fterm) *D/Vb**2, 3*lam**2*D])
    return der #*lam**2

# %%
def stop_event(t,y):
    return y[0]+10 #+de*np.exp(t)
stop_event.terminal = True

zero_hold_func = lambda x: 1+np.heaviside(x-10,0.5)-np.heaviside(x,0.5)

def odefunc_full(l, depvars):
    lam = np.exp(l)
    # lmV,lDt,lMt,lPt = depvars
    mVb,D,M,P = np.exp(depvars)
    Vb = -mVb
    V = Vb + de*lam
    # V,D,M,P = from_btilde(lam, mVb,Dt,Mt,Pt)
    Z0 = 0*V
    ar1 = V/V

    Tv = P/D/Vb**2
    linMat_inv = 1/Vb**2/(gam*Tv-1) * np.array([[-gam*Tv, ar1, -Tv],[ar1,-ar1,Tv],[gam*ar1,-gam*ar1,ar1]])
    linb = np.array([2*Vb* (V-lam), (de-1)*V*lam+2/9*M/lam+10*(-V/(lam/lamdi)**10), Vb*lam*((2-Lam0*D**(2-nu)*P**(nu-1))*(gam-1)+2*(de-1))])

    # if not np.isfinite(V/lam).all():
    #     print(V, lam)
    # print(linMat_inv.shape,linb[:,np.newaxis].transpose((2,0,1)).shape)
    linc = np.array([de/Vb*lam,Z0,Z0])
    if np.isscalar(V):
        der = np.matmul(linMat_inv, linb ) 
        # der[0] *= zero_hold_func(V)
        der -= linc
        # der = np.matmul(linMat_inv, linb )+ np.array([-de/Vb*lam,Z0,Z0])
        # if der[0]<0:
            # print(der, linMat_det1, linb, linMat_cofac1)
    else:
        der = np.matmul(linMat_inv.transpose((2,0,1)), linb[:,np.newaxis].transpose((2,0,1)) )
        # der[:,0] *= zero_hold_func(V)[:,np.newaxis]
        der -= linc[:,np.newaxis].transpose((2,0,1))
        der = der.transpose((1,2,0))[:,0,:]

    derM = 3*D*lam**3 /M


    return der, derM, linMat_inv, linb #*lam**2

def odefunc(l, depvars):
    der3, derM = odefunc_full(l, depvars)[:2]
    der = np.insert(der3, 2, derM, axis=0)
    if not np.isfinite(der).all():
        # print(der,l,depvars)
        return np.nan_to_num(der)
    return der


#%%

def get_soln_gas_full(lamsh):
    res_pre = solve_ivp(odefunc_prof_init_Pless, (1,lamsh), preshock(np.pi)[1:], max_step=0.01 )
    V1, D1, M1 = res_pre.y[0][-1], res_pre.y[1][-1], res_pre.y[2][-1]
    bcs = shock_jump(lamsh, V1, D1, M1)
    bcs[0] = - bcs[0] + de*lamsh
    # print(bcs)
    bcs = np.log(bcs)
    # bcs[3] = 0
    res_post = solve_ivp(odefunc, (np.log(lamsh),np.log(1e-7)), bcs, method='Radau', max_step=0.05, vectorized=True, events=stop_event)
    return res_pre, res_post

def M0_num(lamsh):
    res = get_soln_gas_full(lamsh)[1]
    M0val = res.y[2][-1]
    stopM0 = np.exp(M0val)-1e-3
    return stopM0 

def lam_atM0(lamsh):
    res = get_soln_gas_full(lamsh)[1]
    return res.t[-1]/np.log(10)+6.999


#%%
name = 'cold_vary-s'
# name = 'shocked_vary-s'
name = 'shocked_vary-gam'
# name = 'shocked_vary-cooling'
# name = 'shocked_vary-lamdish'
# name = 'shocked_vary-lamshsp'

with open(f'{name}-rads.txt', 'tr') as file: rads_list = eval(file.read())

s = 1
gam = 5/3
Lam0 = 3e-2
nu=1/2
fb = 0.156837
# fb = 0.5
fd = (1-fb)

lamshsp = 0.9
disk_rad_by_shock = 0.05
lamdish = disk_rad_by_shock #*lamsh

varypars=[]

if name == 'cold_vary-s':
    s_vals = [0.5,1,1.5,2,3,]
    varypars += ['s']
    lamshsp = 0.1
    lamdish = 0.5

if name == 'shocked_vary-s':
    s_vals = [0.5,1,1.5,2,3,]#[:-1]
    varypars += ['s']

if name == 'shocked_vary-gam':
    gam_vals= [2,1.8,5/3,1.5,7/5,4/3,]
    lamshsp_vals = [1.2,1.05,0.9,0.7,0.5,0.3]
    varypars += ['gam','lamshsp']

if name == 'shocked_vary-cooling':
    Lam0_vals = [1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]
    varypars += ['Lam0']

if name == 'shocked_vary-lamdish':
    lamdish_vals = [percent/100 for percent in [2,5,10,15,25]]
    varypars += ['lamdish']

if name == 'shocked_vary-lamshsp':
    lamshsp_vals = [1.1,1,.9,.8,.7,.6,.5]#[0.35,0.3,0.25, 0.2]
    varypars += ['lamshsp']


# name = '_cold_vary-s'
# s_vals = [0.5,1,1.5,2,3,5]
# varypars += ['s']
# lamsh = 0.03

# name = '_shocked_vary-s'
# s_vals = [0.5,1,1.5,2,3,5]
# varypars += ['s']

# name = '_shocked_vary-s+sh'
# s_vals = [0.5,1,1.5,2,3,5]
# lamsh_vals = [0.35,0.32,0.3,0.25,0.2,0.1]
# varypars += ['s','lamsh']

# name = '_shocked_vary-s+sh+di'
# s_vals = [0.5,1,1.5,2,]
# lamsh_vals = [0.35,0.32,0.3,0.25,]
# lamdi_vals = [0.05*lamsh for lamsh in lamsh_vals]
# varypars += ['s','lamsh','lamdi']

# name = '_shocked_vary-gam'
# gam_vals= [5/3,7/5,4/3,]
# varypars += ['gam']

# name = '_shocked_vary-gam+sh'
# gam_vals= [5/3,7/5,4/3,]
# lamsh_vals = [0.35,0.3,0.25]
# varypars += ['gam','lamsh']

# name = '_shocked_vary-cooling'
# Lam0_vals = [1e-3,3e-3,1e-2,3e-2,1e-1]
# varypars += ['Lam0']

# name = '_shocked_vary-cooling_fn'
# nu_vals = [-1/2,1/2]
# varypars += ['nu']

# name = '_shocked_vary-lamdi'
# lamdi_vals = [percent/100*lamsh for percent in [2,5,10,15,25]]
# varypars += ['lamdi']

# name = '_shocked_vary-lamsh'
# lamsh_vals = [0.35,0.3,0.25, 0.2]
# varypars += ['lamsh']

# name = '_shocked_vary-lamsh-di'
# lamsh_vals = [0.35,0.3,0.25, 0.2]
# lamdi_vals = [0.05*lamsh for lamsh in lamsh_vals]
# varypars += ['lamsh','lamdi']

# lamsh_sols = {}
# lam_atM0_sols = {}
# lambins = np.linspace(0.01, 0.5, 8)

# for s in s_vals[::]:
#     t_now = time()
#     de = 2* (1+s/3) /3
#     alpha_D = -9/(s+3)
#     aD, aP, aM = alpha_D, (2*alpha_D+2), alpha_D+3
#     aD, aP, aM = 0,0,0
#     print(s, aD, aP, aM)

#     lamshsol = my_bisect(lam_atM0, lambins[0], lambins[-1], xtol=1e-7)#+1e-5
#     # lamshsol = thetbins[idx_M0neg+1]
#     t_bef, t_now = t_now, time()
#     print(f'{t_now-t_bef:.4g}s', f's={s}: root thetsh obtained')
#     # lamshsol1 = minimize_scalar(M0, method='bounded', bounds=(1.5*np.pi, 1.9*np.pi))
#     lamsh_sols[s] = lamshsol
#     lam_atM0_sols[s] = lam_atM0(lamshsol)
#     print(f's={s}', lamshsol, lam_atM0_sols[s])


fig5, axs5 = plt.subplots(2,2, dpi=100, figsize=(12,8), sharex=True)
fig6, (ax62,ax6) = plt.subplots(1,2, dpi=100, figsize=(14,7))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(10):
    plab=''
    try:
        if 'gam' in varypars: gam = gam_vals[i]; plab+=r'$\gamma=$'+f"{gam:.3g}, "
        if 's' in varypars: s = s_vals[i]; plab+=f"s={s} "
        if 'lamshsp' in varypars: lamshsp = lamshsp_vals[i]; plab+=r'$R_s=$'+f'{lamshsp:g}'#plab+=r'$\lambda_s=$'+f'{lamshsp*100:g} '+r'$\%~ \lambda_{sp}$'
        if 'lamdish' in varypars: lamdish = lamdish_vals[i]; plab+=r'$\lambda_d=$'+f'{lamdish*100:g} '+r'$\%~ \lambda_s$'
        if 'Lam0' in varypars: Lam0 = Lam0_vals[i]; plab+=r'$\Lambda_0=$'+f'{Lam0:g} '
        # if 'nu' in varypars: nu = nu_vals[i]; plab+=r'$\nu=$'+f'{nu} '
    except IndexError: break

    t_now = time()
    de = 2* (1+s/3) /3
    alpha_D = -9/(s+3)
    descr = f'_{name}_lamshsp={lamshsp:.3g}_s={s:.2g}_gam={gam:.3g}_lamdish={lamdish:.3g}_Lam0={Lam0:.1e}_nu={nu:.1g}'

    resdf_prof_gaso_bertshi = pd.read_hdf(f'profiles_gaso_bertshi_s={s:.2g}_gam={gam:.3g}.hdf5', key=f'gas/main', mode='r')
    # resdf_prof_gaso_bertshi = pd.read_hdf(f'profiles_gasdm{descr}.hdf5', key=f'gas/iter0', mode='r')
    # lamsh = resdf_prof_gaso_bertshi.l[np.diff(resdf_prof_gaso_bertshi.Vb).argmax()]
    lamsh = lamshsp*rads_list[i][2] #rads_list[i][1]  #
    lamdi = lamdish*lamsh
    
    # lamshsol = 0.35 #lamsh_sols[s] #+5e-3 # 0.338976 #
    res_pre, res_post = get_soln_gas_full(lamsh)
    print(res_post.y[2][-1])
    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: post shock profiles obtained')

    lamsh_pre = res_pre.t
    V_pre, D_pre, M_pre = res_pre.y

    lamsh_post = np.exp(res_post.t)
    mVb_post, D_post, M_post, P_post = np.exp(res_post.y)
    V_post = de*lamsh_post - mVb_post
    P_pre = lamsh_pre*0

    lamsh = lamsh_pre.min()
    # lam_all = lamsh_pre
    # V_all = V_pre
    # D_all = D_pre
    # M_all = M_pre
    # P_all = P_pre

    lam_all = np.concatenate([lamsh_post, lamsh_pre][::-1])
    V_all = np.concatenate([V_post, V_pre][::-1])
    D_all = np.concatenate([D_post, D_pre][::-1])
    M_all = np.concatenate([M_post, M_pre][::-1])
    P_all = np.concatenate([P_post, P_pre][::-1])
    Vb_all = V_all - de*lam_all

    color_this = colors[i] #plt.cm.turbo(s/4)


    axs5[0,0].plot(lam_all,-Vb_all, color=color_this)
    axs5[0,1].plot(lam_all,D_all, color=color_this, label=plab)
    axs5[1,0].plot(lam_all,M_all, color=color_this)
    axs5[1,1].plot(lam_all,P_all, color=color_this)
    # axs5[0,2].plot(lam_all, P_all/D_all, color=color_this)
    # axs5[1,2].plot(lam_all, P_all/D_all**gam, color=color_this)
    # axs5[1,2].plot(lam_all, D_all*Vb_all**2-gam*P_all, color=color_this)

    # resdf_prof_gaso_bertshi = pd.read_hdf(f'profiles_gaso_bertshi_s={s:.2g}_gam={gam:.3g}.hdf5', key=f'gas/main', mode='r')
    # resdf_prof_gaso_bertshi = pd.read_hdf(f'profiles_gasdm_shocked_vary-s_lamshsp=0.9_s={s:.2g}_gam={gam:.3g}_lamdish=0.05_Lam0=3.0e-02_nu=0.5.hdf5', key=f'gas/main', mode='r')
    axs5[0,0].plot(resdf_prof_gaso_bertshi.l, -resdf_prof_gaso_bertshi.Vb, color=color_this, ls='--')
    axs5[0,1].plot(resdf_prof_gaso_bertshi.l, resdf_prof_gaso_bertshi.D, color=color_this, ls='--')
    axs5[1,0].plot(resdf_prof_gaso_bertshi.l, resdf_prof_gaso_bertshi.M, color=color_this, ls='--')
    axs5[1,1].plot(resdf_prof_gaso_bertshi.l, resdf_prof_gaso_bertshi.P, color=color_this, ls='--')
    print(resdf_prof_gaso_bertshi.l[np.diff(resdf_prof_gaso_bertshi.Vb).argmax()])
    # PderD_post = np.gradient(P_post,lamsh_post)/D_post

    M_intrp = interp1d(lam_all, M_all, fill_value="extrapolate")
    D_intrp = interp1d(lam_all, D_all, fill_value="extrapolate")
    V_intrp = interp1d(lam_all, V_all, fill_value="extrapolate")
    irem = P_pre.shape[0]-1
    # PderD_intrp = interp1d(np.delete(lam_all,irem), np.delete(PderD_all,irem), kind='linear', fill_value="extrapolate")

    # PderD_intrp = interp1d(lamsh_post, PderD_post, kind='linear', fill_value=0, bounds_error=False)

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: all profiles obtained')

    def odefunc_traj(xi, arg):
        lam = arg
        return V_intrp(lam)-de*lam

    # taush = (thtshsol - np.sin(thtshsol)) / np.pi
    # xish = np.log(taush)
    # res = solve_ivp(odefunc_traj, (0,5), (1,), method='Radau', max_step=0.01, dense_output=False, vectorized=True)
    # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: post shock trajectory obtained')
    
    # xires = res.t
    # lamres = res.y[0]
    # vres = res.y[1]
    xires,lamres = cumtrapz(1/(V_all-de*lam_all), x=lam_all), lam_all[1:]

    taures = np.exp(xires)
    lamFres = lamres*taures**de

    ax6.plot(taures,lamFres, color=color_this, label=plab)
    ax62.plot(xires,lamres, color=color_this)
    xio,lamo = cumtrapz(1/(resdf_prof_gaso_bertshi.V-de*resdf_prof_gaso_bertshi.l), x=resdf_prof_gaso_bertshi.l), resdf_prof_gaso_bertshi.l[1:]
    tauo = np.exp(xio)
    lamFo = lamo*tauo**de
    ax62.plot(xio,lamo, c=color_this, ls='-.')
    ax6.plot(tauo,lamFo, color=color_this, ls='-.')

    #trajectory analytical
    thet_range = np.linspace(0.5, 1.2*np.pi,2000)
    tau_anlt = (thet_range - np.sin(thet_range)) / np.pi
    xi_anlt = np.log(tau_anlt)
    lam_anlt = preshock(thet_range)[0]
    lamF_anlt = lam_anlt*tau_anlt**de

    # ax6.plot(xi_anlt, lam_anlt, color=color_this)
    ax6.plot(tau_anlt, lamF_anlt, color=color_this)
    
    # dmo_prfl = pd.read_hdf(f'profiles_dmo_{s}.hdf5')

    # Mta = (3*np.pi/4)**2
    # M_dmo = interp1d(dmo_prfl['l'], dmo_prfl['M']*Mta, fill_value="extrapolate")
    # D_dmo = interp1d(dmo_prfl['l'].iloc[1:], dmo_prfl['rho'].iloc[1:], fill_value="extrapolate")

    # axs5[1,0].plot(lam_all, M_dmo(lam_all), color=color_this, ls='dashed')

#Loop ends

axs5[0,0].plot(lam_all,de*lam_all, c='k', ls='--', label='V=0')

ax6.legend(loc='best')
ax6.set_xlabel(r'$\tau$')
ax6.set_ylabel('$\lambda_F$')
ax6.set_xlim(0,12)
ax6.set_ylim(0.0001,1.1)
# ax6.set_xscale('log')
ax6.set_yscale('log')

ax62.set_xlabel(r'$\xi$')
ax62.set_ylabel('$\lambda$')
# ax62.set_xlim(,)
# ax62.set_ylim(0.01,1.1)
ax62.set_yscale('log')
    
axs5[0,0].set_xscale('log')
axs5[0,0].set_xlim(1e-5,1)
axs5[0,0].legend()
axs5[0,1].legend()
axs5[1,0].set_xlabel('$\lambda$')
axs5[1,1].set_xlabel('$\lambda$')
# axs5[1,2].set_xlabel('$\lambda$')

if gam==5/3:
    axs5[0,0].set_xlim(7e-5,1)
    axs5[0,0].set_ylim(5e-6,1e1)
    axs5[0,1].set_ylim(1e-1,1e11)
    axs5[1,0].set_ylim(1e-3,1e1)
    # axs5[1,1].set_ylim(1e0,1e14)
    # axs5[0,2].set_ylim(1e-1,1e2)
    # axs5[1,2].set_ylim(1e-5,5e-1)
elif gam==4/3:
    axs5[0,0].set_xlim(1e-5,1)
    axs5[0,0].set_ylim(5e-6,1e1)
    axs5[0,1].set_ylim(1e0,1e11)
    axs5[1,0].set_ylim(1e-2,1e1)
    axs5[1,1].set_ylim(1e1,1e14)
    # axs5[0,2].set_ylim(1e0,1e3)
    # axs5[1,2].set_ylim(1e-2,5e-1)

axs5[0,0].set_xlim(7e-5,1)
axs5[0,0].set_ylim(5e-6,1e1)
axs5[0,1].set_ylim(1e-1,1e11)
axs5[1,0].set_ylim(1e-3,1e1)
# axs5[1,1].set_ylim(1e0,1e14)


axs5[0,0].set_ylabel(r'$-\bar{V}$')
axs5[0,1].set_ylabel('D')
axs5[1,0].set_ylabel('M')
axs5[1,1].set_ylabel('P')
# axs5[0,2].set_ylabel('T')
# axs5[1,2].set_ylabel('K')

axs5[0,0].set_yscale('log')
axs5[0,1].set_yscale('log')
axs5[1,0].set_yscale('log')
axs5[1,1].set_yscale('log')
ax62.set_xlim(0,5)
# axs5[0,2].set_yscale('log')
# axs5[1,2].set_yscale('log')

# axs5[1,0].xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10,subs=np.arange(2, 10)))
# axs5[1,0].xaxis.get_ticklocs(minor=True)
# axs5[1,0].minorticks_on()
# axs5[1,0].xaxis.set_tick_params(which='minor', bottom=True)

fig5.tight_layout()
fig6.tight_layout()

fig5.savefig(f'Eds-gaso_profiles_{name}.pdf')
fig6.savefig(f'Eds-gaso_trajectory_{name}.pdf')
# axs5[0,0].set_xlim(1e-6,1)
# axs5[1,0].set_ylim(1e-4,1e1)



#%%
# fig7,(ax71, ax72, ax73) = plt.subplots(3, dpi=120, figsize=(7,10), sharex=True)
# ax71.plot(lamsh_post, res_post.y[0])
# ax72.plot(lamsh_post, odefunc(np.log(lamsh_post),res_post.y)[0])
# ax72.plot(lamsh_post[1:], np.diff(res_post.y[0])/np.diff(np.log(lamsh_post)))

# ax73.plot(lamsh_post, odefunc_full(lamsh_post,res_post.y)[2][1,2]/odefunc_full(lamsh_post,res_post.y)[2][1,0])

# ax71.set_xscale('log')
# ax72.set_yscale('log')
# ax73.set_yscale('log')
# #%%
# plt.loglog(lamsh_post, odefunc_full(np.log(lamsh_post),res_post.y)[2][1,2]/odefunc_full(np.log(lamsh_post),res_post.y)[2][1,0])
# plt.loglog(lamsh_pre, odefunc_full(np.log(lamsh_pre),np.log(to_btilde(lamsh_pre,*shock_jump(lamsh_pre, *res_pre.y))))[2][0,1])

# #%%
# plt.loglog(lamsh_post, odefunc_full(np.log(lamsh_post),res_post.y)[2][1,2]/odefunc_full(np.log(lamsh_post),res_post.y)[2][1,0])
# plt.loglog(lamsh_post, np.exp(res_post.y[3]-res_post.y[0]*2-res_post.y[1]))
# bcs_all = to_btilde(lamsh_pre,*shock_jump(lamsh_pre, *res_pre.y))
# plt.loglog(lamsh_pre, bcs_all[3]/bcs_all[0]**2/bcs_all[1])
# plt.ylabel('$\mathcal{T}$')
# #%%
# plt.plot(lamsh_post, odefunc_full(np.log(lamsh_post),res_post.y)[3][1])
# plt.ylim(-1,2)
# plt.xscale('log')

# #%%
# plt.plot(lamsh_post, odefunc_full(np.log(lamsh_post),res_post.y)[2][1,0])
# plt.ylim(-1,2)
# plt.xscale('log')


## %%


#%%
ts = np.linspace(.25,5,30)
rs = ts**de

rs = np.logspace(-1,1,50)
ts = rs**(1/de)

r = np.outer(lamFres,rs)
t = np.outer(taures,ts)

r_anlt = np.outer(lamF_anlt,rs)
t_anlt = np.outer(tau_anlt,ts)

plt.plot(t,r, lw=1)
plt.plot(t_anlt,r_anlt, lw=1)

plt.grid(visible=True,axis='y', which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()

plt.xlim(0,10)
# plt.ylim(3e-2,1e1)
plt.yscale('log')
plt.ylabel('r')
plt.xlabel('t')
plt.savefig(f'Eds-gas-{gam:.02f}_trajectory_phys{name}.pdf')

#%%