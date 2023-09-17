#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
from scipy.optimize import fsolve, bisect
from time import time
from copy import copy
import dill   
# %%

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
plt.style.use('default')

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

#%%

# %%                           #pip install dill --user
# import dill  
# dill.load_session(f'soln-globalsave_all1.pkl')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
Mta = (3*np.pi/4)**2
fb = 0.156837
fd = 1-fb

#%%
name = 'cold_vary-s'
# name = 'shocked_vary-s'
# name = 'shocked_vary-gam'
# name = 'shocked_vary-cooling'
name = 'shocked_vary-lamdish'
name = 'shocked_vary-lamshsp'

with open(f'{name}-descr.txt', 'tr') as file: descr_list = eval(file.read())
with open(f'{name}-plab.txt', 'tr') as file: plab_list = eval(file.read())
with open(f'{name}-conv_iters.txt', 'tr') as file: conv_iter_list = eval(file.read())

# descr_list = descr_list_dict[name]
# plab_list= plab_list_dict[name]

t_now = time()
# thtshsol = fsolve(M0, 1.5*np.pi)
# fig4, ax4 = plt.subplots(1, dpi=200, figsize=(7,5))
fig5, axs5 = plt.subplots(2,2, figsize=(14,10), sharex=True)
fig6, ax6 = plt.subplots(1)

fig7, (ax71,ax72) = plt.subplots(1,2, figsize=(14,7))
# fig8, (ax8,ax82) = plt.subplots(2, figsize=(7,10))

for i,descr in enumerate(descr_list):
    # s=float(descr.split('_')[4][2:])
    for value in descr.split('_')[3:]: exec(value) 
    de=2* (1+s/3) /3
    t_bef, t_now = t_now, time()
    plab = plab_list[i]
    conv_iter = conv_iter_list[i]
    plot_iters = [0,conv_iter]
    # err_tol = 0.01
    # color_this = plt.cm.turbo(n/30)
    color_this = colors[i]
    linestyles= [':', '--', '-','-.']

    n=0
    resdf_prof_gas = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'gas/iter{n}', mode='r')
    resdf_prof_dm = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter{n}', mode='r')
    resdf_traj_dm = pd.read_hdf(f'traj_gasdm{descr:s}.hdf5', key=f'dm/iter{n}', mode='r')
    #resdf_traj_dm_d = pd.read_hdf(f'traj_gasdm{descr:s}_desktop.hdf5', key=f'dm/iter{n}', mode='r')
    resdf_prof_dmo = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter0', mode='r')

    axs5[0,0].plot(resdf_prof_gas.l, -resdf_prof_gas.Vb, color=color_this, ls='--')
    axs5[0,1].plot(resdf_prof_gas.l, resdf_prof_gas.D, color=color_this, ls='--')
    axs5[1,0].plot(resdf_prof_gas.l, resdf_prof_gas.M/resdf_prof_gas.M[0], ls='--', color=color_this)
    axs5[1,1].plot(resdf_prof_gas.l, resdf_prof_gas.P, color=color_this, ls='--')

    ax6.plot(cumtrapz(1/(resdf_prof_gas.V-de*resdf_prof_gas.l), x=resdf_prof_gas.l), resdf_prof_gas.l[1:], c=color_this, ls=':')

    ax71.plot(resdf_prof_dm.l[1:], resdf_prof_dm.M[1:]/fd, ls='--', color=color_this)

    n=conv_iter
    resdf_prof_gas = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'gas/iter{n}', mode='r')
    resdf_prof_dm = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter{n}', mode='r')
    resdf_traj_dm = pd.read_hdf(f'traj_gasdm{descr:s}.hdf5', key=f'dm/iter{n}', mode='r')
    #resdf_traj_dm_d = pd.read_hdf(f'traj_gasdm{descr:s}_desktop.hdf5', key=f'dm/iter{n}', mode='r')
    # resdf_prof_dmo = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter0', mode='r')

    axs5[0,0].plot(resdf_prof_gas.l, -resdf_prof_gas.Vb, color=color_this)
    axs5[0,1].plot(resdf_prof_gas.l, resdf_prof_gas.D, color=color_this)
    axs5[1,0].plot(resdf_prof_gas.l, resdf_prof_gas.M/resdf_prof_gas.M[0], color=color_this)
    axs5[1,1].plot(resdf_prof_gas.l, resdf_prof_gas.P, color=color_this, label=plab)

    # axs5[1,0].plot(resdf_prof_dm.l, resdf_prof_dm.M, ls='dashdot', color=color_this)

    ax71.plot(resdf_prof_dm.l[1:], resdf_prof_dm.M[1:], ls='-', c=color_this)
    ax71.plot(resdf_prof_gas.l, resdf_prof_gas.M, ls='-.', c=color_this)
    # ax71.plot(ri_pre,Mdr_dmo/Mta, ls='--', c=color_this)
    # plt.plot(r,Mdr+Mbr)
    ax71.set_xscale('log')
    ax71.set_yscale('log')


    # axs5[1,0].plot(lam_all,M_all+M_dm(lam_all), color=color_this, ls='dashed')

    ax6.plot(resdf_traj_dm.xi,resdf_traj_dm.lam, color=color_this, label=plab)
    # ax6.plot(resdf_traj_dm_d.xi,resdf_traj_dm_d.lam, label=f'n={n}_desktop')

    # V_intrp = interp1d(resdf_prof_gas.l, resdf_prof_gas.V, fill_value=np.nan)
    # lamshsol, bcs = get_shock_bcs(thtshsol)
    # taush = (thtshsol - np.sin(thtshsol)) / np.pi
    # xish = np.log(taush)
    # res_traj_gas = solve_ivp(odefunc_traj_gas, (1,4), np.array([1]), method='Radau', max_step=0.001, dense_output=False, vectorized=True)
    # # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

    # t_bef, t_now = t_now, time()
    # print(f'{t_now-t_bef:.4g}s', f's={s}: post shock trajectory obtained')

    # xires = res_traj_gas.t
    # lamres = res_traj_gas.y[0]
    # # vres = res.y[1]

    # taures = np.exp(xires)
    # lamFres = lamres*taures**de

    # # ax6.plot(taures,lamFres, color=color_this, label=f's={s}')
    # ax6.plot(xires,lamres, color=color_this)
    ax6.plot(cumtrapz(1/(resdf_prof_gas.V-de*resdf_prof_gas.l), x=resdf_prof_gas.l), resdf_prof_gas.l[1:], c=color_this, ls='-.')
    
    resdf_relx = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'relx/iter{n}', mode='r')

    MiMf, rfri, rf = resdf_relx.MiMf, resdf_relx.rfri, resdf_relx.rf

    # MiMf_max, MiMf_min = MiMf_stack.max(axis=0), MiMf_stack.min(axis=0)
    # rfri_max, rfri_min = rfri_stack.max(axis=0), rfri_stack.min(axis=0)

    MiMf_err = np.full_like(MiMf,0.0)
    rfri_err = np.full_like(rfri,0.01)#resdf_relx.MiMf_err, resdf_relx.rfri_err




    ax72.errorbar(MiMf[::20],rfri[::20], xerr=MiMf_err[::20], yerr=rfri_err[::20],fmt='.')
    # ax72.fill_between(MiMf, rfri_min, rfri_max, color=color_this, alpha=0.3)  
    # ax71.scatter(MiMf[60:-50],rfri[60:-50],c=rf[60:-50])
    # cplot = ax72.scatter(MiMf,rfri,c=np.log10(rf), s=60, cmap='nipy_spectral')
    # plab=f'n={n}'
    ax72.plot(MiMf,rfri, label=plab, c=color_this, lw=3)
    # ax71.scatter(MiMf[100:],rfri[100:],c=np.log10(rf[100:]), cmap='nipy_spectral')
    # ax71.plot(MiMf,1+0.25*(MiMf-1),'k',label='$q=0.25$')

ax71.plot([],[], ls='-', c='k', label='DM')
ax71.plot([],[], ls='-.', c='k', label='Gas')
ax71.plot([],[], ls='--', c='k', label='DM in DMO' )

ax71.set_xlabel(r'$r/r_{\rm{ta}}$')
ax71.set_ylabel(r'$\mathcal{M}$')
ax71.set_xlim(5e-4,1e0)

# ax72.plot(MiMf,1+0.33*(MiMf-1)-0.02,'k:',label='$q=0.33$, $q_0=0.02$')

ax72.set_xlabel('$M_i/M_f$')
ax72.set_ylabel('$r_f/r_i$')

# fig7.colorbar(cplot, ax=ax72,label=r'$r_f/r_{\rm{ta}}$')
ax71.legend()
ax72.legend()

# axs5[1,0].plot(dmo_prfl['l'], dmo_prfl['M']*Mta, color='k', ls='dashed')
# axs5[1,0].plot(resdf_prof_dmo.l, resdf_prof_dmo.M, color='purple', ls='dashed')

# ax4.set_xlabel(r'$\theta$')
# ax4.set_ylabel(r'$M(\lambda=0)$')
# # ax4.set_ylim(-2,5)
# ax4.legend()
    
axs5[0,0].set_xscale('log')
axs5[0,0].set_xlim(3e-4,1)
axs5[1,1].legend()

# axs5[0,0].set_ylim(1e-60,1)

axs5[1,0].plot([], ls='solid', color='k', label='Gas with DM halo')
# axs5[1,0].plot([], ls='dashdot', color='k', label='DM')
axs5[1,0].plot([], ls='dashed', color='k', label='Self-gravitating gas')
axs5[1,0].legend(loc='lower left')

# if gam>1.67:
# axs5[0,0].set_xlim(1e-2,1)
axs5[0,1].set_ylim(1e-1,1e7)
axs5[1,0].set_ylim(1e-2,2e0)
axs5[1,1].set_ylim(1e0,1e9)

axs5[0,0].set_ylabel(r'$-\bar{V}$')
axs5[0,1].set_ylabel(r'$D$')
axs5[1,0].set_ylabel(r'$M/M_{\rm{ta}}$')
axs5[1,1].set_ylabel(r'$P$')

axs5[0,0].set_yscale('log')
axs5[0,1].set_yscale('log')
axs5[1,0].set_yscale('log')
axs5[1,1].set_yscale('log')




ax6.set_xlim(0,6)
ax6.set_ylim(1e-3,1)
ax6.set_yscale('log')
# ax6.set_ylim(resdf_prof_dm.l[1],1)
ax6.xaxis.get_ticklocs(minor=True)     # []
ax6.minorticks_on()
ax6.grid(visible=True, which='both', axis='x')
ax6.set_xlabel(r'$\xi$')
ax6.set_ylabel(r'$\lambda$')
ax6.legend(loc='upper right')

fig5.savefig(f'profiles_gas_{name}.pdf', bbox_inches='tight')
fig6.savefig(f'trajectory_gasdm_{name}.pdf', bbox_inches='tight')

fig7.savefig(f'relx_reln_{name}.pdf', bbox_inches='tight')

#%%