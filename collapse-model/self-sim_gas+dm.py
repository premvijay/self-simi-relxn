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
# %%

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
    M = fb* Mta * Mn
    lam = (1-np.cos(tht))/2 * Mn**(1/3+eps)
    # D = ( 3*np.pi/8 * (1-np.cos(tht)) * Mn**(3*eps/2-1) * np.sin(tht) * Mta**-2 )**-1
    D = fb* 9*np.pi**2/ ( (2+6*eps) *(1-np.cos(tht))**3 - 9*eps*np.sin(tht) *(tht-np.sin(tht)) *(1-np.cos(tht)) ) * Mn**(-3*eps)
    # D_num = np.gradient(M,lam)/lam**2
    V = np.pi/2 * np.sin(tht)/(1-np.cos(tht)) * Mn**(1/3-eps/2)
    return lam,V,D,M



# %%
def odefunc(l, depvars):
    # lam = 1/laminv
    lam = np.exp(l)
    V,D,M,P = depvars
    Vb = V - de*lam
    linb = -np.array([2*D*V-2*D*lam, (de-1)*V*lam+2/9*M/lam, (2*(gam-1)+2*(de-1))*lam, -3*lam**3*D])
    # der_num = np.transpose(np.linalg.solve(linMat1,linb1), (1,0))
    linMat_det1 = D*Vb**2-gam*P
    # if linMat_det1 == 0: print(depvars)
    linMat_cofac1 = np.array([[-gam*P/D,D*Vb,-P,0],[D*Vb,-D**2,D*P/Vb,0],[0,0,0,linMat_det1],[gam*P*Vb,-gam*D*P,D*P*Vb,0]])
    linMat_inv = linMat_cofac1/ linMat_det1
    der = np.matmul(linMat_inv,linb)
    return der #*lam**2


#%%
# thtsh = 4.58324+1

def get_shock_bcs(thtsh):
    lamsh, V1, D1, M1 = preshock(thtsh)
    return lamsh, shock_jump(lamsh, V1, D1, M1)

def get_soln(thtsh):
    lamsh, bcs = get_shock_bcs(thtsh)
    # print(thtsh)#(lamsh, 1e-9) #np.log(lamsh),np.log(1e-9))
    return solve_ivp(odefunc, (np.log(lamsh),np.log(1e-8)), bcs, method='Radau', max_step=0.1, vectorized=False)
def M0(thtsh):
    res = get_soln(thtsh)
    M0val = res.y[2][-1]
    return M0val-1e-4 #if M0val>0 else -(-M0val)**(1/11)

#%%
def solve_bisect(func,bounds):
    b0, b1 = bounds
    bmid = (b0+b1)/2

def my_bisect(f, a, b, tol=1e-4): 
    m = (a + b)/2
    sfa = -1
    sfb = +1
    f_at_m = f(m)
    sfm = np.sign(f_at_m)
    # print(a,b,m,f_at_m)
    if abs(b-a) < tol:
        return m if f_at_m >0 else b
    elif sfa == sfm:
        return my_bisect(f, m, b, tol)
    elif sfb == sfm:
        return my_bisect(f, a, m, tol)

#%%
def odefunc_traj_dm(xi, arg):
    lam = arg[0]
    v = arg[1]
    return (v, -2/9 * M_tot(np.abs(lam))/(lam**2+1e-6) * np.sign(lam) - de*(de-1)*lam - (2*de-1)*v)

def odefunc_traj_gas(xi, arg):
    lam = arg
    return V_intrp(lam)-de*lam
#     except:
#         print(lam,s, xi, V_intrp(lam))
#         raise Exception

#%%
t_now = time()
# thtshsol = fsolve(M0, 1.5*np.pi)
s = 1
gam = 1.3
fb = 0.156837
# fb = 0.5
fd = (1-fb)
# fig4, ax4 = plt.subplots(1, dpi=200, figsize=(10,7))
thtsh_sols = []
thtbins_alls = []
M0_atbins_alls = []

dmo_prfl = pd.read_hdf(f'profiles_dmo_{s}.hdf5', key='main')

Mta = (3*np.pi/4)**2
M_dmo = interp1d(dmo_prfl['l'], dmo_prfl['M']*Mta, fill_value="extrapolate")
# D_dmo = interp1d(dmo_prfl['l'].iloc[1:], dmo_prfl['rho'].iloc[1:], fill_value="extrapolate")

M_dm = lambda lam: M_dmo(lam)*(1-fb)

de = 2* (1+s/3) /3
upsil = 1 if s >= 3/2 else 3*s/(s+3)

plot_iters = [0]#1,2,3] #,5,6,7]

t_bef, t_now = t_now, time()
print(f'{t_now-t_bef:.4g}s', 'Initialised vals and funcs for iteration')

for n in range(0, 1):
    thtbins_all = [np.linspace(1.2*np.pi, 1.99*np.pi, 8)]
    M0_atbins_all = []
    for nsect_i in range(0,3):
        thtbins = thtbins_all[nsect_i]
        M0_atbins_all.append(list(map(M0,thtbins)))
        # t_bef, t_now = t_now, time()
        # print(f'{t_now-t_bef:.4g}s', f's={s}: grid M0 obtained')
        idx_M0neg = np.where(np.sign(M0_atbins_all[nsect_i])==-1)[0].max()
        # t_bef, t_now = t_now, time()
        # print(f'{t_now-t_bef:.4g}s', f's={s}: grid M0 selected')
        thtshsol = thtbins[idx_M0neg+1]
        thtbins_all.append(np.linspace(thtbins[idx_M0neg], 2*thtshsol-thtbins[idx_M0neg], 8))

    thtshsol = my_bisect(M0, thtbins_all[nsect_i+1][0], thtbins_all[nsect_i+1][-1], tol=1e-4)
    
    thtsh_sols.append(thtshsol)
    thtbins_alls.append(thtbins_all)
    M0_atbins_alls.append(M0_atbins_all)

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f'{n}th iter gas shock radius solved')

    res_prof_gas = get_soln(thtshsol)

    lamsh_post = np.exp(res_prof_gas.t)
    V_post, D_post, M_post, P_post = res_prof_gas.y

    thtsh_preange = np.arange(1*np.pi, thtshsol,0.01)

    lamsh_pre, V_pre, D_pre, M_pre = preshock(thtsh_preange)
    P_pre = lamsh_pre*0

    lamsh = lamsh_pre.min()

    lam_all = np.concatenate([lamsh_post, lamsh_pre][::-1])
    V_all = np.concatenate([V_post, V_pre][::-1])
    D_all = np.concatenate([D_post, D_pre][::-1])
    M_all = np.concatenate([M_post, M_pre][::-1])
    P_all = np.concatenate([P_post, P_pre][::-1])

    M_gas = interp1d(lam_all, M_all, fill_value="extrapolate")

    M_tot = lambda lam : M_dm(lam)+M_gas(lam)

    resdf_gas = pd.DataFrame(data={'l':lam_all, 'M':M_all, 'V':V_all, 'D':D_all, 'P':P_all,})
    resdf_gas.to_hdf(f'profiles_gasdm_s{s:g}_gam{gam:.3g}.hdf5', 'gas/main', mode='a')
    resdf_gas.to_hdf(f'profiles_gasdm_s{s:g}_gam{gam:.3g}.hdf5', f'gas/iter{n}', mode='a')

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f'{n}th iter gas profiles updated')

    xi_max = np.log(2e-4**upsil)*-3/2/s

    res_traj_dm = solve_ivp(odefunc_traj_dm, (0,xi_max), np.array([1,-de]), method='Radau', t_eval=(np.linspace(0,xi_max**3,500000))**(1/4), max_step=np.inf, dense_output=False, vectorized=True)
    # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

    xi = res_traj_dm.t
    lam = np.abs(res_traj_dm.y[0])
    # v = res_traj_dm.y[1]
    loglam = np.log(np.maximum(lam,1e-15))

    resdf_traj_dm = pd.DataFrame(data={'xi':xi, 'lam':lam,})
    resdf_traj_dm.to_hdf(f'traj_gasdm_s{s:g}_gam{gam:.3g}.hdf5', 'dm/main', mode='a')
    resdf_traj_dm.to_hdf(f'traj_gasdm_s{s:g}_gam{gam:.3g}.hdf5', f'dm/iter{n}', mode='a')

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f'{n}th iter DM trajectory obtained')

    l_range = np.zeros(301)
    l_range[1:] = np.logspace(-2.5,0, 300)
    M_vals = np.zeros(301)
    # rho_vals = np.zeros(301)
    # v_xi = interp1d(xi, v, fill_value="extrapolate")
    for i in range(1,301):
        # l_range.append(l)
        l = l_range[i]
        # spl = InterpolatedUnivariateSpline(xi, loglam-np.log(l),k=3)
        # roots = spl.roots()
        roots_ind = np.nonzero(np.diff(np.sign(loglam-np.log(l))))[0]
        roots = (xi[roots_ind]+xi[np.array(roots_ind)+1])/2
        #Based on theory we need odd number of roots, otherwise there is a major error
        n_roots = roots.shape[0]
        last_root_i = n_roots if n_roots%2==1 else n_roots-1
        Int_M = np.exp((-2*s/3)*roots[:last_root_i])
        M_val = np.sum(Int_M[::2]) - np.sum(Int_M[1::2])
        M_vals[i] = M_val
    M_vals[-1] = 1

    M_vals = np.asarray(M_vals)
    # M_vals_er = np.asarray(M_vals_er)
    # rho_vals = np.asarray(rho_vals)

    M_vals *= Mta*(1-fb) / M_vals[-1]

    M_dm = interp1d(l_range, M_vals, fill_value="extrapolate")

    resdf_dm = pd.DataFrame(data={'l':l_range, 'M':M_vals,})
    resdf_dm.to_hdf(f'profiles_gasdm_s{s:g}_gam{gam:.3g}.hdf5', 'dm/main', mode='a')
    resdf_dm.to_hdf(f'profiles_gasdm_s{s:g}_gam{gam:.3g}.hdf5', f'dm/iter{n}', mode='a')

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f'{n}th iter DM mass profile updated')




#%%
del res_traj_dm, lam, loglam, xi
import dill                            #pip install dill --user
filename = f'soln-globalsave_s{s:g}_gam{gam:.3g}.pkl'
dill.dump_session(filename)


# %%
import dill                            #pip install dill --user
filename = f'soln-globalsave_s{s:g}_gam{gam:.3g}.pkl'
dill.load_session(filename)


#%%
t_now = time()
# thtshsol = fsolve(M0, 1.5*np.pi)
fig4, ax4 = plt.subplots(1, dpi=200, figsize=(7,5))
fig5, axs5 = plt.subplots(2,2, dpi=200, figsize=(10,8), sharex=True)
fig6, ax6 = plt.subplots(1)

# plot_iters = [0,] #1,2,3,5,6,7]

t_bef, t_now = t_now, time()
print(f'{t_now-t_bef:.4g}s', 'Initialised plots and figs for iteration')

for n in plot_iters:
    color_this = plt.cm.turbo(n/7)
    linestyles= [':', '--', '-']
    thtbins_all = thtbins_alls[n]
    for nsect_i in range(0,3):
        ax4.plot(thtbins_all[nsect_i],M0_atbins_alls[n][nsect_i], color=color_this, ls=linestyles[nsect_i], label=f'n={n} and nsect={nsect_i}')

    thtshsol = thtsh_sols[n]
    ax4.axvline(thtsh_sols[n], color=color_this, label=r'$\theta_{s}=$'+f'{thtsh_sols[n]:.6g}')
    print(f'n={n}', thtshsol)

    resdf_prof_gas = pd.read_hdf(f'profiles_gasdm_s{s:g}_gam{gam:.3g}.hdf5', key=f'gas/iter{n}', mode='r')
    resdf_prof_dm = pd.read_hdf(f'profiles_gasdm_s{s:g}_gam{gam:.3g}.hdf5', key=f'dm/iter{n}', mode='r')
    resdf_traj_dm = pd.read_hdf(f'traj_gasdm_s{s:g}_gam{gam:.3g}.hdf5', key=f'dm/iter{n}', mode='r')

    axs5[0,0].plot(resdf_prof_gas.l, -resdf_prof_gas.V, color=color_this, label=f'n={n}')
    axs5[0,1].plot(resdf_prof_gas.l, resdf_prof_gas.D, color=color_this)
    axs5[1,0].plot(resdf_prof_gas.l, resdf_prof_gas.M, color=color_this, ls='dashdot')
    axs5[1,1].plot(resdf_prof_gas.l, resdf_prof_gas.P, color=color_this)

    axs5[1,0].plot(resdf_prof_dm.l, resdf_prof_dm.M, ls='solid', color=color_this)


    M_gas = interp1d(resdf_prof_gas.l, resdf_prof_gas.M)
    M_dm = interp1d(resdf_prof_dm.l, resdf_prof_dm.M)

    M_tot = lambda lam : M_dm(lam)+M_gas(lam)

    # axs5[1,0].plot(lam_all,M_all+M_dm(lam_all), color=color_this, ls='dashed')

    ax6.plot(resdf_traj_dm.xi,resdf_traj_dm.lam, color=color_this, label=f'n={n}')

    V_intrp = interp1d(resdf_prof_gas.l, resdf_prof_gas.V, fill_value="extrapolate")
    lamshsol, bcs = get_shock_bcs(thtshsol)
    taush = (thtshsol - np.sin(thtshsol)) / np.pi
    xish = np.log(taush)
    res_traj_gas = solve_ivp(odefunc_traj_gas, (xish,4), np.array([lamshsol]), method='Radau', max_step=0.001, dense_output=False, vectorized=True)
    # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: post shock trajectory obtained')
    
    xires = res_traj_gas.t
    lamres = res_traj_gas.y[0]
    # vres = res.y[1]

    taures = np.exp(xires)
    lamFres = lamres*taures**de

    # ax6.plot(taures,lamFres, color=color_this, label=f's={s}')
    ax6.plot(xires,lamres, color=color_this)


    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f'{n}th iter plotted')

axs5[1,0].plot(dmo_prfl['l'], dmo_prfl['M']*Mta, color='k', ls='dashed')


ax4.set_xlabel(r'$\theta$')
ax4.set_ylabel(r'$M(\lambda=0)$')
# ax4.set_ylim(-2,5)
ax4.legend()
    
axs5[0,0].set_xscale('log')
axs5[0,0].set_xlim(1e-4,1)
axs5[0,0].legend()

# axs5[0,0].set_ylim(1e-60,1)

axs5[1,0].plot([], ls='solid', color='k', label='DM')
axs5[1,0].plot([], ls='dashdot', color='k', label='Gas')
axs5[1,0].plot([], ls='dashed', color='k', label='DMo')
axs5[1,0].legend()

if gam>1.66:
    # axs5[0,0].set_xlim(1e-2,1)
    axs5[0,1].set_ylim(1e-1,1e6)
    axs5[1,0].set_ylim(1e-2,1e1)
    axs5[1,1].set_ylim(1e0,1e7)

axs5[0,0].set_ylabel('-V')
axs5[0,1].set_ylabel('D')
axs5[1,0].set_ylabel('M')
axs5[1,1].set_ylabel('P')

axs5[0,0].set_yscale('log')
axs5[0,1].set_yscale('log')
axs5[1,0].set_yscale('log')
axs5[1,1].set_yscale('log')




ax6.set_xlim(0,4)
ax6.set_ylim(0,1)
ax6.set_yscale('log')
ax6.set_ylim(resdf_prof_dm.l[1],1)
ax6.xaxis.get_ticklocs(minor=True)     # []
ax6.minorticks_on()
ax6.grid(visible=True, which='both', axis='x')
ax6.set_xlabel(r'$\xi$')
ax6.set_ylabel(r'$\lambda$')
ax6.legend()

# fig5.savefig(f'Eds-gas-{gam:.02f}_profiles.pdf')
# fig5.savefig(f'Eds-gas-{gam:.02f}_trajectory.pdf')


#%%
plt.show()

#%%

# %%

# %%
# import dill                            #pip install dill --user
# filename = f'soln-globalsave_s{s:g}_gam{gam:.3g}.pkl'
# dill.load_session(filename)

#%%
fd = (1-fb)
lamr_full = np.logspace(-3.3,-0.005,300)
lamr = np.logspace(-3.3,-0.005,300)

r, Mdr, Mbr, Mdr_dmo = lamr, M_dm(lamr), M_gas(lamr), M_dmo(lamr_full)*fd
ri_pre = lamr_full

#%%
plt.figure()
plt.plot(r,Mdr, label='DM')
plt.plot(r,Mbr*fd/fb, label='baryon')
plt.plot(ri_pre,Mdr_dmo, label='DM in DMO' )
# plt.plot(r,Mdr+Mbr)
plt.xscale('log')
plt.yscale('log')
plt.legend()

#%%
rf = r.copy()

logri_logM = interp1d(np.log10(Mdr_dmo),np.log10(ri_pre), fill_value='extrapolate')

# assert (ri_M(Mdr_dmo) == r).all()

ri = 10**logri_logM(np.log10(Mdr))

Mf = Mdr+Mbr
Mi = Mdr/fd

MiMf = ( fd* (Mbr/ Mdr + 1) )**-1
rfri = rf / ri
 
#%%
plt.figure()
# plt.scatter(MiMf[60:-50],rfri[60:-50],c=rf[60:-50])
plt.scatter(MiMf,rfri,c=np.log10(rf), cmap='nipy_spectral')
# plt.scatter(MiMf[100:],rfri[100:],c=np.log10(rf[100:]), cmap='nipy_spectral')
plt.plot(MiMf,1+0.25*(MiMf-1),'k',label='q=0.25')
plt.colorbar(label='rf (defined as relaxed $\lambda$)')
plt.xlabel('Mi/Mf')
plt.ylabel('rf/ri')
plt.legend()
plt.savefig('ratio_plot_anyl.pdf')
# %%
