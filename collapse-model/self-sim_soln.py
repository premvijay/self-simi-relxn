#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
from time import time
# %%

# %%

# %%
s = 1
fig4, (ax4,ax41) = plt.subplots(2, dpi=200, figsize=(10,12), sharex=True)
fig5, (ax5,ax6) = plt.subplots(1,2, dpi=200, figsize=(14,7))

t_now = time()
for s in [0.5,1,1.5,2,3][::]:
    de = 2* (1+s/3) /3
    upsil = 1 if s >= 3/2 else 3*s/(s+3)

    def M0(l):
        return l**(3/4)
    def M_pred(l):
        return l**upsil

    # def M(l,n):
    #     return 0

    M_func = M_pred

    color_this = plt.cm.jet(s/3)
    # linestyles = [":","-.","--","-"]
    linestyles = [(0, (1, 3)), (0, (2, 3)), (0, (3, 3)), (0, (4, 3)), (0, (5, 1,1,1)), 'solid'][1:]
    ls_cycler = cycle(linestyles[::])

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s} Initialised vals and funcs for iteration')

    for n in range(5):
        def ode_func(xi, arg):
            lam = arg[0]
            v = arg[1]
            # print(lam, (v, -2/9 * M(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v + 1e-50/lam**10))
            # if lam<1e-5: v=-v
            # try:
            return (v, -2/9 * (3*np.pi/4)**2* M_func(np.abs(lam))/(lam**2+1e-6) * np.sign(lam) - de*(de-1)*lam - (2*de-1)*v)
            # except:
            #     print(lam,s, v, xi)
            #     raise Exception

        xi_max = np.log(5e-4**upsil)*-3/2/s

        res = solve_ivp(ode_func, (0,xi_max), np.array([1,-de]), method='Radau', t_eval=np.linspace(0,xi_max**3,500000)**(1/3), max_step=np.inf, dense_output=False, vectorized=True) #np.unique(np.concatenate([np.linspace(0,2,50000),np.log10(np.linspace(1,10000,100000))]))
        # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

        xi = res.t
        lam = np.abs(res.y[0])
        v = res.y[1]
        loglam = np.log(np.maximum(lam,1e-15))

        t_bef, t_now = t_now, time()
        print(f'{t_now-t_bef:.4g}s', f'{n}th iter DM trajectory obtained')

        # @np.vectorize
        # def M(l):
        #     Integ = lambda xi : np.exp(-xi*2*s/3) * np.heaviside(l-res.sol(xi)[0], 1)
        #     return quad(Integ, 0, np.inf)[0]
        
        # l_range = np.linspace(0,1, 200)
        # l_range = np.logspace(-2.5,0, 300)
        
        # @np.vectorize
        # def M(l):

        # Integ_fac_vals = np.exp(-xi*2*s/3)
        # grid_lam, grid_l_range = np.meshgrid(lam, l_range)
        # Integ_vals = Integ_fac_vals * np.heaviside(grid_l_range-grid_lam, 1)
        # M_vals = np.trapz(Integ_vals, xi) #grid_l_range, axis=0)
        # M_vals += quad(lambda xi : np.exp(-xi*2*s/3), xi[-1], np.inf)[0]*l_range

        l_range = np.zeros(301)
        l_range[1:] = np.logspace(-2.5,0, 300)
        n_roots_ar = np.zeros(301)
        # first_root = [0]
        # all_roots = [[None]] #np.zeros(300,70)
        M_vals = np.zeros(301)
        # M_vals_er = []
        if n==4:
            rho_vals = np.zeros(301)
            v_xi = interp1d(xi, v, fill_value="extrapolate")
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
            n_roots_ar[i] = n_roots
            # first_root.append(roots[0])
            # all_roots.append(roots)
            # if roots.shape[0]>1:
            #     M_vals_er.append(Int_M[-2]-Int_M[-1])
            # else:
            #     M_vals_er.append(0)
            if n==4:
                Int_rho = 2/9*s * (3*np.pi/4)**2 * np.exp((-2*s/3)*roots) / np.abs(v_xi(roots)) / l**2
                rho_vals[i] = np.sum(Int_rho)
        M_vals[-1] = 1

        M_vals = np.asarray(M_vals)
        # M_vals_er = np.asarray(M_vals_er)
        if n==4: rho_vals = np.asarray(rho_vals)

        M_vals /= M_vals[-1]

        M_func = interp1d(l_range, M_vals, assume_sorted=True, fill_value="extrapolate")

        t_bef, t_now = t_now, time()
        print(f'{t_now-t_bef:.4g}s', f'{n}th iter DM mass profile obtained')


        df = pd.DataFrame(data={'l':l_range, 'M':M_vals,})
        df.to_hdf(f'profiles_dmo_{s}.hdf5', 'main')
        df.to_hdf(f'profiles_dmo_{s}.hdf5', f'iter{n}')
        

        if n in [0,1,2,3,4,5,7,8]:
            # xi = np.linspace(0,4,100)
            # lam = res.sol(xi)[0]
            # plt.plot(xi,lam)
            # xi = res.t
            # lam = res.y[0]
            tau = np.exp(xi)
            lamF = lam*tau**de
            
            ls = next(ls_cycler)
            if n==6: ls='-'
            
            ax4.plot(xi,lam, color=color_this, ls=ls, lw=1)
            ax41.plot(xi,v, color=color_this, ls=ls, lw=1)
            # plt.plot(xi,lamF, color=color_this, label=f's={s}')
            # plt.plot(res1.t, res1.y[0], color=color_this)
            # plt.plot(res.t, res.y[1], color=color_this)

            # lam = np.linspace(0,1,200)
            if s==1:
                ax5.plot([],[], color='k', ls=ls, label=f'n={n}', lw=1)
                ax4.plot([],[], color='k', ls=ls, label=f'n={n}', lw=1)
                ax41.plot([],[], color='k', ls=ls, label=f'n={n}', lw=1)
            
            ax5.plot(l_range, M_vals, color=color_this, ls=ls, lw=1)
            # ax6.plot(l_range[1:], np.diff(M_vals)/l_range[1:]**2, color=color_this, ls=ls, lw=1)
            if n==4: ax6.plot(l_range[1:], rho_vals[1:], color=color_this, ls='-', lw=1)

            t_bef, t_now = t_now, time()
            print(f'{t_now-t_bef:.4g}s', f'{n}th iter saved and plotted')

    # ax5.plot(lam, M_pred(lam), color=color_this, ls='-', label=f's={s}')
    ax4.plot([],[], color=color_this, label=f's={s}')
    ax5.plot([],[], color=color_this, label=f's={s}')

    eps = 1/s
    Mta = (3*np.pi/4)**2
    tht = np.linspace(np.pi, 2*np.pi,300)
    Mn = ((tht-np.sin(tht))/np.pi)**(-2/3*s)
    M = Mta * Mn
    lam_param = (1-np.cos(tht))/2 * Mn**(1/3+eps)
    ax5.plot(lam_param,Mn, color=color_this, ls='-')

        
    # ax5.plot(l_range, M_vals)
    


# all_roots_ar = np.zeros((int(max(n_roots_ar)),301))
# for i,roots_ar in enumerate(all_roots):
#     all_roots_ar[0:len(roots_ar),i] = roots_ar


ax4.set_xlim(0,10)
ax4.set_ylim(5e-4,1)
ax4.set_yscale('log')
# ax4.set_ylim(l_range[1],1)
ax4.xaxis.get_ticklocs(minor=True)     # []
ax4.minorticks_on()
ax4.grid(visible=True, which='both', axis='x')
ax4.set_xlabel(r'$\xi$')
ax4.set_ylabel(r'$\lambda$')
ax4.legend()

ax41.set_xlabel(r'$\xi$')
ax41.set_ylabel(r'$v$')

ax5.set_xlim(l_range[1]/1.2,1.1)
ax5.set_ylim(M_vals[10]/2,1.1)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlabel(r'$\lambda$')
ax5.set_ylabel(r'$M$')
ax5.legend()

ax6.set_xlim(l_range[1]/1.2,1.1)
ax6.set_ylim(bottom=0.7)
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xlabel(r'$\lambda$')
ax6.set_ylabel(r'$\rho$')
# ax6.legend()
plt.show()





#%%
ts = np.linspace(.25,5,30)
rs = ts**de

rs = np.logspace(-1,1,500)
ts = rs**(1/de)

r = np.outer(lamF,rs)
t = np.outer(tau,ts)

# r_anlt = np.outer(lamF_anlt,rs)
# t_anlt = np.outer(tau_anlt,ts)

plt.plot(t,r, lw=1)
# plt.plot(t_anlt,r_anlt, lw=1)

plt.grid(visible=True,axis='y', which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()

plt.xlim(0,10)
# plt.ylim(3e-2,1e1)
plt.yscale('log')
plt.ylabel('r')
plt.xlabel('t')

#%%
# fig4.tight_layout()
# fig5.tight_layout()
fig4.savefig('Eds-CDM_shells.pdf')
fig5.savefig('Eds-CDM_M_lam.pdf')
# %%

# %%
s = 1
fig7, (ax71,ax72) = plt.subplots(1,2, dpi=200, figsize=(14,7))
for s in [0.2,0.5,1,2,5]:
    eps = 1/s
    Mta = (3*np.pi/4)**2
    tht = np.linspace(0.01, 2*np.pi,300)
    Mn = ((tht-np.sin(tht))/np.pi)**(-2/3*s)
    M = Mta * Mn
    lam = (1-np.cos(tht))/2 * Mn**(1/3+eps)
    D = ( 3*np.pi/8 * (1-np.cos(tht)) * Mn**(3*eps/2-1) * np.sin(tht) * Mta**-2 )**-1
    D_num = np.gradient(M,lam)/lam**2
    ax71.plot(lam,M, label=r'$\epsilon=$'+f'{eps}')
    # ax72.plot(lam,D, label=r'$\epsilon=$'+f'{eps}')
    ax72.plot(lam,D_num, label=r'$\epsilon=$'+f'{eps}')
ax71.legend()
ax72.legend()
ax71.set_ylim(0,30)
ax71.set_xlim(0,3)
ax72.set_xscale('log')
ax72.set_yscale('log')
ax72.set_xlim(1e-2,1e1)
ax72.set_ylim(1e-1,1e3)

# %%
