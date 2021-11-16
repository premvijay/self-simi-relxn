#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d
from itertools import cycle
plt.style.use('seaborn-darkgrid')
# %%

# %%

# %%
s = 1
fig4, ax4 = plt.subplots(1, dpi=120, figsize=(10,7))
fig5, (ax5,ax6) = plt.subplots(1,2, dpi=120, figsize=(14,7))


for s in [0.5,1,1.5,2]:#[:2]:
    de = 2* (1+s/3) /3

    def M0(l):
        return l**(3/4)
    def M_pred(l):
        gam = 1 if s >= 3/2 else 3*s/(s+3)
        return l**gam

    # def M(l,n):
    #     return 0

    M = M_pred

    color_this = plt.cm.jet(s/2)
    # linestyles = [":","-.","--","-"]
    linestyles = [(0, (1, 3)), (0, (2, 3)), (0, (3, 3)), (0, (4, 3)), (0, (5, 3))]
    ls_cycler = cycle(linestyles)
    for n in range(21):
        def ode_func(xi, arg):
            lam = arg[0]
            v = arg[1]
            # print(lam, (v, -2/9 * M(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v + 1e-50/lam**10))
            # if lam<1e-5: v=-v
            try:
                return (v, -2/9 * (3*np.pi/4)**2* M(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v + 1e-50/lam**30)
            except:
                print(lam,s, v, xi)
                raise Exception


        res = solve_ivp(ode_func, (0,10), np.array([1,-de]), max_step=0.01, dense_output=True)
        # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

        xi = res.t
        lam = res.y[0]

        # @np.vectorize
        # def M(l):
        #     Integ = lambda xi : np.exp(-xi*2*s/3) * np.heaviside(l-res.sol(xi)[0], 1)
        #     return quad(Integ, 0, np.inf)[0]
        
        l_range = np.linspace(0,1, 200)
        l_range = np.logspace(-2.5,0, 200)
        Integ_fac_vals = np.exp(-xi*2*s/3)
        # @np.vectorize
        # def M(l):
        grid_lam, grid_l_range = np.meshgrid(lam, l_range)
        Integ_vals = Integ_fac_vals * np.heaviside(grid_l_range-grid_lam, 1)
        M_vals = np.trapz(Integ_vals, xi) #grid_l_range, axis=0)
        M_vals += quad(lambda xi : np.exp(-xi*2*s/3), xi[-1], np.inf)[0]*l_range

        M_vals /= M_vals[-1]

        M = interp1d(l_range, M_vals, fill_value="extrapolate")

        

        if n in [1,3,5,8,20]:
            # xi = np.linspace(0,4,100)
            # lam = res.sol(xi)[0]
            # plt.plot(xi,lam)
            xi = res.t
            lam = res.y[0]
            tau = np.exp(xi)
            lamF = lam*tau**de
            
            ls = next(ls_cycler)
            
            ax4.plot(xi,lam, color=color_this, ls=ls)
            # plt.plot(xi,lamF, color=color_this, label=f's={s}')
            # plt.plot(res1.t, res1.y[0], color=color_this)
            # plt.plot(res.t, res.y[1], color=color_this)

            lam = np.linspace(0,1,200)
            if s==0.5:
                ax5.plot([],[], color='k', ls=ls, label=f'n={n}')
                ax4.plot([],[], color='k', ls=ls, label=f'n={n}')
            
            ax5.plot(l_range, M_vals, color=color_this, ls=ls)
            ax6.plot(l_range[1:], np.diff(M_vals)/l_range[1:]**2, color=color_this, ls=ls)
        
    ax5.plot(lam, M_pred(lam), color=color_this, ls='-', label=f's={s}')
    ax4.plot([],[], color=color_this, label=f's={s}')
    ax6.set_xscale('log')
        
    # ax5.plot(l_range, M_vals)
ax4.xaxis.get_ticklocs(minor=True)     # []
ax4.minorticks_on()
ax4.grid(b=True, which='both', axis='x')
ax4.set_xlabel(r'$\xi$')
ax4.set_ylabel(r'$\lambda$')
ax4.legend()

ax5.set_xlabel(r'$\lambda$')
ax5.set_ylabel(r'$M$')
ax5.legend()

ax6.set_xscale('log')
ax5.set_xlabel(r'$\lambda$')
ax5.set_ylabel(r'$\rho$')
ax5.legend()
plt.show()

# fig4.savefig('Eds-CDM_shells.pdf')
# fig5.savefig('Eds-CDM_M_lam.pdf')
# %%

# %%
