#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
from time import time


#%%
def odefunc(xi, lam):
    return 0

#%%
g = 10
hm = 1
vel_rev = 0
def odefunc_fall(t, h):
    v = -np.sqrt(2*g*(hm-h))
    if np.abs(h)<1e-3: 
        global vel_rev 
        vel_rev +=1
    return v*(-1)**vel_rev

res_fl = solve_ivp(odefunc_fall, (0,.5), (hm*0.9,), max_step=0.001, method='Radau')
# %%
plt.plot(res_fl.t,res_fl.y[0],)


#%%
s = 2
fig4, (ax4,ax41) = plt.subplots(2, dpi=120, figsize=(8,9))
# fig5, (ax5,ax6) = plt.subplots(1,2, dpi=200, figsize=(14,7))

t_now = time()

de = 2* (1+s/3) /3

def M0(l):
    return l**(3/4)
def M_pred(l):
    gam = 1 if s >= 3/2 else 3*s/(s+3)
    return l**gam

# def M(l,n):
#     return 0

M_func = M_pred

color_this = plt.cm.jet(s/2)
# # linestyles = [":","-.","--","-"]
# linestyles = [(0, (1, 3)), (0, (2, 3)), (0, (3, 3)), (0, (4, 3)), (0, (5, 1,1,1)), 'solid']
# ls_cycler = cycle(linestyles[::])

t_bef, t_now = t_now, time()
print(f'{t_now-t_bef:.4g}s', f's={s} Initialised vals and funcs for iteration')

def ode_func(xi, arg):
    lam = arg[0]
    v = arg[1]
    # print(lam, (v, -2/9 * M(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v + 1e-50/lam**10))
    # if lam<1e-5: v=-v
    a_grav = -2/9 * (3*np.pi/4)**2* M_func(np.abs(lam))/(lam**2+1e-6) * np.sign(lam)
    a_shm = - de*(de-1)*lam
    a_drag = - (2*de-1)*v
    return (v, a_grav + a_shm + a_drag )# *np.sign(lam)/np.sign(v))


res = solve_ivp(ode_func, (0,10), np.array([1,-de]), method='Radau', t_eval=np.linspace(0,1000,500000)**(1/3), max_step=np.inf, dense_output=False, vectorized=True) #np.unique(np.concatenate([np.linspace(0,2,50000),np.log10(np.linspace(1,10000,100000))]))
# res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

xi = res.t
lam = res.y[0]
v = res.y[1]
loglam = np.log(np.maximum(lam,1e-15))
tau = np.exp(xi)
lamF = lam*tau**de

ls='-'

ac = -2/9 * (3*np.pi/4)**2* M_func(np.abs(lam))/lam**2 * np.sign(lam) - de*(de-1)*lam - (2*de-1)*v


ax4.plot(xi,np.abs(lam), color=color_this, ls=ls, lw=1)
ax41.plot(xi,v, color=color_this, ls=ls, lw=1)
# plt.plot(xi,lamF, color=color_this, label=f's={s}')

ax4.set_yscale('log')
# %%
# plt.plot(xi,lam)
# %%

#%%
plt.figure(figsize=(9,7))
plt.plot(xi, -2/9 * (3*np.pi/4)**2* M_func(np.abs(lam))/(lam**2+1e-6) * np.sign(lam) - de*(de-1)*lam - (2*de-1)*v, label='accel_total')
plt.plot(xi, -2/9 * (3*np.pi/4)**2* M_func(np.abs(lam))/(lam**2+1e-6) * np.sign(lam), label='accel_mass')
plt.plot(xi, - de*(de-1)*lam, label='accel_shm')
plt.plot(xi, - (2*de-1)*v, label='accel_drag')
# plt.plot(xi, v*10, label='vel')
# plt.plot(xi,lam*10, label='lam')
plt.legend()
plt.show()
# %%
