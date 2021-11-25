#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from itertools import cycle
plt.style.use('seaborn-darkgrid')
# %%


# %%

# %%
gam = 7/5



#%%
def v_iter(R,v):
    return cinf**2 * (1/(gam-1)+1/R) - Mdot/(4*np.pi*R**2*v) * k*gam/(gam-1)

#%%
cinf = 100
Mdot = 0.01
k = 100
R = np.logspace(-3,1)
v = R/R*0
plt.figure()
for Mdot in [0.001,0.01,0.1,1,10,100]:
    for n in range(11):
        v = v_iter(R, v)
        cs = Mdot/(4*np.pi*R**2*v) * k*gam
        # if n%10==0:
    plt.loglog(R,v/cs, label=f'n={n}, Md={Mdot}')
    plt.plot()
    
plt.legend()
# plt.yscale('linear')

# %%
# Mdotc = np.pi * (2/(5-3*gam))**((5-3*gam)/(2*(gam-1))) * 
# %%
