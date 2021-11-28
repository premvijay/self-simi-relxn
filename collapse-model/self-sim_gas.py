#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
# %%

# %%

# %%
s = 1
gam = 5/3
fig4, ax4 = plt.subplots(1, dpi=200, figsize=(10,7))
fig5, (ax5,ax6) = plt.subplots(1,2, dpi=200, figsize=(14,7))

de = 2* (1+s/3) /3


# %%

# %%
def odefunc(lam, depvars):
    V,D,M,P = depvars
    Vb = V - de*lam
    try:
        veclen = len(V)
    except:
        veclen = 1
    linMat = np.array([[D, Vb, 0*V, 0*V], [Vb, 0*V, 0*V, 1/D], [0*V, -Vb*gam/D, 0*V, Vb/P], [0*V, 0*V, 1*V, 0*V]])
    linb = -np.array([2*D*V/lam-2*D, (de-1)*V+2/9*M/lam**2, 2*(gam-1)+2*(de-1)+V-V, -3*lam**2*D])
    try:
        linMat = np.transpose(linMat,(2,0,1)) #linMat.reshape(veclen,4,4)
        linb = np.transpose(linb, (1,0)) #linb.reshape(veclen,4)
    except:
        print(linb.shape)
    try:
        return np.transpose(np.linalg.solve(linMat,linb), (1,0))
    except:
        # print(linMat)
        # raise Exception
        return depvars*0




# %%
res = solve_ivp(odefunc, (3.2e-1,1e-2), [-0.14,1.8e1,4e-1,1e1], max_step=0.01, vectorized=True)
# %%
plt.plot(res.t, res.y[2])
plt.xscale('log')
plt.yscale('log')

# %%
