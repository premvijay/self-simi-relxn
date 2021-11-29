#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
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
    # lam = 1/laminv
    V,D,M,P = depvars
    Vb = V - de*lam
    try:
        veclen = len(V)
    except:
        veclen = 1
    linMat = np.array([[D, Vb, 0*V, 0*V], [Vb, 0*V, 0*V, 1/D], [0*V, -Vb*gam/D, 0*V, Vb/P], [0*V, 0*V, 1+V-V, 0*V]])
    linb = -np.array([2*D*V/lam-2*D, (de-1)*V+2/9*M/lam**2, 2*(gam-1)+2*(de-1)+V-V, -3*lam**2*D])
    try:
        linMat1 = np.transpose(linMat,(2,0,1)) #linMat.reshape(veclen,4,4)
        linb1 = np.transpose(linb, (1,0)) #linb.reshape(veclen,4)
    except:
        print(linb.shape)
    try:
        der = np.transpose(np.linalg.solve(linMat1,linb1), (1,0))
        # if der[2]<0:
        #     print(der)
        #     print(linb1)
        #     print(linMat1)
        #     print(depvars)
        #     raise Exception
        # else:
            # print('good')

        return der #*lam**2
    except:
    #     # print(linMat)
    #     # raise Exception
        return depvars*0






# %%
lams = 3.2e-1
prfl = pd.read_hdf(f'profiles_dmo_{s}.hdf5')
# %%
prfl
# %%
Mta = (3*np.pi/4)**2
M_dmo = interp1d(prfl['l'], prfl['M']*Mta, fill_value="extrapolate")
D_dmo = interp1d(prfl['l'].iloc[1:], prfl['rho'].iloc[1:], fill_value="extrapolate")
# %%
M_dmo(lams)
# %%
D_dmo(lams)
# %%
def shock_jump(lams, V1, D1, M1):
    Vb1 = V1 - de*lams
    fac = (gam-1)/(gam+1)
    D2 = D1/fac
    V2 = Vb1*fac + de*lams
    P2 = D1 * Vb1**2 * (2/(gam+1))
    M2 = M1
    return(np.array([V2,D2,M2,P2]))
# %%
shock_jump(lams, -1.4, D_dmo(lams), M_dmo(lams))
# %%

# %% [-0.14,1.8e1,4e0,1e1]
lams = 3.3e-1
bcs = shock_jump(lams, -1.4, D_dmo(lams), M_dmo(lams))
res = solve_ivp(odefunc, (3.2e-1,1e-2), bcs, max_step=0.001, vectorized=True)
# %%
plt.plot(res.t, res.y[3])
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-2,1)
# %%
