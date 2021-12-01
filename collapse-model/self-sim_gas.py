#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
from scipy.optimize import fsolve, bisect
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
    return(np.array([V2,D2,M2,P2]))

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
        #     # print('good')
        #     pass

        return der #*lam**2
    except:
    #     # print(linMat)
    #     # raise Exception
        return depvars*0


#%%
# thtsh = 4.58324+1

def get_soln(thtsh):
    lamsh, V1, D1, M1 = preshock(thtsh)
    bcs = shock_jump(lamsh, V1, D1, M1)
    # print(thtsh)
    return solve_ivp(odefunc, (lamsh,1e-4), bcs, max_step=0.001, vectorized=True)
def M0(thtsh):
    res = get_soln(thtsh)
    M0val = res.y[2][-1]
    return M0val-1e-1 #if M0val>0 else -(-M0val)**(1/11)

#%%
# thtshsol = fsolve(M0, 1.5*np.pi)
s = 1
gam = 5/3
# fig4, ax4 = plt.subplots(1, dpi=200, figsize=(10,7))
fig5, axs5 = plt.subplots(2,2, dpi=200, figsize=(14,12))

for s in [0.5,1,2,3,5]:
    de = 2* (1+s/3) /3
    thtshsol = bisect(M0, 1.1*np.pi, 1.9*np.pi)
    res = get_soln(thtshsol)

    axs5[0,0].loglog(res.t,-res.y[0], label=f's={s}')
    axs5[0,1].loglog(res.t,res.y[1])
    axs5[1,0].plot(res.t,res.y[2])
    axs5[1,1].loglog(res.t,res.y[3])

# %%
plt.plot(res.t, res.y[2])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1e-2,1)
# plt.ylim(1e0)



#%%
thtshs = np.linspace(4.4,2*np.pi-1.5,10)
M0s = [M0(thtsh) for thtsh in thtshs]
plt.plot(thtshs, M0s)


# %%
lamsh = 3.2e-1
prfl = pd.read_hdf(f'profiles_dmo_{s}.hdf5')
# %%
prfl
# %%
Mta = (3*np.pi/4)**2
M_dmo = interp1d(prfl['l'], prfl['M']*Mta, fill_value="extrapolate")
D_dmo = interp1d(prfl['l'].iloc[1:], prfl['rho'].iloc[1:], fill_value="extrapolate")
# %%
M_dmo(lamsh)
# %%
D_dmo(lamsh)
# %%

# %%
shock_jump(lamsh, -1.4, D_dmo(lamsh), M_dmo(lamsh))
# %%

# %% [-0.14,1.8e1,4e0,1e1]
lamsh = 3.38976e-1
bcs = shock_jump(lamsh, -1.47080, D_dmo(lamsh), M_dmo(lamsh))

# %%
