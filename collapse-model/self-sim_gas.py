#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
from scipy.optimize import fsolve, bisect, minimize_scalar
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
        # print(der)
        # print('lstsq',np.linalg.lstsq(linMat1[0],linb1[0])[0])
        # raise Exception
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
        # print(linMat, linb1)
        der = np.transpose(np.linalg.lstsq(linMat1[0],linb1[0])[0:1], (1,0))
        # print(linMat1)
        # print(linb1, der)
        # raise Exception
        return der #depvars*0


#%%
# thtsh = 4.58324+1

def get_shock_bcs(thtsh):
    lamsh, V1, D1, M1 = preshock(thtsh)
    return lamsh, shock_jump(lamsh, V1, D1, M1)

def get_soln(thtsh):
    lamsh, bcs = get_shock_bcs(thtsh)
    # print(thtsh)
    return solve_ivp(odefunc, (lamsh,1e-6), bcs, max_step=0.001, vectorized=True)
def M0(thtsh):
    res = get_soln(thtsh)
    M0val = res.y[2][-1]
    return M0val #if M0val>0 else -(-M0val)**(1/11)

#%%
def solve_bisect(func,bounds):
    b0, b1 = bounds
    bmid = (b0+b1)/2

def my_bisect(f, a, b, tol=3e-3): 
    # approximates a root, R, of f bounded 
    # by a and b to within tolerance 
    # | f(m) | < tol with m the midpoint 
    # between a and b Recursive implementation

    # get midpoint
    m = (a + b)/2

    sfa = np.sign(f(a))
    sfb = np.sign(f(b))
    f_at_m = f(m)
    sfm = np.sign(f_at_m)
    # check if a and b bound a root
    if sfa == sfb:
        raise Exception(
         "The scalars a and b do not bound a root")
        
    
    print(a,b,m,f_at_m)
    if np.abs(f_at_m) < tol:
        # stopping condition, report m as root
        return m if f_at_m >0 else (m+b)/2
    elif sfa == sfm:
        # case where m is an improvement on a. 
        # Make recursive call with a = m
        return my_bisect(f, m, b, tol)
    elif sfb == sfm:
        # case where m is an improvement on b. 
        # Make recursive call with b = m
        return my_bisect(f, a, m, tol)

#%%
# thtshsol = fsolve(M0, 1.5*np.pi)
s = 1
gam = 4/3
s_vals = [0.5,1,1.5,2,3,5]
fb = 0.2
# fig4, ax4 = plt.subplots(1, dpi=200, figsize=(10,7))
thtsh_sols = {}
for s in s_vals[::]:
    de = 2* (1+s/3) /3
    if s==3:
        thtshsol = my_bisect(M0, 1.75*np.pi, 1.9*np.pi)
    else:
        thtshsol = my_bisect(M0, 1.5*np.pi, 1.9*np.pi)
    # thtshsol1 = minimize_scalar(M0, method='bounded', bounds=(1.5*np.pi, 1.9*np.pi))
    thtsh_sols[s] = thtshsol
    print(s,thtshsol,M0(thtshsol))

#%%


fig5, axs5 = plt.subplots(2,2, dpi=200, figsize=(12,10), sharex=True)
fig6, ax6 = plt.subplots(1)

for s in s_vals[::]:
    de = 2* (1+s/3) /3
    thtshsol = thtsh_sols[s]
    res = get_soln(thtshsol)

    lamsh_post = res.t
    V_post, D_post, M_post, P_post = res.y

    thtsh_preange = np.arange(1.1*np.pi, thtshsol,0.01)

    lamsh_pre, V_pre, D_pre, M_pre = preshock(thtsh_preange)
    P_pre = lamsh_pre*0

    lamsh = lamsh_pre.min()

    lam_all = np.concatenate([lamsh_post, lamsh_pre][::-1])
    V_all = np.concatenate([V_post, V_pre][::-1])
    D_all = np.concatenate([D_post, D_pre][::-1])
    M_all = np.concatenate([M_post, M_pre][::-1])
    P_all = np.concatenate([P_post, P_pre][::-1])

    color_this = plt.cm.turbo(s/4)

    axs5[0,0].plot(lam_all,-V_all, color=color_this, label=f's={s}')
    axs5[0,1].plot(lam_all,D_all, color=color_this)
    axs5[1,0].plot(lam_all,M_all, color=color_this)
    axs5[1,1].plot(lam_all,P_all, color=color_this)


    PderD_post = np.gradient(P_post,lamsh_post)/D_post

    M_intrp = interp1d(lam_all, M_all, fill_value="extrapolate")
    D_intrp = interp1d(lam_all, D_all, fill_value="extrapolate")
    V_intrp = interp1d(lam_all, V_all, fill_value="extrapolate")
    irem = P_pre.shape[0]-1
    # PderD_intrp = interp1d(np.delete(lam_all,irem), np.delete(PderD_all,irem), kind='linear', fill_value="extrapolate")

    PderD_intrp = interp1d(lamsh_post, PderD_post, kind='linear', fill_value=0, bounds_error=False)

    def odefunc_traj(xi, arg):
        lam = arg[0]
        v = arg[1]
        # print(lam, (v, -2/9 * M(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v + 1e-50/lam**10))
        # if lam<1e-5: v=-v
        # if v>0: print(lam, v, V_intrp(lam)-de*lam)
        try:
            return (V_intrp(lam)-de*lam, -2/9 * M_intrp(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v - PderD_intrp(lam))
        except:
            print(lam,s, v, xi, V_intrp(lam))
            raise Exception

    lamshsol, bcs = get_shock_bcs(thtshsol)
    taush = (thtshsol - np.sin(thtshsol)) / np.pi
    xish = np.log(taush)
    res = solve_ivp(odefunc_traj, (xish,2.2), np.array([lamshsol,bcs[0]-de*lamshsol]), method='RK45', max_step=0.001, dense_output=False, vectorized=True)
    # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

    xires = res.t
    lamres = res.y[0]
    vres = res.y[1]

    taures = np.exp(xires)
    lamFres = lamres*taures**de

    ax6.plot(taures,lamFres, color=color_this, label=f's={s}')
    # ax6.plot(xires,lamres, color=color_this)

    #trajectory analytical
    thet_range = np.linspace(0.5, thtshsol,2000)
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

ax6.legend(loc='lower left')
ax6.set_xlabel(r'$\tau$')
ax6.set_ylabel('$\lambda_F$')
ax6.set_xlim(-1,5)
ax6.set_ylim(0,1.1)
    
axs5[0,0].set_xscale('log')
axs5[0,0].set_xlim(1e-4,1)
axs5[0,0].legend()
axs5[1,0].set_xlabel('$\lambda$')
axs5[1,1].set_xlabel('$\lambda$')

if gam>1.66:
    axs5[0,0].set_xlim(1e-2,1)
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

fig5.savefig(f'Eds-gas-{gam:.02f}_profiles.pdf')
fig6.savefig(f'Eds-gas-{gam:.02f}_trajectory.pdf')




#%%


# axs5[0,0].
# axs5[0,1].
# axs5[1,0].
# axs5[1,1].

# axx5.


#%%













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
# lamsh = 3.2e-1
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
