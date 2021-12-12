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
def odefunc(lam, depvars):
    # lam = 1/laminv
    V,D,M,P = depvars
    Vb = V - de*lam
    Mtot = M + M_dm(lam)
    try:
        veclen = len(V)
    except:
        veclen = 1
    linMat = np.array([[D, Vb, 0*V, 0*V], [Vb, 0*V, 0*V, 1/D], [0*V, -Vb*gam/D, 0*V, Vb/P], [0*V, 0*V, 1+V-V, 0*V]])
    linb = -np.array([2*D*V/lam-2*D, (de-1)*V+2/9*Mtot/lam**2, 2*(gam-1)+2*(de-1)+V-V, -3*lam**2*D])
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

def get_shock_bcs(thtsh):
    lamsh, V1, D1, M1 = preshock(thtsh)
    return lamsh, shock_jump(lamsh, V1, D1, M1)

def get_soln(thtsh):
    lamsh, bcs = get_shock_bcs(thtsh)
    # print(thtsh)
    return solve_ivp(odefunc, (lamsh,1e-3), bcs, max_step=0.001, vectorized=True)
def M0(thtsh):
    res = get_soln(thtsh)
    M0val = res.y[2][-1]
    return M0val-1e-2 #if M0val>0 else -(-M0val)**(1/11)







#%%
# thtshsol = fsolve(M0, 1.5*np.pi)
s = 2
gam = 5/3
fb = 0.156837
# fig4, ax4 = plt.subplots(1, dpi=200, figsize=(10,7))
thtsh_sols = {}
fig5, axs5 = plt.subplots(2,2, dpi=200, figsize=(14,12), sharex=True)
# fig6, ax6 = plt.subplots(1)

# for s in [1,1.5,2,3][:]:
dmo_prfl = pd.read_hdf(f'profiles_dmo_{s}.hdf5')

Mta = (3*np.pi/4)**2
M_dmo = interp1d(dmo_prfl['l'], dmo_prfl['M']*Mta, fill_value="extrapolate")
D_dmo = interp1d(dmo_prfl['l'].iloc[1:], dmo_prfl['rho'].iloc[1:], fill_value="extrapolate")

M_dm = lambda lam: M_dmo(lam)*(1-fb)

de = 2* (1+s/3) /3

plot_iters = [0,1,2,3,5,6,7]

for n in range(8):
    thtshsol = bisect(M0, 1.1*np.pi, 1.9*np.pi)
    thtsh_sols[s] = thtshsol

    # for s in [0.5,1,1.5,2,3,5][1:5]:
    de = 2* (1+s/3) /3
    thtshsol = thtsh_sols[s]
    res = get_soln(thtshsol)

    lamsh_post = res.t
    V_post, D_post, M_post, P_post = res.y

    thtsh_preange = np.arange(1*np.pi, thtshsol,0.01)

    lamsh_pre, V_pre, D_pre, M_pre = preshock(thtsh_preange)
    P_pre = lamsh_pre*0

    lamsh = lamsh_pre.min()

    lam_all = np.concatenate([lamsh_post, lamsh_pre][::-1])
    V_all = np.concatenate([V_post, V_pre][::-1])
    D_all = np.concatenate([D_post, D_pre][::-1])
    M_all = np.concatenate([M_post, M_pre][::-1])
    P_all = np.concatenate([P_post, P_pre][::-1])

    # color_this = plt.cm.turbo(s/2)
    color_this = plt.cm.turbo(n/7)

    if n in plot_iters:
        axs5[0,0].plot(lam_all,-V_all, color=color_this, label=f'n={n}')
        axs5[0,1].plot(lam_all,D_all, color=color_this)
        axs5[1,0].plot(lam_all,M_all+M_dm(lam_all), color=color_this)
        axs5[1,1].plot(lam_all,P_all, color=color_this)



    M_gas = interp1d(lam_all, M_all, fill_value="extrapolate")

    M_tot = lambda lam : M_dm(lam)+M_gas(lam)

    def ode_func(xi, arg):
        lam = arg[0]
        v = arg[1]
        # print(lam, (v, -2/9 * M(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v + 1e-50/lam**10))
        # if lam<1e-5: v=-v
        try:
            return (v, -2/9 * (3*np.pi/4)**2* M_tot(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v + 1e-40/lam**9)
        except:
            print(lam,s, v, xi)
            raise Exception


    res = solve_ivp(ode_func, (0,4), np.array([1,-de]), method='Radau', t_eval=(np.arange(0,256,0.005))**(1/4), max_step=0.001, dense_output=False, vectorized=True)
    # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

    xi = res.t
    lam = res.y[0]
    v = res.y[1]

    l_range = [0]
    M_vals = [0]
    for l in np.logspace(-2.5,0, 200):
        l_range.append(l)
        spl = InterpolatedUnivariateSpline(xi, lam-l)
        roots = spl.roots()
        Int_M = np.exp((-2*s/3)*roots)
        M_val = np.sum(Int_M[::2]) - np.sum(Int_M[1::2])
        M_vals.append(M_val)
    M_vals[-1] = 1

    M_vals = np.asarray(M_vals)
    # M_vals_er = np.asarray(M_vals_er)
    # rho_vals = np.asarray(rho_vals)

    M_vals *= Mta*(1-fb) / M_vals[-1]

    M_dm = interp1d(l_range, M_vals, fill_value="extrapolate")




axs5[1,0].plot(lam_all, M_dmo(lam_all), color='k', ls='dashed')

# s-Loop ends
    
axs5[0,0].set_xscale('log')
axs5[0,0].set_xlim(1e-4,1)
axs5[0,0].legend()

if gam>1.66:
    axs5[0,0].set_xlim(1e-2,1)
    axs5[0,1].set_ylim(1e-1,1e6)
    axs5[1,0].set_ylim(1e-2,1e1)
    axs5[1,1].set_ylim(1e0,1e7)

axs5[0,0].set_ylabel('-V')
axs5[0,1].set_ylabel('D')
axs5[1,0].set_ylabel('M')
axs5[1,1].set_ylabel('P')

# axs5[0,0].set_yscale('log')
axs5[0,1].set_yscale('log')
axs5[1,0].set_yscale('log')
axs5[1,1].set_yscale('log')

# fig5.savefig(f'Eds-gas-{gam:.02f}_profiles.pdf')
# fig5.savefig(f'Eds-gas-{gam:.02f}_trajectory.pdf')

#%%
import dill                            #pip install dill --user
filename = 'soln-globalsave-2.pkl'
dill.dump_session(filename)
#%%
plt.show()

#%%

# %%

# %%
import dill                            #pip install dill --user
filename = 'soln-globalsave-1.pkl'
dill.load_session(filename)

#%%
fd = (1-fb)
lamr_full = np.logspace(-3,-0.5,300)
lamr = np.logspace(-3,-0.5,300)

r, Mdr, Mbr, Mdr_dmo = lamr, M_dm(lamr), M_gas(lamr), M_dmo(lamr_full)*fd
ri_pre = lamr_full

#%%
plt.plot(r,Mdr, label='DM')
plt.plot(r,Mbr*fd/fb, label='baryon')
plt.plot(ri_pre,Mdr_dmo, label='DMO_scaled' )
# plt.plot(r,Mdr+Mbr)
# plt.xscale('log')
# plt.yscale('log')
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
plt.plot(MiMf[60:-50],rfri[60:-50])
plt.xlabel('Mi/Mf')
plt.ylabel('rf/ri')
plt.savefig('ratio_plot_anyl.pdf')
# %%
