#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
from scipy.optimize import fsolve, bisect, minimize_scalar
from time import time
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


#%%
def odefunc_prof_init_Pless(lam, depvars):
    # lam = 1/laminv
    # lam = np.exp(l)
    V,D,M = depvars
    Vb = V - de*lam
    # linb = -np.array([2*D*V-2*D*lam, (de-1)*V*lam+2/9*M/lam, -3*lam**3*D])/lam
    # # der_num = np.transpose(np.linalg.solve(linMat1,linb1), (1,0))
    # linMat_det1 = Vb**2
    # # if linMat_det1 == 0: print(depvars)
    # linMat_cofac1 = np.array([[0,Vb,0],[Vb,-D,0],[0,0,linMat_det1]])
    # linMat_inv = linMat_cofac1/ linMat_det1
    # der = np.matmul(linMat_inv,linb)
    Fterm = (de-1)*V + 2/9*M/lam**2
    der = np.array([-Fterm/Vb, (2*Vb*(lam-V)/lam + Fterm) *D/Vb**2, 3*lam**2*D])
    return der #*lam**2

# tht_ran = np.linspace(np.pi,1.5*np.pi)
# res = solve_ivp(odefunc_prof_init_Pless, (1,0.5), preshock(np.pi)[1:], max_step=0.01 )
# plt.plot(res.t,res.y[2])
# plt.plot(preshock(tht_ran)[0], preshock(tht_ran)[3])


# %%
def to_btilde(lam, V,D,M,P):
    Vb = V - de*lam
    Dt, Mt, Pt = D*lam**-aD, M*lam**-aM, P*lam**-aP
    return -Vb, Dt, Mt, Pt

def from_btilde(lam, mVb,Dt,Mt,Pt):
    V = -mVb + de*lam
    D,M,P = Dt*lam**aD, Mt*lam**aM, Pt*lam**aP
    return V,D,M,P

def stop_event(t,y):
    return y[2]+10 #+de*np.exp(t)
stop_event.terminal = True

def odefunc_tilde_full(l, depvars):
    lam = np.exp(l)
    # lmV,lDt,lMt,lPt = depvars
    mVb,Dt,Mt,Pt = np.exp(depvars)
    Vb = -mVb
    V = Vb + de*lam
    # V,D,M,P = from_btilde(lam, mVb,Dt,Mt,Pt)
    Z0 = 0*V
    ar1 = V/V
    # linb1 = -np.array([2*D*V-2*D*lam, (de-1)*V*lam+2/9*M/lam, (2*(gam-1)+2*(de-1))*lam])
    # linb2 = -np.array([Vb*aD*D, aP*P/D, -Vb*(aD*gam-aP)])
    # linb = linb1 +linb2
    # linMat_det1 = D*Vb**2-gam*P
    # # if linMat_det1 == 0: print(depvars)
    # # linMat_cofac1 = np.array([[-gam*P/D,D*Vb,-P,0],[Dt*Vb,-Dt*D,Dt*P/Vb,0],[0,0,0,lam**(-aM)*linMat_det1],[gam*Pt*Vb,-gam*D*Pt,D*Pt*Vb,0]])
    # linMat_cofac1 = np.array([[-gam*P/(D*Vb),D,-P/Vb],[Vb,-D,P/Vb],[gam*Vb,-gam*D,D*Vb]])
    # linMat_inv = linMat_cofac1/ linMat_det1

    Tv = Pt/Dt*lam**(aP-aD) /Vb**2
    linMat_inv = 1/Vb**2/(gam*Tv-1) * np.array([[-gam*Tv, ar1, -Tv],[ar1,-ar1,Tv],[gam*ar1,-gam*ar1,ar1]])
    linb = np.array([2*Vb* (V-lam), (de-1)*V*lam+2/9*Mt*lam**(aM-1), 2*Vb*lam*((gam-1)+(de-1))])

    # print(linMat_inv.shape,linb[:,np.newaxis].transpose((2,0,1)).shape)
    linc = np.array([de/Vb*lam,aD+Z0,aP+Z0])
    if np.isscalar(V):
        der = np.matmul(linMat_inv, linb ) - linc
        # der = np.matmul(linMat_inv, linb )+ np.array([-de/Vb*lam,Z0,Z0])
        # if der[0]<0:
            # print(der, linMat_det1, linb, linMat_cofac1)
    else:
        der = np.matmul(linMat_inv.transpose((2,0,1)), linb[:,np.newaxis].transpose((2,0,1)) ) - linc[:,np.newaxis].transpose((2,0,1))
        der = der.transpose((1,2,0))[:,0,:]

    derM = 3*Dt*lam**(aD+3-aM) /Mt - aM


    return der, derM, linMat_inv #, linb1, linMat_cofac1 #*lam**2

def odefunc_tilde(l, depvars):
    der3, derM = odefunc_tilde_full(l, depvars)[:2]
    der = np.insert(der3, 2, derM, axis=0)
    if not np.isfinite(der).all():
        print(der,l,depvars)
    return der


def get_soln_gas_full_tilde(lamsh):
    res_pre = solve_ivp(odefunc_prof_init_Pless, (1,lamsh), preshock(np.pi)[1:], max_step=0.01 )
    V1, D1, M1 = res_pre.y[0][-1], res_pre.y[1][-1], res_pre.y[2][-1]
    bcs = shock_jump(lamsh, V1, D1, M1) #get_shock_bcs(thtsh_sols[s])[1] #
    # print(bcs)
    bcs = to_btilde(lamsh, *bcs)
    # print(bcs)
    bcs = np.log(bcs)
    res_post = solve_ivp(odefunc_tilde, (np.log(lamsh),np.log(1e-7)), bcs, events=stop_event, method='Radau', max_step=0.05, vectorized=True)
    return res_pre, res_post


def M0_num_tilde(lamsh):
    res = get_soln_gas_full_tilde(lamsh)[1]
    M0val = res.y[2][-1] + (aM*res.t[-1])
    stopM0 = np.exp(M0val)-1e-3
    # stopM0 = res.y[2][-1] + 10
    return stopM0 #, np.exp(res.t[-1]), lamsh #if M0val>0 else -(-M0val)**(1/11)

def lam_atM0(lamsh):
    res = get_soln_gas_full_tilde(lamsh)[1]
    return res.t[-1]/np.log(10)+6.999


#%%

#%%
# thtshsol = fsolve(M0, 1.5*np.pi)
s = 1
gam = 4.5/3
s_vals = [0.5,1,1.5,2,3,5]

#%%
lamsh_sols = {}
lam_atM0_sols = {}
lambins = np.linspace(0.01, 0.5, 8)

# for s in s_vals[::]:
t_now = time()
de = 2* (1+s/3) /3
alpha_D = -9/(s+3)
aD, aP, aM = alpha_D, (2*alpha_D+2), alpha_D+3
aD, aP, aM = 0,0,0
print(s, aD, aP, aM)

lamshsol = bisect(lam_atM0, lambins[0], lambins[-1], xtol=1e-7)#+1e-5
# lamshsol = thetbins[idx_M0neg+1]
t_bef, t_now = t_now, time()
print(f'{t_now-t_bef:.4g}s', f's={s}: root thetsh obtained')
# lamshsol1 = minimize_scalar(M0, method='bounded', bounds=(1.5*np.pi, 1.9*np.pi))
lamsh_sols[s] = lamshsol
lam_atM0_sols[s] = lam_atM0(lamshsol)
print(f's={s}', lamshsol, lam_atM0_sols[s])

#%%

#%%
s = 1
fig4, (ax4,ax41) = plt.subplots(2, dpi=200, figsize=(10,12), sharex=True)
fig5, (ax5,ax6) = plt.subplots(1,2, dpi=200, figsize=(14,7))

t_now = time()
# for s in [0.5,1,1.5,2,3][::]:
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

for n in range(1):
    def ode_func(xi, arg):
        lam = arg[0]
        v = arg[1]
        # print(lam, (v, -2/9 * M(lam)/lam**2 - de*(de-1)*lam - (2*de-1)*v + 1e-50/lam**10))
        # if lam<1e-5: v=-v
        # try:
        der = (v, -2/9 * (3*np.pi/4)**2* M_func(lam)/(lam**2) - de*(de-1)*lam - (2*de-1)*v + 1e-5/lam**3) #- Pp_D(lam)
        # if der[1]-0.1:
        # print(der)# except:
        return der
        #     print(lam,s, v, xi)
        #     raise Exception

    xi_max = np.log(5e-4**upsil)*-3/2/s

    res = solve_ivp(ode_func, (0,xi_max), np.array([1,-de]), method='Radau', t_eval=np.linspace(0,xi_max**3,50000)**(1/3), max_step=np.inf, dense_output=False, vectorized=True) #np.unique(np.concatenate([np.linspace(0,2,50000),np.log10(np.linspace(1,10000,100000))]))
    # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

    xi = res.t
    lam = res.y[0] #np.abs(res.y[0])
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
            Int_rho = 2/9*s * np.exp((-2*s/3)*roots) / np.abs(v_xi(roots)) / l**2   #* (3*np.pi/4)**2
            rho_vals[i] = np.sum(Int_rho)
    M_vals[-1] = 1

    M_vals = np.asarray(M_vals)
    # M_vals_er = np.asarray(M_vals_er)
    if n==4: rho_vals = np.asarray(rho_vals)

    M_vals /= M_vals[-1]

    rho_vals_diff = np.diff(M_vals)/np.diff(l_range)/(3*l_range[1:]**2)

    M_func = interp1d(l_range, M_vals, assume_sorted=True, fill_value="extrapolate")
    D_func = interp1d(l_range[1:], rho_vals_diff, assume_sorted=True, fill_value="extrapolate")

    def Pp_D(lam):
        return D_func(lam)

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
        ax41.plot(xi[1:],np.diff(lam)/np.diff(xi), color=color_this, ls='-', lw=1)
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
        ax6.plot(l_range[1:], rho_vals_diff, color=color_this, ls=ls, lw=1)
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
# %%
