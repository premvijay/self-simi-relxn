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
def odefunc(l, depvars):
    # lam = 1/laminv
    lam = np.exp(l)
    V,D,M,P = depvars
    Vb = V - de*lam
    # try:
    #     veclen = len(V)
    # except:
    #     veclen = 1
    # linMat = np.array([[D, Vb, 0*V, 0*V], [Vb, 0*V, 0*V, 1/D], [0*V, -Vb*gam/D, 0*V, Vb/P], [0*V, 0*V, 1+V-V, 0*V]])
    linb = -np.array([2*D*V-2*D*lam, (de-1)*V*lam+2/9*M/lam, (2*(gam-1)+2*(de-1))*lam, -3*lam**3*D])
    # try:
    #     linMat1 = np.transpose(linMat,(2,0,1)) #linMat.reshape(veclen,4,4)
    #     linb1 = np.transpose(linb, (1,0)) #linb.reshape(veclen,4)
    # except:
    #     print(linb.shape)
    if True:
        # der_num = np.transpose(np.linalg.solve(linMat1,linb1), (1,0))
        linMat_det1 = D*Vb**2-gam*P
        # if linMat_det1 == 0: print(depvars)
        linMat_cofac1 = np.array([[-gam*P/D,D*Vb,-P,0],[D*Vb,-D**2,D*P/Vb,0],[0,0,0,linMat_det1],[gam*P*Vb,-gam*D*P,D*P*Vb,0]])
        linMat_inv = linMat_cofac1/ linMat_det1
        # print(linMat_inv.shape,linb1.shape, linMat.shape, linb.shape)
        # print(linMat_inv, np.linalg.inv(linMat[:,:,0]))
        der = np.matmul(linMat_inv,linb)
        # print(der[0][0][0],der_num[0][0])
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
    # except:
    #     # print(linMat, linb1)
    #     der = np.transpose(np.linalg.lstsq(linMat1[0],linb1[0])[0:1], (1,0))
    #     # print(linMat1)
    #     # print(linb1, der)
    #     # raise Exception
    #     return der #depvars*0

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


#%%
# thtsh = 4.58324+1

def get_shock_bcs(thtsh):
    lamsh, V1, D1, M1 = preshock(thtsh)
    return lamsh, shock_jump(lamsh, V1, D1, M1)

def get_soln(thtsh):
    lamsh, bcs = get_shock_bcs(thtsh)
    # print(thtsh)#(lamsh, 1e-9) #np.log(lamsh),np.log(1e-9))
    return solve_ivp(odefunc, (np.log(lamsh),np.log(1e-9)), bcs, method='Radau', max_step=np.inf, vectorized=False)
def M0(thtsh):
    res = get_soln(thtsh)
    lamsh_post = np.exp(res.t)
    V_post, D_post, M_post, P_post = res.y
    M0_expected = M_post[0]*(lamsh_post[-1]/lamsh_post[0])**(alpha_D+3)
    M0val = res.y[2][-1]
    return M0val-M0_expected #3e-3 #if M0val>0 else -(-M0val)**(1/11)

def P0(thtsh):
    res = get_soln(thtsh)
    return res.y[3][-1]
    # return M0val-M0_expected #3e-3 #if M0val>0 else -(-M0val)**(1/11)

#%%
def get_soln_gas_full(lamsh):
    res_pre = solve_ivp(odefunc_prof_init_Pless, (1,lamsh), preshock(np.pi)[1:], max_step=0.01 )
    V1, D1, M1 = res_pre.y[0][-1], res_pre.y[1][-1], res_pre.y[2][-1]
    bcs = shock_jump(lamsh, V1, D1, M1) #get_shock_bcs(thtsh_sols[s])[1] #
    res_post = solve_ivp(odefunc, (np.log(lamsh),np.log(1e-12)), bcs, method='Radau', max_step=np.inf, vectorized=False)
    return res_pre, res_post

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

def M0_num(lamsh):
    res = get_soln_gas_full(lamsh)[1]
    M0val = res.y[2][-1]
    return M0val-3e-4 #if M0val>0 else -(-M0val)**(1/11)

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
def solve_bisect(func,bounds):
    b0, b1 = bounds
    bmid = (b0+b1)/2

def my_bisect(f, a, b, xtol=1e-4): 
    # approximates a root, R, of f bounded 
    # by a and b to within tolerance 
    # | f(m) | < tol with m the midpoint 
    # between a and b Recursive implementation

    # get midpoint
    m = (a + b)/2

    sfa = -1
    sfb = +1
    f_at_m = f(m)
    sfm = np.sign(f_at_m)

    # print(a,b,m,f_at_m)
    if abs(b-a) < xtol:
        # stopping condition, report m as root
        return m if f_at_m >0 else b
    elif sfa == sfm:
        # case where m is an improvement on a. 
        # Make recursive call with a = m
        return my_bisect(f, m, b, xtol)
    elif sfb == sfm:
        # case where m is an improvement on b. 
        # Make recursive call with b = m
        return my_bisect(f, a, m, xtol)

#%%
# thtshsol = fsolve(M0, 1.5*np.pi)
s = 1
gam = 5/3
s_vals = [0.5,1,1.5,2,3,5]

#%%
fig4, ax4 = plt.subplots(1, dpi=120, figsize=(10,7))
lamsh_sols = {}
# lambins = np.linspace(1.2*np.pi, 1.99*np.pi, 8)
M0_atbins = {}
M0_sols = {}

for s in s_vals[::]:
    t_now = time()
    de = 2* (1+s/3) /3
    alpha_D = -9/(s+3)
    aD, aP, aM = alpha_D, (2*alpha_D+2), alpha_D+3
    print(s, aD, aP, aM)
    lambins = np.linspace(0.7, 0.01, 8)
    for nsect_i in range(0,4):
        M0_atbins[s] = list(map(M0_num_tilde,lambins))
        t_bef, t_now = t_now, time()
        print(f'{t_now-t_bef:.4g}s', f's={s}: grid M0 obtained')
        ax4.plot(lambins,M0_atbins[s], label=f's={s} and nsect={nsect_i}')
        idx_M0neg = np.where(np.sign(M0_atbins[s])==-1)[0].max()
        t_bef, t_now = t_now, time()
        print(f'{t_now-t_bef:.4g}s', f's={s}: grid M0 selected')
        lamshsol = lambins[idx_M0neg+1]
        lambins = np.linspace(lambins[idx_M0neg], max(2*lamshsol-lambins[idx_M0neg],0.01), 8)

    lamshsol = my_bisect(M0_num_tilde, lambins[0], lambins[-1], xtol=1e-6)#+1e-5
    # lamshsol = thetbins[idx_M0neg+1]
    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: root thetsh obtained')
    # lamshsol1 = minimize_scalar(M0, method='bounded', bounds=(1.5*np.pi, 1.9*np.pi))
    lamsh_sols[s] = lamshsol
    M0_sols[s] = M0_num_tilde(lamshsol)
    ax4.scatter(lamshsol, M0_sols[s])
    print(f's={s}', lamshsol, M0_sols[s])
ax4.set_xlabel(r'$\lambda_{sh}}$')
ax4.set_ylabel(r'$M(\lambda=0)$')
ax4.set_ylim(-2,5)
ax4.legend()

#%%
lamsh_sols = {}
lam_atM0_sols = {}
lambins = np.linspace(0.01, 0.5, 8)

for s in s_vals[::]:
    t_now = time()
    de = 2* (1+s/3) /3
    alpha_D = -9/(s+3)
    aD, aP, aM = alpha_D, (2*alpha_D+2), alpha_D+3
    # aD, aP, aM = 0,0,0
    print(s, aD, aP, aM)

    lamshsol = my_bisect(lam_atM0, lambins[0], lambins[-1], xtol=1e-8)#+1e-5
    # lamshsol = thetbins[idx_M0neg+1]
    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: root thetsh obtained')
    # lamshsol1 = minimize_scalar(M0, method='bounded', bounds=(1.5*np.pi, 1.9*np.pi))
    lamsh_sols[s] = lamshsol
    lam_atM0_sols[s] = lam_atM0(lamshsol)
    print(f's={s}', lamshsol, lam_atM0_sols[s])


#%%
fig5, axs5 = plt.subplots(2,3, dpi=100, figsize=(18,12), sharex=True)
fig6, (ax62,ax6) = plt.subplots(1,2, dpi=100, figsize=(10,5))

for s in s_vals[::]:
    t_now = time()
    de = 2* (1+s/3) /3
    alpha_D = -9/(s+3)
    aD, aP, aM = alpha_D, 1*(2*alpha_D+2), alpha_D+3
    # aD, aP, aM = 0,0,0
    lamshsol = lamsh_sols[s] #-1e-8 # 0.338976 #
    res_pre, res_post = get_soln_gas_full_tilde(lamshsol)
    print(res_post.y[2][-1])
    # print(M0(lamshsol))
    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: post shock profiles obtained')

    lamsh_pre = res_pre.t
    V_pre, D_pre, M_pre = res_pre.y

    lamsh_post = np.exp(res_post.t)
    mVb_post, Dt_post, Mt_post, Pt_post = np.exp(res_post.y)
    # V_post = -mVb_post + de*lamsh_post
    V_post, D_post, M_post, P_post = from_btilde(lamsh_post, mVb_post, Dt_post, Mt_post, Pt_post) #Dt_post*lamsh_post**aD, Mt_post*lamsh_post**aM, Pt_post*lamsh_post**aP
    # lamsh_preange = np.arange(1.1*np.pi, lamshsol,0.01)

    # lamsh_pre, V_pre, D_pre, M_pre = preshock(thtsh_preange)
    P_pre = lamsh_pre*0

    lamsh = lamsh_pre.min()

    lam_all = np.concatenate([lamsh_post, lamsh_pre][::-1])
    V_all = np.concatenate([V_post, V_pre][::-1])
    D_all = np.concatenate([D_post, D_pre][::-1])
    M_all = np.concatenate([M_post, M_pre][::-1])
    P_all = np.concatenate([P_post, P_pre][::-1])
    Vb_all = V_all - de*lam_all

    resdf_gas = pd.DataFrame(data={'l':lam_all, 'M':M_all, 'V':V_all, 'D':D_all, 'P':P_all, 'Vb':Vb_all,})
    descr = f'_s={s:.2g}_gam={gam:.3g}'
    resdf_gas.to_hdf(f'profiles_gaso_bertshi{descr:s}.hdf5', 'gas/main', mode='a')

    color_this = plt.cm.turbo(s/4)

    axs5[0,0].plot(lam_all,-V_all, color=color_this, label=f's={s}')
    axs5[0,1].plot(lam_all,D_all, color=color_this)
    axs5[1,0].plot(lam_all,M_all, color=color_this)
    axs5[1,1].plot(lam_all,P_all, color=color_this)
    axs5[0,2].plot(lam_all, P_all/D_all, color=color_this)
    axs5[1,2].plot(lam_all, P_all/D_all**gam, color=color_this)
    # axs5[1,2].plot(lam_all, D_all*Vb_all**2-gam*P_all, color=color_this)


    PderD_post = np.gradient(P_post,lamsh_post)/D_post

    M_intrp = interp1d(lam_all, M_all, fill_value="extrapolate")
    D_intrp = interp1d(lam_all, D_all, fill_value="extrapolate")
    V_intrp = interp1d(lam_all, V_all, fill_value="extrapolate")
    irem = P_pre.shape[0]-1
    # PderD_intrp = interp1d(np.delete(lam_all,irem), np.delete(PderD_all,irem), kind='linear', fill_value="extrapolate")

    PderD_intrp = interp1d(lamsh_post, PderD_post, kind='linear', fill_value=0, bounds_error=False)

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: all profiles obtained')

    def odefunc_traj(xi, arg):
        lam = arg
        return V_intrp(lam)-de*lam

    # V1, D1, M1 = res_pre.y[0][-1], res_pre.y[1][-1], res_pre.y[2][-1]
    # bcs = shock_jump(lamshsol, V1, D1, M1)
    # taush = (thtshsol - np.sin(thtshsol)) / np.pi
    # xish = np.log(taush)
    res = solve_ivp(odefunc_traj, (0,2.2), (1,), method='RK45', max_step=0.01, dense_output=False, vectorized=True)
    # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: post shock trajectory obtained')
    
    xires = res.t
    lamres = res.y[0]
    # vres = res.y[1]

    taures = np.exp(xires)
    lamFres = lamres*taures**de

    ax6.plot(taures,lamFres, color=color_this, label=f's={s}')
    ax62.plot(xires,lamres, color=color_this)

    #trajectory analytical
    thet_range = np.linspace(0.5, 1.2*np.pi,2000)
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
ax6.set_ylim(-0.01,1.1)
# ax6.set_yscale('log')

ax62.set_xlabel(r'$\xi$')
ax62.set_ylabel('$\lambda$')
# ax62.set_xlim(,)
# ax62.set_ylim(0.01,1.1)
ax62.set_yscale('log')
    
axs5[0,0].set_xscale('log')
axs5[0,0].set_xlim(1e-5,1)
axs5[0,0].legend()
axs5[1,0].set_xlabel('$\lambda$')
axs5[1,1].set_xlabel('$\lambda$')

if gam==5/3:
    axs5[0,0].set_xlim(9e-5,1)
    axs5[0,0].set_ylim(5e-6,1e1)
    axs5[0,1].set_ylim(1e-1,1e11)
    axs5[1,0].set_ylim(1e-3,1e1)
    axs5[1,1].set_ylim(1e0,1e14)
    axs5[0,2].set_ylim(1e-1,1e2)
    axs5[1,2].set_ylim(1e-5,5e-1)
elif gam==4/3:
    axs5[0,0].set_xlim(1e-5,1)
    axs5[0,0].set_ylim(5e-6,1e1)
    axs5[0,1].set_ylim(1e0,1e11)
    axs5[1,0].set_ylim(1e-2,1e1)
    axs5[1,1].set_ylim(1e1,1e14)
    axs5[0,2].set_ylim(1e0,1e3)
    axs5[1,2].set_ylim(1e-2,5e-1)


axs5[0,0].set_ylabel('-V')
axs5[0,1].set_ylabel('D')
axs5[1,0].set_ylabel('M')
axs5[1,1].set_ylabel('P')
axs5[0,2].set_ylabel('T')
axs5[1,2].set_ylabel('K')

axs5[0,0].set_yscale('log')
axs5[0,1].set_yscale('log')
axs5[1,0].set_yscale('log')
axs5[1,1].set_yscale('log')
axs5[0,2].set_yscale('log')
axs5[1,2].set_yscale('log')

fig5.savefig(f'Eds-gas-{gam:.02f}_profiles.pdf')
fig6.savefig(f'Eds-gas-{gam:.02f}_trajectory.pdf')
axs5[0,0].set_xlim(1e-6,1)
# axs5[1,0].set_ylim(1e-4,1e1)



#%%
fig7,(ax71, ax72, ax73) = plt.subplots(3, dpi=120, figsize=(7,10), sharex=True)
ax71.plot(lamsh_post, res_post.y[0])
ax72.plot(lamsh_post, odefunc_tilde(np.log(lamsh_post),res_post.y)[0])
ax72.plot(lamsh_post[1:], np.diff(res_post.y[0])/np.diff(np.log(lamsh_post)))

ax73.plot(lamsh_post, odefunc_tilde_full(lamsh_post,res_post.y)[2][0,1])

ax71.set_xscale('log')
#%%
plt.loglog(lamsh_post, odefunc_tilde_full(np.log(lamsh_post),res_post.y)[2][0,0])
plt.loglog(lamsh_pre, odefunc_tilde_full(np.log(lamsh_pre),np.log(to_btilde(lamsh_pre,*shock_jump(lamsh_pre, *res_pre.y))))[2][0,0])
#%%




#%%
ts = np.linspace(.25,5,30)
rs = ts**de

rs = np.logspace(-1,1,50)
ts = rs**(1/de)

r = np.outer(lamFres,rs)
t = np.outer(taures,ts)

r_anlt = np.outer(lamF_anlt,rs)
t_anlt = np.outer(tau_anlt,ts)

plt.plot(t,r, lw=1)
plt.plot(t_anlt,r_anlt, lw=1)


plt.grid(visible=True,axis='y', which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()

plt.xlim(0,10)
plt.ylim(3e-2,1e1)
plt.yscale('log')
plt.ylabel('r')
plt.xlabel('t')


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

#%%
# thtshsol = fsolve(M0, 1.5*np.pi)
s = 1
gam = 5/3
s_vals = [0.5,1,1.5,2,3,5]
fb = 0.2
fig4, ax4 = plt.subplots(1, dpi=120, figsize=(10,7))
thtsh_sols = {}
lamsh_sols = {}
thetbins = np.linspace(1.2*np.pi, 1.99*np.pi, 8)
M0_atbins = {}
P0_atbins = {}
M0_sols = {}

for s in s_vals[::]:
    t_now = time()
    de = 2* (1+s/3) /3
    alpha_D = -9/(s+3)
    thetbins = np.linspace(1.2*np.pi, 1.99*np.pi, 8)
    for nsect_i in range(0,3):
        M0_atbins[s] = list(map(M0,thetbins))
        P0_atbins[s] = list(map(P0,thetbins))
        t_bef, t_now = t_now, time()
        print(f'{t_now-t_bef:.4g}s', f's={s}: grid M0 obtained')
        ax4.plot(thetbins,M0_atbins[s], label=f's={s} and nsect={nsect_i}')
        ax4.plot(thetbins,P0_atbins[s], label=f'P0 s={s} and nsect={nsect_i}')
        idx_M0neg = np.where(np.sign(M0_atbins[s])==-1)[0].max()
        t_bef, t_now = t_now, time()
        print(f'{t_now-t_bef:.4g}s', f's={s}: grid M0 selected')
        thtshsol = thetbins[idx_M0neg+1]
        thetbins = np.linspace(thetbins[idx_M0neg], 2*thtshsol-thetbins[idx_M0neg], 8)

    thtshsol = my_bisect(M0, thetbins[0], thetbins[-1], tol=1e-4)#+1e-5
    # thtshsol = thetbins[idx_M0neg+1]
    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: root thetsh obtained')
    # thtshsol1 = minimize_scalar(M0, method='bounded', bounds=(1.5*np.pi, 1.9*np.pi))
    thtsh_sols[s] = thtshsol
    M0_sols[s] = M0(thtshsol)
    ax4.scatter(thtshsol, M0_sols[s])
    print(f's={s}', thtshsol, M0_sols[s])
ax4.set_xlabel(r'$\theta$')
ax4.set_ylabel(r'$M(\lambda=0)$')
# ax4.set_ylim(-2,5)
ax4.legend()

#%%


fig5, axs5 = plt.subplots(2,2, dpi=100, figsize=(12,10), sharex=True)
fig6, (ax6,ax62) = plt.subplots(1,2, dpi=100, figsize=(10,5))

for s in s_vals[::]:
    t_now = time()
    de = 2* (1+s/3) /3
    alpha_D = -9/(s+3)
    thtshsol = thtsh_sols[s]
    res = get_soln(thtshsol)
    print(res.y[2][-1])
    # print(M0(thtshsol))
    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: post shock profiles obtained')

    lamsh_post = np.exp(res.t)
    V_post, D_post, M_post, P_post = res.y

    thtsh_preange = np.arange(1.1*np.pi, thtshsol,0.01)

    lamsh_pre, V_pre, D_pre, M_pre = preshock(thtsh_preange)
    P_pre = lamsh_pre*0

    lamsh = lamsh_pre.min()
    lamsh_sols[s] = lamsh

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

    axs5[1,0].plot(lamsh_post, M_post[0]*(lamsh_post/lamsh_post[0])**(alpha_D+3), color=color_this)

    PderD_post = np.gradient(P_post,lamsh_post)/D_post

    M_intrp = interp1d(lam_all, M_all, fill_value="extrapolate")
    D_intrp = interp1d(lam_all, D_all, fill_value="extrapolate")
    V_intrp = interp1d(lam_all, V_all, fill_value="extrapolate")
    irem = P_pre.shape[0]-1
    # PderD_intrp = interp1d(np.delete(lam_all,irem), np.delete(PderD_all,irem), kind='linear', fill_value="extrapolate")

    PderD_intrp = interp1d(lamsh_post, PderD_post, kind='linear', fill_value=0, bounds_error=False)

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: all profiles obtained')

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

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', f's={s}: post shock trajectory obtained')
    
    xires = res.t
    lamres = res.y[0]
    vres = res.y[1]

    taures = np.exp(xires)
    lamFres = lamres*taures**de

    ax6.plot(taures,lamFres, color=color_this, label=f's={s}')
    ax62.plot(xires,lamres, color=color_this)

    #trajectory analytical
    thet_range = np.linspace(np.pi, thtshsol,2000)
    tau_anlt = (thet_range - np.sin(thet_range)) / np.pi
    xi_anlt = np.log(tau_anlt)
    lam_anlt = preshock(thet_range)[0]
    lamF_anlt = lam_anlt*tau_anlt**de

    ax62.plot(xi_anlt, lam_anlt, color=color_this)


    

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

ax62.legend()
ax62.set_xlabel(r'$\xi$')
ax62.set_ylabel(r'$\lambda$')
# ax62.set_yscale('log')
ax62.set_ylim(0,1.1)
    
axs5[0,0].set_xscale('log')
axs5[0,0].set_xlim(1e-5,1)
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
axs5[0,0].set_xlim(1e-10,1)
axs5[1,0].set_ylim(1e-4,1e1)

# %%
