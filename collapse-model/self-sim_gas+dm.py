#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from itertools import cycle
plt.style.use('seaborn-darkgrid')
import pandas as pd
from scipy.optimize import fsolve, bisect
from time import time
from copy import copy
import dill   
# %%

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
plt.style.use('default')

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
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.minor.size'] = 3

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
    M = fgas* Mta * Mn
    lam = (1-np.cos(tht))/2 * Mn**(1/3+eps)
    # D = ( 3*np.pi/8 * (1-np.cos(tht)) * Mn**(3*eps/2-1) * np.sin(tht) * Mta**-2 )**-1
    D = fgas* 9*np.pi**2/ ( (2+6*eps) *(1-np.cos(tht))**3 - 9*eps*np.sin(tht) *(tht-np.sin(tht)) *(1-np.cos(tht)) ) * Mn**(-3*eps)
    # D_num = np.gradient(M,lam)/lam**2
    V = np.pi/2 * np.sin(tht)/(1-np.cos(tht)) * Mn**(1/3-eps/2)
    return lam,V,D,M



# %%
def odefunc(l, depvars):
    # lam = 1/laminv
    lam = np.exp(l)
    V,D,M,P = depvars
    Vb = V - de*lam
    linb = -np.array([2*D*V-2*D*lam, (de-1)*V*lam+2/9*(M+M_bg(lam))/lam+3e-4/lam**3, (2*(gam-1)+2*(de-1))*lam, -3*lam**3*D])
    # der_num = np.transpose(np.linalg.solve(linMat1,linb1), (1,0))
    linMat_det1 = D*Vb**2-gam*P
    # if linMat_det1 == 0: print(depvars)
    linMat_cofac1 = np.array([[-gam*P/D,D*Vb,-P,0],[D*Vb,-D**2,D*P/Vb,0],[0,0,0,linMat_det1],[gam*P*Vb,-gam*D*P,D*P*Vb,0]])
    linMat_inv = linMat_cofac1/ linMat_det1
    der = np.matmul(linMat_inv,linb)
    return der #*lam**2

#%%
def odefunc_prof_init_Pless(lam, depvars):
    # lam = 1/laminv
    # lam = np.exp(l)
    V,D,M = depvars
    Vb = V - de*lam
    linb = -np.array([2*D*V-2*D*lam, (de-1)*V*lam+2/9*(M+M_bg(lam))/lam, -3*lam**3*D])/lam
    # der_num = np.transpose(np.linalg.solve(linMat1,linb1), (1,0))
    linMat_det1 = D*Vb**2
    # if linMat_det1 == 0: print(depvars)
    linMat_cofac1 = np.array([[0,D*Vb,0],[D*Vb,-D**2,0],[0,0,linMat_det1]])
    linMat_inv = linMat_cofac1/ linMat_det1
    der = np.matmul(linMat_inv,linb)
    return der #*lam**2

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
    Fterm = (de-1)*V + 2/9*(M+M_bg(lam))/lam**2
    der = np.array([-Fterm/Vb, (2*Vb*(lam-V)/lam + Fterm) *D/Vb**2, 3*lam**2*D])
    # if n_i==0 and lam>0.99: print(der[0])
    return der #*lam**2

# %%
def stop_event(t,y):
    return y[0]+8 #+de*np.exp(t)
stop_event.terminal = True

zero_hold_func = lambda x: 1+np.heaviside(x-10,0.5)-np.heaviside(x,0.5)

def odefunc_full(l, depvars):
    lam = np.exp(l)
    # lmV,lDt,lMt,lPt = depvars
    mVb,D,M,P = np.exp(depvars)
    Vb = -mVb
    V = Vb + de*lam
    # V,D,M,P = from_btilde(lam, mVb,Dt,Mt,Pt)
    Z0 = 0*V
    ar1 = V/V

    Tv = P/D/Vb**2
    linMat_inv = 1/Vb**2/(gam*Tv-1) * np.array([[-gam*Tv, ar1, -Tv],[ar1,-ar1,Tv],[gam*ar1,-gam*ar1,ar1]])
    linb = np.array([2*Vb* (V-lam), (de-1)*V*lam+2/9*(M+M_bg(lam))/lam+10*(-V/(lam/lamdi)**10), Vb*lam*((2-Lam0*D**(2-nu)*P**(nu-1))*(gam-1)+2*(de-1))])

    # if not np.isfinite(V/lam).all():
    #     print(V, lam)
    # print(linMat_inv.shape,linb[:,np.newaxis].transpose((2,0,1)).shape)
    linc = np.array([de/Vb*lam,Z0,Z0])
    if np.isscalar(V):
        der = np.matmul(linMat_inv, linb ) 
        # der[0] *= zero_hold_func(V)
        der -= linc
        # der = np.matmul(linMat_inv, linb )+ np.array([-de/Vb*lam,Z0,Z0])
        # if der[0]<0:
            # print(der, linMat_det1, linb, linMat_cofac1)
    else:
        der = np.matmul(linMat_inv.transpose((2,0,1)), linb[:,np.newaxis].transpose((2,0,1)) )
        # der[:,0] *= zero_hold_func(V)[:,np.newaxis]
        der -= linc[:,np.newaxis].transpose((2,0,1))
        der = der.transpose((1,2,0))[:,0,:]

    derM = 3*D*lam**3 /M


    return der, derM, linMat_inv, linb #*lam**2

def odefunc(l, depvars):
    der3, derM = odefunc_full(l, depvars)[:2]
    der = np.insert(der3, 2, derM, axis=0)
    if not np.isfinite(der).all():
        # print(der,l,depvars)
        return np.nan_to_num(der)
    return der

def get_soln_gas_full(lamsh):
    res_pre = solve_ivp(odefunc_prof_init_Pless, (1,lamsh), preshock(np.pi)[1:], max_step=0.001 )
    V1, D1, M1 = res_pre.y[0][-1], res_pre.y[1][-1], res_pre.y[2][-1]
    bcs = shock_jump(lamsh, V1, D1, M1)
    bcs[0] = - bcs[0] + de*lamsh
    # print(bcs)
    bcs = np.log(bcs)
    res_post = solve_ivp(odefunc, (np.log(lamsh),np.log(1e-7)), bcs, method='Radau', max_step=0.05, vectorized=True, events=stop_event)
    return res_pre, res_post

def M0_num(lamsh):
    res = get_soln_gas_full(lamsh)[1]
    M0val = res.y[2][-1]
    return M0val-3e-4 #if M0val>0 else -(-M0val)**(1/11)



#%%
def odefunc_traj_dm(xi, arg):
    lam = arg[0]
    v = arg[1]
    return (v, -2/9 * M_tot(np.abs(lam))/(lam**2+1e-4) * np.sign(lam) - de*(de-1)*lam - (2*de-1)*v)

# def odefunc_traj_gas(xi, arg):
#     lam = arg
#     return V_intrp(lam)-de*lam
#     except:
#         print(lam,s, xi, V_intrp(lam))
#         raise Exception

#%%
descr_list_dict = {}
plab_list_dict = {}

#%%
name = 'cold_vary-s'
name = 'shocked_vary-s'
# name = 'shocked_vary-gam'
# name = 'shocked_vary-cooling'
# name = 'shocked_vary-lamdi'
# name = 'shocked_vary-lamshsp'
# name = 'shocked_vary-lamsh-di'

names = ['cold_vary-s', 'shocked_vary-s', 'shocked_vary-gam', 'shocked_vary-cooling', 'shocked_vary-lamdish', 'shocked_vary-lamshsp']

for name in names[2:3]:
    try:
        t_now = time()
        # thtshsol = fsolve(M0, 1.5*np.pi)
        s = 1
        gam = 5/3
        Lam0 = 3e-2
        nu=1/2
        fb = 0.156837
        # fb = 0.5
        fd = (1-fb)

        lamshsp = 0.9
        disk_rad_by_shock = 0.05
        lamdish = disk_rad_by_shock #*lamsh

        varypars=[]

        if name == 'cold_vary-s':
            s_vals = [0.5,1,1.5,2,3,]
            varypars += ['s']
            lamshsp = 0.1
            lamdish = 0.5

        if name == 'shocked_vary-s':
            s_vals = [0.5,1,1.5,2,3,]
            varypars += ['s']

        if name == 'shocked_vary-gam':
            gam_vals= [5/3,7/5,4/3,]
            lamshsp_vals = [0.9,0.5,0.3]
            varypars += ['gam','lamshsp']

        if name == 'shocked_vary-cooling':
            Lam0_vals = [1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]
            varypars += ['Lam0']

        if name == 'shocked_vary-lamdish':
            lamdish_vals = [percent/100 for percent in [2,5,10,15,25]]
            varypars += ['lamdish']

        if name == 'shocked_vary-lamshsp':
            lamshsp_vals = [1,.9,.8,.7,.6,.5]#[0.35,0.3,0.25, 0.2]
            varypars += ['lamshsp']

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        descr_list, plab_list, conv_iters, rads = [], [], [], []
        for i in range(10):
            # i=1
            plab=''
            try:
                if 'gam' in varypars: gam = gam_vals[i]; plab+=r'$\gamma=$'+f"{gam:.3g} "
                if 's' in varypars: s = s_vals[i]; plab+=f"s={s} "
                if 'lamshsp' in varypars: lamshsp = lamshsp_vals[i]; plab+=r'$\lambda_s=$'+f'{lamshsp*100:g} '+r'$\%~ \lambda_{sp}$'
                if 'lamdish' in varypars: lamdish = lamdish_vals[i]; plab+=r'$\lambda_d=$'+f'{lamdish*100:g} '+r'$\%~ \lambda_s$'
                if 'Lam0' in varypars: Lam0 = Lam0_vals[i]; plab+=r'$\Lambda_0=$'+f'{Lam0:g} '
                # if 'nu' in varypars: nu = nu_vals[i]; plab+=r'$\nu=$'+f'{nu} '
            except IndexError: break

            descr = f'_{name}_lamshsp={lamshsp:.3g}_s={s:.2g}_gam={gam:.3g}_lamdish={lamdish:.3g}_Lam0={Lam0:.1e}_nu={nu:.1g}'
            descr_list.append(descr)
            plab_list.append(plab)
            with open(f'{name}-descr.txt', 'tw') as file: file.write(str(descr_list))
            with open(f'{name}-plab.txt', 'tw') as file: file.write(str(plab_list))
            # continue

            dmo_prfl = pd.read_hdf(f'profiles_dmo_{s}.hdf5', key='main')

            Mta = (3*np.pi/4)**2
            M_dmo = interp1d(dmo_prfl['l'], dmo_prfl['M']*Mta, fill_value=np.nan)
            # D_dmo = interp1d(dmo_prfl['l'].iloc[1:], dmo_prfl['rho'].iloc[1:], fill_value=np.nan)

            M_dm = lambda lam: M_dmo(lam)*(1-fb)

            de = 2* (1+s/3) /3
            upsil = 1 if s >= 3/2 else 3*s/(s+3)

            # plot_iters = [0,1,2,3,5,6]

            t_bef, t_now = t_now, time()
            print(f'{t_now-t_bef:.4g}s', 'Initialised vals and funcs for iteration')

            # fig_conv, ax_conv = plt.subplots(1,2,figsize=(10,7))
            err = 1
            err_tol = 0.01
            conv_iter = 1000
            for n_i in range(-2, 50):
                print('starting iter ', n_i)
                if n_i>=0:
                    if n_i==0:
                        fgas = 1
                        M_bg = lambda lam: 0
                    else:
                        fgas = fb
                        M_bg = M_dm

                    resdf_prof_gaso_bertshi = pd.read_hdf(f'profiles_gaso_bertshi_s={s:.2g}_gam={gam:.3g}.hdf5', key=f'gas/main', mode='r')
                    lamsh_bert = resdf_prof_gaso_bertshi.l[np.diff(resdf_prof_gaso_bertshi.Vb).argmax()]
                    if n_i==0: lamsh = lamshsp*spl_rad #lamsh_bert
                    lamdi = lamdish*lamsh

                    res_prof_gas_pre, res_prof_gas_post = get_soln_gas_full(lamsh=lamsh)
                    # print(f'changed from {n_true} to {n_i}')

                    lamsh_pre = res_prof_gas_pre.t
                    V_pre, D_pre, M_pre = res_prof_gas_pre.y
                    P_pre = lamsh_pre*0

                    lamsh_post = np.exp(res_prof_gas_post.t)
                    mVb_post, D_post, M_post, P_post = np.exp(res_prof_gas_post.y)
                    V_post = de*lamsh_post - mVb_post

                    lam_all = np.concatenate([lamsh_post, lamsh_pre][::-1])
                    V_all = np.concatenate([V_post, V_pre][::-1])
                    D_all = np.concatenate([D_post, D_pre][::-1])
                    M_all = np.concatenate([M_post, M_pre][::-1])
                    P_all = np.concatenate([P_post, P_pre][::-1])
                    Vb_all = V_all - de*lam_all

                    if n_i>=1:
                        iter_change = np.abs(M_all-M_gas(lam_all)/M_gas(1)*M_all[0])/M_all
                        # ax_conv[0].loglog(lam_all, iter_change, label=f'n={n_i}')

                    M_gas = interp1d(np.append(lam_all,0), np.append(M_all,0), fill_value=np.nan)

                    M_tot = lambda lam : M_dm(lam)+M_gas(lam)

                    resdf_gas = pd.DataFrame(data={'l':lam_all, 'M':M_all, 'V':V_all, 'D':D_all, 'P':P_all, 'Vb':Vb_all,})
                    resdf_gas.to_hdf(f'profiles_gasdm{descr:s}.hdf5', 'gas/main', mode='a')
                    resdf_gas.to_hdf(f'profiles_gasdm{descr:s}.hdf5', f'gas/iter{n_i}', mode='a')
                    # resdf_gas = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'gas/iter{n}', mode='r')
                    # M_gas = interp1d(resdf_gas.l, resdf_gas.M, fill_value=np.nan)

                    t_bef, t_now = t_now, time()
                    print(f'{t_now-t_bef:.4g}s', f'{n_i}th iter gas profiles updated')
                    # print('M', M_tot(1), M_dm(1),M_gas(1))
                
                if n_i<=0:
                    M_tot = M_dmo #lambda lam : M_dm(lam)/fd

                xi_max = np.log(1e-4**upsil)*-3/2/s/1.5

                res_traj_dm = solve_ivp(odefunc_traj_dm, (0,xi_max), np.array([1,-de]), method='Radau', t_eval=np.concatenate([np.linspace(0,1-1e-10,1000), np.linspace(1,xi_max**4,2000000)**(1/4)]), max_step=0.02, dense_output=False, vectorized=True)
                # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

                xi = res_traj_dm.t
                lam = np.abs(res_traj_dm.y[0])
                # v = res_traj_dm.y[1]
                loglam = np.log(np.maximum(lam,1e-15))

                resdf_traj_dm = pd.DataFrame(data={'xi':xi, 'lam':lam,})
                resdf_traj_dm.to_hdf(f'traj_gasdm{descr:s}.hdf5', 'dm/main', mode='a')
                resdf_traj_dm.to_hdf(f'traj_gasdm{descr:s}.hdf5', f'dm/iter{n_i}', mode='a')

                t_bef, t_now = t_now, time()
                print(f'{t_now-t_bef:.4g}s', f'{n_i}th iter DM trajectory obtained')

                spl_ind = np.where(np.abs(np.diff(resdf_traj_dm.lam))<1e-4)[0][0]
                spl_rad = resdf_traj_dm.lam[spl_ind]
                print(s, n_i, spl_rad)

                Dm_prof_lbins = 300
                l_range = np.zeros(Dm_prof_lbins+1)
                l_range[1:] = np.logspace(-2.5,0, Dm_prof_lbins)
                M_vals = np.zeros(Dm_prof_lbins+1)
                # rho_vals = np.zeros(301)
                # v_xi = interp1d(xi, v, fill_value=np.nan)
                for i in range(1,Dm_prof_lbins+1):
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
                M_vals[-1] = 1

                M_vals = np.asarray(M_vals)
                # M_vals_er = np.asarray(M_vals_er)
                # rho_vals = np.asarray(rho_vals)

                M_vals *= Mta*(1-fb) / M_vals[-1]

                if n_i>=-1:
                    iter_change = np.abs(M_vals-M_dm(l_range))/M_vals
                    # ax_conv[1].loglog(l_range, iter_change, label=f'n={n_i}')

                M_dm = interp1d(l_range, M_vals, fill_value=np.nan)
                if n_i<=0:  #Start backreaction at iter 1
                    M_dmo = interp1d(l_range, M_vals/(1-fb), fill_value=np.nan)

                resdf_dm = pd.DataFrame(data={'l':l_range, 'M':M_vals,})
                resdf_dm.to_hdf(f'profiles_gasdm{descr:s}.hdf5', 'dm/main', mode='a')
                resdf_dm.to_hdf(f'profiles_gasdm{descr:s}.hdf5', f'dm/iter{n_i}', mode='a')

                t_bef, t_now = t_now, time()
                print(f'{t_now-t_bef:.4g}s', f'{n_i}th iter DM mass profile updated')

                if n_i>=0:
                    resdf_prof_gas = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'gas/iter{n_i}', mode='r')
                    resdf_prof_dm = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter{n_i}', mode='r')
                    resdf_prof_dmo = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter{0}', mode='r')
                    # lamr_full = np.logspace(-2.3,-0.001,400)
                    # lamr = np.logspace(-2.3,-0.01,100)

                    # r, ri_pre = lamr, lamr_full
                    if n_i==0: min_lam = 1e-2
                    if n_i<=1: min_lam = max(resdf_prof_gas.l.iloc[-1],min_lam)
                    # if n<=10: min_lam = resdf_prof_gas.l.iloc[-1]
                    # print(resdf_prof_gas.l.iloc[-1])
                    r, ri_pre = resdf_prof_dm.l[2:-3][resdf_prof_dm.l>min_lam], resdf_prof_dmo.l[1:]

                    Mdr, Mbr, Mdr_dmo = M_dm(r), M_gas(r), M_dmo(ri_pre)*fd
                    # Mdr, Mbr, Mdr_dmo = resdf_prof_dm.M, M_gas(r), resdf_prof_dmo.M 
                    rf = r.copy()
                    logri_logM = interp1d(np.log10(Mdr_dmo),np.log10(ri_pre)) #, fill_value='extrapolate')

                    ri = 10**logri_logM(np.log10(Mdr))
                    Mf = Mdr+Mbr
                    Mi = Mdr/fd

                    if n_i>=2: MiMf_prev, rfri_prev = MiMf, rfri
                    MiMf = ( fd* (Mbr/ Mdr + 1) )**-1
                    rfri = rf / ri

                    resdf_relx = pd.DataFrame(data={'rf':rf, 'MiMf':MiMf, 'rfri':rfri})

                    if n_i>=2: 
                        MiMf_err, rfri_err = np.abs(MiMf-MiMf_prev), np.abs(rfri-rfri_prev)
                        resdf_relx['MiMf_err'] = MiMf_err
                        resdf_relx['rfri_err'] = rfri_err
                    
                    resdf_relx.to_hdf(f'profiles_gasdm{descr:s}.hdf5', 'relx/main', mode='a')
                    resdf_relx.to_hdf(f'profiles_gasdm{descr:s}.hdf5', f'relx/iter{n_i}', mode='a')
                    if n_i>2: err_prev = err
                    if n_i>=2: err = np.median(rfri_err)
                    if n_i>2:
                        # print(err_prev, err)
                        if err_prev<err_tol and err<err_tol:
                            print('converged at ',n_i)
                            conv_iter = n_i
                            break
            rads.append((lamsh,lamsh_bert,spl_rad,))
            conv_iters.append(conv_iter)
            # ax_conv[0].legend()
            # ax_conv[1].legend()
        with open(f'{name}-rads.txt', 'tw') as file: file.write(str(rads))
        with open(f'{name}-conv_iters.txt', 'tw') as file: file.write(str(conv_iters))
        with open(f'{name}-descr.txt', 'tw') as file: file.write(str(descr_list))
        with open(f'{name}-plab.txt', 'tw') as file: file.write(str(plab_list))
        descr_list_dict[name] = descr_list
        plab_list_dict[name] = plab_list
        print(descr_list_dict, plab_list_dict)

        del res_traj_dm, lam, loglam, xi
        dill.dump_session(f'soln-globalsave_all.pkl')

    except:
        print(f'Error occured {name}-{descr}')
        continue

## %%                    #pip install dill --user
# dill.dump_session(f'soln-globalsave_all.pkl')


# %%                           #pip install dill --user
# import dill  
# dill.load_session(f'soln-globalsave_all1.pkl')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
Mta = (3*np.pi/4)**2
fb = 0.156837
fd = 1-fb

#%%
name = 'cold_vary-s'
name = 'shocked_vary-s'
name = 'shocked_vary-gam'
# name = 'shocked_vary-cooling'
# name = 'shocked_vary-lamdish'
# name = 'shocked_vary-lamshsp'

with open(f'{name}-descr.txt', 'tr') as file: descr_list = eval(file.read())
with open(f'{name}-plab.txt', 'tr') as file: plab_list = eval(file.read())

# descr_list = descr_list_dict[name]
# plab_list= plab_list_dict[name]

t_now = time()
# thtshsol = fsolve(M0, 1.5*np.pi)
# fig4, ax4 = plt.subplots(1, dpi=200, figsize=(7,5))
fig5, axs5 = plt.subplots(2,2, figsize=(10,8), sharex=True)
fig6, ax6 = plt.subplots(1)

fig7, (ax71,ax72) = plt.subplots(1,2, figsize=(10,5))
fig8, (ax8,ax82) = plt.subplots(2, figsize=(7,10))

for i,descr in enumerate(descr_list):
    MiMf_stack, rfri_stack = [], []

    plot_iters = [3,4,5,6,40,49] #,10,20,28,29] #1,2,3,5,6,7]

    t_bef, t_now = t_now, time()
    print(f'{t_now-t_bef:.4g}s', 'Initialised plots and figs for iteration')
    plab = plab_list[i]
    err = 1
    err_tol = 0.01

    for n in range(0,50): #plot_iters:
        # color_this = plt.cm.turbo(n/30)
        color_this = colors[i]
        linestyles= [':', '--', '-','-.']

        resdf_prof_gas = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'gas/iter{n}', mode='r')
        resdf_prof_dm = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter{n}', mode='r')
        resdf_traj_dm = pd.read_hdf(f'traj_gasdm{descr:s}.hdf5', key=f'dm/iter{n}', mode='r')
        #resdf_traj_dm_d = pd.read_hdf(f'traj_gasdm{descr:s}_desktop.hdf5', key=f'dm/iter{n}', mode='r')
        resdf_prof_dmo = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter0', mode='r')

        if n in plot_iters:
            axs5[0,0].plot(resdf_prof_gas.l, -resdf_prof_gas.Vb, color=color_this, label=plab)
            axs5[0,1].plot(resdf_prof_gas.l, resdf_prof_gas.D, color=color_this)
            axs5[1,0].plot(resdf_prof_gas.l, resdf_prof_gas.M, color=color_this)
            axs5[1,1].plot(resdf_prof_gas.l, resdf_prof_gas.P, color=color_this)

            axs5[1,0].plot(resdf_prof_dm.l, resdf_prof_dm.M, ls='dashdot', color=color_this)


        M_gas = interp1d(resdf_prof_gas.l, resdf_prof_gas.M) #, fill_value='extrapolate')
        M_dm = interp1d(resdf_prof_dm.l, resdf_prof_dm.M)
        M_dmo = interp1d(resdf_prof_dmo.l, resdf_prof_dmo.M)

        M_tot = lambda lam : M_dm(lam)+M_gas(lam)

        # axs5[1,0].plot(lam_all,M_all+M_dm(lam_all), color=color_this, ls='dashed')

        if n in plot_iters: 
            ax6.plot(resdf_traj_dm.xi,resdf_traj_dm.lam, color=color_this, label=plab)
            # ax6.plot(resdf_traj_dm_d.xi,resdf_traj_dm_d.lam, label=f'n={n}_desktop')

            # V_intrp = interp1d(resdf_prof_gas.l, resdf_prof_gas.V, fill_value=np.nan)
            # lamshsol, bcs = get_shock_bcs(thtshsol)
            # taush = (thtshsol - np.sin(thtshsol)) / np.pi
            # xish = np.log(taush)
            # res_traj_gas = solve_ivp(odefunc_traj_gas, (1,4), np.array([1]), method='Radau', max_step=0.001, dense_output=False, vectorized=True)
            # # res1 = solve_ivp(fun, (res.t[-1],15), np.array([res.y[0][-1],-res.y[1][-1]]), max_step=0.1, dense_output=True)

            # t_bef, t_now = t_now, time()
            # print(f'{t_now-t_bef:.4g}s', f's={s}: post shock trajectory obtained')

            # xires = res_traj_gas.t
            # lamres = res_traj_gas.y[0]
            # # vres = res.y[1]

            # taures = np.exp(xires)
            # lamFres = lamres*taures**de

            # # ax6.plot(taures,lamFres, color=color_this, label=f's={s}')
            # ax6.plot(xires,lamres, color=color_this)
            # ax6.plot(cumtrapz(1/(resdf_prof_gas.V-de*resdf_prof_gas.l), x=resdf_prof_gas.l), resdf_prof_gas.l[1:], c=color_this, ls='-.')


            t_bef, t_now = t_now, time()
            print(f'{t_now-t_bef:.4g}s', f'{n}th iter plotted')

        if n>=0:
            # lamr_full = np.logspace(-2.3,-0.001,400)
            # lamr = np.logspace(-2.3,-0.01,100)

            # r, ri_pre = lamr, lamr_full
            if n==0: min_lam = 1e-2
            if n<=1: min_lam = max(resdf_prof_gas.l.iloc[-1],min_lam)
            # if n<=10: min_lam = resdf_prof_gas.l.iloc[-1]
            # print(resdf_prof_gas.l.iloc[-1])
            r, ri_pre = resdf_prof_dm.l[2:-3][resdf_prof_dm.l>min_lam], resdf_prof_dmo.l[1:]

            Mdr, Mbr, Mdr_dmo = M_dm(r), M_gas(r), M_dmo(ri_pre)
            # Mdr, Mbr, Mdr_dmo = resdf_prof_dm.M, M_gas(r), resdf_prof_dmo.M 


            if n in plot_iters:
                ax71.plot(r,Mdr/Mta, ls='-', c=color_this)
                ax71.plot(r,Mbr*fd/fb/Mta, ls='-.', c=color_this)
                ax71.plot(ri_pre,Mdr_dmo/Mta, ls='--', c=color_this)
                # plt.plot(r,Mdr+Mbr)
                ax71.set_xscale('log')
                ax71.set_yscale('log')

            rf = r.copy()

            logri_logM = interp1d(np.log10(Mdr_dmo),np.log10(ri_pre), fill_value='extrapolate')

            # assert (ri_M(Mdr_dmo) == r).all()

            ri = 10**logri_logM(np.log10(Mdr))

            Mf = Mdr+Mbr
            Mi = Mdr/fd

            if n>=2: MiMf_prev, rfri_prev = MiMf, rfri

            MiMf = ( fd* (Mbr/ Mdr + 1) )**-1
            rfri = rf / ri
            # leng = MiMf.shape[0]

            if n>=2: 
                MiMf_err, rfri_err = np.abs(MiMf-MiMf_prev), np.abs(rfri-rfri_prev)
                if n in plot_iters: ax8.plot(MiMf, rfri_err, color=color_this, label=plab+f' n={n}', ls=linestyles[n%4])

            if n>0:
                MiMf_stack.append(MiMf)
                rfri_stack.append(rfri)

            if n>2: err_prev = err
            if n>=2: err = np.median(rfri_err)
            if n>2:
                # print(err_prev, err)
                if err_prev<err_tol and err<err_tol:
                    print('converged at ',n)
                    break


    # MiMf_err, rfri_err = np.abs(MiMf-MiMf_prev), np.abs(rfri-rfri_prev)
    MiMf_stack, rfri_stack = np.vstack(MiMf_stack), np.vstack(rfri_stack)
    
    # MiMf, rfri = MiMf_stack.mean(axis=0), rfri_stack.mean(axis=0)

    # MiMf_max, MiMf_min = MiMf_stack.max(axis=0), MiMf_stack.min(axis=0)
    # rfri_max, rfri_min = rfri_stack.max(axis=0), rfri_stack.min(axis=0)

    # MiMf_err, rfri_err = np.abs(MiMf_max-MiMf_min)/2, np.abs(rfri_max-rfri_min)/2
    # ax8.plot(MiMf, rfri_err, color=color_this, label=plab)
    iter_diff = np.median(np.abs(rfri_stack[1:]-rfri_stack[:-1]), axis=1)
    conv_bool = iter_diff<err_tol
    iter_end = np.where(conv_bool[1:]*conv_bool[:-1])[0][0]+3
    print('converged at', iter_end)
    ax82.scatter(iter_end, iter_diff[iter_end-2], color=color_this)
    ax82.plot(np.arange(2,rfri_stack.shape[0]+1), iter_diff, color=color_this)


    ax72.errorbar(MiMf[::20],rfri[::20], xerr=MiMf_err[::20], yerr=rfri_err[::20],fmt='.')
    # ax72.fill_between(MiMf, rfri_min, rfri_max, color=color_this, alpha=0.3)  
    # ax71.scatter(MiMf[60:-50],rfri[60:-50],c=rf[60:-50])
    cplot = ax72.scatter(MiMf,rfri,c=np.log10(rf), s=60, cmap='nipy_spectral')
    # plab=f'n={n}'
    ax72.plot(MiMf,rfri, label=plab, c=color_this, lw=3)
    # ax71.scatter(MiMf[100:],rfri[100:],c=np.log10(rf[100:]), cmap='nipy_spectral')
    # ax71.plot(MiMf,1+0.25*(MiMf-1),'k',label='$q=0.25$')

ax71.plot([],[], ls='-', c='k', label='DM')
ax71.plot([],[], ls='-.', c='k', label='Gas')
ax71.plot([],[], ls='--', c='k', label='DM in DMO' )

ax71.set_xlabel(r'$r/r_{\rm{ta}}$')
ax71.set_ylabel(r'$M/M_{\rm{ta}}$')

# ax72.plot(MiMf,1+0.33*(MiMf-1)-0.02,'k:',label='$q=0.33$, $q_0=0.02$')

ax72.set_xlabel('$M_i/M_f$')
ax72.set_ylabel('$r_f/r_i$')

fig7.colorbar(cplot, ax=ax72,label=r'$r_f/r_{\rm{ta}}$')
ax71.legend()
ax72.legend()

# axs5[1,0].plot(dmo_prfl['l'], dmo_prfl['M']*Mta, color='k', ls='dashed')
axs5[1,0].plot(resdf_prof_dmo.l, resdf_prof_dmo.M, color='purple', ls='dashed')

# ax4.set_xlabel(r'$\theta$')
# ax4.set_ylabel(r'$M(\lambda=0)$')
# # ax4.set_ylim(-2,5)
# ax4.legend()
    
axs5[0,0].set_xscale('log')
axs5[0,0].set_xlim(1e-4,1)
axs5[0,0].legend()

# axs5[0,0].set_ylim(1e-60,1)

axs5[1,0].plot([], ls='solid', color='k', label='Gas')
axs5[1,0].plot([], ls='dashdot', color='k', label='DM')
axs5[1,0].plot([], ls='dashed', color='k', label='DMo')
axs5[1,0].legend()

# if gam>1.67:
# axs5[0,0].set_xlim(1e-2,1)
axs5[0,1].set_ylim(1e-1,1e6)
axs5[1,0].set_ylim(1e-2,1e1)
axs5[1,1].set_ylim(1e0,1e7)

axs5[0,0].set_ylabel('-Vb')
axs5[0,1].set_ylabel('D')
axs5[1,0].set_ylabel('M')
axs5[1,1].set_ylabel('P')

axs5[0,0].set_yscale('log')
axs5[0,1].set_yscale('log')
axs5[1,0].set_yscale('log')
axs5[1,1].set_yscale('log')




ax6.set_xlim(0,6)
ax6.set_ylim(1e-3,1)
ax6.set_yscale('log')
# ax6.set_ylim(resdf_prof_dm.l[1],1)
ax6.xaxis.get_ticklocs(minor=True)     # []
ax6.minorticks_on()
ax6.grid(visible=True, which='both', axis='x')
ax6.set_xlabel(r'$\xi$')
ax6.set_ylabel(r'$\lambda$')
ax6.legend(loc='upper right')

ax8.legend(loc='upper right')
ax8.set_xlabel('Mi/Mf')
ax8.set_ylabel('error in rf/ri at nth iter')
    # fig8.savefig('relx_reln_converge_issue.png', bbox_inches='tight')

# fig5.savefig(f'Eds-gas-{gam:.02f}_profiles.pdf')
# fig5.savefig(f'Eds-gas-{gam:.02f}_trajectory.pdf')


#%%
# plt.show()
# %%
# import dill                            #pip install dill --user
# filename = f'soln-globalsave{descr:s}.pkl'
# dill.load_session(filename)

#%%
s = [0.5]+[0.5,1,1.5]
gam = [5/3,]*4
Lam0 = [3e-2,]*3+[3e-3,]
nu = [1/2,]*4

lamsh = [0.035]+[0.35,0.35,0.25]
disk_rad_by_shock = [0.5]+[0.05,]*3 #[0.15]+

fb = 0.156837
# fb = 0.5
fd = (1-fb)

fig7, (ax71,ax72) = plt.subplots(1,2, figsize=(13,6))


# colors = plt.cm.get_cmap('hsv', 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(len(s)):
    descr = f'_s={s[i]:.2g}_gam={gam[i]:.3g}_shk={lamsh[i]:.1g}_Rd={disk_rad_by_shock[i]*100:.2g}%_Lam={Lam0[i]:.1e}_nu={nu[i]:.1g}'
    resdf_prof_gas = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'gas/main', mode='r')
    resdf_prof_dm = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/main', mode='r')
    resdf_prof_dmo = pd.read_hdf(f'profiles_gasdm{descr:s}.hdf5', key=f'dm/iter-1', mode='r')

    M_gas = interp1d(resdf_prof_gas.l, resdf_prof_gas.M)
    M_dm = interp1d(resdf_prof_dm.l, resdf_prof_dm.M)
    M_dmo = interp1d(resdf_prof_dmo.l, resdf_prof_dmo.M)

    lamr_full = np.logspace(-2.3,-0.005,300)
    lamr = np.logspace(-2.3,-0.005,300)

    r, ri_pre = lamr, lamr_full
    r, ri_pre = resdf_prof_dm.l[1:], resdf_prof_dmo.l[1:]

    Mdr, Mbr, Mdr_dmo = M_dm(r), M_gas(r), M_dmo(ri_pre)
    # Mdr, Mbr, Mdr_dmo = resdf_prof_dm.M, resdf_prof_gas.M, resdf_prof_dmo.M 

    print(f'{descr[1:]}')


    ax71.plot(r,Mdr/Mta, ls='-', c=colors[i])
    ax71.plot(r,Mbr*fd/fb/Mta, ls='-.', c=colors[i])
    ax71.plot(ri_pre,Mdr_dmo/Mta, ls='--', c=colors[i])
    # plt.plot(r,Mdr+Mbr)
    ax71.set_xscale('log')
    ax71.set_yscale('log')


    ##%%
    rf = r.copy()

    logri_logM = interp1d(np.log10(Mdr_dmo),np.log10(ri_pre), fill_value='extrapolate')

    # assert (ri_M(Mdr_dmo) == r).all()

    ri = 10**logri_logM(np.log10(Mdr))

    Mf = Mdr+Mbr
    Mi = Mdr/fd

    MiMf = ( fd* (Mbr/ Mdr + 1) )**-1
    rfri = rf / ri
    
    # ax71.scatter(MiMf[60:-50],rfri[60:-50],c=rf[60:-50])
    cplot = ax72.scatter(MiMf,rfri,c=np.log10(rf), s=60, cmap='nipy_spectral')
    plab = f"s={s[i]} "
    plab += r'$\lambda_s=$'+f'{lamsh[i]}'
    # plab += r'$\gamma=$'+f"{gam[i]:.3g}"
    ax72.plot(MiMf,rfri, label=plab, c=colors[i], lw=3)
    # ax71.scatter(MiMf[100:],rfri[100:],c=np.log10(rf[100:]), cmap='nipy_spectral')
    # ax71.plot(MiMf,1+0.25*(MiMf-1),'k',label='$q=0.25$')

ax71.plot([],[], ls='-', c='k', label='DM')
ax71.plot([],[], ls='-.', c='k', label='Gas')
ax71.plot([],[], ls='--', c='k', label='DM in DMO' )

ax71.set_xlabel(r'$r/r_{\rm{ta}}$')
ax71.set_ylabel(r'$M/M_{\rm{ta}}$')

ax72.plot(MiMf,1+0.33*(MiMf-1)-0.02,'k:',label='$q=0.33$, $q_0=0.02$')

ax72.set_xlabel('$M_i/M_f$')
ax72.set_ylabel('$r_f/r_i$')

fig7.colorbar(cplot, ax=ax72,label=r'$r_f/r_{\rm{ta}}$')
ax71.legend()
ax72.legend()
fig7.savefig(f'ratio_plot_anyl.pdf', bbox_inches='tight')
# %%
plt.show()
# %%
