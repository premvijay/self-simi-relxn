#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz, quad, trapezoid, cumulative_trapezoid
plt.style.use('seaborn-darkgrid')

#%%
def exponential_decay(t, y1): return -0.5 * y1
sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], max_step=0.01)
print(sol.t)
# %%
plt.plot(sol.t, sol.y[1])
# %%

# %%
def dIdy(y, beta):
    return 1/np.sqrt(y**-1+y**2-3*beta/2**(2/3))

y = np.linspace(1e-8,1)
beta = 1.5
dIdyvals = dIdy(y, beta)
# %%
I = cumulative_trapezoid(dIdyvals, y)
# %%
plt.plot(I,y[1:])
# plt.plot(I,y[1:])
# %%
u = np.sinh(3*I/2)**(2/3)
# %%
plt.plot(u,y[1:])
# %%
sol = solve_ivp(lambda I,y: np.sqrt(y**-1+y**2-3*beta/2**(2/3)), [1e-8, 0.5], [1e-8], max_step=0.01)
sol1 = solve_ivp(lambda I,y: -np.sqrt(y**-1+y**2-3*beta/2**(2/3)), [0.35, 0.6], [0.35], max_step=0.01)
# print(sol.t)
# %%
plt.plot( sol.t,sol.y[0],)
plt.plot( sol1.t,sol1.y[0],)
plt.plot(I,y[1:])
# %%
def y_ta(beta):
    return 2**(2/3)*beta**(1/2)*np.sin(1/3*np.arcsin(beta**(-3/2)))
# %%
def dIdy(y, beta):
    return 1/np.sqrt(y**-1+y**2-3*beta/2**(2/3))

def u(I):
    return np.sinh(3*I/2)**(2/3)

beta = np.linspace(1,50)
beta = 1.2
logbetas1 = np.linspace(0.03,0.4, 15)
logbetas2 = np.linspace(0.4,1.0, 12)
logbetas = np.concatenate([logbetas1,logbetas2])
# betas = 10**logbetas
plt.figure(figsize=(8,7), dpi=130)
for logbeta in logbetas:
    beta = 10**logbeta
    y1 = np.linspace(1e-8,y_ta(beta)-1e-8, 1000)
    y2 = np.linspace(y_ta(beta)/2, y_ta(beta)-1e-8, 2000)[::-1]
    y1_cen = (y1[:-1]+y1[1:])/2
    y2_cen = (y2[:-1]+y2[1:])/2
    # dIdyvals = dIdy(y, beta)

    # I1 = cumulative_trapezoid(dIdy(y1, beta), y1, initial=0)
    # I2 = cumulative_trapezoid(-dIdy(y2, beta), y2, initial=I1[-1]) - dIdy(y2[0], beta)*(y2[1]-y2[0])

    I1 = np.cumsum(dIdy(y1_cen, beta)*np.diff(y1))
    I2 = np.cumsum(-dIdy(y2_cen, beta)*np.diff(y2))+I1[-1]
    # I2 = np.cumtrapz(-dIdy(y2, beta)*np.diff(y2)[0])+I1[-1]

    color_this = plt.cm.nipy_spectral((logbeta)/1.1)
    plt.plot(u(I1),y1[1:], color=color_this, lw=1)
    plt.plot(u(I2),y2[1:], color=color_this, lw=1)
    plt.scatter(u(I1[-1]),y_ta(beta), color=color_this, s=5)
    plt.scatter(u(I2[-1]),y_ta(beta)/2, color=color_this, s=8)

plt.xlabel('u')
plt.ylabel('y')
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap='nipy_spectral'))
cb1.set_label(r'$\log (\beta)$')
# beta = np.linspace(1,50)
# plt.scatter()
# %%

# %%
