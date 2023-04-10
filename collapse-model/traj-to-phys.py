#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
tau = np.linspace(1,2,200)
lam_F = (2-tau)**0.5

#%%
plt.plot(tau,lam_F)
# %%
de = 8/9
# rs = np.linspace(.5,2,10)
ts = np.linspace(.5,2,10)
rs = ts**de

#%%
r = np.outer(lam_F,rs)
t = np.outer(tau,ts)
# %%
plt.plot(t,r)
# %%
