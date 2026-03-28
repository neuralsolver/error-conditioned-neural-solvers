#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[13]:


get_ipython().system('pip install scipy')


# In[14]:


import torch
import os
import math

import matplotlib.pyplot as plt
import matplotlib

from random_fields import GaussianRF

from timeit import default_timer

import scipy.io
from ns_2d import navier_stokes_2d
from scipy.io import loadmat
import torch.nn.functional as F
device = torch.device('cpu')


# In[15]:


device = torch.device('cpu')

for i in range(1):

    #Resolution
    s = 64
    sub = 1

    #Number of solutions to generate
    N = 100

    #Set up 2d GRF with covariance parameters
    #torch.manual_seed(42)
    #GRF = GaussianRF(2, s, alpha=2.5, tau=3, device=device)
    GRF = GaussianRF(2, s, alpha=2.5, tau=7.0, device=device)

    #Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s+1, device=device)
    t = t[0:-1]

    X,Y = torch.meshgrid(t, t)
    f = 0.1*(torch.sin(4*math.pi*(X + Y)) + torch.cos(4*math.pi*(X + Y)))
    #f = (0.1 * torch.sin(2*math.pi*(X + Y)) + 0.2 * torch.cos(4*math.pi*X) + 0.1 * torch.sin(6*math.pi*Y)) 

    #Number of snapshots from solution
    record_steps = 10

    #Inputs
    a = torch.zeros(N, s, s)
    #Solutions
    u = torch.zeros(N, s, s, record_steps)

    #Solve equations in batches (order of magnitude speed-up)

    #Batch sizea
    bsize = 50

    c = 0
    t0 = default_timer()
    for j in range(N//bsize):

        #Sample random feilds
        w0 = GRF.sample(bsize)
        
        #Solve NS
        sol, sol_t = navier_stokes_2d(w0, f, 1e-4, 1.0, 1e-4, record_steps)

        a[c:(c+bsize),...] = w0
        u[c:(c+bsize),...] = sol

        c += bsize
        t1 = default_timer()
        print(j, c, t1-t0)
        
    os.makedirs("data", exist_ok=True)
    filename = 'data/ns_64_test_2(306-f).mat'.format(i+51)
    scipy.io.savemat(filename, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
    print('Saved:', filename)


# In[16]:


data = scipy.io.loadmat('data/ns_64_test_2(306-f).mat')


# In[17]:


print(data['u'].shape)


# In[18]:


a = data['a']
u = data['u']
u = torch.from_numpy(u).float()
t = data['t'].squeeze()
t = torch.from_numpy(t).float() 
print(a.shape)
print(u.shape)
print(t)


# In[19]:


plt.imshow(u[2,:,:,0], cmap='viridis', origin='lower')
plt.colorbar()
plt.title("w(x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.imshow(u[2,:,:,9], cmap='viridis', origin='lower')
plt.colorbar()
plt.title("w(x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[33]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation

sample = 2
data = u[sample].cpu().numpy()  

fig, ax = plt.subplots()

im = ax.imshow(data[:, :, 0], cmap='viridis', origin='lower')
cbar = plt.colorbar(im)
ax.set_title("w(x,y)")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    im.set_array(data[:, :, frame])
    ax.set_title(f"t = {frame}")
    return [im]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=data.shape[2],   
    interval=200           
)

from IPython.display import HTML
HTML(ani.to_jshtml())


# In[12]:


time = torch.linspace(0, 1, 64+1, device=device)
time = time[0:-1]

X,Y = torch.meshgrid(time, time)
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
plt.imshow(f, cmap='viridis', origin='lower')


# In[13]:


import torch, math

def residual_map_vorticity(w, sol_t, f, nu):
    """
    w:    (N, N, T) vorticity snapshots (one sample)
    sol_t:(T,)      times for each snapshot
    f:    (N, N)    forcing in vorticity equation
    nu:   float     viscosity

    returns:
      R: (N, N, T) residual map at each snapshot time
    """
    assert w.ndim == 3, "w should be (N,N,T)"
    N, N2, T = w.shape
    assert N == N2

    #device = w.device
    dtype  = w.dtype
    #sol_t = sol_t.to(device=device, dtype=dtype)
    #f = f.to(device=device, dtype=dtype)

    k = torch.fft.fftfreq(N, d=1.0 / N)  
    kx = k[None, :].repeat(N, 1)   
    ky = k[:, None].repeat(1, N)   

    lap_pos = 4 * (math.pi**2) * (kx**2 + ky**2)  # positive "lap" like your code
    lap_pos[0, 0] = 1.0  # avoid divide by zero for psi

    R = torch.empty((N, N, T))

    wt = torch.empty_like(w)
    wt[:, :, 0]    = (w[:, :, 1] - w[:, :, 0]) / (sol_t[1] - sol_t[0])
    wt[:, :, -1]   = (w[:, :, -1] - w[:, :, -2]) / (sol_t[-1] - sol_t[-2])
    for n in range(1, T-1):
        wt[:, :, n] = (w[:, :, n+1] - w[:, :, n-1]) / (sol_t[n+1] - sol_t[n-1])

    for n in range(T):
        wn = w[:, :, n]                         
        wh = torch.fft.fft2(wn)                 

        psi_h = wh / lap_pos

        u = torch.fft.ifft2(1j * 2 * math.pi * ky * psi_h).real
        v = torch.fft.ifft2(-1j * 2 * math.pi * kx * psi_h).real

        wx  = torch.fft.ifft2(1j * 2 * math.pi * kx * wh).real
        wy  = torch.fft.ifft2(1j * 2 * math.pi * ky * wh).real

        lap_w = torch.fft.ifft2((-lap_pos) * wh).real

        R[:, :, n] = wt[:, :, n] + u * wx + v * wy - nu * lap_w - f

    return R


# In[15]:


print(u[0].shape)
R = residual_map_vorticity(u[1], t, f, 1e-3)


# In[27]:


plt.imshow(R[:,:, 9], cmap='viridis', origin='lower', vmax = 0.3, vmin = -0.25)
plt.colorbar(label="R")
plt.title("R")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
print(torch.mean(R**2))


# In[ ]:




