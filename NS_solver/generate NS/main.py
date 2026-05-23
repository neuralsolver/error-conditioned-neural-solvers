#!/usr/bin/env python
# coding: utf-8

# In[12]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[13]:


get_ipython().system('pip install scipy')


# In[14]:


import torch
import os
import math
import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from random_fields import GaussianRF

from timeit import default_timer

import scipy.io
from ns_2d import navier_stokes_2d
from scipy.io import loadmat
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[15]:


device = torch.device('cpu')

#set_seed(42)
for i in range(1):

    #Resolution
    s = 128
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
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y))) #original forcing 
    #f = 0.1*(torch.sin(4*math.pi*(X + Y)) + torch.cos(4*math.pi*(X + Y))) #forcing shift
    #f = -4.0 * torch.cos(2 * math.pi * 4 * Y) #KF

    #Number of snapshots from solution
    record_steps = 20
    #Inputs
    a = torch.zeros(N, s, s)
    #Solutions
    u = torch.zeros(N, s, s, record_steps)

    #Solve equations in batches (order of magnitude speed-up)

    #Batch size
    bsize = 10

    c = 0
    t0 = default_timer()
    for j in range(N//bsize):

        #Sample random feilds
        w0 = 2 * GRF.sample(bsize)
        #w0 = GRF.sample(bsize)
        
        sol_warm, _ = navier_stokes_2d(w0, f, 1e-3, 2.0, 1e-4, 10) #viscosity = 1e-3, record start from t = 2s
        w_start = sol_warm[..., -1]
        
        #Solve NS
        sol, sol_t = navier_stokes_2d(w_start, f, 1e-3, 4.0, 1e-4, record_steps, t0 = 2.0)  #vrecord start from t = 2s, end at t = 6s
        
        a[c:(c+bsize),...] = w_start 
        u[c:(c+bsize),...] = sol

        c += bsize
        t1 = default_timer()
        print(j, c, t1-t0)
        
    '''os.makedirs("320", exist_ok=True)
    filename = '320/KF_test_visc(1e-3-t40).mat'.format(i+51) 
    scipy.io.savemat(filename, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()}) 
    print('Saved:', filename)'''

    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'KF_128(1e-3)(t40)_test.h5')

    with h5py.File(filename, 'w') as f:
        f.create_dataset('a', data=a.cpu().numpy().astype('float32'))
        f.create_dataset('u', data=u.cpu().numpy().astype('float32'))
        f.create_dataset('t', data=sol_t.cpu().numpy())
    
    print('Saved:', filename)
