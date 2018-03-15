# -*- coding: utf-8 -*-
"""
Runs a CUDA accelerate normalized cross correlation 
between a reference signal and a batch of signals. 
device_obj and run are imported from cross_correlate.py, 
which utilizes pycu_interface.


"""
import numpy as np
from time import time
from scipy.signal import chirp

# Imports that utilize pycu_interfaces
from cross_correlate import device_obj, run

cpu_enable = False
plot = True

def gen_return_signals(s, ny, nx):
    """
    Generates image return data.
    
    """
    r = np.tile(s.reshape((len(s),1,1)),(1,ny,nx))
    base_noise = np.random.normal(loc=0.0, scale=1./(ny*nx), size=len(s))
    noise_multi = np.arange(0, ny*nx, 1).reshape(ny,nx)
    noise_cube = base_noise[:,None,None]*noise_multi
    return r+noise_cube
    

def gen_ref_signal(nz):
    t = np.linspace(0.,1.,nz)
    return chirp(t, 5., t[-1], 20.).astype('f4')


# Reference signal to cross correlate aga
nz, ny, nx = 1024, 512, 480
s = gen_ref_signal(nz)
r = gen_return_signals(s, ny, nx).astype('f4').reshape(nz, ny*nx).T


"""
From this point on, you can assume the data is already 
avaiable in your pipeline.

Below, the with statement is used so that automatic 
context clean up is done afterward. This means that 
the user created device_obj pointers are removed, 
and all GPU resources that were used by device_obj 
freed.
"""

# Cross correlate s against every r
corr_coefs = np.empty(r.shape[0], r.dtype)
t0 = time()
with device_obj(r.shape, r.dtype, n_streams=4) as d:
    run(d, s, r, corr_coefs)
print("GPU time = %.2f seconds"%(time()-t0))


if cpu_enable:
    t0 = time()
    corr_coefs2 = np.empty(r.shape[0], r.dtype)
    for idx in np.ndindex(r.shape[0]):
        corr_coefs2[idx] = np.corrcoef(s,r[idx])[1,0]
    print("CPU time = %.2f seconds"%(time()-t0))
    print("GPU all close to CPU? %s"%(np.allclose(corr_coefs,corr_coefs2)))


if plot:
    import matplotlib.pyplot as plt
    plt.imshow(corr_coefs.reshape(ny,nx))
    plt.title("Normalized Cross Correlation Result")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.show()