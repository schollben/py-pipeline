# %% look at data

import h5py
with h5py.File('TSeries-04022026-1315-003.h5', 'r') as f:
    dff  = f['dff'][:]
    imgs = f['opto_delta_images'][:]

