# %% 1. load dataset to view
import h5py
import os

os.chdir(r'/mnt/bigdata/PROCESSED') #change if necessary

fname = 'TSeries-07132025-1042-002.h5' #choose file

with h5py.File(fname, 'r') as f:
    dff  = f['dff'][:]
    imgs = f['opto_delta_images'][:]
    
print('done loading')

# %% 2. display image and ROIs with labels


# %% 