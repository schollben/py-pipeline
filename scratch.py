# %% view whats in data

import h5py
import os
os.chdir(r'/mnt/bigdata/PROCESSED')

with h5py.File('TSeries-04022026-1315-003.h5', 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name} Shape: {obj.shape} Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
    
    f.visititems(print_structure)


#%% load 
import h5py
import os
os.chdir(r'/mnt/bigdata/PROCESSED')
with h5py.File('TSeries-04022026-1315-003.h5', 'r') as f:
    dff  = f['dff'][:]
    imgs = f['opto_delta_images'][:]
    

# %%
