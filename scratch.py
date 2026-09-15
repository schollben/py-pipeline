# %% view whats in data

import h5py
import os
os.chdir(r'/mnt/bigdata/PROCESSED')

with h5py.File('TSeries-07132025-1042-002.h5', 'r') as f:
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
    

# %% test filterbaseline_dF_comp

import numpy as np
from img_utils import filter_baseline_dF_comp, plot_raw_dff

rng = np.random.default_rng(0)
n = 6000          # frames
fs = 30.0         # Hz

# slow baseline wander + giant sigmoid step in the middle (jumps up, stays up)
t = np.arange(n) / fs
baseline = 500 + 40*np.sin(2*np.pi*t/220)
baseline += 1000/(1 + np.exp(-1*(np.arange(n) - n/2)/(n/60)))

# calcium events: random onsets with exponential decay, shrinking over the trace
kernel = np.exp(-np.arange(int(8*0.6*fs)) / (0.6*fs))
spikes = np.zeros(n)
onsets = np.sort(rng.integers(0, n, 50))
spikes[onsets] = rng.gamma(2.0, 0.25, 50) * np.linspace(1.0, 0.1, 50)
events = np.convolve(spikes, kernel)[:n]

# brief transient artifacts, including one right at the start
artifacts = np.zeros(n)
artifacts[3:8] += 2000
for f in rng.integers(300, n-300, 6):
    artifacts[f:f+rng.integers(2, 6)] += rng.choice([-2, 2]) * rng.uniform(300, 600)

# noise scales with sqrt(F) (shot noise) plus read noise
raw = baseline*(1 + events) + artifacts
raw += rng.normal(0, 1, n)*np.sqrt(np.maximum(raw, 1))*0.8 + rng.normal(0, 8, n)

dff_cc = filter_baseline_dF_comp(raw, 301)
plot_raw_dff(raw[:, np.newaxis], dff_cc[:, np.newaxis], [0], pts=301)
