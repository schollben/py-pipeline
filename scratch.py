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
    

#%% test new versions of filter_baseline_dF_comp
import numpy as np
import h5py
import matplotlib.pyplot as plt
from img_utils import (filter_baseline_dF_comp, plot_raw_dff)

with h5py.File('/mnt/bigdata/SCANIMAGE_LOCAL/test/test2p_ROI.h5', 'r') as f:
    dff_saved = f['dff'][:]          # computed by the old version
    raw       = f['raw_cell_traces'][:]

fs = 30.0      
cells = [0, 2, 4, 6]

# run the new version on every cell, keeping the baseline it used
dff_new = np.zeros_like(raw)
base_new = np.zeros_like(raw)
for cc in range(raw.shape[1]):
    dff_new[:, cc], base_new[:, cc] = filter_baseline_dF_comp(raw[:, cc], fs=fs, win_sec=60.0)

# plot with the ACTUAL baseline overlaid (not a recomputed guess)
plot_raw_dff(raw, dff_new, cells, baselines=base_new)

# old vs new on the same axes, per cell
fig, axes = plt.subplots(len(cells), 1, figsize=(14, 2.5*len(cells)), sharex=True)
for ax, cc in zip(np.atleast_1d(axes), cells):
    ax.plot(dff_saved[:, cc], lw=0.4, color='r', alpha=0.7, label='old')
    ax.plot(dff_new[:, cc], lw=0.4, color='b', alpha=0.7, label='new')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel(f'cell {cc}')
axes[0].legend(loc='upper right')
plt.tight_layout()
plt.show()


    

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

dff_cc = filter_baseline_dF_comp(raw, fs=fs, win_sec=45.0)
plot_raw_dff(raw[:, np.newaxis], dff_cc[:, np.newaxis], [0], pts=301)
