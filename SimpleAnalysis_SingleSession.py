# %% 1. initialize

%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.session import load_session
from analysis import load_session, plot_avg_rois, plot_stim_traces, plot_tuning_curves, compute_selectivity, plot_preference_maps

sns.set_theme(context='notebook', style='white')

FNAME = '/mnt/bigdata/PROCESSED/TSeries-07132025-1042-002.h5'

s = load_session(FNAME)

print(f'{s.exp_id}: {s.n_rois} ROIs, '
      f'{len(s.directions)} directions, {len(s.contrasts)} contrasts')

# show average image with ROI mask
plot_avg_rois(s,vmax_frac=0.6) 


# %% 2. trial-averaged time-varying responses (one or more cells)
plot_stim_traces(s, [5,50,70]) #example: cells 5, 50, 70


# %% 4. tuning curves + preferred direction (double-Gaussian fit)
plot_tuning_curves(s)


# %% 5. direction / orientation selectivity
compute_selectivity(s)
print(f'gDSI median {np.nanmedian(s.gdsi):.3f}  |  gOSI median {np.nanmedian(s.gosi):.3f}')


# %% 6. preference maps (direction | orientation)
plot_preference_maps(s, thr=0.1)

