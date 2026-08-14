# %% 1. initialize

%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis import (load_session, plot_avg_rois, plot_stim_traces, plot_tuning_curves,
                      compute_selectivity, plot_preference_maps, describe_photostim_groups,
                      plot_photostim_group_heatmaps, compute_influence, plot_influence_maps,
                      plot_influence_by_contrast)

sns.set_theme(context='notebook', style='white')


# TEST: 07132025, good day for oricont and photostim
folderName = '/Users/benjaminscholl/Dropbox/projects/2poptostim/PROCESSED/V1/'
# FNAME = '/mnt/bigdata/PROCESSED/TSeries-07132025-1042-003.h5'
FNAME = folderName + 'TSeries-07132025-1042-003.h5'

dat = load_session(FNAME)

print(f'{dat.exp_id}: {dat.n_rois} ROIs, '
      f'{len(dat.directions)} directions, {len(dat.contrasts)} contrasts')

# show average image with ROI mask
plot_avg_rois(dat,vmax_frac=0.6) 


# %% 2. trial-averaged time-varying responses (one or more cells)
plot_stim_traces(dat, [1,2,3,4,5]); #example: cells 5, 50, 70


# %% 3. tuning curves + preferred direction (double-Gaussian fit)
# and compute direction / orientation selectivit
plot_tuning_curves(dat);
compute_selectivity(dat);
print(f'gDSI median {np.nanmedian(dat.gdsi):.3f}  |  gOSI median {np.nanmedian(dat.gosi):.3f}')


# %% 5. preference maps (direction | orientation)
plot_preference_maps(dat, thr=0.1);



# %% 6. photostimulation group dF/F activity
describe_photostim_groups(dat)
plot_photostim_group_heatmaps(dat, mode='zscore');


# %% 7. influence maps (photostim vs sham)
compute_influence(dat);
plot_influence_maps(dat);


# %% 8. influence maps by stimulus contrast
plot_influence_by_contrast(dat);


