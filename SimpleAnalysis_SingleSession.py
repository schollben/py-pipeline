# %% 1. initialize

%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis import (load_session, check_event_alignment, dropFirstEvents,
                      rebuild_cyc, compute_responses, plot_avg_rois,
                      plot_stim_traces, plot_tuning_curves, compute_selectivity,
                      plot_preference_maps, describe_photostim_groups,
                      plot_photostim_group_heatmaps, plot_photostim_target_traces,
                      influence_grand, influence_by_stim, influence_bootstrap,
                      plot_influence_maps, plot_influence_by_contrast)

sns.set_theme(context='notebook', style='white')

################################################################

# TEST: 07132025, good day for oricont and photostim

folderName = '/Users/benjaminscholl/Dropbox/projects/2poptostim/PROCESSED/V1/'

# FNAME = '/mnt/bigdata/PROCESSED/TSeries-07132025-1042-003.h5'

FNAME = folderName + 'TSeries-07132025-1042-003.h5'

################################################################

dat = load_session(FNAME)
print(f'{dat.exp_id}: {dat.n_rois} ROIs, '
      f'{len(dat.directions)} directions, {len(dat.contrasts)} contrasts')

# show average image with ROI mask
plot_avg_rois(dat,vmax_frac=0.6)


# %% check for a spurious first TTL pair before building cyc
if check_event_alignment(dat):
    dropFirstEvents(dat)


# %%  rebuild cyc from raw dff (un-blanked, wider) and inspect it to pick windows
# pre/post: seconds before/after stimulus onset to include in cyc (for plotting and response computation)
rebuild_cyc(dat, preStim=1, postStim=2);

# trial-averaged time-varying responses (one or more cells)
# always plots the full cyc window (as built by rebuild_cyc)
plot_stim_traces(dat, [1,5,10,15,20],
                 mask_artifact=False,
                 baseline_subtract=False,
                 trials='sham');

# recompute peak-minus-baseline responses from cyc: the pipeline's FFT-based
# resp/resps are all-NaN for opto sessions (FFT propagates the photostim blank).
# baseline/peak: windows read off the plot above (seconds relative to onset).
compute_responses(dat, baseline=(-1.0, -0.5), peak=(0.4, 1.2));


# %% 3. tuning curves + preferred direction (double-Gaussian fit)
# and compute direction / orientation selectivit
plot_tuning_curves(dat);
compute_selectivity(dat);
print(f'gDSI median {np.nanmedian(dat.gdsi):.3f}  |  gOSI median {np.nanmedian(dat.gosi):.3f}')


# %% 5. preference maps (direction | orientation)
plot_preference_maps(dat, thr=0.1);



# %% 6. photostimulation group dF/F activity
describe_photostim_groups(dat)
# plot_photostim_group_heatmaps(dat, mode='zscore');
plot_photostim_target_traces(dat, window=(-0.5, 1.5));


# %% 7. influence: grand average across all stimulus conditions
influence_grand(dat, baseline=(-1.0, -0.5), peak=(0.4, 1.2));
plot_influence_maps(dat);


# %% 8. influence maps by stimulus contrast
influence_by_stim(dat, baseline=(-1.0, -0.5), peak=(0.4, 1.2));
plot_influence_by_contrast(dat);


# %% 9. bootstrap influence (mean/SEM/CI over resampled trials)
influence_bootstrap(dat, by='grand', n_boot=1000, seed=0);
{tn: (v['grand'], v['sem']) for tn, v in dat.influence.items()}


