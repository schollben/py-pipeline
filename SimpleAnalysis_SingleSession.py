# %% initialize
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from analysis.photostim import influence_by_contrast, photostim_group_map
sns.set_theme(context='notebook', style='white')
from analysis import (load_session, check_event_alignment, dropFirstEvents,
                      rebuild_cyc, compute_responses, compute_snr,plot_avg_rois,
                      plot_stim_traces, plot_tuning_curves, compute_selectivity,
                      plot_preference_maps, describe_photostim_groups,
                      plot_photostim_target_traces, influence_grand, influence_by_contrast, 
                      influence_by_stim, influence_bootstrap,
                      plot_influence_maps, plot_influence_by_contrast)
from analysis import info


################################################################

# 07132025, good day for oricont and photostim, offsetFrames=-15
# 
session_name = 'TSeries-07132025-1042-003.h5'
folderName = '/Users/benjaminscholl/Dropbox/projects/2poptostim/PROCESSED/V1/'
FNAME = folderName + session_name


################################################################

dat = load_session(FNAME)
print(f'{dat.exp_id}: {dat.n_rois} ROIs, '
      f'{len(dat.directions)} directions, {len(dat.contrasts)} contrasts')
if dat.has_photostim==False:
    print("NO photostimulation data")

# resolution and pull hardcoded values from info.py
h, w = dat.avg_image.shape
dat.microns_per_pixel = info.getZoom(dat.optical_zoom) / min([h, w])
dat.offsetFrames = info.getOffsetFrames(session_name)

# build ROI locations in microns and get pairwise distances
ROI_locations = np.array([
    np.argwhere(dat.mask2d[c] > 0.5).mean(axis=0)[[1, 0]] 
    for c in range(dat.n_rois)
])
dat.roiLocs = ROI_locations * dat.microns_per_pixel
dat.dist = squareform(pdist(dat.roiLocs, metric='euclidean'))

# show average image with ROI mask
plot_avg_rois(dat,vmax_frac=0.6)

# check for a spurious first TTL pair before building cyc
if check_event_alignment(dat):
    dropFirstEvents(dat)
    

# %%  rebuild cyc from raw dff (un-blanked, wider) and inspect it to pick windows
# pre/post: seconds before/after stimulus onset to include in cyc (for plotting and response computation)
# offsetFrames: move the window earlier/later by this many frames (can be negative)
# for example TSeries-07132025-1042-003.h5 appears to have a ~15 frame lead in the event timing (PMT shutter begins BEFORE stimulus, which is not possible)

rebuild_cyc(dat, preStim=0.25, postStim=2, offsetFrames=dat.offsetFrames)

# trial-averaged time-varying responses (one or more cells)
# always plots the full cyc window (as built by rebuild_cyc)

plot_stim_traces(dat, [1,2,10,54],
                 mask_artifact=False,
                 baseline_subtract=True,
                 trials='sham');

# recompute peak-minus-baseline responses from cyc
# baseline/peak: windows read off the plot above, in seconds from the START of the cyc window (t=0 is the left edge of the plot)
# With preStim=0.25, postStim=2 the window spans 0 -> 2.25 s and visual onset sits at 0.25 s.----> eventually this will fixed once we understand the issues

baseline=info.getWindow(session_name)[0]
peak=info.getWindow(session_name)[1]
compute_responses(dat, baseline=baseline, peak=peak);
compute_snr(dat, baseline=baseline, peak=peak, thresh = 1);

remove_rois = info.removeROIs(session_name)
for n in remove_rois:
    dat.is_good_cell[n] = False


# %% 3. tuning curves + preferred direction (double-Gaussian fit) + preference map

# and compute direction / orientation selectivit
plot_tuning_curves(dat)
compute_selectivity(dat) 
#preference maps (direction | orientation)
plot_preference_maps(dat, thr=0.1)


# %% 4. photostimulation group dF/F activity

describe_photostim_groups(dat)
plot_photostim_target_traces(dat, baseline=baseline, peak=peak)


# %% examine nontarget-target relationships (independent of contrast) 
# influence: grand average across all stimulus conditions -> 
# windows are inherited from compute_responses above (via dat.resps), so influence
# and resp always measure the same thing; pass baseline=/peak= here only to override (not recommended)

influence_grand(dat, good_only=True, mode='dprime') # mode: diff or dprim
group_map = photostim_group_map(dat)
for tn, info in group_map.items():

    is_target = np.zeros(dat.n_rois, dtype=bool)
    is_target[info['target_rois']] = True

    print(is_target)

    grand = dat.influence[tn]['grand']
    valid = ~np.isnan(grand)

    print(grand)

# plot_influence_maps(dat, vlim=0.5)


# %% examine nontarget-target relationships -> DEPENDENT on contrast
# influence maps by stimulus contrast

influence_by_contrast(dat, good_only=True, mode='dprime') # mode: diff or dprime
plot_influence_by_contrast(dat,vlim=0.5); 

