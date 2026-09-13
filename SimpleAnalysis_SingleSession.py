# %% initialize
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import circmean
import seaborn as sns
from analysis.photostim import influence_by_contrast, photostim_group_map
sns.set_theme(context='notebook', style='white')
sns.set_theme(style='ticks')
from analysis import (load_session, check_event_alignment, dropFirstEvents,
                      rebuild_cyc, compute_responses, compute_snr,plot_avg_rois,
                      plot_stim_traces, plot_tuning_curves, compute_selectivity,
                      plot_preference_maps, describe_photostim_groups,
                      plot_photostim_target_traces, influence_grand, influence_by_contrast, 
                      influence_by_stim, influence_bootstrap,
                      plot_influence_maps, plot_influence_by_contrast)
from analysis import info


################################################################

# session_name = 'TSeries-07132025-1042-002.h5'
session_name = 'TSeries-11032024-1313-001.h5' # 11032024-1313-003, -007, -012, -014, -017.
folderName = '/Users/benjaminscholl/Dropbox/projects/2poptostim/PROCESSED/V1/'
FNAME = folderName + session_name


################################################################

dat = load_session(FNAME)
print(f'{dat.exp_id}: {dat.n_rois} ROIs, {dat.session_type} session')
if dat.has_visual:
    print(f'   {len(dat.directions)} directions, {len(dat.contrasts)} contrasts')
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
plot_avg_rois(dat,vmax_frac=0.6); 

# check for a spurious first TTL pair before building cyc
if check_event_alignment(dat):
    dropFirstEvents(dat)

if dat.has_photostim==False:
    print("NO photostimulation data")


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


# %% photostimulation group dF/F activity

describe_photostim_groups(dat)
plot_photostim_target_traces(dat, baseline=baseline, peak=peak); 


# %% tuning curves + preferred direction (double-Gaussian fit) + preference map
# skipped on a photostim-only session: no visual drive, so no tuning to measure

# and compute direction / orientation selectivity
plot_tuning_curves(dat);
compute_selectivity(dat)
#preference maps (direction | orientation)
plot_preference_maps(dat, thr=0.1);


# %% examine nontarget-target relationships (independent of contrast) 
# influence: grand average across all stimulus conditions -> 
# windows are inherited from compute_responses above (via dat.resps), so influence
# and resp always measure the same thing; pass baseline=/peak= here only to override (not recommended)

influence_grand(dat, good_only=True, mode='dprime') # mode: diff or dprim
group_map = photostim_group_map(dat)

# plot_influence_maps(dat, vlim=0.5); 

# loop through ensembles
for tn, info in group_map.items():

    is_target = np.zeros(dat.n_rois, dtype=bool)
    is_target[info['target_rois']] = True

    grand = dat.influence[tn]['grand']
    validinfl = ~np.isnan(grand) # nontarget cells valid influence measure

    # enesmble spatial and functional properties
    targetPoints = dat.roiLocs[is_target]

    avgLoc = np.mean(targetPoints, axis=0)
    spatSpread = np.sqrt(((targetPoints - avgLoc)**2).sum(1).mean()) #RMS

    avgCorr = np.corrcoef(dat.resp[is_target]) 
    avgCorr = np.tril(np.corrcoef(dat.resp[is_target]),k=-1)
    avgCorr = np.mean(avgCorr[avgCorr != 0])

    avgPref = (0.5) * circmean( 2 * (dat.pref_dir[is_target] % 180), high=360, low=0)

    # nontarget cell comparisons: distance, functional similarity, preference
    nontarget = ~is_target
    nontargetPoints = dat.roiLocs[nontarget]

    nonTargetEnsembleCorr = []
    nonTargetNearestCorr = []
    nonTargetEnsemblePref = []
    nonTargetNearestPref = []
    nonTargetEnsembleDist = [] 
    nonTargetNearestDist = [] 
    infl = []

    cells = np.where(nontarget & validinfl)[0]
    for n in cells:

        infl.append(grand[n])

        # average correlation with the target ensemble
        r = np.tril(np.corrcoef(dat.resp[n], dat.resp[is_target]),k=-1)
        r = r[r != 0].mean()
        nonTargetEnsembleCorr.append(r)

        # average correlation with the target ensemble
        angDiff = (dat.pref_dir[n] % 180 - avgPref + 90) % 180 - 90
        nonTargetEnsemblePref.append(np.abs(angDiff))

        d = np.linalg.norm(dat.roiLocs[n] - avgLoc)
        nonTargetEnsembleDist.append(d)

        # which target is closest to the nontarget?
        d = np.linalg.norm(dat.roiLocs[n] - targetPoints, axis=1)
        idx = np.argmin(d)
        nonTargetNearestDist.append(d[idx])

        # correlation with the nearest target
        r = np.corrcoef(dat.resp[n], dat.resp[info['target_rois'][idx]])
        nonTargetNearestCorr.append(r[0,1])

        # preference difference with the nearest target
        angDiff = ( (dat.pref_dir[n] % 180) - (dat.pref_dir[info['target_rois'][idx]] % 180) + 90) % 180 - 90
        nonTargetNearestPref.append(np.abs(angDiff))


    # scatter plots per group
    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); 

    sns.scatterplot(x=nonTargetEnsembleDist,    y=infl, ax=axes[0,0], 
                    s=80, marker='o',color='b',alpha=0.8,edgecolor='w',linewidth=1); 
    sns.scatterplot(x=nonTargetEnsemblePref,    y=infl, ax=axes[0,1], 
                    s=80, marker='o',color='b',alpha=0.8,edgecolor='w',linewidth=1); 
    sns.scatterplot(x=nonTargetEnsembleCorr,    y=infl, ax=axes[0,2], 
                    s=80, marker='o',color='b',alpha=0.8,edgecolor='w',linewidth=1); 
    sns.scatterplot(x=nonTargetNearestDist,     y=infl, ax=axes[1,0], 
                    s=80, marker='o',color='b',alpha=0.8,edgecolor='w',linewidth=1); 
    sns.scatterplot(x=nonTargetNearestPref,     y=infl, ax=axes[1,1], 
                    s=80, marker='o',color='b',alpha=0.8,edgecolor='w',linewidth=1); 
    sns.scatterplot(x=nonTargetNearestCorr,     y=infl, ax=axes[1,2],
                    s=80, marker='o',color='b',alpha=0.8,edgecolor='w',linewidth=1); 

    sns.lineplot(x=[0, 300], y=[0,0],ax=axes[0,0], linestyle='--',color='k'); 
    sns.lineplot(x=[0, 100], y=[0,0],ax=axes[0,1], linestyle='--',color='k'); 
    sns.lineplot(x=[0, 1],   y=[0, 0],ax=axes[0,2], linestyle='--',color='k'); 
    sns.lineplot(x=[0, 300], y=[0, 0],ax=axes[1,0], linestyle='--',color='k'); 
    sns.lineplot(x=[0, 100], y=[0, 0],ax=axes[1,1], linestyle='--',color='k'); 
    sns.lineplot(x=[0, 1],   y=[0, 0],ax=axes[1,2], linestyle='--',color='k'); 

    for ax, lbl in zip(axes.ravel(), ['ensemble dist','ensemble angd','ensemble r',
                                    'nearest dist','nearest angd','nearest r']):
        ax.set_xlabel(lbl, fontsize=18, fontname='Arial')
        ax.set_ylabel('influence', fontsize=18, fontname='Arial')
        ax.set_box_aspect(1)

    fig.tight_layout()
    sns.despine(fig=fig)
    axes[0,1].set_title(f'spread {spatSpread:.1f} µm , avg r {avgCorr:.2f} , pref {avgPref:.0f}°',
                        fontsize=18, pad=10);


# %% same relationships, split by contrast
# influence per contrast bin -> x-variables are contrast-independent, so they are
# computed once per nontarget cell and reused for every contrast series

influence_by_contrast(dat, good_only=True, mode='dprime');  # mode: diff or dprime
group_map = photostim_group_map(dat)

plot_influence_by_contrast(dat,vlim=0.5); 

# marker style per contrast: 0 = white face/black edge, mid = gray face, max = black face
def contrast_style(c, contrasts):
    if c == 0:
        return dict(color='w', edgecolor='k')
    if c == max(contrasts):
        return dict(color='k', edgecolor='w')
    return dict(color='0.6', edgecolor='w')

# loop through ensembles
for tn, info in group_map.items():

    is_target = np.zeros(dat.n_rois, dtype=bool)
    is_target[info['target_rois']] = True

    inflAll = dat.influence[tn]['influence']   # (n_cells, n_contrasts)
    contrasts = dat.influence[tn]['labels']
    validinfl = ~np.isnan(inflAll).any(axis=1) # valid at every contrast

    # enesmble spatial and functional properties
    targetPoints = dat.roiLocs[is_target]

    avgLoc = np.mean(targetPoints, axis=0)
    spatSpread = np.sqrt(((targetPoints - avgLoc)**2).sum(1).mean()) #RMS

    avgCorr = np.tril(np.corrcoef(dat.resp[is_target]),k=-1)
    avgCorr = np.mean(avgCorr[avgCorr != 0])

    avgPref = (0.5) * circmean( 2 * (dat.pref_dir[is_target] % 180), high=360, low=0)

    # nontarget cell comparisons: distance, functional similarity, preference
    nontarget = ~is_target

    nonTargetEnsembleCorr = []
    nonTargetNearestCorr = []
    nonTargetEnsemblePref = []
    nonTargetNearestPref = []
    nonTargetEnsembleDist = []
    nonTargetNearestDist = []

    cells = np.where(nontarget & validinfl)[0]
    for n in cells:

        # average correlation with the target ensemble
        r = np.tril(np.corrcoef(dat.resp[n], dat.resp[is_target]),k=-1)
        r = r[r != 0].mean()
        nonTargetEnsembleCorr.append(r)

        # preference difference with the target ensemble
        angDiff = (dat.pref_dir[n] % 180 - avgPref + 90) % 180 - 90
        nonTargetEnsemblePref.append(np.abs(angDiff))

        d = np.linalg.norm(dat.roiLocs[n] - avgLoc)
        nonTargetEnsembleDist.append(d)

        # which target is closest to the nontarget?
        d = np.linalg.norm(dat.roiLocs[n] - targetPoints, axis=1)
        idx = np.argmin(d)
        nonTargetNearestDist.append(d[idx])

        # correlation with the nearest target
        r = np.corrcoef(dat.resp[n], dat.resp[info['target_rois'][idx]])
        nonTargetNearestCorr.append(r[0,1])

        # preference difference with the nearest target
        angDiff = ( (dat.pref_dir[n] % 180) - (dat.pref_dir[info['target_rois'][idx]] % 180) + 90) % 180 - 90
        nonTargetNearestPref.append(np.abs(angDiff))

    # scatter plots per group, one series per contrast
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    panels = [(axes[0,0], nonTargetEnsembleDist, 'ensemble dist',  300),
              (axes[0,1], nonTargetEnsemblePref, 'ensemble angd',  100),
              (axes[0,2], nonTargetEnsembleCorr, 'ensemble r',       1),
              (axes[1,0], nonTargetNearestDist,  'nearest dist',   300),
              (axes[1,1], nonTargetNearestPref,  'nearest angd',   100),
              (axes[1,2], nonTargetNearestCorr,  'nearest r',        1)]

    for ax, x, lbl, xmax in panels:
        for ci, c in enumerate(contrasts):
            sns.scatterplot(x=x, y=inflAll[cells, ci], ax=ax, s=80, marker='o',
                            alpha=0.8, linewidth=1, label=f'{c:g}%',
                            **contrast_style(c, contrasts))
        sns.lineplot(x=[0, xmax], y=[0, 0], ax=ax, linestyle='--', color='k')
        ax.set_xlabel(lbl, fontsize=18, fontname='Arial')
        ax.set_ylabel('influence', fontsize=18, fontname='Arial')
        ax.set_box_aspect(1)
        ax.legend_.remove() if ax is not axes[0,2] else ax.legend(title='contrast', frameon=False)

    fig.tight_layout()
    sns.despine(fig=fig)
    axes[0,1].set_title(f'spread {spatSpread:.1f} µm , avg r {avgCorr:.2f} , pref {avgPref:.0f}°',
                        fontsize=18, pad=10);


# The pairing holds because both sides are indexed by the same cells array, in the same order:

# cells = np.where(nontarget & validinfl)[0] — sorted ROI indices, fixed order
# The for n in cells: loop appends one value per iteration, so nonTargetEnsembleDist[k] corresponds to cells[k]
# inflAll[cells, ci] uses that same array as a fancy index, so element k is inflAll[cells[k], ci] — the same ROI
# Both are length len(cells) with matching positions. The contrast is correct because ci comes from enumerate(contrasts) where contrasts = dat.influence[tn]['labels'], and labels is built in column order by influence_by_contrast (bins = [... for c in s.contrasts] at photostim.py:569), so column ci is contrast labels[ci].

# To verify rather than trust it, assert inside the plotting loop:


# assert len(x) == len(cells)
# k = 5  # any index
# assert np.isclose(nonTargetEnsembleDist[k],
#                   np.linalg.norm(dat.roiLocs[cells[k]] - avgLoc))
# The fragile part isn't the indexing — it's that the lists are built by append in one loop and consumed in another. If you ever add a continue inside the cell loop, the lists would silently go out of sync with cells while staying plausible. Converting to arrays computed directly from cells removes that failure mode entirely:


# nonTargetEnsembleDist = np.linalg.norm(dat.roiLocs[cells] - avgLoc, axis=1)
# That one is a direct vectorization; the nearest-target and correlation quantities need the loop, but you could assemble them as np.array(...) right after and assert the lengths match once.
