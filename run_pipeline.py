# run_pipeline.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from bruker_pipeline import process_experiment

######## Set parameters for the experiment to process ########
DATE            = '04032026'    # acquisition date, format MMDDYYYY
FILE_NUM        = 1             # TSeries number (e.g. 3 → matches folder ending in -003)
STIM_FILE       = 1             # PsychoPy file, Set to -1 if there is no stimulus file.
IS_SPONTANEOUS  = True          # True  → experiment has NO visual stimulus (spontaneous activity only)
USE_INFERENCE   = True          # True  → use inference_results.h5
DUR_RESP        = 2             # Response window duration in seconds to build the trial-averaged response matrix (cyc)

IS_2P_OPTO      = True          # True  → experiment includes 2-photon photostimulation (optogenetics), Will read MarkPoints XML and compute opto % change images
OPTO_POST_SEC   = 0.2           # seconds after the blanking window to average (response)
OPTO_PRE_SEC    = 0.5           # seconds before trigger onset to average (baseline)
OPTO_DUR        = 0.4           # how long is opto (when shutter CLOSED)

DO_PLOT         = True          # True → generate and save a summary figure (ROIs, MarkPoints, opto images).

# Known acquisition bug: photostim trigger stream is offset by 1 row relative
# to the PsychoPy file. Keep True until the bug is fixed in the acquisition software.
OPTO_OFFSET     = True

######## Run the processing pipeline with the specified parameters ########
if __name__ == '__main__':
    result = process_experiment(
        date                 = DATE,
        file_num             = FILE_NUM,
        stim_file            = STIM_FILE,
        is_spontaneous       = IS_SPONTANEOUS,
        is_2p_opto           = IS_2P_OPTO,
        use_inference        = USE_INFERENCE,
        do_neuropil          = False,
        dur_resp             = DUR_RESP,
        opto_post_sec        = OPTO_POST_SEC,
        opto_pre_sec         = OPTO_PRE_SEC,
        opto_blank_sec       = OPTO_DUR,
        do_plot              = DO_PLOT,
        do_vrec_diagnostic   = False,
        opto_offset_trigger  = OPTO_OFFSET,
    )
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import ndimage

    SHAM_IDX  = 0
    OPTO_IDX  = 1
    UM_PER_PX = 1.11   # from TSeries XML micronsPerPixel (XAxis/YAxis)

    dff      = result['dff']                  # (n_frames, n_rois)
    mask2d   = result['mask2d']               # (n_rois, H, W)
    cyc_opto = result['cyc_photostim_only']   # (n_rois, n_groups, max_trials)
    rpp      = result['roi_photostim_point']  # (n_rois,)  -1 = nontarget
    exp_id   = result.get('experiment_id', result.get('data_dir', 'unknown'))

    nontarget_mask = rpp == -1
    n_nt  = nontarget_mask.sum()
    n_tgt = (~nontarget_mask).sum()

    # ── centroids and distance matrix (nontargets only) ───────────────────────
    centroids_pix = np.array([ndimage.center_of_mass(m) for m in mask2d])
    nt_centroids  = centroids_pix[nontarget_mask]
    nt_dff        = dff[:, nontarget_mask]

    diff        = nt_centroids[:, np.newaxis, :] - nt_centroids[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2)) * UM_PER_PX   # (n_nt, n_nt)

    # ── 1. Correlation matrix ─────────────────────────────────────────────────
    corr_matrix = np.corrcoef(nt_dff.T)   # (n_nt, n_nt)

    # ── 2. Pairwise correlation vs distance ───────────────────────────────────
    tri_idx   = np.triu_indices(n_nt, k=1)
    corr_vals = corr_matrix[tri_idx]
    dist_vals = dist_matrix[tri_idx]

    bin_edges   = np.arange(0, min(np.nanmax(dist_matrix) + 25, 600), 25)
    bin_centers, bin_mean, bin_sem = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        v = corr_vals[(dist_vals >= lo) & (dist_vals < hi)]
        bin_centers.append((lo + hi) / 2)
        if len(v) > 1:
            bin_mean.append(np.mean(v))
            bin_sem.append(np.std(v, ddof=1) / np.sqrt(len(v)))
        else:
            bin_mean.append(np.nan)
            bin_sem.append(np.nan)
    bin_centers = np.array(bin_centers)
    bin_mean    = np.array(bin_mean)
    bin_sem     = np.array(bin_sem)

    # ── 3. Influence = opto_mean − sham_mean (nontargets) ────────────────────
    opto_mean    = np.nanmean(cyc_opto[:, OPTO_IDX, :], axis=1)
    sham_mean    = np.nanmean(cyc_opto[:, SHAM_IDX, :], axis=1)
    influence    = opto_mean - sham_mean
    nt_influence = influence[nontarget_mask]
    tgt_influence = influence[~nontarget_mask]

    n_trials_opto = int((~np.isnan(cyc_opto[0, OPTO_IDX, :])).sum())

    print(f"Trials     : opto={n_trials_opto}  sham={int((~np.isnan(cyc_opto[0, SHAM_IDX, :])).sum())}")
    print(f"Nontarget influence — mean: {np.nanmean(nt_influence):.4f}  median: {np.nanmedian(nt_influence):.4f}")
    print(f"Target influence    — mean: {np.nanmean(tgt_influence):.4f}  (should be positive)")

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # Panel 1 — correlation matrix
    ax1  = fig.add_subplot(gs[0])
    vmax = np.nanpercentile(np.abs(corr_matrix[~np.eye(n_nt, dtype=bool)]), 99)
    im   = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      interpolation='nearest')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Pearson r')
    ax1.set_title(f'Correlation matrix\n(nontarget cells, n={n_nt})', fontsize=11)
    ax1.set_xlabel('Cell index')
    ax1.set_ylabel('Cell index')

    # Panel 2 — correlation vs distance
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(dist_vals, corr_vals, s=1, alpha=0.2,
                color='steelblue', rasterized=True, zorder=1)
    valid = ~np.isnan(bin_mean)
    ax2.errorbar(bin_centers[valid], bin_mean[valid], yerr=bin_sem[valid],
                 fmt='o-', color='k', ms=4, lw=1.5, capsize=3, zorder=2,
                 label='Mean ± SEM')
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel('Cortical distance (µm)')
    ax2.set_ylabel('Pearson r')
    ax2.set_title('Correlation vs distance\n(nontarget cells)', fontsize=11)
    ax2.legend(fontsize=8)

    # Panel 3 — influence distribution
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(nt_influence, bins=max(15, int(np.sqrt(n_nt) * 1.5)),
             color='steelblue', edgecolor='white', lw=0.4)
    ax3.axvline(0, color='k', lw=1.2, ls='--', label='zero')
    ax3.axvline(np.nanmean(nt_influence), color='tomato', lw=1.8,
                label=f'mean = {np.nanmean(nt_influence):.4f}')
    frac_supp = (nt_influence < 0).mean()
    ax3.text(0.97, 0.97,
             f'Suppressed: {frac_supp:.1%}\nActivated: {1-frac_supp:.1%}\n'
             f'n={n_nt} cells | {n_trials_opto} opto trials',
             transform=ax3.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    ax3.set_xlabel('Influence (opto − sham dF/F)')
    ax3.set_ylabel('Number of cells')
    ax3.set_title('Nontarget cell influence\ndistribution', fontsize=11)
    ax3.legend(fontsize=8)

    fig.suptitle(exp_id, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()

