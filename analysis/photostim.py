import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from .session import cyc_response_windows, cyc_onset

_TARGET_COLOR = 'purple'
_NONTARGET_COLOR = 'gray'


def photostim_group_map(s):
    """Map each REAL (power > 0) photostim group to its paired sham group.

    Returns a dict keyed by real `target_number` (float), each value a dict:
        power        - laser power (mW) of the real group
        sham         - target_number of the paired sham group (power == 0)
        target_rois  - sorted array of ROI indices targeted by this group

    Pairing uses `markpoints_group_info` columns [condition_idx, unique_group_id,
    n_targets, dispersion]: conditions sharing a unique_group_id target the same
    cells, so a power>0 condition and its power==0 sibling are the real/sham pair.
    `target_number` is assumed to be `condition_idx + offset` where `offset` is
    derived from the data (min unique target_number), not hard-coded.
    """
    if not s.has_photostim:
        raise ValueError(f'{s.exp_id}: no photostimulation data in this session.')

    group_info = s.markpoints_group_info      # (n_conds, 4)
    powers = s.markpoints_laser_power         # (n_conds,)
    cond_idx = group_info[:, 0].astype(int)
    unique_group_id = group_info[:, 1].astype(int)

    tn_unique = np.unique(s.target_number)
    offset = int(round(tn_unique.min())) - int(cond_idx.min())

    def cond_to_tn(ci):
        return ci + offset

    def tn_to_cond(tn):
        return int(round(tn)) - offset

    mp_cond = s.markpoints_condition_idx      # (n_points,) condition per markpoint
    mp_roi = s.markpoint_assigned_roi         # (n_points,) ROI per markpoint, -1=none

    result = {}
    for ci in cond_idx:
        if powers[ci] <= 0:
            continue
        uid = unique_group_id[ci]
        sibling_conds = cond_idx[unique_group_id == uid]
        sham_conds = [c for c in sibling_conds if powers[c] == 0]
        if not sham_conds:
            continue
        sham_cond = sham_conds[0]

        target_rois = set()
        for c in sibling_conds:
            pts = np.where(mp_cond == c)[0]
            rois = mp_roi[pts]
            target_rois.update(int(r) for r in rois if r >= 0)

        real_tn = cond_to_tn(ci)
        result[real_tn] = dict(
            power=float(powers[ci]),
            sham=cond_to_tn(sham_cond),
            target_rois=np.array(sorted(target_rois), dtype=int),
        )
    return result


def cyc_trial_group(s):
    """(n_stims, n_trials) array: target_number of each cyc trial slot, NaN if none.

    Reproduces gen_stim_cyc's trial-fill ordering: iterate presentations in
    stim_id order, and for the k-th occurrence of a given stim, that cyc trial
    slot [stim_index, k] is filled from that presentation. `target_number` is
    directly index-aligned with `stim_id` (no offset) — verified empirically:
    this pairing gives real photostim groups a much larger response in their
    targeted cells than their sham pair; a shifted pairing gives the reverse.
    """
    stim_id = s.stim_id
    unique_stims = s.unique_stims
    target_number = s.target_number
    n_stims = len(unique_stims)
    n_trials = s.cyc.shape[2]
    n = min(len(stim_id), len(target_number))

    grp = np.full((n_stims, n_trials), np.nan)
    trial_count = np.zeros(n_stims, dtype=int)
    for i in range(n):
        ind = np.where(unique_stims == stim_id[i])[0][0]
        k = trial_count[ind]
        if k < n_trials:
            grp[ind, k] = target_number[i]
        trial_count[ind] += 1
    return grp


def describe_photostim_groups(s):
    """Print the real/sham group pairing, powers, target ROIs, and trial counts."""
    if not s.has_photostim:
        print(f'{s.exp_id}: no photostimulation data in this session.')
        return

    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)

    print(f'=== Photostimulation groups: {s.exp_id} ===')
    for real_tn, info in sorted(gmap.items()):
        n_real = int(np.sum(grp == real_tn))
        n_sham = int(np.sum(grp == info['sham']))
        print(f'  group {real_tn:g} (power={info["power"]:.0f} mW)  '
              f'paired sham={info["sham"]:g}  '
              f'targets={info["target_rois"].tolist()}  '
              f'trials: real={n_real} sham={n_sham}')


def plot_photostim_group_heatmaps(s, mode='raw'):
    """Trial-averaged per-cell activity heatmap for each real photostim group.

    mode: 'raw' (cell-by-cell trial average), 'norm' (per-cell peak-normalized),
    or 'zscore' (per-cell z-scored across time).
    """
    if mode not in ('raw', 'norm', 'zscore'):
        raise ValueError("mode must be 'raw', 'norm', or 'zscore'")

    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    real_groups = sorted(gmap.keys())
    n_frames = s.cyc.shape[3]
    pre_frames = int(round(1.0 / s.frame_period)) if s.frame_period else 0

    fig, axes = plt.subplots(1, len(real_groups),
                              figsize=(6 * len(real_groups), 8), squeeze=False)

    for ax, real_tn in zip(axes[0], real_groups):
        target_rois = gmap[real_tn]['target_rois']
        mask = grp == real_tn
        stim_idx, trial_idx = np.where(mask)
        traces = s.cyc[:, stim_idx, trial_idx, :]        # (n_cells, n_sel, n_frames)
        avg = np.nanmean(traces, axis=1)                 # (n_cells, n_frames)

        is_target = np.zeros(s.n_rois, dtype=bool)
        is_target[target_rois] = True
        order = np.argsort(~is_target, kind='stable')    # targets first
        avg_sorted = avg[order]
        is_target_sorted = is_target[order]

        # frames fully NaN across all cells = the photostim-pulse blanking window;
        # exclude them from norm/zscore statistics so the artifact-adjacent dip
        # doesn't wash out genuine post-stimulus response structure.
        valid_frame = ~np.all(np.isnan(avg_sorted), axis=0)

        if mode == 'norm':
            peak = np.nanmax(np.abs(avg_sorted[:, valid_frame]), axis=1, keepdims=True)
            peak[peak == 0] = 1
            data = avg_sorted / peak
            vlim = 1.0
        elif mode == 'zscore':
            mu = np.nanmean(avg_sorted[:, valid_frame], axis=1, keepdims=True)
            sd = np.nanstd(avg_sorted[:, valid_frame], axis=1, keepdims=True)
            sd[sd == 0] = 1
            data = (avg_sorted - mu) / sd
            vlim = np.nanpercentile(np.abs(data[:, valid_frame]), 99)
        else:
            data = avg_sorted
            vlim = np.nanpercentile(np.abs(data[:, valid_frame]), 99)

        cmap = plt.get_cmap('coolwarm').copy()
        cmap.set_bad('lightgray')
        im = ax.imshow(data, aspect='auto', cmap=cmap,
                        vmin=-vlim, vmax=vlim, interpolation='nearest')
        ax.axvline(pre_frames, color='black', lw=1, ls='--')
        for row in np.where(is_target_sorted)[0]:
            ax.plot(-1.5, row, marker='o', color=_TARGET_COLOR,
                    clip_on=False, markersize=5)
        ax.set_xlim(-3, n_frames)
        ax.set_title(f'group {real_tn:g} ({mode})')
        ax.set_xlabel('frame (photostim onset = dashed line)')
        if ax is axes[0, 0]:
            ax.set_ylabel('cells (targets first)')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(s.exp_id)
    fig.tight_layout()
    return fig


def group_trial_resp(s, target_tn, grp, base_sl, peak_sl):
    """(n_cells, n_stims, n_trials) per-trial peak-minus-baseline response.

    Trial-preserving replacement for the old average-then-peak measurement:
    baseline and peak are computed per trial (not on the trial-averaged trace)
    so the trial axis survives for mean/SEM and bootstrap use downstream.
    NaN where a (stim, trial) slot isn't filled for this group.

    grp : (n_stims, n_trials) target_number per cyc trial slot, from
        `cyc_trial_group`. base_sl, peak_sl : frame slices from
        `cyc_response_windows`.
    """
    n_stims = len(s.unique_stims)
    n_trials = s.cyc.shape[2]
    resp = np.full((s.n_rois, n_stims, n_trials), np.nan)
    for si in range(n_stims):
        trial_idx = np.where(grp[si] == target_tn)[0]
        if len(trial_idx) == 0:
            continue
        traces = s.cyc[:, si, trial_idx, :]                   # (n_cells, n_sel, n_frames)
        baseline = np.nanmean(traces[..., base_sl], axis=-1)  # (n_cells, n_sel)
        post = np.nanmax(traces[..., peak_sl], axis=-1)       # (n_cells, n_sel)
        resp[:, si, trial_idx] = post - baseline
    return resp


def influence_grand(s, baseline_guard_sec=0.5, post_sec=1.0,
                    baseline=None, peak=None):
    """Grand-average influence of each target group on all nontargets.

    Diagnostic entry point: pools every trial of the real group and every trial
    of its paired sham (across all stimulus conditions), takes the mean of each,
    and forms:

        influence[cell] = (mean(resp_real) - mean(resp_sham)) / mean(resp_sham)

    Per-trial responses come from `group_trial_resp` (peak-minus-baseline
    computed per trial, not on the trial-averaged trace — see that function's
    docstring for why). Baseline / peak windows come from `cyc_response_windows`;
    pass explicit `baseline=`/`peak=` (seconds from the start of the cyc window) for windows read
    off a `rebuild_cyc`-produced cyc.

    Sets s.influence = {real_tn: {'grand': (n_cells,), 'kind': 'grand'}} and
    returns the same dict.
    """
    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    base_sl, peak_sl = cyc_response_windows(s, baseline_guard_sec, post_sec,
                                             baseline, peak)

    influence = {}
    for real_tn, info in gmap.items():
        resp_real = group_trial_resp(s, real_tn, grp, base_sl, peak_sl)
        resp_sham = group_trial_resp(s, info['sham'], grp, base_sl, peak_sl)
        mean_real = np.nanmean(resp_real, axis=(1, 2))    # (n_cells,)
        mean_sham = np.nanmean(resp_sham, axis=(1, 2))    # (n_cells,)
        with np.errstate(invalid='ignore', divide='ignore'):
            grand = np.where(np.abs(mean_sham) > 1e-6,
                             (mean_real - mean_sham) / mean_sham, np.nan)
        influence[real_tn] = dict(grand=grand, kind='grand')

    s.influence = influence
    return influence


def influence_by_stim(s, baseline_guard_sec=0.5, post_sec=1.0,
                      baseline=None, peak=None):
    """Influence of each target group on all nontargets, per stimulus condition.

    Same measure as `influence_grand`, but trials are pooled within each
    stimulus id rather than across all of them:

        influence[cell, stim] = (mean(resp_real) - mean(resp_sham)) / mean(resp_sham)

    Sets s.influence = {real_tn: {'influence': (n_cells, n_stims),
    'grand': (n_cells,), 'kind': 'stim'}} and returns the same dict. `grand` is
    the mean of `influence` across stims, for `plot_influence_maps`.
    """
    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    base_sl, peak_sl = cyc_response_windows(s, baseline_guard_sec, post_sec,
                                             baseline, peak)

    influence = {}
    for real_tn, info in gmap.items():
        resp_real = group_trial_resp(s, real_tn, grp, base_sl, peak_sl)
        resp_sham = group_trial_resp(s, info['sham'], grp, base_sl, peak_sl)
        mean_real = np.nanmean(resp_real, axis=2)    # (n_cells, n_stims)
        mean_sham = np.nanmean(resp_sham, axis=2)    # (n_cells, n_stims)
        with np.errstate(invalid='ignore', divide='ignore'):
            inf = np.where(np.abs(mean_sham) > 1e-6,
                           (mean_real - mean_sham) / mean_sham, np.nan)
        influence[real_tn] = dict(
            influence=inf,
            grand=np.nanmean(inf, axis=1),
            kind='stim',
        )

    s.influence = influence
    return influence


def influence_bootstrap(s, by='grand', n_boot=1000, seed=None,
                        baseline_guard_sec=0.5, post_sec=1.0,
                        baseline=None, peak=None):
    """Bootstrap distribution of influence over trials.

    by : 'grand' (trials pooled across all stimulus conditions, matching
        `influence_grand`) or 'stim' (trials pooled per stimulus condition,
        matching `influence_by_stim`).

    For each cell (and, when by='stim', each stimulus condition), resamples the
    pooled real trials and the pooled sham trials *independently* with
    replacement (they are separate trial sets, not paired), recomputes
    mean(real) / mean(sham) and the ratio, and repeats `n_boot` times. Reports
    the point estimate (from `influence_grand`/`influence_by_stim`, unresampled),
    the bootstrap SEM (SD of the resampled distribution), and the 2.5/97.5
    percentile CI.

    Sets s.influence = {real_tn: {
        'grand' or 'influence': point estimate (as in influence_grand/by_stim),
        'sem': same shape, bootstrap SD,
        'ci_lo', 'ci_hi': same shape, 2.5th/97.5th percentiles,
        'kind': 'bootstrap', 'by': by, 'n_boot': n_boot, 'seed': seed,
    }} and returns the same dict.
    """
    if by not in ('grand', 'stim'):
        raise ValueError("by must be 'grand' or 'stim'")

    point = (influence_grand if by == 'grand' else influence_by_stim)(
        s, baseline_guard_sec, post_sec, baseline, peak)

    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    base_sl, peak_sl = cyc_response_windows(s, baseline_guard_sec, post_sec,
                                             baseline, peak)
    rng = np.random.default_rng(seed)

    influence = {}
    for real_tn, info in gmap.items():
        resp_real = group_trial_resp(s, real_tn, grp, base_sl, peak_sl)
        resp_sham = group_trial_resp(s, info['sham'], grp, base_sl, peak_sl)
        # (n_cells, n_stims, n_trials) -> pool trials per the 'by' scope
        if by == 'grand':
            real_pool = resp_real.reshape(s.n_rois, -1)   # (n_cells, n_stims*n_trials)
            sham_pool = resp_sham.reshape(s.n_rois, -1)
            boot = np.full((s.n_rois, n_boot), np.nan)
            for b in range(n_boot):
                r = rng.choice(real_pool.shape[1], real_pool.shape[1], replace=True)
                h = rng.choice(sham_pool.shape[1], sham_pool.shape[1], replace=True)
                mr = np.nanmean(real_pool[:, r], axis=1)
                mh = np.nanmean(sham_pool[:, h], axis=1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    boot[:, b] = np.where(np.abs(mh) > 1e-6, (mr - mh) / mh, np.nan)
            influence[real_tn] = dict(
                grand=point[real_tn]['grand'],
                sem=np.nanstd(boot, axis=1),
                ci_lo=np.nanpercentile(boot, 2.5, axis=1),
                ci_hi=np.nanpercentile(boot, 97.5, axis=1),
                kind='bootstrap', by=by, n_boot=n_boot, seed=seed,
            )
        else:
            n_stims = resp_real.shape[1]
            sem = np.full((s.n_rois, n_stims), np.nan)
            ci_lo = np.full((s.n_rois, n_stims), np.nan)
            ci_hi = np.full((s.n_rois, n_stims), np.nan)
            for si in range(n_stims):
                real_pool = resp_real[:, si, :]    # (n_cells, n_trials)
                sham_pool = resp_sham[:, si, :]
                boot = np.full((s.n_rois, n_boot), np.nan)
                for b in range(n_boot):
                    r = rng.choice(real_pool.shape[1], real_pool.shape[1], replace=True)
                    h = rng.choice(sham_pool.shape[1], sham_pool.shape[1], replace=True)
                    mr = np.nanmean(real_pool[:, r], axis=1)
                    mh = np.nanmean(sham_pool[:, h], axis=1)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        boot[:, b] = np.where(np.abs(mh) > 1e-6, (mr - mh) / mh, np.nan)
                sem[:, si] = np.nanstd(boot, axis=1)
                ci_lo[:, si] = np.nanpercentile(boot, 2.5, axis=1)
                ci_hi[:, si] = np.nanpercentile(boot, 97.5, axis=1)
            influence[real_tn] = dict(
                influence=point[real_tn]['influence'],
                grand=point[real_tn]['grand'],
                sem=sem, ci_lo=ci_lo, ci_hi=ci_hi,
                kind='bootstrap', by=by, n_boot=n_boot, seed=seed,
            )

    s.influence = influence
    return influence


def _draw_influence_map(ax, s, grand_influence, target_rois, title, vlim=None):
    """Facecolor-by-influence overlay: PRGn fill for nontargets, purple/gray outlines."""
    if vlim is None:
        vlim = np.nanpercentile(np.abs(grand_influence), 99)
        if not np.isfinite(vlim) or vlim == 0:
            vlim = 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    cmap = plt.get_cmap('PRGn')

    ax.imshow(s.avg_image, cmap='gray', vmax=s.avg_image.max() / 2)
    is_target = np.zeros(s.n_rois, dtype=bool)
    is_target[target_rois] = True

    for cc in range(s.n_rois):
        m = s.mask2d[cc] > 0.5
        if is_target[cc]:
            ax.contour(s.mask2d[cc], levels=[0.5], colors=[_TARGET_COLOR], linewidths=1.2)
            continue
        ax.contour(s.mask2d[cc], levels=[0.5], colors=[_NONTARGET_COLOR], linewidths=0.6)
        val = grand_influence[cc]
        if np.isnan(val):
            continue
        rgb = np.array(cmap(norm(val))[:3])
        ax.imshow(np.dstack([np.ones_like(s.avg_image)] * 3) * rgb,
                  alpha=m.astype(float) * 0.7)

    ax.set_title(title)
    ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, fraction=0.046, label='influence')


def plot_influence_maps(s, influence=None):
    """One spatial influence map per real photostim group (grand mean across stims)."""
    if influence is None:
        if s.influence is None:
            raise ValueError(
                f'{s.exp_id}: no influence computed; run rebuild_cyc() -> '
                'compute_responses() -> influence_grand(...) first')
        influence = s.influence
    gmap = photostim_group_map(s)
    real_groups = sorted(influence.keys())

    fig, axes = plt.subplots(1, len(real_groups),
                              figsize=(6 * len(real_groups), 6), squeeze=False)
    for ax, real_tn in zip(axes[0], real_groups):
        _draw_influence_map(ax, s, influence[real_tn]['grand'],
                            gmap[real_tn]['target_rois'], f'group {real_tn:g}')
    fig.suptitle(f'{s.exp_id}  |  influence (grand mean across stims)')
    fig.tight_layout()
    return fig


def plot_influence_by_contrast(s, influence=None):
    """Spatial influence maps split by stimulus contrast (rows=groups, cols=contrast)."""
    if influence is None:
        if s.influence is None:
            raise ValueError(
                f'{s.exp_id}: no influence computed; run rebuild_cyc() -> '
                'compute_responses() -> influence_by_stim(...) first')
        influence = s.influence
    gmap = photostim_group_map(s)
    real_groups = sorted(influence.keys())
    contrasts = s.contrasts
    stim_contrast = s.stim_table[:, 1]

    fig, axes = plt.subplots(len(real_groups), len(contrasts),
                              figsize=(5 * len(contrasts), 5 * len(real_groups)),
                              squeeze=False)
    for row, real_tn in enumerate(real_groups):
        inf = influence[real_tn]['influence']              # (n_cells, n_stims)
        for col, contrast in enumerate(contrasts):
            stim_mask = stim_contrast == contrast
            grand = np.nanmean(inf[:, stim_mask], axis=1)
            _draw_influence_map(axes[row, col], s, grand, gmap[real_tn]['target_rois'],
                                f'group {real_tn:g}  |  contrast {contrast:g}')
    fig.suptitle(f'{s.exp_id}  |  influence by contrast')
    fig.tight_layout()
    return fig


def plot_photostim_target_traces(s, window=None):
    """Trial-averaged cyc timecourse of each targeted ROI, real vs sham.

    Grid of line plots: one row per real photostim group/ensemble, one column per
    targeted ROI in that group. Each subplot overlays the target's real (power>0)
    and paired sham (power=0) trial-averaged traces (mean +/- SEM), collapsed
    across visual stims (as plot_photostim_group_heatmaps does).

    window : (t0, t1) in seconds relative to visual onset to crop the displayed
        frames; None shows the full cyc window.
    """
    if not s.has_photostim:
        print(f'{s.exp_id}: no photostimulation data in this session.')
        return

    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    real_groups = sorted(gmap.keys())
    n_frames = s.cyc.shape[3]
    onset = cyc_onset(s)

    # display frame slice from the requested time window (relative to onset)
    if window is not None:
        f0 = int(np.clip(onset + round(window[0] / s.frame_period), 0, n_frames))
        f1 = int(np.clip(onset + round(window[1] / s.frame_period), f0 + 1, n_frames))
    else:
        f0, f1 = 0, n_frames
    xf = np.arange(f0, f1)

    n_rows = len(real_groups)
    n_cols = max(len(gmap[tn]['target_rois']) for tn in real_groups)
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False,
                             figsize=(3 * n_cols, 2.5 * n_rows))

    def mean_sem(roi, target_tn):
        si, ti = np.where(grp == target_tn)
        tr = s.cyc[roi, si, ti, :]                       # (n_sel, n_frames)
        mean = np.nanmean(tr, axis=0)
        n = max(np.sum(~np.isnan(tr[:, 0])), 1)
        sem = np.nanstd(tr, axis=0) / np.sqrt(n)
        return mean, sem

    legended = False
    for row, real_tn in enumerate(real_groups):
        target_rois = gmap[real_tn]['target_rois']
        sham_tn = gmap[real_tn]['sham']
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(target_rois):
                ax.axis('off')
                continue
            roi = int(target_rois[col])
            for tn, color, label in ((real_tn, _TARGET_COLOR, 'real'),
                                     (sham_tn, _NONTARGET_COLOR, 'sham')):
                mean, sem = mean_sem(roi, tn)
                ax.plot(xf, mean[f0:f1], color=color, lw=1,
                        label=(label if not legended else None))
                ax.fill_between(xf, (mean - sem)[f0:f1], (mean + sem)[f0:f1],
                                color=color, alpha=0.2, lw=0)
            ax.axvline(onset, color='black', lw=0.8, ls='--')
            ax.set_title(f'grp {real_tn:g} · ROI {roi}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'group {real_tn:g}\ndF/F', fontsize=8)
            if not legended:
                ax.legend(fontsize=7, frameon=False)
                legended = True
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel('frame (onset = dashed)', fontsize=8)

    sns.despine(fig=fig)
    fig.suptitle(f'{s.exp_id} · target-ROI timecourses (real vs sham)')
    fig.tight_layout()
    return fig
