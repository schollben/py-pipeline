import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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


def compute_influence(s, pre_sec=1.0, post_sec=0.5):
    """Per-group influence of photostimulation relative to its paired sham.

    For each real group, and each visual stim id, computes a peak response
    (post-onset peak minus pre-onset baseline, from the trial-averaged cyc
    trace) separately for the real group's trials and its sham's trials, then:

        influence[cell, stim] = (resp_real - resp_sham) / resp_sham

    Peak responses are recomputed per (cell, stim, group) from the group-specific
    subset of cyc trials — the pipeline's precomputed `resp` mixes all groups
    together and cannot be used here.

    Sets s.influence = {real_tn: {'influence': (n_cells, n_stims), 'grand': (n_cells,)}}
    and returns the same dict.
    """
    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    n_stims = len(s.unique_stims)
    pre_frames = int(round(pre_sec / s.frame_period))
    post_frames = int(round(post_sec / s.frame_period))

    def group_peak_resp(target_tn):
        """(n_cells, n_stims) peak-minus-baseline response for one group."""
        resp = np.full((s.n_rois, n_stims), np.nan)
        for si in range(n_stims):
            trial_idx = np.where(grp[si] == target_tn)[0]
            if len(trial_idx) == 0:
                continue
            traces = s.cyc[:, si, trial_idx, :]              # (n_cells, n_sel, n_frames)
            avg = np.nanmean(traces, axis=1)                  # (n_cells, n_frames)
            baseline = np.nanmean(avg[:, :pre_frames], axis=1)
            post = np.nanmax(avg[:, pre_frames:pre_frames + post_frames], axis=1)
            resp[:, si] = post - baseline
        return resp

    influence = {}
    for real_tn, info in gmap.items():
        resp_real = group_peak_resp(real_tn)
        resp_sham = group_peak_resp(info['sham'])
        with np.errstate(invalid='ignore', divide='ignore'):
            inf = np.where(np.abs(resp_sham) > 1e-6,
                           (resp_real - resp_sham) / resp_sham, np.nan)
        influence[real_tn] = dict(
            influence=inf,
            grand=np.nanmean(inf, axis=1),
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
        influence = s.influence if s.influence is not None else compute_influence(s)
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
        influence = s.influence if s.influence is not None else compute_influence(s)
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
