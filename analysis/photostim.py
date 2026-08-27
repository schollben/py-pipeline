import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from .session import cyc_response_windows, cyc_onset, _window_slice

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
    slot [stim_index, k] is filled from that presentation.

    `target_number` is taken as index-aligned with `stim_id` (no offset). That
    holds on a session as loaded, and on one where `dropFirstEvents` has trimmed
    both consistently — real photostim groups then show a much larger response in
    their targeted cells than their sham pair, and a shifted pairing gives the
    reverse. Note the empirical check behind that claim only exercises the
    as-loaded case; it is `dropFirstEvents`' job to preserve the alignment, and a
    bug there previously broke it silently (real/sham came out swapped). The
    real>sham post-condition is now checked by `check_real_sham_ordering`.
    """
    stim_id = s.stim_id
    unique_stims = s.unique_stims
    target_number = s.target_number
    n_stims = len(unique_stims)
    n_trials = s.cyc.shape[2]
    n = min(len(stim_id), len(target_number))
    if abs(len(stim_id) - len(target_number)) > 1:
        print(f'[!] {s.exp_id}: stim_id ({len(stim_id)}) and target_number '
              f'({len(target_number)}) differ by more than one entry; pairing them '
              f'by index and truncating to {n}. Photostim group labels may be '
              f'misassigned — check the event alignment for this session.')

    grp = np.full((n_stims, n_trials), np.nan)
    trial_count = np.zeros(n_stims, dtype=int)
    for i in range(n):
        ind = np.where(unique_stims == stim_id[i])[0][0]
        k = trial_count[ind]
        if k < n_trials:
            grp[ind, k] = target_number[i]
        trial_count[ind] += 1
    return grp


def check_real_sham_ordering(s, base_sl=None, peak_sl=None, verbose=True):
    """Sanity-check that each real group drives its targets harder than its sham.

    A sham fires at 0 mW, so it cannot drive its targeted ROIs. If a sham's mean
    target-ROI response exceeds its real partner's, the trial<->target_number
    pairing is off (typically a leading-event/off-by-one bug upstream in
    `dropFirstEvents`) and every real/sham comparison downstream is inverted.

    Returns True when all groups are ordered correctly, False otherwise. Warns on
    each violation when `verbose`.
    """
    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    if base_sl is None or peak_sl is None:
        base_sl, peak_sl = cyc_response_windows(s)

    def target_resp(rois, tn):
        si, ti = np.where(grp == tn)
        if not len(si):
            return np.nan
        tr = s.cyc[np.ix_(np.asarray(rois))][:, si, ti, :]
        return float(np.nanmean(np.nanmax(tr[..., peak_sl], axis=-1)
                                - np.nanmean(tr[..., base_sl], axis=-1)))

    ok = True
    for real_tn, info in sorted(gmap.items()):
        rois = info['target_rois']
        if not len(rois):
            continue
        r, sh = target_resp(rois, real_tn), target_resp(rois, info['sham'])
        if np.isnan(r) or np.isnan(sh):
            continue
        if r <= sh:
            ok = False
            if verbose:
                print(f'[!] {s.exp_id}: group {real_tn:g} sham response ({sh:+.4f}) '
                      f'exceeds real ({r:+.4f}) in its own targets — real/sham are '
                      f'likely swapped; check dropFirstEvents / event alignment.')
    return ok


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


def plot_photostim_group_heatmaps(s, mode='raw', baseline_guard_sec=0.5,
                                  post_sec=1.0, baseline=None, peak=None,
                                  baseline_subtract=True, mask_artifact=True):
    """Trial-averaged per-cell activity heatmap for each real photostim group.

    mode: 'raw' (cell-by-cell trial average), 'norm' (per-cell peak-normalized),
    or 'zscore' (per-cell z-scored across time).

    baseline_guard_sec, post_sec, baseline, peak : identical to
        `compute_responses` / `cyc_response_windows` — pass the same values you
        passed to `compute_responses` so the heatmap shows what was measured.
    baseline_subtract : subtract each trial's own mean over the `baseline` slice
        before averaging (per-trial, as `compute_responses` does), so each cell's
        dF/F offset is removed and the colour scale reflects evoked change.
    mask_artifact : NaN the frames between the end of the baseline window and the
        start of the peak window (the photostim artifact + blank span) so they are
        drawn as gray and excluded from the norm/zscore statistics.
    """
    if mode not in ('raw', 'norm', 'zscore'):
        raise ValueError("mode must be 'raw', 'norm', or 'zscore'")

    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    real_groups = sorted(gmap.keys())
    n_frames = s.cyc.shape[3]
    pre_frames = int(round(1.0 / s.frame_period)) if s.frame_period else 0
    base_sl, peak_sl = cyc_response_windows(s, baseline_guard_sec, post_sec,
                                            baseline, peak)

    art = np.zeros(n_frames, dtype=bool)
    if mask_artifact:
        art[base_sl.stop:peak_sl.start] = True

    fig, axes = plt.subplots(1, len(real_groups),
                              figsize=(6 * len(real_groups), 8), squeeze=False)

    for ax, real_tn in zip(axes[0], real_groups):
        target_rois = gmap[real_tn]['target_rois']
        mask = grp == real_tn
        stim_idx, trial_idx = np.where(mask)
        traces = s.cyc[:, stim_idx, trial_idx, :]        # (n_cells, n_sel, n_frames)
        if baseline_subtract:
            traces = traces - np.nanmean(traces[:, :, base_sl], axis=2,
                                         keepdims=True)
        avg = np.nanmean(traces, axis=1)                 # (n_cells, n_frames)
        avg[:, art] = np.nan

        is_target = np.zeros(s.n_rois, dtype=bool)
        is_target[target_rois] = True
        order = np.argsort(~is_target, kind='stable')    # targets first
        avg_sorted = avg[order]
        is_target_sorted = is_target[order]

        # frames fully NaN across all cells = the photostim-pulse blanking window
        # plus (when mask_artifact) the baseline->peak gap; exclude them from
        # norm/zscore statistics so the artifact-adjacent dip doesn't wash out
        # genuine post-stimulus response structure.
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


def group_trial_resp(s, target_tn, grp, base_sl=None, peak_sl=None, resps=None):
    """(n_cells, n_stims, n_trials) per-trial peak-minus-baseline response.

    Trial-preserving measurement: baseline and peak are computed per trial (not on
    the trial-averaged trace) so the trial axis survives for mean/SEM and bootstrap
    use downstream. NaN where a (stim, trial) slot isn't filled for this group.

    `resps` : precomputed (n_cells, n_stims, n_trials) per-trial responses — pass
        `s.resps` from `compute_responses` so influence measures exactly what
        `resp` measured. Selecting this group is then just a mask over the trial
        axis, not a recomputation. When None, recomputes from `s.cyc` using
        `base_sl`/`peak_sl` (the explicit-window override path).

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
        if resps is not None:
            resp[:, si, trial_idx] = resps[:, si, trial_idx]
        else:
            traces = s.cyc[:, si, trial_idx, :]                   # (n_cells, n_sel, n_frames)
            baseline = np.nanmean(traces[..., base_sl], axis=-1)  # (n_cells, n_sel)
            post = np.nanmax(traces[..., peak_sl], axis=-1)       # (n_cells, n_sel)
            resp[:, si, trial_idx] = post - baseline
    return resp


def _influence_trial_resps(s, baseline_guard_sec, post_sec, baseline, peak):
    """Resolve the per-trial response source shared by the influence functions.

    Default: reuse `s.resps` from `compute_responses`, so influence and `resp`
    always reflect the same measurement windows. Passing any window argument
    explicitly overrides that and recomputes from `s.cyc`.

    Returns (resps_or_None, base_sl, peak_sl).
    """
    explicit = (baseline is not None or peak is not None
                or baseline_guard_sec is not None or post_sec is not None)
    if explicit:
        kw = {}
        if baseline_guard_sec is not None:
            kw['baseline_guard_sec'] = baseline_guard_sec
        if post_sec is not None:
            kw['post_sec'] = post_sec
        base_sl, peak_sl = cyc_response_windows(s, baseline=baseline, peak=peak, **kw)
        return None, base_sl, peak_sl

    if s.resps is None:
        raise ValueError(
            f'{s.exp_id}: no per-trial responses on the session; run '
            f'compute_responses(...) first so influence uses the same baseline/peak '
            f'windows as resp, or pass baseline=/peak= explicitly to override.')
    if s.resps.shape[:3] != (s.n_rois, len(s.unique_stims), s.cyc.shape[2]):
        raise ValueError(
            f'{s.exp_id}: s.resps {s.resps.shape} does not match the current cyc '
            f'geometry ({s.n_rois}, {len(s.unique_stims)}, {s.cyc.shape[2]}); '
            f're-run compute_responses(...) after rebuild_cyc().')
    return s.resps, None, None


def influence_grand(s, baseline_guard_sec=None, post_sec=None,
                    baseline=None, peak=None, mode='dprime', good_only=True,
                    clip_sham=True):
    """Grand-average influence of each target group on all nontargets.

    Diagnostic entry point: pools every trial of the real group (across all
    stimulus conditions) and compares it against a common sham reference — every
    sham (0 mW) trial of every group pooled — then forms:

        mode='dprime' (default):
            influence[cell] = (mean(resp_real) - mean(sham_all)) / std(sham_all)
        mode='diff':
            influence[cell] =  mean(resp_real) - mean(sham_all)

    All shams are pooled into one reference per cell: they differ only in which
    targets were addressed at zero power, so they measure the same null, and
    pooling gives a better-sampled mean and sigma than any single group's shams.

    'dprime' is a standardised effect size: the real-minus-sham difference in
    units of the sham's own trial-to-trial SD — a z-score against the null
    condition. Only the sham SD is used, not a real/sham pooled SD, because
    photostim can itself inflate response variability, which would shrink the
    effect size for exactly the cells being driven most strongly. The denominator
    is an SD, so it is always positive and never near zero — unlike normalising
    by the sham mean, which is statistically indistinguishable from zero for most
    cells and produces both huge values and spurious sign flips. 'diff' is the
    same numerator left in dF/F units, for when absolute magnitude matters.

    Per-trial responses default to `s.resps` from `compute_responses`, so influence
    is measured over exactly the same baseline/peak windows as `resp` — run
    `compute_responses(...)` first or this raises. Passing any of
    `baseline`/`peak`/`baseline_guard_sec`/`post_sec` overrides that and recomputes
    from `s.cyc` with those windows (seconds from the start of the cyc window).

    clip_sham : floor a negative `mean(resp_sham)` at 0 before taking the
        difference. A sham (0 mW) response should not be negative; where it is,
        it is noise around a true zero, so flooring it stops that noise adding
        to the real-minus-sham difference. Applies to both modes, since both
        share the same numerator. Pass False to use the raw sham mean.
    good_only : set the influence of cells failing `s.is_good_cell` to NaN, so
        low-SNR cells are excluded from the maps and summary statistics. Run
        `compute_snr(...)` first to reassign that flag from the responses;
        without it the pipeline's skewness-based flag is used. Pass False to
        keep every cell.

    Sets s.influence = {real_tn: {'grand': (n_cells,), 'kind': 'grand',
    'mode': mode}} and returns the same dict.
    """
    if mode not in ('dprime', 'diff'):
        raise ValueError("mode must be 'dprime' or 'diff'")
    if good_only and s.is_good_cell is None:
        raise ValueError(
            f'{s.exp_id}: good_only=True but the session has no is_good_cell '
            f'flag; run compute_snr(...) first or pass good_only=False.')
    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    resps, base_sl, peak_sl = _influence_trial_resps(
        s, baseline_guard_sec, post_sec, baseline, peak)

    # one common sham reference for every group: all sham (0 mW) trials pooled.
    # Shams differ only in which targets were addressed at zero power, so they
    # measure the same null; pooling them gives a single per-cell reference and a
    # far better-sampled sigma than any one group's shams could.
    sham_tns = sorted({info['sham'] for info in gmap.values()})
    sham_all = np.concatenate(
        [group_trial_resp(s, tn, grp, base_sl, peak_sl, resps).reshape(s.n_rois, -1)
         for tn in sham_tns], axis=1)                     # (n_cells, n_sham_trials)
    mean_sham = np.nanmean(sham_all, axis=1)              # (n_cells,)
    sigma_sham = np.nanstd(sham_all, axis=1, ddof=1)      # (n_cells,)
    if clip_sham:
        # a negative sham response is not physical; for these cells the sham is
        # statistically indistinguishable from zero, so floor it there rather
        # than let noise inflate the real-minus-sham difference.
        mean_sham = np.where(mean_sham < 0, 0.0, mean_sham)

    influence = {}
    for real_tn, info in gmap.items():
        resp_real = group_trial_resp(s, real_tn, grp, base_sl, peak_sl, resps)
        mean_real = np.nanmean(resp_real, axis=(1, 2))    # (n_cells,)
        if mode == 'diff':
            grand = mean_real - mean_sham
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                grand = np.where(sigma_sham > 0,
                                 (mean_real - mean_sham) / sigma_sham, np.nan)
        if good_only:
            grand = np.where(np.asarray(s.is_good_cell, dtype=bool), grand, np.nan)
        influence[real_tn] = dict(grand=grand, kind='grand', mode=mode)

    s.influence = influence
    return influence


def _influence_by_bins(s, bins, kind, baseline_guard_sec, post_sec, baseline,
                       peak, mode, good_only, clip_sham):
    """Influence per stimulus bin, against a sham reference matched to each bin.

    `bins` is a list of (label, stim_mask, sham_mask) triples. `stim_mask`
    selects the columns of the (n_cells, n_stims, n_trials) response arrays that
    make up the bin's real trials; `sham_mask` selects the columns forming its
    sham reference — usually the same, but broader where several stimulus ids are
    the same condition (e.g. every contrast-0 stim is a blank, so their shams all
    pool into one reference). Shams are pooled across every sham group — they
    measure the same null — but stay matched to the bin's visual drive, so that
    drive is present in both terms and cancels in the difference. Shared by
    `influence_by_stim` (one bin per stimulus) and `influence_by_contrast` (one
    bin per contrast level).
    """
    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    resps, base_sl, peak_sl = _influence_trial_resps(
        s, baseline_guard_sec, post_sec, baseline, peak)

    sham_tns = sorted({info['sham'] for info in gmap.values()})
    sham_by_group = [group_trial_resp(s, tn, grp, base_sl, peak_sl, resps)
                     for tn in sham_tns]                  # each (cells, stims, trials)

    # per-bin sham reference: all sham groups pooled, restricted to the bin
    n_bins = len(bins)
    mean_sham = np.full((s.n_rois, n_bins), np.nan)
    sigma_sham = np.full((s.n_rois, n_bins), np.nan)
    for bi, (_, _, sham_mask) in enumerate(bins):
        pooled = np.concatenate(
            [r[:, sham_mask, :].reshape(s.n_rois, -1) for r in sham_by_group], axis=1)
        mean_sham[:, bi] = np.nanmean(pooled, axis=1)
        sigma_sham[:, bi] = np.nanstd(pooled, axis=1, ddof=1)
    if clip_sham:
        mean_sham = np.where(mean_sham < 0, 0.0, mean_sham)

    good = (np.asarray(s.is_good_cell, dtype=bool) if good_only else None)
    labels = np.array([lbl for lbl, _, _ in bins], dtype=float)

    influence = {}
    for real_tn in gmap:
        resp_real = group_trial_resp(s, real_tn, grp, base_sl, peak_sl, resps)
        mean_real = np.full((s.n_rois, n_bins), np.nan)
        for bi, (_, stim_mask, _) in enumerate(bins):
            mean_real[:, bi] = np.nanmean(
                resp_real[:, stim_mask, :].reshape(s.n_rois, -1), axis=1)

        if mode == 'diff':
            inf = mean_real - mean_sham
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                inf = np.where(sigma_sham > 0,
                               (mean_real - mean_sham) / sigma_sham, np.nan)
        if good_only:
            inf = np.where(good[:, None], inf, np.nan)

        influence[real_tn] = dict(influence=inf, grand=np.nanmean(inf, axis=1),
                                  labels=labels, kind=kind, mode=mode)

    s.influence = influence
    return influence


def _check_influence_args(s, mode, good_only):
    if mode not in ('dprime', 'diff'):
        raise ValueError("mode must be 'dprime' or 'diff'")
    if good_only and s.is_good_cell is None:
        raise ValueError(
            f'{s.exp_id}: good_only=True but the session has no is_good_cell '
            f'flag; run compute_snr(...) first or pass good_only=False.')


def influence_by_stim(s, baseline_guard_sec=None, post_sec=None,
                      baseline=None, peak=None, mode='dprime', good_only=True,
                      clip_sham=True):
    """Influence of each target group on all nontargets, per stimulus condition.

    Same measure as `influence_grand`, but real trials are pooled within each
    stimulus id rather than across all of them, and each stimulus is compared
    against its OWN sham reference:

        mode='dprime' (default):
            influence[cell, stim] = (mean(real[stim]) - mean(sham[stim]))
                                    / std(sham[stim])
        mode='diff':
            influence[cell, stim] =  mean(real[stim]) - mean(sham[stim])

    The sham reference is pooled across every sham group (they measure the same
    null) but kept matched to the stimulus, so the visual response is present in
    both terms and cancels in the difference — what remains is the photostim
    effect, not the cell's visual tuning. The exception is contrast 0: every
    contrast-0 stimulus is the same blank regardless of its nominal direction, so
    their sham trials all pool into a single reference shared by those stimuli.

    Away from contrast 0, `sigma` is estimated from one stimulus' worth of sham
    trials, so it is noisier than in `influence_grand` and inflates d' where it
    comes out small; prefer `influence_by_contrast` for interpretation, or
    `mode='diff'`, which has no denominator.

    Window handling, `mode`, `good_only` and `clip_sham` all match
    `influence_grand`.

    Sets s.influence = {real_tn: {'influence': (n_cells, n_stims),
    'grand': (n_cells,), 'labels': stim ids, 'kind': 'stim', 'mode': mode}} and
    returns the same dict. `grand` is the mean across stims, for
    `plot_influence_maps`.
    """
    _check_influence_args(s, mode, good_only)
    # every contrast-0 stimulus is the same blank regardless of its nominal
    # direction, so both its real and its sham trials pool across all contrast-0
    # stimuli: those columns are identical to each other, and to the contrast-0
    # column of influence_by_contrast. At other contrasts real and sham are both
    # kept per-stimulus, since the visual drive has to cancel in the difference.
    blank = s.stim_table[:, 1] == 0
    bins = [(sid,
             blank if blank[i] else (s.unique_stims == sid),
             blank if blank[i] else (s.unique_stims == sid))
            for i, sid in enumerate(s.unique_stims)]
    return _influence_by_bins(s, bins, 'stim', baseline_guard_sec, post_sec,
                              baseline, peak, mode, good_only, clip_sham)


def influence_by_contrast(s, baseline_guard_sec=None, post_sec=None,
                          baseline=None, peak=None, mode='dprime',
                          good_only=True, clip_sham=True):
    """Influence of each target group on all nontargets, per contrast level.

    As `influence_by_stim`, but stimuli are binned by contrast rather than taken
    individually: every stimulus at a given contrast is pooled, and the sham
    reference is pooled across sham groups within that same contrast.

        influence[cell, contrast] = (mean(real[contrast]) - mean(sham[contrast]))
                                    / std(sham[contrast])

    Pooling over directions within a contrast gives many more trials per bin than
    `influence_by_stim`, so mean and sigma are far better estimated, while the
    reference still matches the visual drive of the trials it is compared against.

    Sets s.influence = {real_tn: {'influence': (n_cells, n_contrasts),
    'grand': (n_cells,), 'labels': contrasts, 'kind': 'contrast',
    'mode': mode}} and returns the same dict.
    """
    _check_influence_args(s, mode, good_only)
    stim_contrast = s.stim_table[:, 1]
    bins = [(c, stim_contrast == c, stim_contrast == c) for c in s.contrasts]
    return _influence_by_bins(s, bins, 'contrast', baseline_guard_sec, post_sec,
                              baseline, peak, mode, good_only, clip_sham)


def influence_bootstrap(s, by='grand', n_boot=1000, seed=None,
                        baseline_guard_sec=None, post_sec=None,
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
    resps, base_sl, peak_sl = _influence_trial_resps(
        s, baseline_guard_sec, post_sec, baseline, peak)
    rng = np.random.default_rng(seed)

    influence = {}
    for real_tn, info in gmap.items():
        resp_real = group_trial_resp(s, real_tn, grp, base_sl, peak_sl, resps)
        resp_sham = group_trial_resp(s, info['sham'], grp, base_sl, peak_sl, resps)
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


def _draw_influence_map(ax, s, grand_influence, target_rois, title, vlim=None,
                        show_image=True, colorbar=True):
    """Facecolor-by-influence overlay: PRGn fill for nontargets, black dashed
    outlines for targets, gray for nontargets.

    show_image : draw the average 2P image behind the ROIs. False leaves a blank
        background (ROI outlines and influence fills only).
    colorbar : attach a colorbar to this axis. False when the caller draws a
        single shared colorbar for the whole figure.

    Returns the ScalarMappable used for the fills, so a caller can build a shared
    colorbar from it.
    """
    if vlim is None:
        vlim = np.nanpercentile(np.abs(grand_influence), 99)
        if not np.isfinite(vlim) or vlim == 0:
            vlim = 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    cmap = plt.get_cmap('PRGn')

    if show_image:
        ax.imshow(s.avg_image, cmap='gray', vmax=s.avg_image.max() / 2)
    else:
        # no background image to set the extent/orientation, so pin them to the
        # frame ourselves and keep imshow's top-left origin for the overlays
        ny, nx = s.avg_image.shape
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(ny - 0.5, -0.5)
        ax.set_aspect('equal')
    is_target = np.zeros(s.n_rois, dtype=bool)
    is_target[target_rois] = True

    for cc in range(s.n_rois):
        m = s.mask2d[cc] > 0.5
        if is_target[cc]:
            ax.contour(s.mask2d[cc], levels=[0.5], colors=['black'],
                       linewidths=1.2, linestyles='dashed')
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
    if colorbar:
        plt.colorbar(sm, ax=ax, fraction=0.046, label='influence')
    return sm


def plot_influence_maps(s, influence=None, show_image=False, vlim=None):
    """One spatial influence map per real photostim group (grand mean across stims).

    All groups share one symmetric colour scale and one figure-level colorbar, so
    fills are directly comparable across subplots.

    show_image : draw the average 2P image behind the ROIs (default False —
        blank background, ROI outlines and influence fills only).
    vlim : symmetric colour limit (+/- vlim). None (default) uses the 99th
        percentile of |influence| pooled over every group.
    """
    if influence is None:
        if s.influence is None:
            raise ValueError(
                f'{s.exp_id}: no influence computed; run rebuild_cyc() -> '
                'compute_responses() -> influence_grand(...) first')
        influence = s.influence
    gmap = photostim_group_map(s)
    real_groups = sorted(influence.keys())

    # one symmetric scale pooled over every group so subplots are comparable
    if vlim is None:
        allvals = np.concatenate([np.asarray(influence[tn]['grand']).ravel()
                                  for tn in real_groups])
        vlim = np.nanpercentile(np.abs(allvals), 99)
        if not np.isfinite(vlim) or vlim == 0:
            vlim = 1.0

    fig, axes = plt.subplots(1, len(real_groups),
                              figsize=(6 * len(real_groups), 6), squeeze=False)
    for ax, real_tn in zip(axes[0], real_groups):
        sm = _draw_influence_map(ax, s, influence[real_tn]['grand'],
                                 gmap[real_tn]['target_rois'], f'group {real_tn:g}',
                                 vlim=vlim, show_image=show_image, colorbar=False)
    fig.suptitle(f'{s.exp_id}  |  influence (grand mean across stims)')
    fig.tight_layout()
    modes = {influence[tn].get('mode', 'dprime') for tn in real_groups}
    lbl = ('influence (dF/F, real - sham)' if modes == {'diff'}
           else "influence (d', sham SD)")
    fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.046, label=lbl)
    return fig


def plot_influence_by_contrast(s, influence=None, show_image=False, vlim=None):
    """Spatial influence maps split by stimulus contrast (rows=groups, cols=contrast).

    Reads the per-bin `influence` produced by `influence_by_contrast` (one column
    per contrast) or by `influence_by_stim` (stims averaged within each contrast).
    All panels share one symmetric colour scale and one figure-level colorbar, so
    fills are comparable across both groups and contrasts.

    show_image : draw the average 2P image behind the ROIs (default False).
    vlim : symmetric colour limit (+/- vlim). None uses the 99th percentile of
        |influence| pooled over every panel.
    """
    if influence is None:
        if s.influence is None:
            raise ValueError(
                f'{s.exp_id}: no influence computed; run rebuild_cyc() -> '
                'compute_responses() -> influence_by_contrast(...) first')
        influence = s.influence
    gmap = photostim_group_map(s)
    real_groups = sorted(influence.keys())
    contrasts = s.contrasts

    first = influence[real_groups[0]]
    if 'influence' not in first:
        raise ValueError(
            f'{s.exp_id}: s.influence holds a grand average, not per-stimulus '
            f'values; run influence_by_contrast(...) or influence_by_stim(...).')

    # panel values: already per-contrast, or per-stim averaged within contrast
    by_contrast = first.get('kind') == 'contrast'
    stim_contrast = s.stim_table[:, 1]

    def panel(real_tn, contrast):
        inf = influence[real_tn]['influence']
        if by_contrast:
            col = int(np.argmin(np.abs(influence[real_tn]['labels'] - contrast)))
            return inf[:, col]
        return np.nanmean(inf[:, stim_contrast == contrast], axis=1)

    panels = {(tn, c): panel(tn, c) for tn in real_groups for c in contrasts}

    if vlim is None:
        allvals = np.concatenate([v.ravel() for v in panels.values()])
        vlim = np.nanpercentile(np.abs(allvals), 99)
        if not np.isfinite(vlim) or vlim == 0:
            vlim = 1.0

    fig, axes = plt.subplots(len(real_groups), len(contrasts),
                              figsize=(5 * len(contrasts), 5 * len(real_groups)),
                              squeeze=False)
    for row, real_tn in enumerate(real_groups):
        for col, contrast in enumerate(contrasts):
            sm = _draw_influence_map(
                axes[row, col], s, panels[(real_tn, contrast)],
                gmap[real_tn]['target_rois'],
                f'group {real_tn:g}  |  contrast {contrast:g}',
                vlim=vlim, show_image=show_image, colorbar=False)
    fig.suptitle(f'{s.exp_id}  |  influence by contrast')
    fig.tight_layout()
    modes = {influence[tn].get('mode', 'dprime') for tn in real_groups}
    lbl = ('influence (dF/F, real - sham)' if modes == {'diff'}
           else "influence (d', sham SD)")
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.046, label=lbl)
    return fig


def plot_photostim_target_traces(s, window=None, baseline_guard_sec=0.5,
                                 post_sec=1.0, baseline=None, peak=None,
                                 show_windows=True, ymin=-0.05,
                                 baseline_subtract=True, trial_range=None):
    """Trial-averaged cyc timecourse of each targeted ROI, real vs sham.

    Grid of line plots: one row per real photostim group/ensemble, one column per
    targeted ROI in that group. Each subplot overlays the target's real (power>0)
    and paired sham (power=0) trial-averaged traces (mean +/- SEM), collapsed
    across visual stims (as plot_photostim_group_heatmaps does).

    window : (t0, t1) in seconds from the START of the cyc window (the same
        convention as `baseline`/`peak` and `compute_responses`, NOT relative to
        onset) to crop the displayed frames; None shows the full cyc window. For
        a `rebuild_cyc(preStim=1, postStim=2)` cyc the window spans 0 -> 3.0 s
        with onset at 1.0 s, so `window=(0.5, 2.5)` is -0.5 to 1.5 s around onset.

    baseline_guard_sec, post_sec, baseline, peak : identical to
        `compute_responses` / `cyc_response_windows` — the measurement windows
        that responses are computed over. Pass the same values you passed to
        `compute_responses` so the plot shows what was actually measured.
    show_windows : shade the baseline (gray) and peak (yellow) measurement
        windows behind each trace.
    ymin : lower y-axis limit for every subplot (upper limit autoscales);
        None leaves the y-axis fully autoscaled.
    baseline_subtract : subtract each trial's own mean over the `baseline` slice
        before averaging, matching what `compute_responses` measures (per-trial,
        not per-averaged-trace). Removes the dF/F offset and shrinks the SEM band
        by the amount of across-trial baseline drift.
    trial_range : (t0, t1) half-open range of cyc trial indices to include,
        e.g. (0, 10) for the first ten repeats of each stim or (10, None) to
        drop them. Either bound may be None for open-ended. The index is the
        repeat number within a stimulus (cyc axis 2), not a presentation number
        in acquisition order. None (default) uses every trial.
    """
    if not s.has_photostim:
        print(f'{s.exp_id}: no photostimulation data in this session.')
        return

    gmap = photostim_group_map(s)
    grp = cyc_trial_group(s)
    real_groups = sorted(gmap.keys())
    n_frames = s.cyc.shape[3]
    fp = s.frame_period
    onset = cyc_onset(s)
    base_sl, peak_sl = cyc_response_windows(s, baseline_guard_sec, post_sec,
                                            baseline, peak)
    check_real_sham_ordering(s, base_sl, peak_sl)

    if trial_range is not None:
        t_lo, t_hi = trial_range
        t_lo = 0 if t_lo is None else int(t_lo)
        t_hi = s.cyc.shape[2] if t_hi is None else int(t_hi)

    # display frame slice from the requested time window (from cyc-window start)
    if window is not None:
        disp = _window_slice(s, window, n_frames, 'window')
        f0, f1 = disp.start, disp.stop
    else:
        f0, f1 = 0, n_frames
    xf = np.arange(f0, f1) * fp                          # seconds from cyc start

    n_rows = len(real_groups)
    n_cols = max(len(gmap[tn]['target_rois']) for tn in real_groups)
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False,
                             figsize=(3 * n_cols, 2.5 * n_rows))

    def mean_sem(roi, target_tn):
        si, ti = np.where(grp == target_tn)
        if trial_range is not None:
            keep = (ti >= t_lo) & (ti < t_hi)
            si, ti = si[keep], ti[keep]
        if si.size == 0:
            nan = np.full(n_frames, np.nan)
            return nan, nan
        tr = s.cyc[roi, si, ti, :]                       # (n_sel, n_frames)
        if baseline_subtract:
            tr = tr - np.nanmean(tr[:, base_sl], axis=1, keepdims=True)
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
            if show_windows:
                ax.axvspan(base_sl.start * fp, base_sl.stop * fp,
                           color='0.5', alpha=0.15, lw=0)
                ax.axvspan(peak_sl.start * fp, peak_sl.stop * fp,
                           color='gold', alpha=0.2, lw=0)
            for tn, color, label in ((real_tn, _TARGET_COLOR, 'real'),
                                     (sham_tn, _NONTARGET_COLOR, 'sham')):
                mean, sem = mean_sem(roi, tn)
                ax.plot(xf, mean[f0:f1], color=color, lw=1,
                        label=(label if not legended else None))
                ax.fill_between(xf, (mean - sem)[f0:f1], (mean + sem)[f0:f1],
                                color=color, alpha=0.2, lw=0)
            ax.axvline(onset * fp, color='black', lw=0.8, ls='--')
            ax.set_xlim(f0 * fp, (f1 - 1) * fp)
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            ax.set_title(f'grp {real_tn:g} · ROI {roi}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'group {real_tn:g}\ndF/F', fontsize=8)
            if not legended:
                ax.legend(fontsize=7, frameon=False)
                legended = True
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel('time from cyc start (s; onset = dashed)', fontsize=8)

    sns.despine(fig=fig)
    tr_note = '' if trial_range is None else f' · trials [{t_lo}, {t_hi})'
    fig.suptitle(f'{s.exp_id} · target-ROI timecourses (real vs sham){tr_note}')
    fig.tight_layout()
    return fig
