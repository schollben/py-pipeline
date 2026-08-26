import warnings

import numpy as np

from .session import cyc_response_windows


def compute_snr(s, thresh=1.0, baseline_guard_sec=0.5, post_sec=1.0,
                baseline=None, peak=None, verbose=True):
    """Per-cell SNR from peak responses, and reassign s.is_good_cell.

    Recomputes cell quality from the evoked response itself. The pipeline's
    skewness-based `is_good_cell` is measured on the raw dF/F trace, which the
    photostim blank distorts; this measures the same thing on the trial-resolved
    responses that the analysis actually uses.

    For each cell, the preferred stimulus is the one with the largest mean peak
    response, and

        SNR = mu_pref / (sigma_pref + sigma_base)

    where mu_pref / sigma_pref are the mean and across-trial SD of the per-trial
    responses at that preferred stimulus (from `s.resps`, already peak-minus-
    baseline, so mu_pref is a change from baseline), and sigma_base is the
    across-trial SD of the per-trial baseline means, pooled over every stimulus.
    Both denominator terms are trial-to-trial SDs, so they are in the same units.

    Cells with SNR > `thresh` are good. Overwrites `s.is_good_cell` in place and
    sets `s.snr` and `s.pref_stim`.

    thresh : SNR cutoff for a good cell (default 1.0).
    baseline_guard_sec, post_sec, baseline, peak : passed to
        `cyc_response_windows` to locate the baseline window sigma_base is
        measured over. Pass the same values you passed to `compute_responses`.
    verbose : print the count and percentage of good cells.

    Returns the (n_cells,) SNR array.
    """
    if s.resps is None:
        raise ValueError(
            f'{s.exp_id}: no per-trial responses; run compute_responses(...) first.')

    base_sl, _ = cyc_response_windows(s, baseline_guard_sec, post_sec,
                                      baseline, peak)

    # preferred stimulus per cell = largest mean peak response
    mean_by_stim = np.nanmean(s.resps, axis=2)               # (cells, stims)
    all_nan = np.all(np.isnan(mean_by_stim), axis=1)
    safe = np.where(all_nan[:, None], 0.0, mean_by_stim)
    pref = np.argmax(safe, axis=1)                           # (cells,)

    cells = np.arange(s.n_rois)
    pref_resps = s.resps[cells, pref, :]                     # (cells, trials)
    mu_pref = np.nanmean(pref_resps, axis=1)
    sigma_pref = np.nanstd(pref_resps, axis=1, ddof=1)

    # per-trial baseline means, pooled over stims -> across-trial SD. Unfilled
    # cyc trial slots are all-NaN (a stim with fewer repeats than n_trials);
    # they come through as NaN here and are dropped by the nanstd below.
    with np.errstate(invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            base_means = np.nanmean(s.cyc[..., base_sl], axis=3)  # (cells, stims, trials)
    sigma_base = np.nanstd(base_means.reshape(s.n_rois, -1), axis=1, ddof=1)

    denom = sigma_pref + sigma_base
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = np.where(denom > 0, mu_pref / denom, np.nan)
    snr[all_nan] = np.nan

    is_good = snr > thresh
    s.snr, s.pref_stim, s.is_good_cell = snr, pref, is_good

    if verbose:
        n_good = int(np.sum(is_good))
        print(f'{s.exp_id}: SNR > {thresh:g} -> {n_good}/{s.n_rois} '
              f'({100 * n_good / s.n_rois:.1f}%) cells good.')
    return snr
