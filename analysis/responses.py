import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .session import cyc_response_windows, cyc_onset


def compute_responses(s, baseline_guard_sec=0.5, post_sec=1.0):
    """Recompute peak-minus-baseline responses from cyc, robust to NaN blanking.

    Replaces the pipeline's FFT-based `resp`/`resps`/`resp_err`, which are all
    NaN for 2p-opto sessions (the FFT propagates the photostim NaN-blank) and,
    even without NaNs, integrate the whole window including the pre-onset
    artifact. Here:

        response = max(cyc over post-blank peak window) - mean(cyc over baseline)

    with windows from `cyc_response_windows` (baseline before the artifact, peak
    after the blank). Sets and returns s.resp; also sets s.resps, s.resp_err.
    Shapes match the pipeline's: resp/resp_err (n_cells, n_stims),
    resps (n_cells, n_stims, n_trials).
    """
    base_sl, peak_sl = cyc_response_windows(s, baseline_guard_sec, post_sec)
    baseline = np.nanmean(s.cyc[..., base_sl], axis=3)     # (cells, stims, trials)
    peak = np.nanmax(s.cyc[..., peak_sl], axis=3)          # (cells, stims, trials)
    resps = peak - baseline
    resp = np.nanmean(resps, axis=2)                       # (cells, stims)
    n = np.sum(~np.isnan(resps), axis=2)
    resp_err = np.nanstd(resps, axis=2) / np.sqrt(np.maximum(n, 1))

    s.resp, s.resps, s.resp_err = resp, resps, resp_err
    return resp


def plot_stim_traces(s, cells, gap=10, mask_artifact=True, window=None,
                     baseline_subtract=False, trials='all'):
    """Concatenated trial-averaged cyc traces (mean +/- SEM) for each cell.

    mask_artifact : NaN the photostim artifact + blank span so the plot autoscales
        to the real baseline/response instead of the large negative photostim dip.
    window : (t0, t1) in seconds relative to visual onset, e.g. (-0.5, 2.0), to
        crop each stim's displayed frames. None shows the full cyc window.
    baseline_subtract : subtract each stim's pre-onset baseline (from the
        cyc_response_windows baseline slice) so the dF/F offset is removed and
        stims are zeroed / comparable.
    trials : 'all' (default) averages every trial; 'sham' averages only the
        power=0 sham trials, giving the clean visual response uncontaminated by
        the photostim drive. Falls back to 'all' (with a note) for sessions
        without photostim / sham trials.

    A small `n=<count>` above each stim shows how many trials went into that mean.
    """
    cells = np.atleast_1d(cells)
    n_stims, n_trials, frames = s.cyc.shape[1], s.cyc.shape[2], s.cyc.shape[3]
    base_sl, peak_sl = cyc_response_windows(s)

    # per (stim, trial) selection mask + effective mode label
    sel = np.ones((n_stims, n_trials), dtype=bool)
    mode = 'all'
    if trials == 'sham':
        if s.has_photostim:
            from .photostim import cyc_trial_group, photostim_group_map
            grp = cyc_trial_group(s)
            sham_tns = list({info['sham'] for info in photostim_group_map(s).values()})
            cand = np.isin(grp, sham_tns)
            if cand.any():
                sel, mode = cand, 'sham'
            else:
                print(f'{s.exp_id}: no sham trials found — showing all.')
        else:
            print(f'{s.exp_id}: no photostim in this session — showing all.')

    # display frame slice from the requested time window (relative to onset)
    if window is not None:
        onset = cyc_onset(s)
        f0 = int(np.clip(onset + round(window[0] / s.frame_period), 0, frames))
        f1 = int(np.clip(onset + round(window[1] / s.frame_period), f0 + 1, frames))
    else:
        f0, f1 = 0, frames
    disp = slice(f0, f1)
    dframes = f1 - f0
    starts = np.arange(n_stims) * (dframes + gap)

    art = np.zeros(frames, dtype=bool)
    if mask_artifact:
        art[base_sl.stop:peak_sl.start] = True    # dip + blank between windows

    fig, axes = plt.subplots(len(cells), 1, figsize=(14, 1.8 * len(cells)),
                             sharex=True, squeeze=False)
    for ax, cc in zip(axes[:, 0], cells):
        for si in range(n_stims):
            tr = s.cyc[cc, si][sel[si]]
            mean = np.nanmean(tr, axis=0)
            sem = np.nanstd(tr, axis=0) / np.sqrt(
                max(np.sum(~np.isnan(tr[:, 0])), 1))
            if baseline_subtract:
                mean = mean - np.nanmean(mean[base_sl])
            mean = mean.copy()
            mean[art] = np.nan
            x = starts[si] + np.arange(dframes)
            ax.plot(x, mean[disp], color='k', lw=0.8)
            ax.fill_between(x, (mean - sem)[disp], (mean + sem)[disp],
                            color='k', alpha=0.2, lw=0)
        ax.set_ylabel(f'cell {cc}\ndF/F', fontsize=8)
        ax.margins(x=0)

    # per-stim trial-count annotation (denominator of each average, in-window)
    top_ax = axes[0, 0]
    for si in range(n_stims):
        tr = s.cyc[cells[0], si][sel[si]]
        n = int(np.sum(~np.isnan(tr[:, disp]).all(axis=1)))
        top_ax.annotate(f'n={n}', xy=(starts[si] + dframes / 2, 1.02),
                        xycoords=('data', 'axes fraction'), ha='center',
                        va='bottom', fontsize=5, color='0.4', clip_on=False)

    axes[-1, 0].set_xticks(starts + dframes / 2)
    axes[-1, 0].set_xticklabels(s.unique_stims.astype(int), fontsize=6, rotation=90)
    axes[-1, 0].set_xlabel(f'stimulus ID  ({mode} trials)')
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig
