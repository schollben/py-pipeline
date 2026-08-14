import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .session import cyc_response_windows


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


def plot_stim_traces(s, cells, gap=10, mask_artifact=True):
    """Concatenated trial-averaged cyc traces (mean +/- SEM) for each cell.

    With `mask_artifact` (default), the photostim artifact + blank span is NaN'd
    so the plot autoscales to the real baseline/response instead of the large
    negative photostim dip.
    """
    cells = np.atleast_1d(cells)
    n_stims, frames = s.cyc.shape[1], s.cyc.shape[3]
    starts = np.arange(n_stims) * (frames + gap)

    art = np.zeros(frames, dtype=bool)
    if mask_artifact:
        base_sl, peak_sl = cyc_response_windows(s)
        art[base_sl.stop:peak_sl.start] = True    # dip + blank between windows

    fig, axes = plt.subplots(len(cells), 1, figsize=(14, 1.8 * len(cells)),
                             sharex=True, squeeze=False)
    for ax, cc in zip(axes[:, 0], cells):
        for si in range(n_stims):
            trials = s.cyc[cc, si]
            mean = np.nanmean(trials, axis=0)
            mean[art] = np.nan
            n = np.sum(~np.isnan(trials[:, 0]))
            sem = np.nanstd(trials, axis=0) / np.sqrt(max(n, 1))
            x = starts[si] + np.arange(frames)
            ax.plot(x, mean, color='k', lw=0.8)
            ax.fill_between(x, mean - sem, mean + sem, color='k', alpha=0.2, lw=0)
        ax.set_ylabel(f'cell {cc}\ndF/F', fontsize=8)
        ax.margins(x=0)
    axes[-1, 0].set_xticks(starts + frames / 2)
    axes[-1, 0].set_xticklabels(s.unique_stims.astype(int), fontsize=6, rotation=90)
    axes[-1, 0].set_xlabel('stimulus ID')
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig
