import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_stim_traces(s, cells, gap=10):
    """Concatenated trial-averaged cyc traces (mean +/- SEM) for each cell."""
    cells = np.atleast_1d(cells)
    n_stims, frames = s.cyc.shape[1], s.cyc.shape[3]
    starts = np.arange(n_stims) * (frames + gap)

    fig, axes = plt.subplots(len(cells), 1, figsize=(14, 1.8 * len(cells)),
                             sharex=True, squeeze=False)
    for ax, cc in zip(axes[:, 0], cells):
        for si in range(n_stims):
            trials = s.cyc[cc, si]
            mean = np.nanmean(trials, axis=0)
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
