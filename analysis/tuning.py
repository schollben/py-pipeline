import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from .session import resp_grid


def _wrap180(x):
    return (x + 180) % 360 - 180


def double_gaussian(theta, A1, A2, mu, sigma, dc):
    g1 = A1 * np.exp(-_wrap180(theta - mu) ** 2 / (2 * sigma ** 2))
    g2 = A2 * np.exp(-_wrap180(theta - mu - 180) ** 2 / (2 * sigma ** 2))
    return g1 + g2 + dc


def fit_direction_tuning(dirs, resps_top):
    """Fit per-trial responses (n_dir, n_trials) at the top contrast.

    Returns dict with params, pref_dir (deg, [0,360) or NaN), success.
    """
    x = np.repeat(dirs, resps_top.shape[1])
    y = resps_top.ravel()
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    fail = dict(params=None, pref_dir=np.nan, success=False)
    if len(np.unique(x)) < 3:
        return fail

    mean_by_dir = np.array([np.nanmean(resps_top[i]) for i in range(len(dirs))])
    mu0 = dirs[np.nanargmax(mean_by_dir)]
    amp0 = np.nanmax(mean_by_dir) - np.nanmin(mean_by_dir)
    p0 = [amp0, amp0 / 2, mu0, 30, np.nanmin(mean_by_dir)]
    bounds = ([0, 0, -360, 5, -np.inf], [np.inf, np.inf, 720, 180, np.inf])
    try:
        popt, _ = curve_fit(double_gaussian, x, y, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError):
        return fail
    A1, A2, mu, _, _ = popt
    pref = mu if A1 >= A2 else mu + 180
    return dict(params=popt, pref_dir=pref % 360, success=True)


def plot_tuning_curves(s, cells=None):

    if cells is None:
        cells = np.arange(s.n_rois)
    
    cells = np.atleast_1d(cells)
    dirs, cons = s.directions, s.contrasts
    shades = sns.color_palette('Greys', n_colors=len(cons) + 1)[1:]

    ncol = min(6, len(cells)) #minimum number of cells per row is 8 or whatever you want to set
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2 * nrow), squeeze=False)
    axes = axes.ravel()
    fine = np.linspace(0, 360, 361)

    pref_dir = np.full(s.n_rois, np.nan)
    fit_params = np.full((s.n_rois, 5), np.nan)
    top_ids = np.where(s.stim_table[:, 1] == cons[-1])[0]
    top_dirs = s.stim_table[top_ids, 0]
    order = np.argsort(top_dirs)
    top_ids, top_dirs = top_ids[order], top_dirs[order]

    for ax, cc in zip(axes, cells):
        resp, err = resp_grid(s, cc)
        for ci in range(len(cons)):
            ax.errorbar(dirs, resp[:, ci], yerr=err[:, ci], color=shades[ci],
                        lw=1, marker='o', ms=2, capsize=0)
        fit = fit_direction_tuning(top_dirs, s.resps[cc, top_ids])
        if fit['success']:
            # ax.plot(fine, double_gaussian(fine, *fit['params']),
            #         color='crimson', lw=1)
            ax.axvline(fit['pref_dir'], color='crimson', ls='--', lw=1)
            pref_dir[cc] = fit['pref_dir']
            fit_params[cc] = fit['params']
        ax.set_title(f'cell {cc}', fontsize=8)
        ax.set_xticks([0, 180, 360])
        ax.tick_params(labelsize=6)
    for ax in axes[len(cells):]:
        ax.axis('off')
    sns.despine(fig=fig)
    fig.supxlabel('direction (deg)')
    fig.supylabel('response')
    fig.tight_layout()

    s.pref_dir = pref_dir
    s.fit_params = fit_params
    return pref_dir
