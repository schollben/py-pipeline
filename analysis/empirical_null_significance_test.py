# %%
# empirical-null significance of grand-mean d' influence (nontargets) Standalone cell. 
# Efron local-fdr against an empirical null built from sham-vs-sham
# d' values. Run after influence_grand(dat, mode='dprime').

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
import statsmodels.api as sm
from patsy import dmatrix
from analysis.photostim import (photostim_group_map, cyc_trial_group,
                                group_trial_resp, _influence_trial_resps)

FDR_THRESH = 0.20
# Bin count / spline flexibility scale with sample size. The plan's 120 bins and
# df=7 assume thousands of scores; a single FOV yields ~n_nontargets x n_ensembles
# (order 100), where 120 bins are mostly empty and the spline fits noise -- which
# in turn makes the Tweedie log-density derivative explode.
N_BINS = None      # None -> chosen from n below
SPLINE_DF = None   # None -> chosen from n below
rng = np.random.default_rng(0)

gmap = photostim_group_map(dat)
grp = cyc_trial_group(dat)
resps, base_sl, peak_sl = _influence_trial_resps(dat, None, None, None, None)
good = np.asarray(dat.is_good_cell, dtype=bool)

# ---- pooled sham reference, exactly as influence_grand builds it ----------------
sham_tns = sorted({info['sham'] for info in gmap.values()})
sham_all = np.concatenate(
    [group_trial_resp(dat, tn, grp, base_sl, peak_sl, resps).reshape(dat.n_rois, -1)
     for tn in sham_tns], axis=1)                       # (n_cells, n_sham_trials)
sham_all = sham_all[:, ~np.all(np.isnan(sham_all), axis=0)]
mean_sham = np.nanmean(sham_all, axis=1)
sigma_sham = np.nanstd(sham_all, axis=1, ddof=1)
mean_sham_clip = np.where(mean_sham < 0, 0.0, mean_sham)

# ---- d_real: one entry per (nontarget cell, real ensemble) ----------------------
d_real, real_cell, real_tn_lbl = [], [], []
for real_tn, info in gmap.items():
    g = dat.influence[real_tn]['grand'] if dat.influence is not None else None
    if g is None:                       # recompute if influence_grand wasn't run
        mr = np.nanmean(group_trial_resp(dat, real_tn, grp, base_sl, peak_sl, resps),
                        axis=(1, 2))
        with np.errstate(invalid='ignore', divide='ignore'):
            g = np.where(sigma_sham > 0, (mr - mean_sham_clip) / sigma_sham, np.nan)
        g = np.where(good, g, np.nan)
    nontarget = good.copy()
    nontarget[info['target_rois']] = False              # drop the directly-stimulated cells
    cells = np.where(nontarget & np.isfinite(g))[0]
    d_real.append(np.asarray(g)[cells])
    real_cell.append(cells)
    real_tn_lbl.append(np.full(cells.size, real_tn))
d_real = np.concatenate(d_real)
real_cell = np.concatenate(real_cell)
real_tn_lbl = np.concatenate(real_tn_lbl)

# ---- d_sham: same statistic where the "real" trials are also sham ---------------
# Draw a pseudo-ensemble of sham trials the SAME SIZE as a real ensemble, score it
# against the full sham reference with the identical formula, and repeat. Matching
# the trial count matters: d' here divides by the single-trial sham SD, so a
# pseudo-ensemble averaging a different number of trials would be shrunk by a
# different sqrt(n) and the null would come out the wrong width.
n_real_trials = int(np.median([
    np.sum(np.isfinite(group_trial_resp(dat, tn, grp, base_sl, peak_sl, resps)
                       .reshape(dat.n_rois, -1)[np.argmax(good)]))
    for tn in gmap]))
n_sham_trials = sham_all.shape[1]
n_draw_trials = min(n_real_trials, n_sham_trials)
n_draws = 200
d_sham = []
for _ in range(n_draws):
    sel = rng.choice(n_sham_trials, size=n_draw_trials, replace=False)
    mr = np.nanmean(sham_all[:, sel], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(sigma_sham > 0, (mr - mean_sham_clip) / sigma_sham, np.nan)
    d_sham.append(g[good & np.isfinite(g)])
d_sham = np.concatenate(d_sham)
print(f'sham pseudo-ensembles: {n_draw_trials} trials each (real ensembles have '
      f'~{n_real_trials}), {n_draws} draws')

print(f'{dat.exp_id}: d_real n={d_real.size} (nontarget x ensemble)  |  '
      f'd_sham n={d_sham.size}')


# ---- steps 1-5, applied to an arbitrary score vector ---------------------------
def _run(d_scores, d_null):
    """(fdr per score, corrected scores, pi0, grid, f_dens, f0_dens) — inline, no reuse."""
    n = d_scores.size
    n_bins = N_BINS if N_BINS else int(np.clip(np.sqrt(n) * 2.5, 20, 120))
    spline_df = SPLINE_DF if SPLINE_DF else int(np.clip(n_bins // 6, 3, 7))

    # 1. empirical null: Gaussian fit to the sham scores
    mu0 = np.nanmean(d_null)
    sd0 = np.nanstd(d_null, ddof=1)

    # 2. Poisson-GLM spline smooth of the real-score histogram
    lo = min(np.nanmin(d_scores), np.nanmin(d_null))
    hi = max(np.nanmax(d_scores), np.nanmax(d_null))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(d_scores, bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    binw = edges[1] - edges[0]
    X = dmatrix(f'cr(x, df={spline_df})', {'x': ctr}, return_type='dataframe')
    fit = sm.GLM(counts, X, family=sm.families.Poisson()).fit()

    grid = np.linspace(lo, hi, 1000)
    Xg = dmatrix(X.design_info, {'x': grid}, return_type='dataframe')
    lam = np.asarray(fit.predict(Xg), dtype=float)
    f = lam / (lam.sum() * (grid[1] - grid[0]))          # normalized density
    f = np.maximum(f, 1e-12)
    f0 = norm.pdf(grid, mu0, sd0)

    # 3. null proportion from the central region of the sham distribution
    q1, q3 = np.nanpercentile(d_null, [25, 75])
    central = (grid >= q1) & (grid <= q3)
    # pi0 f0 must sit at or below f in the null region, so pi0 = median(f/f0)
    pi0 = float(np.clip(np.median(f[central] / np.maximum(f0[central], 1e-12)), 0, 1))

    # 4. local fdr on the grid, interpolated back onto each score
    fdr_grid = np.clip(pi0 * f0 / f, 0, 1)
    fdr = interp1d(grid, fdr_grid, bounds_error=False,
                   fill_value=(fdr_grid[0], fdr_grid[-1]))(d_scores)

    # 5. Tweedie correction: z + d/dz log f(z).
    # DIAGNOSTIC ONLY at single-FOV n. Tweedie needs a well-estimated log-density
    # slope; with ~100 scores even a stiff spline gives a derivative that is large
    # and sign-unstable in the tails, so d_corrected can exceed the raw d' instead
    # of shrinking it. Refit here with a deliberately stiff (df=3) density and
    # limit the shift to |z|, which bounds the correction at "shrink to zero".
    Xs = dmatrix('cr(x, df=3)', {'x': ctr}, return_type='dataframe')
    fit_s = sm.GLM(counts, Xs, family=sm.families.Poisson()).fit()
    lam_s = np.asarray(fit_s.predict(dmatrix(Xs.design_info, {'x': grid},
                                             return_type='dataframe')), dtype=float)
    f_s = np.maximum(lam_s / (lam_s.sum() * (grid[1] - grid[0])), 1e-12)
    dlogf = np.gradient(np.log(f_s), grid)
    k = max(3, len(grid) // 40)
    dlogf = np.convolve(dlogf, np.ones(k) / k, mode='same')
    shift = interp1d(grid, dlogf, bounds_error=False,
                     fill_value=(dlogf[0], dlogf[-1]))(d_scores)
    shift = np.clip(shift, -np.abs(d_scores), np.abs(d_scores))
    d_corr = d_scores + shift
    # how often the bound binds -- if most points saturate, the log-density slope
    # is not estimable at this n and d_corrected carries no information.
    tweedie_saturated = float(np.mean(np.isclose(d_corr, 0.0, atol=1e-9)))

    return (fdr, d_corr, pi0, grid, f, f0, mu0, sd0, counts, ctr, binw,
            tweedie_saturated)


(fdr, d_corr, pi0, grid, f, f0, mu0, sd0, counts, ctr, binw,
 tweedie_sat) = _run(d_real, d_sham)

sig = fdr < FDR_THRESH
act = sig & (d_real > 0)
sup = sig & (d_real < 0)
print(f'empirical null: mean={mu0:+.3f}  SD={sd0:.3f}  (N(0,1) would be 0.000 / 1.000)')
print(f'pi0 = {pi0:.3f}')
print(f'significant (local fdr < {FDR_THRESH}): {sig.sum()}/{sig.size} '
      f'({100*sig.mean():.1f}%)  |  activated {act.sum()} ({100*act.mean():.1f}%)  '
      f'suppressed {sup.sum()} ({100*sup.mean():.1f}%)')

# ---- 6. validation: same machinery run on the sham scores themselves ------------
v_fdr, _, v_pi0, *_ = _run(d_sham, d_sham)
if tweedie_sat > 0.5:
    print(f'[!] Tweedie: {100*tweedie_sat:.0f}% of points hit the shrink-to-zero '
          f'bound -- n={d_real.size} is too small to estimate the log-density '
          f'slope. Treat d_corrected as uninformative here; use d_raw + fdr.')
print(f'[validation] sham-as-real: pi0={v_pi0:.3f} (expect ~1), '
      f'significant {(v_fdr < FDR_THRESH).sum()}/{v_fdr.size} '
      f'({100*(v_fdr < FDR_THRESH).mean():.2f}%, expect ~0)')

# ---- 7. views ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.bar(ctr, counts, width=binw, color='0.8', edgecolor='none', label='d_real')
scale = counts.sum() * binw
ax.plot(grid, pi0 * f0 * scale, 'r-', lw=2, label=r'$\pi_0 f_0$ (null)')
ax.plot(grid, f * scale, 'k-', lw=1.5, label='f (spline fit)')
ax.axvline(0, color='0.5', ls=':', lw=1)
ax.set_xlabel("influence d'"); ax.set_ylabel('count')
ax.set_title(f'excess over null  |  $\\pi_0$={pi0:.3f}')
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
o = np.argsort(d_real)
ax.plot(d_real[o], fdr[o], 'k-', lw=1.5)
ax.axhline(FDR_THRESH, color='r', ls='--', lw=1, label=f'fdr = {FDR_THRESH}')
ax.axvline(0, color='0.5', ls=':', lw=1)
ax.set_xlabel("influence d'"); ax.set_ylabel('local fdr'); ax.set_ylim(0, 1.05)
ax.set_title(f'{sig.sum()} significant  ({act.sum()} act / {sup.sum()} sup)')
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
ax.scatter(d_real[~sig], d_corr[~sig], s=8, c='0.75', edgecolor='none', label='n.s.')
ax.scatter(d_real[sig], d_corr[sig], s=10, c='crimson', edgecolor='none',
           label=f'fdr < {FDR_THRESH}')
lims = [np.nanmin(d_real), np.nanmax(d_real)]
ax.plot(lims, lims, 'k--', lw=1)
ax.axhline(0, color='0.5', ls=':', lw=1); ax.axvline(0, color='0.5', ls=':', lw=1)
ax.set_xlabel("raw d'"); ax.set_ylabel("Tweedie-corrected d'")
ax.set_title('shrinkage')
ax.legend(frameon=False, fontsize=8)

fig.suptitle(f'{dat.exp_id}  |  grand-mean influence significance (nontargets)')
fig.tight_layout()

# per-(cell, ensemble) table for inspection
sig_table = np.column_stack([real_tn_lbl, real_cell, d_real, d_corr, fdr, sig])
print('\ncolumns: group, cell, d_raw, d_corrected, fdr, significant')
print(sig_table[np.argsort(fdr)][:20])

