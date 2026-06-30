import numpy as np


def compute_selectivity(s):
    """Normalized vector strength at the top contrast (rectified responses).

    gDSI uses k=1 (direction), gOSI uses k=2 (orientation). Sets s.gdsi, s.gosi.
    """
    cons = s.contrasts
    top_ids = np.where(s.stim_table[:, 1] == cons[-1])[0]
    theta = np.deg2rad(s.stim_table[top_ids, 0])

    gdsi = np.full(s.n_rois, np.nan)
    gosi = np.full(s.n_rois, np.nan)
    for cc in range(s.n_rois):
        r = np.clip(s.resp[cc, top_ids], 0, None)
        total = r.sum()
        if total > 0:
            gdsi[cc] = np.abs(np.sum(r * np.exp(1j * theta))) / total
            gosi[cc] = np.abs(np.sum(r * np.exp(2j * theta))) / total

    s.gdsi, s.gosi = gdsi, gosi
    return gdsi, gosi
