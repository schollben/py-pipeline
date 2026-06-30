import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb


def _pastel(frac, sat):
    """Map a [0,1) hue to a pastel-HSV RGB (blended toward white)."""
    rgb = np.array(hsv_to_rgb([frac, 1.0, 1.0]))
    return rgb * sat + (1 - sat)


def _draw_map(ax, s, pref, period, flag, sat, title, cmap_name):
    ax.imshow(s.avg_image, cmap='gray', vmax=s.avg_image.max() / 2)
    for cc in range(s.n_rois):
        m = s.mask2d[cc] > 0.5
        if flag[cc] and not np.isnan(pref[cc]):
            rgb = _pastel((pref[cc] % period) / period, sat)
        else:
            rgb = (0.35, 0.35, 0.35)
        ax.imshow(np.dstack([np.ones_like(s.avg_image)] * 3) * rgb,
                  alpha=m.astype(float))
    ax.set_title(title)
    ax.axis('off')
    sm = cm.ScalarMappable(cmap=cmap_name)
    sm.set_array([0, period])
    plt.colorbar(sm, ax=ax, fraction=0.046, label='deg')


def plot_preference_maps(s, thr=0.1, sat=0.5):
    fig, (axd, axo) = plt.subplots(1, 2, figsize=(13, 6))
    _draw_map(axd, s, s.pref_dir, 360, s.gdsi >= thr, sat,
              'Direction preference', 'hsv')
    _draw_map(axo, s, s.pref_dir, 180, s.gosi >= thr, sat,
              'Orientation preference', 'hsv')
    fig.suptitle(s.exp_id)
    fig.tight_layout()
    return fig
