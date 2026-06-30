import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROI_COLORS = {'soma': 'cyan', 'dendrite': 'yellow', 'spine': 'magenta'}


def _roi_type_color(s, cc):
    if s.is_soma is not None and s.is_soma[cc]:
        return _ROI_COLORS['soma']
    if s.is_dendrite is not None and s.is_dendrite[cc]:
        return _ROI_COLORS['dendrite']
    if s.is_spine is not None and s.is_spine[cc]:
        return _ROI_COLORS['spine']
    return 'lime'


def plot_avg_rois(s, label=True, ax=None, vmax_frac=0.5):

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(s.avg_image, cmap='gray', interpolation='nearest',
              vmax=s.avg_image.max() * vmax_frac)

    for cc in range(s.n_rois):
        color = _roi_type_color(s, cc)
        ax.contour(s.mask2d[cc], levels=[0.5], colors=[color],
                   linewidths=0.6, alpha=0.8)
        if label:
            pts = np.argwhere(s.mask2d[cc] > 0.5)
            if len(pts):
                cy, cx = pts.mean(axis=0)
                ax.text(cx, cy, str(cc), ha='center', va='center',
                        fontsize=6, color='black', fontweight='bold', clip_on=True)
    ax.legend(handles=[mpatches.Patch(color=c, label=k.capitalize())
                       for k, c in _ROI_COLORS.items()],
              loc='upper right', fontsize=8)
    
    ax.set_title(f'{s.exp_id}  |  {s.n_rois} ROIs')
    ax.axis('off')
    return ax
