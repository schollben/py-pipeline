# %% 1. initialize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.session import load_session
from analysis import rois, responses, tuning, selectivity, maps

sns.set_theme(context='notebook', style='white')

FNAME = '/home/schollab-gaga/Desktop/test/TSeries-07132025-1042-002.h5'

s = load_session(FNAME)

print(f'{s.exp_id}: {s.n_rois} ROIs, '
      f'{len(s.directions)} directions, {len(s.contrasts)} contrasts')

# %% 2. average image with ROI masks
rois.plot_avg_rois(s)
plt.show()

# %% 3. trial-averaged time-varying responses (one or more cells)
responses.plot_stim_traces(s, 5)
plt.show()

# %% 4. tuning curves + preferred direction (double-Gaussian fit)
tuning.plot_tuning_curves(s)
plt.show()

# %% 5. direction / orientation selectivity
selectivity.compute_selectivity(s)
print(f'gDSI median {np.nanmedian(s.gdsi):.3f}  |  gOSI median {np.nanmedian(s.gosi):.3f}')

# %% 6. preference maps (direction | orientation)
maps.plot_preference_maps(s, thr=0.1)
plt.show()
