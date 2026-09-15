# %% plot data with TTLs to check timing of photostim vs visual stim with dff
# loads the procesed h5 file and plots the dff for a single ROI, with vertical lines for stim_on and photostim events
import h5py
import numpy as np
import plotly.graph_objects as go

dirLoc = '/mnt/bigdata/PROCESSED/' # update for your computer
# FNAME = dirLoc + 'TSeries-07132025-1042-003.h5'
# FNAME = dirLoc + 'TSeries-07212026-1350-001.h5'
FNAME = dirLoc + 'TSeries-11032024-1313-012.h5'

OFFSET = 15
ROI = 21
frameStart, frameEnd = 0, 2000

with h5py.File(FNAME, 'r') as f:
    dff = f['dff'][frameStart:frameEnd, ROI]
    dff_nan = f['dff_nan'][frameStart:frameEnd, ROI]
    stim_on = f['stim_on_2p_frame'][:].ravel() - OFFSET
    photostim = f['photostim_2p_frame'][:].ravel() - OFFSET
    stim_id = f['stim_id'][:].ravel()
    target_number = f['target_number'][:].ravel()
    frame_period = float(f['Bruker_Acq']['frame_period'][()])

n = min(len(stim_on), len(photostim))

print(f'stim_on={len(stim_on)}  photostim={len(photostim)}  '
      f'stim_id={len(stim_id)}  target_number={len(target_number)}')
print(f'photostim - stim_on: unique={np.unique(photostim[:n] - stim_on[:n])}  '
      f'frame_period={frame_period:.4f}s')

x = np.arange(frameStart, frameEnd)

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=dff, name=f'ROI {ROI}',
                         line=dict(color='black', width=1)))
# fig.add_trace(go.Scatter(x=x, y=dff_nan, name='dff_nan', opacity=0.4,
#                          line=dict(color='orange', width=2)))

# stim_id / target_number are empty on sessions whose trial table was never built,
# so label with them only where an entry exists
for i in np.where((stim_on >= frameStart) & (stim_on < frameEnd))[0]:
    lbl = f'#{i} sid={stim_id[i]:g}' if i < len(stim_id) else f'#{i}'
    fig.add_vline(x=stim_on[i], line=dict(color='royalblue', width=1, dash='dot'),
                  annotation_text=lbl, annotation_font_size=8)

for i in np.where((photostim >= frameStart) & (photostim < frameEnd))[0]:
    lbl = f'tn={target_number[i]:g}' if i < len(target_number) else None
    fig.add_vline(x=photostim[i], line=dict(color='crimson', width=1, dash='dot'),
                  annotation_text=lbl, annotation_font_size=8)

fig.update_xaxes(title_text='2P frame')
fig.update_yaxes(title_text='dF/F')
fig.show()

# %%
