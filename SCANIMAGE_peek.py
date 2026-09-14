# dff_quicklook.py — ScanImage stand-in for bruker_pipeline.process_experiment's
# DFF extraction, for data that has no TSeries XML (no frame_period, no
# markpoints, no vrec). Trace extraction / dF/F code below is copied directly
# from img_utils.py and bruker_pipeline.py — only the XML-derived inputs are
# substituted.
#
# SUBSTITUTIONS (everything else matches the original):
#   1. frame_period — normally parsed from TSeries XML. ScanImage has no
#      equivalent file here, so this is set from FRAME_RATE below (~30 Hz,
#      approximate — not read from metadata).
#   2. h5 movie key — process_experiment uses dat_name = list(h.keys())[0]
#      (first key in the file). Kept identical here.
#
# No opto, no visual stim (per this experiment) — Steps 9-12 of
# process_experiment (stim cycles, opto delta images, vrec) are skipped
# entirely, not substituted.

import os
import math
import h5py
import numpy as np
import roifile
from scipy import signal
from statistics import median
from skimage.draw import polygon2mask
from matplotlib import path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

FOLDER = "/mnt/bigdata/BRUKER_LOCAL/TSeries-08212026-1111-002"
FRAME_RATE = 30.0                    # SUBSTITUTION — approximate, not from metadata
frame_period = 1.0 / FRAME_RATE
chunk_size = 1000
n_plot = 30


# ── copied from img_utils.py ────────────────────────────────────────────

def gen_polyline_roi(nm_coord, d_width=10.0, size_x=512, size_y=512):
	x = nm_coord[:,1]
	y = nm_coord[:,0]
	r = np.sqrt((x[1:] - x[0:-1])**2 + (y[1:] - y[0:-1])**2)

	delta_x = np.multiply((d_width / 2) / r, (y[:-1] - y[1:]))
	delta_y = np.multiply((d_width / 2) / r, (x[1:] - x[:-1]))
	new_x = x + np.append(delta_x, delta_x[-1])
	new_y = y + np.append(delta_y, delta_y[-1])

	delta_x = np.multiply((-d_width / 2) / r, (y[:-1] - y[1:]))
	delta_y = np.multiply((-d_width / 2) / r, (x[1:] - x[:-1]))
	new_x = np.append(new_x, np.flip(x + (np.append(delta_x, delta_x[-1]))))
	new_y = np.append(new_y, np.flip(y + (np.append(delta_y, delta_y[-1]))))

	new_x = np.append(new_x, new_x[0])
	new_y = np.append(new_y, new_y[0])

	polyline_mask = polygon2mask((size_x, size_y), np.array([new_x, new_y]).T)
	return polyline_mask.T


def in_polygon(xq, yq, xv, yv):
	shape = xq.shape
	xq = xq.reshape(-1); yq = yq.reshape(-1)
	xv = xv.reshape(-1); yv = yv.reshape(-1)
	q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
	p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
	return p.contains_points(q).reshape(shape)


def prctfilt1(x, n=3, blksz=1000, dim=None):
	x = np.asarray(x)
	if dim is None:
		dim = 0
		while dim < x.ndim and x.shape[dim] == 1:
			dim += 1
		if dim == x.ndim:
			return x.copy()
	if dim != 0:
		perm = np.arange(x.ndim)
		perm[0], perm[dim] = dim, 0
		x = np.transpose(x, perm)
		restore_perm = np.argsort(perm)
	if blksz is None:
		blksz = x.shape[0]
	m = (n - 1) // 2 if n % 2 == 1 else n // 2

	y = np.zeros_like(x)
	for idx in np.ndindex(x.shape[1:]):
		vector = x[(slice(None),) + idx]
		padded = np.zeros(len(vector) + 2*m)
		padded[m:m+len(vector)] = vector
		for i in range(0, len(vector), blksz):
			end_idx = min(i + blksz, len(vector))
			chunk_size_ = end_idx - i
			windows = np.zeros((n, chunk_size_))
			for j in range(n):
				windows[j] = padded[i+j:i+j+chunk_size_]
			y[(slice(i, end_idx),) + idx] = np.percentile(windows, 20, axis=0)
	if dim != 0:
		y = np.transpose(y, restore_perm)
	return y


def filter_baseline_dF_comp(raw, pts=99):
	F_temp = raw
	F_temp = np.concatenate((np.repeat(np.mean(F_temp[2:5]), pts), F_temp))
	F_temp = np.concatenate((F_temp, np.repeat(np.mean(F_temp[-5:-2]), pts)))
	F_temp = signal.medfilt(F_temp, pts)
	raw_new = F_temp[pts:-pts]
	raw_new = np.divide((raw - raw_new), raw_new)
	raw_newlpf = prctfilt1(raw_new, 91)
	raw_new = raw_new - raw_newlpf
	return raw_new


# ── adapted from bruker_pipeline.process_experiment ─────────────────────

roi_zip = os.path.join(FOLDER, "RoiSet.zip")
roi_list = roifile.roiread(roi_zip)
num_cells = len(roi_list)
print(f'{num_cells} ROIs loaded')

is_dendrite = np.array([r.roitype == 5 for r in roi_list])
if any(is_dendrite):
	is_spine = np.array([r.roitype == 7 for r in roi_list])
	is_soma  = np.zeros(num_cells, dtype=bool)
else:
	is_spine = np.zeros(num_cells, dtype=bool)
	is_soma  = np.array([r.roitype == 7 for r in roi_list])

inference_path = os.path.join(FOLDER, "inference.h5")
h = h5py.File(inference_path, 'r')
dat_name = list(h.keys())[0]
num_frames, size_x, size_y = h[dat_name].shape
print(f'Movie: {num_frames} frames, {size_x}×{size_y} px')

xx = np.linspace(0, size_x - 1, size_x)
yy = np.linspace(0, size_y - 1, size_y)
xx, yy = np.meshgrid(xx, yy)

mask2d = np.zeros((num_cells, size_x, size_y), dtype=float)
for cc in tqdm(range(num_cells), desc='Building masks', ncols=75):
	nm_coord = roi_list[cc].coordinates()
	if roi_list[cc].roitype == 5:
		mask2d[cc] = gen_polyline_roi(nm_coord=nm_coord, d_width=roi_list[cc].stroke_width,
		                               size_x=size_x, size_y=size_y)
	else:
		mask2d[cc] = in_polygon(xx, yy, nm_coord[:, 0], nm_coord[:, 1])

raw_traces = np.zeros((num_frames, num_cells))
nz_per_cell = [np.nonzero(mask2d[cc]) for cc in range(num_cells)]
n_chunks = math.ceil(num_frames / chunk_size)

for f_i in tqdm(range(n_chunks), desc='Extracting traces', ncols=75):
	start = f_i * chunk_size
	stop = min((f_i + 1) * chunk_size, num_frames)
	chunk = h[dat_name][start:stop]
	for cc in range(num_cells):
		raw_traces[start:stop, cc] = np.mean(chunk[:, nz_per_cell[cc][0], nz_per_cell[cc][1]], axis=1)

h.close()

dff_baseline_sec = 13.2
dff_window = int(round(dff_baseline_sec / frame_period))
if dff_window % 2 == 0:
	dff_window += 1
print(f'dF/F baseline window: {dff_window} frames ({dff_window * frame_period:.1f} s at {1/frame_period:.1f} Hz)')

dff = np.zeros((num_frames, num_cells))
for cc in tqdm(range(num_cells), desc='Computing dF/F', ncols=75):
	dff[:, cc] = filter_baseline_dF_comp(raw_traces[:, cc], dff_window)

n_bad = int(np.sum(~np.isfinite(dff)))
if n_bad:
	print(f'[dF/F] Replacing {n_bad} non-finite samples (divide-by-~0) with NaN.')
	dff[~np.isfinite(dff)] = np.nan

# ── trace plot panel, copied from plot_experiment_summary's right panel ──

n_plot_frames = min(num_frames, 12000)
t_axis = np.arange(n_plot_frames) * frame_period
n_plot = min(n_plot, num_cells)

spacing = 2.0 * np.nanpercentile(np.abs(dff[:n_plot_frames, :n_plot]), 99)
spacing = max(spacing, 0.05)

fig, ax_dff = plt.subplots(figsize=(14, 6))
for cc in range(n_plot):
	offset = cc * spacing
	ax_dff.plot(t_axis, dff[:n_plot_frames, cc] + offset, linewidth=0.6, color='black', alpha=0.75)
	ax_dff.text(t_axis[-1], offset, f' {cc}', va='center', fontsize=6, color='black')

ax_dff.set_xlabel('Time (s)')
ax_dff.set_ylabel('dF/F  (offset per cell)')
ax_dff.set_title(f'dF/F traces — first {n_plot} cells — {os.path.basename(FOLDER)}')
ax_dff.set_xlim(t_axis[0], t_axis[-1])
ax_dff.set_yticks([])
plt.tight_layout()

out_path = os.path.join(FOLDER, 'dff_quicklook_first30.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()

