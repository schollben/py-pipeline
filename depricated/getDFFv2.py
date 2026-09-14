# %% Imports and Configuration
# getDFFv2 — Bruker calcium imaging pipeline
# Edit the configuration below, then run cell-by-cell to validate each step.

import os
import sys
import math
from glob import glob
import roifile
import h5py
import numpy as np
from scipy.signal import medfilt, find_peaks
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from tqdm import tqdm
from img_utils import *

save_location = '/mnt/bigdata/'  # where does BRUKER or SCANIMAGE folders live
date          = '12172024'       # date of acquisition, format MMDDYYYY
file_num      = 18               # TSeries number
stim_file     = 19               # Psychopy file number. Set to -1 for no stimulus.
optical_zoom  = 12               # Optical zoom used during acquisition
dur_resp      = 2.5              # response duration in seconds
frame_period  = 0.033            # frame period (0.033 for 30Hz, 0.05 for 20Hz)
chnk          = int(1e3)         # Number of frames to process at a time

inference     = True             # True = inference_results.h5, False = registered.h5
is_2p_opto    = True            # True if 2p optogenetic dataset
do_neuropil   = False            # True to run neuropil subtraction
is_spontaneous = False            # True if no stimulus, just spontaneous activity

output_dir    = '/mnt/PROCESSED/'

# %% Resolve data folder and open output H5

folder_list = get_target_folders_v2(os.path.join(save_location, 'BRUKER', ''), date, file_num, 'TSeries')
assert len(folder_list) > 0, 'No matching TSeries folder found.'
data_folder = folder_list[0]
print(f'Data folder: {data_folder}')

os.makedirs(output_dir, exist_ok=True)
out_h5_name = os.path.join(output_dir, f'TSeries-{date}-{file_num:03d}.h5')
outfile = h5py.File(out_h5_name, 'w')
outfile.create_dataset('frame_period', data=np.array([frame_period]))
outfile.create_dataset('do_cascade', data=np.array([False]))  # needed by gen_stim_cyc
outfile.create_dataset('stim_file', data=stim_file)

# %% Load ROIs

roi_path = os.path.join(data_folder, 'RoiSet.zip')
try:
    roi_list = roifile.roiread(roi_path)
except FileNotFoundError:
    print(f'Halting. No ROI file found at {roi_path}')
    sys.exit(1)

num_cells = len(roi_list)
print(f'{num_cells} ROIs loaded')

is_dendrite = np.array([roi.roitype == 5 for roi in roi_list])
if any(is_dendrite):
    is_spine = np.array([roi.roitype == 7 for roi in roi_list])
    is_soma  = np.array([False] * num_cells)
else:
    is_spine = np.array([False] * num_cells)
    is_soma  = np.array([roi.roitype == 7 for roi in roi_list])

outfile.create_dataset('is_dendrite', data=is_dendrite)
outfile.create_dataset('is_spine',    data=is_spine)
outfile.create_dataset('is_soma',     data=is_soma)

# %% Open calcium movie and compute average projection

if inference:
    h5_movie_path = os.path.join(data_folder, 'inference_results.h5')
else:
    h5_movie_path = os.path.join(data_folder, 'registered.h5')

try:
    h = h5py.File(h5_movie_path, 'r')
except FileNotFoundError:
    print(f'Halting. Calcium movie not found at {h5_movie_path}')
    sys.exit(1)

dat_name = list(h.keys())[0]
num_frames, size_x, size_y = h[dat_name].shape
print(f'Movie: {num_frames} frames, {size_x}x{size_y}')

# Average projection from first 100 frames
n_proj = min(100, num_frames)
avg_projection = np.mean(h[dat_name][:n_proj, :, :], axis=0)

# %% Build cell masks

x = np.linspace(0, size_x - 1, size_x)
y = np.linspace(0, size_y - 1, size_y)
x, y = np.meshgrid(x, y)
mask2d = np.zeros((num_cells, size_x, size_y))
neuropil_mask = np.zeros((size_x, size_y))

for cc in tqdm(range(num_cells), desc="Getting masks", ncols=75):
    nm_coord = roi_list[cc].coordinates()
    if roi_list[cc].roitype == 5:
        mask2d[cc, :, :] = gen_polyline_roi(nm_coord=nm_coord, d_width=roi_list[cc].stroke_width)
    else:
        mask2d[cc, :, :] = in_polygon(x, y, nm_coord[:, 0], nm_coord[:, 1])
    neuropil_mask += mask2d[cc, :, :]

outfile.create_dataset('mask2d', data=mask2d)
outfile.create_dataset('neuropil_mask', data=neuropil_mask)

# %% Extract raw traces

raw_cell_traces = np.zeros((num_frames, num_cells))
raw_neuropil    = np.zeros((num_frames,))

for f_i in tqdm(range(math.ceil(num_frames / chnk)), desc="Extracting traces", ncols=75):
    start = int(f_i * chnk)
    stop = min(int((f_i + 1) * chnk), num_frames)
    imgstack = h[dat_name][start:stop, :, :]

    for cc in range(num_cells):
        nz = np.nonzero(mask2d[cc, :, :])
        raw_cell_traces[start:stop, cc] = np.mean(imgstack[:, nz[0], nz[1]], axis=1)

    if do_neuropil:
        nz = np.nonzero(neuropil_mask)
        raw_neuropil[start:stop] = np.mean(imgstack[:, nz[0], nz[1]], axis=1)

# %% Compute dF/F

dff = np.zeros((num_frames, num_cells))
for cc in tqdm(range(num_cells), desc="Computing dF/F", ncols=75):
    dff[:, cc] = filter_baseline_dF_comp(raw_cell_traces[:, cc], 99 * 4 + 1)

dff_neuropil = None
if do_neuropil:
    dff_neuropil = filter_baseline_dF_comp(raw_neuropil, 99 * 4 + 1)

outfile.create_dataset('raw_cell_traces', data=raw_cell_traces)
outfile.create_dataset('dff', data=dff)
if do_neuropil:
    outfile.create_dataset('raw_neuropil', data=raw_neuropil)
    outfile.create_dataset('dff_neuropil', data=dff_neuropil)

# %% Stimulus processing

has_stim = stim_file > -1

if has_stim:
    print('Grabbing two-photon frametimes...')
    xml_dest = os.path.join(data_folder, os.path.basename(data_folder)) + '.xml'
    frame_triggers = read_xml_file(xml_dest)
    frame_triggers = frame_triggers * 1e4  # Convert seconds to 10kHz sampling
    if frame_triggers[0] > 340:
        print('First 2p frame was dropped!')
    frame_triggers = replace_missing_frame_triggers(frame_triggers)
    outfile.create_dataset('frame_triggers', data=frame_triggers)

    # Voltage recording
    voltage_files = glob(os.path.join(data_folder, '*VoltageRecording*.csv'))
    assert len(voltage_files) == 1, \
        f'Expected 1 voltage file, found {len(voltage_files)} in {data_folder}'
    vrec = genfromtxt_with_progress(voltage_files[0], delimiter=',', skip_header=1)
    outfile.create_dataset('vrec', data=vrec)

    # Detect stimulus onsets
    print('Getting stimulus times and syncing with two-photon frame times...')
    stim_trig = medfilt(vrec[:, 1], 101)
    stim_trig[stim_trig < 0] = 0
    stim_trig = np.diff(stim_trig)

    stim_on, _  = find_peaks(stim_trig, distance=1e3,
                              height=(max(stim_trig) - max(stim_trig) * 0.9))
    stim_off, _ = find_peaks(-stim_trig, distance=1e3,
                              height=(max(-stim_trig) - max(-stim_trig) * 0.9))
    outfile.create_dataset('stim_on', data=stim_on)
    outfile.create_dataset('stim_off', data=stim_off)

    # 2p opto photostim triggers
    if is_2p_opto:
        print('Processing 2pOpto triggers...')
        photostim_trig = medfilt(vrec[:, 1], 51)
        photostim_trig[photostim_trig < 0] = 0
        photostim_triggers = find_peaks(photostim_trig, distance=1e4,
                                        height=(max(photostim_trig) - max(photostim_trig) * 0.9))
        photostim_triggers = photostim_triggers[0]
        outfile.create_dataset('photostim_triggers', data=photostim_triggers)

    # Read psychopy file
    psychopy_loc = os.path.join(save_location, 'BRUKER_PSYCHOPY')
    psychopy_date_dir = '-'.join([date[4:], date[0:2], date[2:4]])
    psychopy_path = os.path.join(psychopy_loc, psychopy_date_dir, f'T{stim_file:03d}.txt')
    psychopy_data = np.genfromtxt(psychopy_path)

    if not is_2p_opto:
        stim_id = psychopy_data[:, 0]
        unique_stims = np.unique(stim_id)
        stim_properties = psychopy_data[:, 1:]
        outfile.create_dataset('stim_properties', data=stim_properties)
    else:
        if psychopy_data.shape[1] == 2:
            stim_id = psychopy_data[:, 0]
        else:
            stim_id = psychopy_data[:, 2]
        unique_stims = np.unique(stim_id)
        if psychopy_data.shape[1] > 3:
            outfile.create_dataset('stim_properties', data=psychopy_data[:, 3:])
        outfile.create_dataset('target_number', data=psychopy_data[:, 0])
        outfile.create_dataset('target_trial', data=psychopy_data[:, 1])

    outfile.create_dataset('stim_id', data=stim_id)
    outfile.create_dataset('unique_stims', data=unique_stims)

    assert len(stim_on) == len(stim_id), \
        f'Mismatch: {len(stim_on)} stimulus onsets detected, but {len(stim_id)} stim IDs. ' \
        f'Check voltage recording channels.'

    # Sync stim onsets to 2p frames
    stim_on_2p_frame = np.zeros(len(stim_id))
    for s in range(len(stim_id)):
        stim_on_2p_frame[s] = np.argmin(abs(stim_on[s] - frame_triggers))
    outfile.create_dataset('stim_on_2p_frame', data=stim_on_2p_frame)

    # 2p opto frame synchronization
    if is_2p_opto:
        target_stim_2p_frame = np.zeros(len(photostim_triggers))
        for ss in range(len(photostim_triggers)):
            target_stim_2p_frame[ss] = np.argmin(abs(photostim_triggers[ss] - frame_triggers))
        outfile.create_dataset('target_stim_2p_frame', data=target_stim_2p_frame)

else:
    print('No stimulus file specified. Skipping stimulus processing.')

# %% Stimulus cycles and peak responses

if has_stim:
    if do_neuropil:
        neuropil_subtraction(outfile=outfile, roi_list=roi_list)

    gen_stim_cyc(outfile=outfile, pre=0, slag=0, dur_resp=dur_resp)

    # Peak responses
    n_stims = len(unique_stims)
    if np.floor(len(stim_id) / n_stims) > 2:
        cyc = outfile['cyc'][:]
        n_trials = cyc.shape[2]
        resp_all     = np.zeros((num_cells, n_stims))
        resps_all    = np.zeros((num_cells, n_stims, n_trials))
        resp_err_all = np.zeros((num_cells, n_stims))

        for cc in range(num_cells):
            resp, resps, resp_err = compute_peak_resp(cyc[cc, :, :, :])
            resp_all[cc, :] = resp
            resps_all[cc, :, :] = resps
            resp_err_all[cc, :] = resp_err

        outfile.create_dataset('resp', data=resp_all)
        outfile.create_dataset('resps', data=resps_all)
        outfile.create_dataset('resp_err', data=resp_err_all)
    else:
        print('Too few trials per stimulus to compute peak responses.')

# %% Per-stimulus average images

if has_stim:
    print('Computing per-stimulus average images...')
    stim_on_2p = outfile['stim_on_2p_frame'][:].astype(int)
    n_avg_frames = 30

    stim_avg_images = np.zeros((len(unique_stims), size_x, size_y))
    for si, stim_val in enumerate(tqdm(unique_stims, desc="Stim avg images", ncols=75)):
        trial_indices = np.where(stim_id == stim_val)[0]
        acc = np.zeros((size_x, size_y))
        count = 0
        for ti in trial_indices:
            onset = stim_on_2p[ti]
            if onset + n_avg_frames <= num_frames:
                acc += np.mean(h[dat_name][onset:onset + n_avg_frames, :, :], axis=0)
                count += 1
        if count > 0:
            stim_avg_images[si] = acc / count

    outfile.create_dataset('stim_avg_images', data=stim_avg_images)
    print(f'Saved stim_avg_images: {stim_avg_images.shape}')

# %% Save and close

h.close()
outfile.close()
print(f'Results saved to {out_h5_name}')

# %% Visualization

# Figure 1: Average projection
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
ax1.imshow(avg_projection, cmap='gray')
ax1.set_title('Average Projection (first 100 frames)')
ax1.axis('off')

# Figure 2: First 10 ROI dF/F traces
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
n_to_plot = min(10, dff.shape[1])
offsets = np.arange(n_to_plot) * 2
for i in range(n_to_plot):
    ax2.plot(dff[:, i] + offsets[i], linewidth=0.5, label=f'ROI {i}')
ax2.set_xlabel('Frame')
ax2.set_ylabel('dF/F (offset)')
ax2.set_title('dF/F Traces (first 10 ROIs)')
ax2.legend(loc='upper right', fontsize=7)

# Figure 3: Average trace per stimulus ID (only if stim processing ran)
if has_stim:
    with h5py.File(out_h5_name, 'r') as f:
        cyc = f['cyc'][:]           # (cells, stims, trials, frames)
        unique_stims_plot = f['unique_stims'][:]

    avg_per_stim = np.nanmean(cyc, axis=(0, 2))  # -> (stims, frames)

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    for si in range(avg_per_stim.shape[0]):
        ax3.plot(avg_per_stim[si, :], label=f'Stim {unique_stims_plot[si]:.0f}')
    ax3.set_xlabel('Frame (relative to stim onset)')
    ax3.set_ylabel('dF/F')
    ax3.set_title('Average Trace per Stimulus')
    ax3.legend(fontsize=7)

plt.show()
