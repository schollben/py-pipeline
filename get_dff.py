# getDFF
# processing script for extracting traces from calcium movies

import os
import sys
import code
from glob import glob
import tifffile
import roifile
import math
import h5py
import numpy as np
from scipy.signal import medfilt, find_peaks
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm
from img_utils import *


save_location = '/mnt/md0/'
data_type = 'BRUKER'
date = '12172024'
file_num = 18
stim_file= 19
optical_zoom = 12
dur_resp  = 2.5 # seconds

# Uncomment as each is verified in behavior
do_neuropil   = False
do_cascade    = False
cascade_model = ''
is_2p_opto    = False
# is_voltage = False
# extract_hotspot = False

# 30 frames added to movie start at .h5 conversion.
#   If DeepInterpolation is not run, these must be accounted for.
#   DeepInterpolation removes these when run.
is_deep_interp = True

frame_period = 0.033
chnk = int(1e3) # Number of frames to process at a time

# Cascade filenames here ~~~~~~~~~~~~~~~~~~~~~~
cascade_dir = '/home/schollab-dion/Documents/Cascade-1.0.0'
cascade_model_dir = os.path.join(cascade_dir, 'Pretrained_models')
assert os.path.isdir(os.path.join(cascade_model_dir, cascade_model)),  \
    f'Specified cascade model {cascade_model} not found.' \
    'Install, or check if a valid pretrained cascade model was specified.'
sys.path.append(cascade_dir)
from cascade2p.cascade import predict as cascade_predict


# Storing results so far
# This may not be the best way of passing around information globally.
out_h5_name = 'TSeries-'+date+'-'+f'{file_num:03d}.h5'
outfile = h5py.File(out_h5_name, 'w')
outfile.create_dataset('frame_period', data=np.array([frame_period]))
outfile.create_dataset('stim_file', data=stim_file)
outfile.create_dataset('do_cascade', data=np.array([do_cascade]))

# Processing starting...


folder_list = get_target_folders_v2(save_location+data_type+'/', date,file_num, 'TSeries')

target_folder = folder_list[0]
os.chdir(target_folder)
print(f'Starting CE creation for {target_folder}')

## ROI file handling
try:
    roi_list = roifile.roiread('RoiSet.zip')
except FileNotFoundError:
    print(f'Halting. No ROI file for {target_folder}')
    sys.exit(1) # kill
else:
    num_cells = len(roi_list)
    located_dend_roi = any(roi.roitype == 5 for roi in roi_list)

is_dendrite = [roi.roitype == 5 for roi in roi_list]
if any(is_dendrite):
    is_spine = np.array([roi.roitype == 7 for roi in roi_list])
    is_soma  = np.array([False] * num_cells)
else:
    is_spine = np.array([False] * num_cells)
    is_soma  = np.array([roi.roitype == 7 for roi in roi_list])

outfile.create_dataset('is_dendrite', data=is_dendrite)
outfile.create_dataset('is_spine',    data=is_spine)
outfile.create_dataset('is_soma',     data=is_soma)


## Grabbing .h5 processed calcium movie handling - 
try:
    if is_deep_interp:
        h = h5py.File('inference_results.h5', 'r')
    else:
        h = h5py.File('registered.h5', 'r')
except FileNotFoundError:
    print(f'Halting. Processed calcium movie file not found. \n Check destination and/or "is_deep_interp" flag.')
    sys.exit(1) # kill

dat_name = [key for key in h.keys()][0]
num_frames, size_x, size_y = h[dat_name].shape


## Grabbing cell masks -
x = np.linspace(0, size_x-1, size_x)
y = np.linspace(0, size_y-1, size_y)
x, y = np.meshgrid(x, y)
mask2d = np.zeros((num_cells, size_x, size_y))
neuropil_mask = np.zeros((size_x,size_y))

for cc in tqdm(range(num_cells), desc="getting masks", ncols=75):
    nm_coord = roi_list[cc].coordinates()
    if roi_list[cc].roitype == 5:       
        mask2d[cc,:,:] = gen_polyline_roi(nm_coord=nm_coord, d_width=roi_list[cc].stroke_width)
    else:
        mask2d[cc,:,:] = in_polygon(x, y, nm_coord[:,0], nm_coord[:,1])
    neuropil_mask += mask2d[cc,:,:]
outfile.create_dataset('mask2d', data=mask2d)
outfile.create_dataset('neuropil_mask', data=neuropil_mask)


## Using cell masks to extract raw cell traces -
raw_cell_traces = np.zeros((num_frames, num_cells))
raw_neuropil    = np.zeros((num_frames))

for f_i in tqdm(range(math.ceil(num_frames / chnk)), desc="Extracting...", ncols=75):
    start = int(f_i * chnk)
    stop = min(int((f_i + 1) * chnk), num_frames)
    imgstack = h[dat_name][start:stop, :, :]
    
    for cc in range(num_cells):
        nz = np.nonzero(mask2d[cc,:,:])
        raw_cell_traces[start:stop, cc] = np.mean(imgstack[:,nz[0], nz[1]], axis=1)
    
    # Need to get nz_neuropil as well
    if do_neuropil:
        nz = np.nonzero(neuropil_mask)
        raw_neuropil[start:stop] = np.mean(imgstack[:,nz[0], nz[1]], axis=1)


# Because of immutability of NumPy arrays, have to perform a deep copy and temporarily duplicate in memory.
if not is_deep_interp:
    raw_cell_traces_ = raw_cell_traces[30:, :]
    raw_neuropil_ = raw_neuropil[30:]
    raw_cell_traces = raw_cell_traces_
    raw_neuropil = raw_neuropil_
    del raw_neuropil_
    del raw_cell_traces_
    num_frames = num_frames - 30


## Getting dF/F from raw traces
dff = np.zeros((num_frames, num_cells))
for cc in tqdm(range(num_cells), desc="Getting dF/F per cell...", ncols=75):
    dff[:,cc] = filter_baseline_dF_comp(raw_cell_traces[:,cc], 99*4+1)

if do_neuropil:
    dff_neuropil = filter_baseline_dF_comp(raw_neuropil, 99*4+1)

# Save results so far
outfile.create_dataset('raw_cell_traces', data=raw_cell_traces)
outfile.create_dataset('dff', data=dff)
if do_neuropil:
    outfile.create_dataset('raw_neuropil', data=raw_neuropil)
    outfile.create_dataset('dff_neuropil', data=dff_neuropil)

# Stimulus information, frame triggers, stimulus triggers, etc.
if stim_file > -1:
    print('Grabbing two-photon frametimes')
    if 'BRUKER' in data_type:
        xml_dest = os.path.join(target_folder, os.path.basename(target_folder))+'.xml'
        frame_triggers = read_xml_file(xml_dest)
        frame_triggers = frame_triggers * 1e4 # Convert seconds to 10kHz sampling
        if frame_triggers[0] > 340:
            print('First 2p frame was dropped!')
        
        frame_triggers = replace_missing_frame_triggers(frame_triggers)

        psychopy_loc = data_type + '_PSYCHOPY'
        voltage_files = glob('*VoltageRecording*.csv') # already in datadir
        assert len(voltage_files) == 1, f'Unique, singular voltage file not found. Check {target_folder}'
        vrec = genfromtxt_with_progress(voltage_files[0], delimiter=',', skip_header=1) # skip one row. uses np.genfromtxt

        outfile.create_dataset('frame_triggers', data=frame_triggers)
        outfile.create_dataset('vrec', data=vrec)

    elif 'SCANIMAGE' in data_type:
        print('scanimage two-photon timeframes must be implemented')
        sys.exit(1)

    #print(f'Num frametriggers detected: {frame_triggers.shape[1]}')
    print('Getting stimulus times and syncing with two-photon frame times...')
    stim_triggers = medfilt(vrec[:,1],101)
    stim_triggers[stim_triggers < 0] = 0
    stim_triggers = np.diff(stim_triggers)

    stim_on, _  = find_peaks(stim_triggers, distance=1e3, height=(max(stim_triggers) - max(stim_triggers)*0.9))
    stim_off, _ = find_peaks(stim_triggers, distance=1e3, height=(max(stim_triggers) - max(stim_triggers)*0.9))

    outfile.create_dataset('stim_on', data=stim_on)
    outfile.create_dataset('stim_off', data=stim_off)

    if is_2p_opto:
        print('Processing 2pOpto triggers...')
        stim_triggers = medfilt(vrec[:,1],51)
        stim_triggers[stim_triggers < 0] = 0
        photostim_triggers = find_peaks(stim_triggers, distance=1e4, height=(max(stim_triggers) - max(stim_triggers)*0.9))
        photostim_triggers = photostim_triggers[0]
        outfile.create_dataset('photostim_triggers', photostim_triggers)

    psychopy_file_str = os.path.join(psychopy_loc,'-'.join([date[4:], date[0:2], date[2:4]]))
    os.chdir(save_location+psychopy_file_str)

    psychopy_file = np.genfromtxt('T'+'{:03d}'.format(stim_file)+'.txt')


    if not is_2p_opto:
        stim_id = psychopy_file[:,0]
        unique_stims = np.unique(stim_id)
        stim_properties = psychopy_file[:,1:]
    else:
        if psychopy_file.shape[1] == 2:
            stim_id = psychopy_file[:,0]
            #stim_id[0] = None # First target is often lost because of PrairieView.
        else:
            stim_id = psychopy_file[:,2]
        unique_stims = np.unique(stim_id)
        if psychopy_file.shape[1] > 3:
            stim_properties = psychopy_file[:,3:]
            outfile.create_dataset('stim_properties', data=stim_properties)
        target_number = psychopy_file[:,0]
        target_trial  = psychopy_file[:,1]

        outfile.create_dataset('target_number',data=target_number)
        outfile.create_dataset('target_trial',data=target_trial)


    outfile.create_dataset('stim_id',data=stim_id)
    outfile.create_dataset('unique_stims', data=unique_stims)
    # If this assertion fails, may have to see how your voltage
    #   recording information is being handled. The correct
    #   channel changes depending on the age of the acquisition.
    
    assert len(stim_on) == len(stim_id),f'Mismatch: {len(stim_on)} stimlus onsets detected, but {len(stim_id)} stim IDs. \n Check channels of voltage recording file to be sure the right one is selected.'


    stim_on_2p_frame = np.zeros(len(stim_id))

    for s in range(len(stim_id)):
        stim_on_2p_frame[s] = np.argmin(abs(stim_on[s] - frame_triggers))
    outfile.create_dataset('stim_on_2p_frame', data=stim_on_2p_frame)

    ## Target Stim 2p Frame synchronization
    if is_2p_opto:
        target_stim_2p_frame = np.zeros((len(photostim_triggers)))
        for ss in range(len(photostim_triggers)):
            target_stim_2p_frame[ss] = np.argmin(abs(photostim_triggers[ss] - frame_triggers))
        outfile.create_dataset('target_stim_2p_frame', data=target_stim_2p_frame)

else:
    print('No stimulus triggers recorded for this dataset.')

if do_neuropil:
    neuropil_subtraction(outfile=outfile, roi_list=roi_list)

if do_cascade:
#    cascade_results = cascade_predict('Universal_30Hz_smoothing100ms', dff.T, model_folder=cascade_model_dir)
    spike_inference = cascade_predict('Global_EXC_30Hz_smoothing50ms_causalkernel', dff.T, model_folder=cascade_model_dir).T
    outfile.create_dataset('spike_inference', data=spike_inference)

gen_stim_cyc(outfile=outfile, pre=0, slag=0, dur_resp=dur_resp)

dendrite_subtraction(outfile=outfile, dend_sub_flag=1)


## Extract basic responses from cyc or cyc_res
# Need to get expected size of resp, resps, resperr
if stim_file > -1 and np.floor(len(stim_id) / len(unique_stims)) > 2:
    n_trials = outfile['cyc'].shape[2]
    resp_all     = np.zeros(num_cells, len(unique_stims))
    resps_all    = np.zeros(num_cells, len(unique_stims),n_trials)
    resp_err_all = np.zeros(num_cells, len(unique_stims))
    outfile.create_dataset()
    for cc in range(num_cells):
       # This is not the same way as saying something's a spine.
        if roi_list[cc].roitype == 7 and dend_count > 0:
           resp, resps, resp_err = compute_peak_resp(outfile['cyc'][cc,:])
        else:
           resp, resps, resp_err = compute_peak_resp(outfile['cyc_res'][cc,:])
        resp_all[cc,:] = resp
        resps_all[cc,:,:] = resps
        resp_err_all[cc,:] = resp_err
    
    outfile.create_dataset('resp', data=resp_all)
    outfile.create_dataset('resps', data=resps_all)
    outfile.create_dataset('resp_err', data=resp_err_all)
#        W/in loop, collect resp, resps, resp_err into .h5
## Saving - 
outfile.close()
code.interact(local=dict(globals(), **locals())) 
sys.exit(0) # Signal correct exit