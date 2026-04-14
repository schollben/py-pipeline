# getDFF only - quick script to extract dF/F traces from registered (and optionally denoised) calcium movies, using ROIs defined in a .zip file. Saves results in an .h5 file.
# processing script for extracting traces from calcium movies
# presupposes registration, denoising optional
import os
import sys
import roifile
import math
import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from tqdm import tqdm
from img_utils import *
import tkinter as tk
from tkinter import filedialog

# Ask user to select a folder within /mnt/md0/
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
save_location = filedialog.askdirectory(
    initialdir='/mnt/md0/',
    title='Select folder for analysis'
)

# Exit if user cancelled the dialog
if not save_location:
    print('No folder selected. Exiting.')
    sys.exit(0)

print(f'Selected folder: {save_location}')

do_neuropil= True
chnk = int(1e3)
os.chdir(save_location)

out_h5_name = f'{save_location}2p_ROI.h5' #'TSeries-'+date+'-'+f'{file_num:03d}.h5'
outfile = h5py.File(out_h5_name, 'w')

## ROI file handling
try:
    roi_list = roifile.roiread('RoiSet.zip')
except FileNotFoundError:
    print(f'Halting. No ROI file ')
    sys.exit(1) # kill
else:
    num_cells = len(roi_list)
    located_dend_roi = any(roi.roitype == 5 for roi in roi_list)


## Grabbing .h5 processed calcium movie handling - 
try:
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

outfile.close()