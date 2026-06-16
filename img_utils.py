# A module to handle common formats for calcium movies.
import os
import sys
import shutil
import xml.etree.ElementTree as ET
import code
from glob import glob
import h5py
import tifffile
import numpy as np

from matplotlib import path

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from scipy import signal
from statistics import median
from skimage.draw import polygon2mask

from tqdm import tqdm

from sklearn.linear_model import HuberRegressor

def gen_polyline_roi(nm_coord, d_width=10.0, size_x=512, size_y=512):
    '''
    Create a mask for dendrite ROI
    Parameters:
        nm_coord (np.array): 2D NumPy array listing x and y coordinates indicating member pixels.
        d_width (float): Width to draw mask for
        size_x (int): Size of mask to use in first dimension.
        size_y (int): Size of mask to use in second dimension.
    Returns:
        polyline_mask (np.array): 2D NumPy array encoding mask for polyline
    '''
    x = nm_coord[:,1] # These are reversed from MATLAB
    y = nm_coord[:,0]

    r = np.sqrt( (x[1:] - x[0:-1])**2 + (y[1:] - y[0:-1])**2 )

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
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


def prctfilt1(x, n=3, blksz=1000, dim=None):
    """
    One-dimensional percentile filter with support for chunking and multidimensional arrays.
    Equivalent to the MATLAB custom prctfilt1 function.
    
    Parameters:
    x : array_like
        Input signal or array
    n : int, optional
        Window size (default = 3)
    blksz : int, optional
        Block size for chunking. If None, processes all data at once.
    dim : int, optional
        Dimension along which to apply the filter. If None, applies along the first non-singleton dimension.
        
    Returns:
    y : ndarray
        Filtered signal or array
    """
    x = np.asarray(x)
    
    # Handle dimension argument
    if dim is None:
        # Find first non-singleton dimension
        dim = 0
        while dim < x.ndim and x.shape[dim] == 1:
            dim += 1
        if dim == x.ndim:
            return x.copy()  # All dimensions are singleton
            
    # Move the desired dimension to the front
    if dim != 0:
        perm = np.arange(x.ndim)
        perm[0], perm[dim] = dim, 0
        x = np.transpose(x, perm)
        restore_perm = np.argsort(perm)
    
    # Set default block size if not specified
    if blksz is None:
        blksz = x.shape[0]
    
    # Determine padding size
    if n % 2 == 1:  # odd
        m = (n - 1) // 2
    else:  # even
        m = n // 2
    
    # Create output array
    y = np.zeros_like(x)
    
    # Process each "column" (i.e., along all other dimensions)
    for idx in np.ndindex(x.shape[1:]):
        # Extract the vector to filter
        vector = x[(slice(None),) + idx]
        
        # Pad the vector
        padded = np.zeros(len(vector) + 2*m)
        padded[m:m+len(vector)] = vector
        
        # Process in chunks to save memory
        for i in range(0, len(vector), blksz):
            end_idx = min(i + blksz, len(vector))
            chunk_size = end_idx - i
            
            # Create window slices for this chunk
            windows = np.zeros((n, chunk_size))
            for j in range(n):
                windows[j] = padded[i+j:i+j+chunk_size]
            
            # Calculate 20th percentile for each window
            y[(slice(i, end_idx),) + idx] = np.percentile(windows, 20, axis=0)
    
    # Restore original dimensions if needed
    if dim != 0:
        y = np.transpose(y, restore_perm)
        
    return y

def filter_baseline_dF_comp(raw, pts = 99):
    F_temp = raw
    F_temp = np.concatenate((np.repeat(np.mean(F_temp[2:5]),pts),F_temp))
    F_temp = np.concatenate((F_temp, np.repeat(np.mean(F_temp[-5:-2]),pts)))
    # 25th percentile medfilt
    F_temp = signal.medfilt(F_temp, pts)
    # remove padding
    raw_new = F_temp[pts:-pts]
    #code.interact(local=dict(globals(), **locals()))
    raw_new = np.divide((raw - raw_new), raw_new)
    
    #raw_newlpf = signal.medfilt(raw_new, 91)
    raw_newlpf = prctfilt1(raw_new, 91)

    raw_new = raw_new - raw_newlpf
    
    return raw_new

def replace_missing_frame_triggers(frame_triggers):
    '''
    Replace any relative frametriggers that may be missing from a series

    Parameters:
        frame_triggers (np.array): 1D array of relative frame times from
            start of recording. Assumes 10kHz sampling rate.
    Returns:
        frame_triggers_adj (np.array): 1D array of relative frame times from
            start of recording. If any triggers are missing, now included.
    '''
    new_frame_triggers = [frame_triggers[0]]
    med_period = median(np.diff(frame_triggers))
    for k in range(1,frame_triggers.shape[0]):
        if (frame_triggers[k] - frame_triggers[k-1]) > (med_period + med_period/4):
            new_frame_triggers.append(frame_triggers[k-1] + med_period)
        new_frame_triggers.append(frame_triggers[k])
    return np.array(new_frame_triggers)
 
def genfromtxt_with_progress(filename, **kwargs):
    '''
    np.genfromtxt call, but wrapping file with tqdm to add a progress bar
        while iterating over the file lines.
    
    Parameters:
        filename (str): Destination to read from. Local or absolute.
        **kwargs - arguments for np.genfromtxt.
    Returns:
        (np.array): Results of genfromtxt call.
    '''
    with open(filename, 'r') as f:
        total_lines = sum(1 for line in f)
    with open(filename, 'r') as f:
        return np.genfromtxt(tqdm(f, total=total_lines, ncols=75), **kwargs)
        #return np.genfromtxt(tqdm(f, total=total_lines), **kwargs)
 
def neuropil_subtraction(outfile, roi_list):
    '''
    Remove out the neuropil signal from somas.
    Uses statsmodels 'RLM'.

    Parameters:
        outfile (h5py._hl.files.File) File handle for .h5 beign used to store results.
        roi_list (list of ImagejRoi) List of ImageJ ROIs
    Returns:
        Nothing, just writes the neuropil-subtracted traces to the output .h5
    '''
    neuro_sub_slopes = np.zeros((len(roi_list)))
    dff_neuro_subtracted = np.zeros((outfile['dff'].shape))

    robust_reg = HuberRegressor()
    for iter in range(len(roi_list)):
        if roi_list[iter].roitype == 7:
            #robust_reg.fit(den.reshape(-1,1), sp)
            robust_reg.fit(outfile['dff'][:,iter].reshape(-1,1), outfile['dff_neuropil'])
            quick_slope = robust_reg.coef_[0]
            neuro_sub_slopes[iter] = quick_slope
            r = outfile['dff'][:,iter] - (quick_slope * outfile['dff_neuropil'])
            dff_neuro_subtracted[:,iter] = r
        else:
            neuro_sub_slopes[iter] = None
    outfile.create_dataset('neuro_subtraction_slopes', data=neuro_sub_slopes)
    outfile.create_dataset('dff_neuro_subtracted', data=dff_neuro_subtracted)

def gen_stim_cyc(outfile, pre=0, slag=0, dur_resp=2.5):
    '''
    Remove dendrite signal from spine ROIs.
    Navigate it here once you have a way to get it out of main script.
    Parameters:
        outfile (h5py._hl.files.File) File handle for .h5 beign used to store results.
    Returns:
        Nothing, just writes the neuropil-subtracted traces to the output .h5
    '''
    # Extract values out of outfile
    #code.interact(local=dict(globals(), **locals())) 
    frame_period = outfile['frame_period'][:][0] # Strongly dislike this way of passing data.
    num_cells = outfile['dff'].shape[1]
    unique_stims = outfile['unique_stims'][:]
    stim_on_2p_frame = outfile['stim_on_2p_frame'][:]
    stim_id = outfile['stim_id'][:]
    dff = outfile['dff'][:]
    n_trials = int(np.floor(len(outfile['stim_id']) / len(outfile['unique_stims'])))
    stim_dur = int(np.round(dur_resp / frame_period))
    cyc_pre  = int(np.round(pre     / frame_period))
    cyc_slag = int(np.round(slag   / frame_period))
    cyc         = np.zeros((num_cells, len(unique_stims), n_trials, stim_dur + pre))
    cyc_spk_inf = np.zeros((num_cells, len(unique_stims), n_trials, stim_dur + pre))
    for cc in tqdm(range(num_cells), desc='Generating stimulus cycles...'):
        #print(f'cell-iter {cc}')
        trial_list = np.zeros(len(unique_stims))
        for ii in range(len(stim_on_2p_frame)):
            #print(f'stim-iter {ii}')
            # The original MATLAB for this one is confusing to me
            if sum(trial_list==n_trials) != len(unique_stims):
                tt = np.arange(stim_on_2p_frame[ii]- cyc_pre + 1 + cyc_slag, stim_on_2p_frame[ii] + stim_dur + cyc_slag + 1)
                tt = tt.astype('int')
                ind = np.argwhere(unique_stims == stim_id[ii])[0][0] # <- horrendous notation
                if tt[-1] < dff.shape[0]: # Don't want to continue beyond the end of our timeseries
                    f = dff[tt,cc]
                    # Isn't the same check as finding if the ce struct has this field.
                    if outfile['do_cascade'][0]:
                        s = outfile['spike_inference'][tt,cc]
                else:
                    f = np.full(stim_dur + pre, np.nan)
                    s = np.full(stim_dur + pre, np.nan)
                cyc[cc, ind, int(trial_list[ind]), :] = f
                if outfile['do_cascade'][0]:
                    cyc_spk_inf[cc, ind, int(trial_list[ind]), :] = s
                trial_list[ind] += 1
    # Package information for further analysis
    outfile.create_dataset('cyc',data=cyc)
    if 'cyc_spk_inf' in locals():
        outfile.create_dataset('cyc_spk_inf',data=cyc_spk_inf)

def dendrite_subtraction(outfile, dend_sub_flag):
    '''
    For sparse imaging, subtract the dendrite signal out of
    the dF/F for every spine. Gives more spine-specific characterization
    of their responses.

    Parameters:
        outfile (h5py.File) - handle to the outfile currently open.
        dend_sub_flag (int) - 
    '''
    robust_reg = HuberRegressor()
    is_dendrite = outfile['is_dendrite'][:]
    is_spine    = outfile['is_spine'][:]
    if is_dendrite[0] == 1:
        dendcnt = 0
    else:
        dendcnt = 1

    dend_indices = np.where(is_dendrite)[0]

    dend_sub_slopes = np.zeros(is_dendrite.shape[0])
    dff_res = np.zeros(outfile['dff'].shape)

    dend_corrs = np.zeros(is_dendrite.shape[0])

    if 'cyc' in outfile.keys():
        cyc_res = np.zeros((outfile['cyc'].shape))


    for cc in tqdm(range(is_dendrite.shape[0]), desc='Dendrite subtraction', ncols=75):
        if is_dendrite[cc]:
            dendcnt += 1
        elif is_spine[cc]:
            # dff might not be correct field.
            # Check in matlab whether dff can be mutated from raw dff
            sp_dff = outfile['dff'][:]
            sp_dff = sp_dff[0:int(np.round(len(sp_dff)*0.9))] # Kick out the last 10%
            sp_dff[np.isinf(sp_dff)] = 0
            sp_dff_sub = sp_dff[sp_dff < np.nanmedian(sp_dff) + np.abs(np.min(sp_dff))]
            noise_m  = np.nanmedian(sp_dff_sub)
            noise_sd = np.nanstd(sp_dff_sub)

            if dend_sub_flag == 1:
                robust_reg.fit( outfile['dff'][:, is_dendrite[dendcnt]].reshape(-1,1), outfile['dff'][:,cc])
                slope = robust_reg.coef_[0]
            # cyc existence check isn't trivial
            # sp_cyc flattening into a single-dimension needs to be checked.
            elif dend_sub_flag == 2 and 'cyc' in outfile.keys():
                sp_cyc = outfile['cyc'][cc,:,:,:]
                sp_cyc[np.isnan[sp_cyc]] = 0
                robust_reg.fit(outfile['cyc'][is_dendrite[dendcnt],:,:,:], sp_cyc)
                slope = robust_reg.coef_[0]
                
            else:
                robust_reg.fit(outfile['dff'][:,is_dendrite[dendcnt]].reshape(-1,1), outfile['dff'][:,cc])
                slope = robust_reg.coef_[0]
        
        dend_sub_slopes[cc] = slope        
        # Check for this should be at start so we can preallocate before loop.
        if 'cyc' in outfile.keys():
            cyc_res[cc,:,:,:] = outfile['cyc'][cc,:,:,:] - (slope * outfile['cyc'][is_dendrite[dendcnt],:,:,:])
            cyc_res[cc,:,:,:]
        
        r = outfile['dff'][:,cc] - (slope * outfile['dff'][:,is_dendrite[dendcnt]])
        
        dff_res[:,cc] = r
        
        r_sp = outfile['dff'][:,cc]
        r_dn = outfile['dff'][:,is_dendrite[dendcnt]]
        r_sp = r_sp - (slope * r_dn)
        r_sp[r_sp < (-1*noise_sd)] = (-1 * noise_sd)
        r_sp[np.isinf(r_sp)] = 0
        r_dn[np.isinf(r_dn)] = 0
        dff_res[:,cc] = r_sp
        r_sp[r_sp <= 0] = np.nan
        r_dn[r_dn <= 0] = np.nan
        
        # Have to handle missing elements for pairwise deletion same as in MATLAB
        data = np.column_stack((r_sp, r_dn))
        mask = ~np.isnan(data[:,0]) & ~np.isnan(data[:,1])
        #if np.sum(mask) > 1:
        corr = np.corrcoef(data[mask,0], data[mask,1])[0,1]
        dend_corrs[cc] = corr
    else: 
        # is alrady zeros
        #cyc_res[cc,:,:,:] = 0 
        #slope
        dend_corrs[cc] = -1

    outfile.create_dataset('slope', data = dend_sub_slopes)
    outfile.create_dataset('dff_res', data = dff_res)
    outfile.create_dataset('dend_corrs', data = dend_corrs)


def compute_peak_resp(data):
    '''
    data is 3d matrix of ce
    '''
    resp = np.zeros(data.shape[0])
    resps = np.zeros(data.shape[0:2])
    resperr = np.zeros(data.shape[0])

    for ii in range(data.shape[0]):
        # No queeze for python
        data2 = data[ii,:,:]
        temp_f1, temp_f1s, temp_dc, temp_dcs, _ = compf1wdev(data2)
        resp[ii] = temp_f1 + temp_dc
        resps[ii,:] = temp_f1s + temp_dcs
        resperr[ii] = np.nanstd(temp_f1s + temp_dcs) / np.sqrt(data.shape[1])
    return resp, resps, resperr

def compf1wdev(data):
    """
    Python translation of the MATLAB compf1wdev function.
    This function computes FFT components from cycle data.
    
    Parameters:
    data (numpy.ndarray): Array containing rows of single cycle data.
    
    Returns:
    tuple: (f1, f1s, dc, dcs, f2, f2s)
        f1: First harmonic amplitude
        f1s: Individual first harmonic amplitudes
        dc: DC component
        dcs: Individual DC components
        f2: Second harmonic amplitude
        f2s: Individual second harmonic amplitudes
    """
    n, m = data.shape
    
    if n == 1:
        ff = np.fft.fft(data.flatten())
    else:
        # Use nanmean to handle NaN values in the data
        ff = np.fft.fft(np.nanmean(data, axis=0))
    
    dc = ff[0] / m
    f1 = 2 * abs(ff[1] / m)
    f2 = 2 * abs(ff[2] / m)
    angf1 = np.angle(ff[1])
    angf2 = np.angle(ff[2])
    
    if n == 1:
        f1s = 0
        f2s = 0
        dcs = 0
        return f1, f1s, dc, dcs, f2, f2s
    
    # Calculate individual amplitudes
    indf1amp = np.zeros(n)
    inddcamp = np.zeros(n)
    indf2amp = np.zeros(n)
    
    for j in range(n):
        ff = np.fft.fft(data[j, :])
        singleangle = np.angle(ff[1])
        indf1amp[j] = np.cos(singleangle - angf1) * abs(ff[1] * 2 / m)
        inddcamp[j] = ff[0] / m
        
        singleangle = np.angle(ff[2])
        indf2amp[j] = np.cos(singleangle - angf2) * abs(ff[2] * 2 / m)
    
    f1s = indf1amp
    f2s = indf2amp
    dcs = inddcamp
    
    return f1, f1s, dc, dcs, f2, f2s



def read_xml_file(fname):
    '''
    Extract frame timing information from BRUKER output file.
    
    Parameters:
        fname (str): String indicating path to read XML data from.
    Returns:
        rec_info (np.array): Array containing relative frame timing information.
    '''
    try:
        tree = ET.parse(fname)
        root = tree.getroot()
        # Count the number, then get the values
        rel_frametimes = [child.attrib['relativeTime'] for child in root[2] if child.tag == 'Frame']
        return np.array(rel_frametimes).astype('float')
    
    except FileNotFoundError:
        print(f'File {fname} was not found.')
    except ET.ParseError:
        print(f'Error parsing XML file {fname}')
    except Exception as e:
        print('Exception during read_xml_file: ', e)

def get_target_folders_v2(loc, date, fnames, filetype='TSeries'):
    '''
    Grab folder(s) that match given patterns.
    loc (str): Location to parent directory within which to search.
    date (str): MMDDYYYY date string to search for
    fnames (list or int): List of names to gather based on this querey.
    filetype (str): Specify which kind of file from the scanner is required.
    '''
    if 'BRUKER' in loc:
        folder_list = sorted(glob(loc + filetype + '-' + date + '*'))
    elif 'SCANIMAGE' in loc:
        date_str = '-'.join([date[4:], date[:2], date[2:4]])
        folder_list = sorted(glob(loc + filetype + '_' + date_str + '*'))
    
    assert len(folder_list) > 0, 'No target folders matched given specification.'

    if isinstance(fnames, int):
        suffixes = ["{:03d}".format(fnames)]
    else:
        suffixes = ["{:03d}".format(fname) for fname in fnames]

    dir_list = []
    for f in folder_list:
        if os.path.isdir(f) and any(f.endswith(suffix) for suffix in suffixes):
            dir_list.append(f)
    
    return dir_list



def pearson_r_from_movie(h5_path, template_start=0,template_end=1000):
    '''
    Characterizes amount of movement across a calcium movie over time.
    Not recommended anymore. After using with enough movies, it's not
    that informative of a measure compared to viewing sample tif stacks.

    Parameters:
        h5_path (str): String encoding the path to the movie to see.
    Returns:
        corrs (np.array): 1D array for Pearson's r per frame of given movie.
    '''
    with h5py.File(h5_path, 'r') as f:
        key = 'mov' if 'mov' in f.keys() else 'data'
        data = f[key]
        n_frames = data.shape[0]
        
        template = np.mean(data[template_start:min(template_end, n_frames) :, :], axis=0)
        template = template.flatten()

        r_vec = np.empty((n_frames,))
        for i in range(n_frames):
            r_vec[i] = np.corrcoef(template, data[i,:,:].flatten())[0,1]
    return r_vec

def grab_visuals(dir, start_ind=0, end_ind=1000):
    '''
    ImageJ/FIJI refuses to load singular .ome.tif files anymore.
    This makes samples for directories involved so I can do my job.
    
    Parameters:
        dir (str): Directory to take samples from per channel.
        start_ind (int): Time index to start sample from.
        end_ind (int): Time index to end sample from.

    Returns:
        Nothing, just creates files in sample directory of the given dir.

    Ex.
    ~ workdirs = sorted(glob("/mnt/md0/BRUKER/TSeries-11032024"))
    ~ for w in workdirs:
    ~   grab_visuals(w)
    '''
    try:
        print(f'Creating samples for {dir}')
        os.chdir(dir)
        ch1_fnames = sorted(glob("TSeries*Ch1*.ome.tif"))
        ch2_fnames = sorted(glob("TSeries*Ch2*.ome.tif"))
        
        if  not os.path.isdir(os.path.join(dir, 'samples')) and (len(ch1_fnames) > 0 or len(ch2_fnames) > 0):
            os.mkdir(os.path.join(dir, 'samples'))
            print('New samples directory created')

        if len(ch1_fnames) > 0:
            ch1_savename = os.path.join('samples', f'ch1_f{start_ind}-{end_ind}.tif')
            ch1_first = tifffile.imread(ch1_fnames[0], is_ome=False)
            tifffile.imwrite(ch1_savename, ch1_first[start_ind:end_ind, :, :])
            print('Saved Ch1 sample')
        
        if len(ch2_fnames) > 0:
            ch2_savename = os.path.join('samples', f'ch2_f{start_ind}-{end_ind}.tif')
            ch2_first = tifffile.imread(ch2_fnames[0], is_ome=False)
            tifffile.imwrite(ch2_savename, ch2_first[start_ind:end_ind, :, :])
            print('Saved Ch2 sample')

    except IndexError:
        print(f'Indices of {start_ind} and {end_ind} not working. :(')

def sample_stack_from_h5(h5_path, save_path, start_frame, end_frame):
    '''
    Load the main 'data'/'mov' time x width x height data from a given h5 file.
    Slice from start_frame -> end_frame in time.
    Save this data at the save_path as a viewable .tif stack.

    Large .h5 files are not viewable as memory-loadable objects for ImageJ/Fiji,
    so samples are created as drag-and-drop-'able .tifs.
    
        Parameters:
            h5_path (str): Absolute path to the .h5 file to be read.
            save_path (str): Absolute path for the .tif file to be saved.
            start_frame (int): Starting time index for the sample.
            end_frame (int): Ending time index for the sample.
        Returns:
            e (exception): If exception is raised while writing data.
    
    '''
    with h5py.File(h5_path, 'r') as f:
        key = 'mov' if 'mov' in f.keys() else 'data'
        data = f[key]
        data_slice = data[start_frame:end_frame, :, :]
        
        try:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tifffile.imwrite(save_path, data_slice)
        except Exception as e:
            print("Error while saving: ", e)
        print(f"Saved sample at {save_path}")


def std_dev_from_h5(h5_path, start_frame, end_frame, std_width, std_jmps):
    '''
    Load the main 'data'/'mov' [time, width, height] data from a given h5 file's path.
    Slice from start_frame -> end_frame in time.
    Take standard deviations through time with described std-dev widths and jumps per.
    Save frames at the specified save location.

        Parameters:
            h5_path (str): Absolute path to the .h5 file to be read.
            save_path (str): Absolute path to the .tif file to be saved.
            start_frame (int): Starting time index to take samples.
            end_frame (int): Ending time index to take samples.
            std_width (int): Width to take std_deviations from.
            std_jmps (int): What distance forward in time to take between samples.

        Returns:
            e (exception): If exception is raised while writing data.
    '''
    #f = h5py.File(h5_path, "r")
    #data = f['data']
    with h5py.File(h5_path, 'r') as f:
        key = 'mov' if 'mov' in f.keys() else 'data'
        data = f[key]

        curr_strt = start_frame
        curr_end = start_frame + std_width
        termination_val = min(end_frame, data.shape[0]) # Don't go beyond the data

        std_dev_sample_list = []

        while curr_end < termination_val:
            sample = data[curr_strt:curr_end, :, :]
            std_dev_sample_list.append(np.std(sample, axis=0))

        std_dev_sample_array = np.array(std_dev_sample_list)

        try:
            tifffile.imwrite(save_path, std_dev_sample_array)
        except Exception as e:
            print("Error while saving: ", e)


def deinterleave_movies(parent_dir, scope_format):
    """
    Splits tiff stacks where two channels have been interleaved.
    Matches how our SCANIMAGE acquisitions are done for 2ch.

    TODO : 
        - escaped slashes aren't system-agnostic. Change this w/ os.path?
        - Confirm shutil behavior is system-agnostic. Known differences between
            Windows, Ubutu behavior
    """
    is_bruker = 'BRUKER' in parent_dir
    is_scanimage = 'SCANIMAGE' in parent_dir

    fnames = glob(os.path.join(parent_dir, '*.tif'))
    assert len(fnames) > 0, "No valid tif files found in {}".format(parent_dir)
    
    #os.mkdir(parent_dir + '//ch1')
    #os.mkdir(parent_dir + '//ch2')

    os.mkdir(os.path.join(parent_dir, 'ch1'))
    os.mkdir(os.path.join(parent_dir, 'ch2'))

    if scope_format == "BRUKER":
        # Separate files for each
        ch_1_fnames = [f for f in fnames if 'Ch1' in f]
        ch_2_fnames = [f for f in fnames if 'Ch2' in f]
        # Not presuming same number for both, but there usually should be.
        for i in range(len(ch_1_fnames)):
            shutil.move(ch_1_fnames[i], os.path.join(parent_dir, 'ch1', f"{i:05d}.tif"))
        for i in range(len(ch_2_fnames)):
            shutil.move(ch_2_fnames[i], os.path.join(parent_dir, 'ch2', f"{i:05d}.tif"))
            
    elif scope_format == "SCANIMAGE":
        # Interleaved frames in the same file. Read, split, write separate channels.
        for i in range(len(fnames)):
            mov = tifffile.imread(fnames[i])
            ch_1 = mov[::2,:,:] # Even frames, including first
            ch_2 = mov[1::2,:,:]# Odd frames
            tifffile.imwrite(os.path.join(parent_dir, 'ch1', f"{i:05d}.tif"), ch_1)
            tifffile.imwrite(os.path.join(parent_dir, 'ch2', f"{i:05d}.tif"), ch_2)
            
    else:
        assert False, "{} is improper scope format. \
                Give either BRUKER or SCANIMAGE as second argument.".format(scope_format)
        
        s

