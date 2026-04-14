# Checking deepinterpolation signal for whole session, just to verify
import os
import code
from glob import glob
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import h5py
import roifile

from img_utils import sample_stack_from_h5

def compare_reg_inf_ends(dirname, chnk=1000, middle=True):
    '''
    Grab 1000 frame samples from the end.
    Also, grab 1000 frames from the middle if requested
    I was doing this enough times that I wanted it to be quick functionality.
    '''
    reg_name = os.path.join(dirname, 'registered.h5')
    inf_name = os.path.join(dirname, 'inference_results.h5')
    
    reg = h5py.File(reg_name, 'r')
    key = 'data' if 'data' in reg.keys() else 'mov'
    num_frames = int(reg['data'].shape[0])
    reg.close()

    reg_end_savename = os.path.join(dirname, 'samples', f'reg_{num_frames-chnk}_{num_frames}.tif')
    inf_end_savename = os.path.join(dirname, 'samples', f'inf_{num_frames-chnk}_{num_frames}.tif')
        
    if num_frames > chnk:
        sample_stack_from_h5(reg_name, reg_end_savename, num_frames-chnk, num_frames)
        sample_stack_from_h5(inf_name, inf_end_savename, num_frames-chnk, num_frames)
        
        if middle:
            if num_frames < 4000:
                print(f'Only {num_frames} frames, skipping middle sample.')
            else:
                mid_frame = int(num_frames / 2)
                mid_final = min(mid_frame + chnk, num_frames - chnk) # No reason for overlap of data?

                reg_mid_savename = os.path.join(dirname, 'samples', f'reg_{mid_frame}_{mid_final}.tif')
                inf_mid_savename = os.path.join(dirname, 'samples', f'inf_{mid_frame}_{mid_final}.tif')    
                
                sample_stack_from_h5(reg_name, reg_mid_savename, mid_frame, mid_final) 
                sample_stack_from_h5(inf_name, inf_mid_savename, mid_frame, mid_final) 
                
    else:
        print(f'No sample, movie has {num_frames} frames, asking for {chnk}')
        

if __name__ == '__main__':
    just_check_size = False
    workdirs = sorted(glob('/mnt/md0/BRUKER/TSeries-11042024*'))

    for w in workdirs:
        try:
            # Check that files exist for this one
            if not (os.path.isfile(os.path.join(w, 'registered.h5')) and
                     os.path.isfile(os.path.join(w, 'inference_results.h5'))):
                raise FileNotFoundError

        except FileNotFoundError:
            print(f'File not found for {w}')
            continue

        if not just_check_size:
            compare_reg_inf_ends(w)
        else:
            f = h5py.File(os.path.join(w, 'registered.h5'), 'r')
            print(w)
            print(f['data'].shape)
            f.close()

            