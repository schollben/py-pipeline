# bruker_pipeline.py — Bruker two-photon calcium imaging pipeline
#
# Main entry point: process_experiment(...)
# Returns a Python dict containing all processed variables.
#
# Usage example:
#   from bruker_pipeline import process_experiment
#   result = process_experiment(
#       data_dir   = '/mnt/bigdata/BRUKER/TSeries-04022026-1315-003',
#       stim_file  = 3,
#       is_2p_opto = True,
#       do_plot    = True,
#   )

import os
import sys
import math
from glob import glob

import numpy as np
import h5py
import roifile
from scipy.signal import medfilt, find_peaks
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from img_utils import (
    gen_polyline_roi,
    in_polygon,
    filter_baseline_dF_comp,
    read_xml_file,
    replace_missing_frame_triggers,
    genfromtxt_with_progress,
    neuropil_subtraction,
    gen_stim_cyc,
    compute_peak_resp,
    get_target_folders_v2,
)
from xml_utils import (
    parse_tseries_xml,
    parse_markpoints_xml,
    parse_vrec_xml,
    find_experiment_files,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRUKER_ROOT    = '/mnt/bigdata/BRUKER'
PSYCHOPY_ROOT  = '/mnt/bigdata/BRUKER_PSYCHOPY'
DEFAULT_OUTPUT = '/mnt/PROCCESSED/'


# ===========================================================================
# Main pipeline function
# ===========================================================================

def process_experiment(
    data_dir,
    stim_file      = -1,
    is_spontaneous = False,
    is_2p_opto     = False,
    use_inference  = False,
    do_neuropil    = False,
    dur_resp       = 2.5,
    opto_post_sec  = 1.0,
    opto_pre_sec   = 0.5,
    opto_blank_sec = 0.4,
    do_plot               = False,
    opto_offset_trigger   = True,
    chunk_size            = 1000,
    output_dir            = DEFAULT_OUTPUT,
):
    """
    Process a single Bruker TSeries experiment and return a dict of all
    processed variables.

    Parameters
    ----------
    data_dir : str
        Full path to the TSeries folder.
    stim_file : int
        PsychoPy T-file number (e.g. 3 → T003.txt). -1 = no stim file.
    is_spontaneous : bool
        True if no visual stimulus triggers (spontaneous activity recording).
    is_2p_opto : bool
        True if dataset contains 2P photostimulation.
    use_inference : bool
        True = use inference_results.h5, False = use registered.h5.
    do_neuropil : bool
        True to extract and subtract neuropil signal.
    dur_resp : float
        Response window duration in seconds.
    opto_post_sec : float
        Seconds after blanking window to average for opto response image.
    opto_pre_sec : float
        Seconds before trigger onset to average for opto baseline image.
    opto_blank_sec : float
        Blanking period after trigger onset (microscope shutter, ~400 ms).
    do_plot : bool
        True to generate and save a diagnostic summary figure.
    opto_offset_trigger : bool
        True (default) to discard the first row of the PsychoPy file for 2P
        opto experiments. A known acquisition bug causes the trigger stream to
        be offset by one: photostim trigger 0 corresponds to psychopy row 1,
        trigger 1 to row 2, etc. Set to False once the bug is fixed upstream.
    chunk_size : int
        Number of frames to process at once during trace extraction.
    output_dir : str
        Directory for output H5 and figure files.

    Returns
    -------
    dict
        All processed variables keyed as documented in PIPELINE_PLAN.md.
    """
    data_dir   = data_dir.rstrip('/')
    experiment_id = os.path.basename(data_dir)
    os.makedirs(output_dir, exist_ok=True)

    result = {
        'experiment_id': experiment_id,
        'data_dir':      data_dir,
        # flags
        'is_spontaneous': is_spontaneous,
        'is_2p_opto':     is_2p_opto,
        'do_neuropil':    do_neuropil,
        'do_plot':        do_plot,
    }

    # -----------------------------------------------------------------------
    # Step 1 — Locate all experiment files
    # -----------------------------------------------------------------------
    files = find_experiment_files(data_dir)
    _require(files['tseries_xml'],
             f'TSeries XML not found in {data_dir}')

    # -----------------------------------------------------------------------
    # Step 2 — Parse XML metadata
    # -----------------------------------------------------------------------
    print('Parsing TSeries XML...')
    xml_params = parse_tseries_xml(files['tseries_xml'])
    result.update({
        'date':             xml_params['acquisition_date'],
        'frame_period':     xml_params['frame_period'],
        'optical_zoom':     xml_params['optical_zoom'],
        'microns_per_pixel':xml_params['microns_per_pixel'],
        'pixels_per_line':  xml_params['pixels_per_line'],
        'lines_per_frame':  xml_params['lines_per_frame'],
        'objective_lens':   xml_params['objective_lens'],
        'objective_mag':    xml_params['objective_mag'],
        'objective_na':     xml_params['objective_na'],
        'laser_power':      xml_params['laser_power'],
        'pmt_gain':         xml_params['pmt_gain'],
        'bit_depth':        xml_params['bit_depth'],
    })

    frame_period = xml_params['frame_period']
    if frame_period is None or frame_period <= 0:
        print('[WARNING] frame_period not found in XML. Defaulting to 0.033 s.')
        frame_period = 0.033
        result['frame_period'] = frame_period

    vrec_meta = None
    if files['vrec_xml']:
        vrec_meta = parse_vrec_xml(files['vrec_xml'])

    # -----------------------------------------------------------------------
    # Step 3 — Parse MarkPoints XML (if 2P opto)
    # -----------------------------------------------------------------------
    mp_data = None
    if is_2p_opto:
        if files['markpoints_xml']:
            print('Parsing MarkPoints XML...')
            mp_data = parse_markpoints_xml(files['markpoints_xml'])
        else:
            print('[WARNING] is_2p_opto=True but no MarkPoints XML found.')

    # -----------------------------------------------------------------------
    # Step 4 — Load ROIs
    # -----------------------------------------------------------------------
    _require(files['roi_zip'], f'RoiSet.zip not found in {data_dir}')
    roi_list  = roifile.roiread(files['roi_zip'])
    num_cells = len(roi_list)
    print(f'{num_cells} ROIs loaded')

    is_dendrite = np.array([r.roitype == 5 for r in roi_list])
    if any(is_dendrite):
        is_spine = np.array([r.roitype == 7 for r in roi_list])
        is_soma  = np.zeros(num_cells, dtype=bool)
    else:
        is_spine = np.zeros(num_cells, dtype=bool)
        is_soma  = np.array([r.roitype == 7 for r in roi_list])

    result.update({
        'n_rois':      num_cells,
        'is_dendrite': is_dendrite,
        'is_spine':    is_spine,
        'is_soma':     is_soma,
    })

    # -----------------------------------------------------------------------
    # Step 5 — Open calcium movie
    # -----------------------------------------------------------------------
    if use_inference:
        _require(files['inference_h5'],
                 f'inference_results.h5 not found in {data_dir}')
        h5_path = files['inference_h5']
    else:
        _require(files['registered_h5'],
                 f'registered.h5 not found in {data_dir}')
        h5_path = files['registered_h5']

    h = h5py.File(h5_path, 'r')
    dat_name = list(h.keys())[0]
    num_frames, size_x, size_y = h[dat_name].shape
    print(f'Movie: {num_frames} frames, {size_x}×{size_y} px')

    n_proj     = min(100, num_frames)
    avg_image  = np.mean(h[dat_name][:n_proj], axis=0)
    result['avg_image'] = avg_image

    # Frame time axis in seconds
    frame_times_sec = np.arange(num_frames) * frame_period
    result['frame_times_sec'] = frame_times_sec

    # -----------------------------------------------------------------------
    # Step 6 — Build cell masks
    # -----------------------------------------------------------------------
    xx = np.linspace(0, size_x - 1, size_x)
    yy = np.linspace(0, size_y - 1, size_y)
    xx, yy = np.meshgrid(xx, yy)

    mask2d       = np.zeros((num_cells, size_x, size_y), dtype=float)
    neuropil_mask = np.zeros((size_x, size_y), dtype=float)

    for cc in tqdm(range(num_cells), desc='Building masks', ncols=75):
        nm_coord = roi_list[cc].coordinates()
        if roi_list[cc].roitype == 5:
            mask2d[cc] = gen_polyline_roi(nm_coord=nm_coord,
                                          d_width=roi_list[cc].stroke_width,
                                          size_x=size_x, size_y=size_y)
        else:
            mask2d[cc] = in_polygon(xx, yy, nm_coord[:, 0], nm_coord[:, 1])
        neuropil_mask += mask2d[cc]

    result['mask2d']        = mask2d
    result['neuropil_mask'] = neuropil_mask

    # -----------------------------------------------------------------------
    # Step 7 — Extract raw traces
    # -----------------------------------------------------------------------
    raw_traces   = np.zeros((num_frames, num_cells))
    raw_neuropil = np.zeros((num_frames,)) if do_neuropil else None

    # Pre-compute nonzero indices for each mask to avoid repeated calls
    nz_per_cell = [np.nonzero(mask2d[cc]) for cc in range(num_cells)]
    nz_neuropil = np.nonzero(neuropil_mask) if do_neuropil else None

    n_chunks = math.ceil(num_frames / chunk_size)
    
    for f_i in tqdm(range(n_chunks), desc='Extracting traces', ncols=75):
        start = f_i * chunk_size
        stop  = min((f_i + 1) * chunk_size, num_frames)
        chunk = h[dat_name][start:stop]
        for cc in range(num_cells):
            raw_traces[start:stop, cc] = np.mean(
                chunk[:, nz_per_cell[cc][0], nz_per_cell[cc][1]], axis=1)
        if do_neuropil:
            raw_neuropil[start:stop] = np.mean(
                chunk[:, nz_neuropil[0], nz_neuropil[1]], axis=1)

    result['raw_traces'] = raw_traces
    if do_neuropil:
        result['raw_neuropil'] = raw_neuropil

    # -----------------------------------------------------------------------
    # Step 8 — Compute dF/F
    # -----------------------------------------------------------------------
    # Baseline window: ~13 seconds, scaled to actual frame rate.
    # Must be odd for the percentile filter.
    dff_baseline_sec = 13.2
    dff_window = int(round(dff_baseline_sec / frame_period))
    if dff_window % 2 == 0:
        dff_window += 1
    print(f'dF/F baseline window: {dff_window} frames '
          f'({dff_window * frame_period:.1f} s at {1/frame_period:.1f} Hz)')
    dff = np.zeros((num_frames, num_cells))
    for cc in tqdm(range(num_cells), desc='Computing dF/F', ncols=75):
        dff[:, cc] = filter_baseline_dF_comp(raw_traces[:, cc], dff_window)

    dff_neuropil = None
    if do_neuropil:
        dff_neuropil = filter_baseline_dF_comp(raw_neuropil, dff_window)

    result['dff'] = dff
    result['dff_window_frames'] = dff_window
    result['dff_window_sec']    = dff_window * frame_period
    if do_neuropil:
        result['dff_neuropil'] = dff_neuropil

    # -----------------------------------------------------------------------
    # Step 9-11 — Stimulus processing (skip if is_spontaneous)
    # -----------------------------------------------------------------------
    has_stim = (stim_file > -1) and (not is_spontaneous)

    # Initialise stimulus keys to None so they always exist in the dict
    for key in ('frame_triggers', 'vrec', 'vrec_sample_rate',
                'vrec_channel_layout', 'visual_trigger_ch', 'photostim_ch',
                'stim_on', 'stim_on_sec', 'stim_off', 'stim_off_sec',
                'stim_on_2p_frame', 'stim_id', 'unique_stims',
                'stim_properties', 'target_number', 'target_trial',
                'cyc', 'resp', 'resps', 'resp_err', 'stim_avg_images'):
        result[key] = None

    vrec = None   # needed later for opto

    if not is_spontaneous:
        # --- Frame triggers from XML ---
        print('Reading 2P frame triggers from XML...')
        frame_triggers = read_xml_file(files['tseries_xml'])
        frame_triggers = frame_triggers * 1e4   # → 10 kHz samples
        if frame_triggers[0] > 340:
            print('[WARNING] First 2P frame trigger appears to be missing — interpolating.')
        frame_triggers = replace_missing_frame_triggers(frame_triggers)
        result['frame_triggers'] = frame_triggers

        # --- Voltage recording ---
        _require(files['vrec_csv'],
                 f'VoltageRecording CSV not found in {data_dir}')
        print('Loading voltage recording...')
        vrec = genfromtxt_with_progress(files['vrec_csv'],
                                        delimiter=',', skip_header=1)
        vrec_sample_rate = (vrec_meta['sample_rate']
                            if vrec_meta else 10000)
        result['vrec']             = vrec
        result['vrec_sample_rate'] = vrec_sample_rate

        # --- Auto-detect channel layout ---
        ch_layout = detect_vrec_channel_layout(vrec)
        result['vrec_channel_layout'] = ch_layout['layout']
        result['visual_trigger_ch']   = ch_layout['visual_trigger_ch']
        result['photostim_ch']        = ch_layout['photostim_ch']

        vis_ch    = ch_layout['visual_trigger_ch']
        opto_ch   = ch_layout['photostim_ch']

        if has_stim:
            # --- Detect visual stimulus onsets from vrec ---
            vis_signal = medfilt(vrec[:, vis_ch], 101)
            vis_signal[vis_signal < 0] = 0
            vis_diff = np.diff(vis_signal)

            stim_on, _  = find_peaks(vis_diff, distance=1e3,
                                     height=(max(vis_diff) - max(vis_diff) * 0.9))
            stim_off, _ = find_peaks(-vis_diff, distance=1e3,
                                     height=(max(-vis_diff) - max(-vis_diff) * 0.9))

            stim_on_sec  = stim_on  / vrec_sample_rate
            stim_off_sec = stim_off / vrec_sample_rate

            result['stim_on']      = stim_on
            result['stim_on_sec']  = stim_on_sec
            result['stim_off']     = stim_off
            result['stim_off_sec'] = stim_off_sec

            # --- Read PsychoPy stimulus file ---
            psychopy_date = _bruker_date_to_psychopy(experiment_id)
            psychopy_path = os.path.join(
                PSYCHOPY_ROOT, psychopy_date, f'T{stim_file:03d}.txt')
            if not os.path.isfile(psychopy_path):
                print(f'[WARNING] PsychoPy file not found: {psychopy_path}')
                stim_id = None
            else:
                psychopy_data = np.genfromtxt(psychopy_path)
                if psychopy_data.ndim == 1:
                    psychopy_data = psychopy_data[np.newaxis, :]

                # Known acquisition bug: for 2P opto experiments the trigger
                # stream is offset by one row relative to the psychopy file.
                # Dropping row 0 realigns triggers → psychopy rows.
                if is_2p_opto and opto_offset_trigger:
                    print('[opto_offset_trigger=True] Dropping first PsychoPy '
                          f'row (was: {psychopy_data.shape[0]} rows → '
                          f'{psychopy_data.shape[0] - 1} rows).')
                    psychopy_data = psychopy_data[1:]

                if not is_2p_opto:
                    stim_id         = psychopy_data[:, 0]
                    stim_properties = psychopy_data[:, 1:]
                    result['stim_properties'] = stim_properties
                else:
                    # 2P opto: columns are [target_number, target_trial, stim_id, ...]
                    if psychopy_data.shape[1] == 2:
                        stim_id = psychopy_data[:, 0]
                    else:
                        stim_id = psychopy_data[:, 2]
                    result['target_number'] = psychopy_data[:, 0]
                    result['target_trial']  = psychopy_data[:, 1]
                    if psychopy_data.shape[1] > 3:
                        result['stim_properties'] = psychopy_data[:, 3:]

                unique_stims = np.unique(stim_id)
                result['stim_id']      = stim_id
                result['unique_stims'] = unique_stims

                # Validate trigger count
                if len(stim_on) != len(stim_id):
                    print(f'[WARNING] Stim trigger mismatch: '
                          f'{len(stim_on)} detected vs '
                          f'{len(stim_id)} in stim file!')
                else:
                    print(f'Stim triggers: {len(stim_on)} ✓')

                # Map stim onsets to 2P frame indices
                stim_on_2p_frame = np.array([
                    np.argmin(np.abs(s - frame_triggers))
                    for s in stim_on
                ], dtype=float)
                result['stim_on_2p_frame'] = stim_on_2p_frame

    # -----------------------------------------------------------------------
    # Step 11 — Response analysis (stimulus present, not spontaneous)
    # -----------------------------------------------------------------------
    if has_stim and result['stim_id'] is not None:
        # Write a minimal H5 in memory for gen_stim_cyc / neuropil_subtraction
        # (those functions expect an open h5py.File handle called `outfile`)
        out_h5_path = os.path.join(
            output_dir, f'{experiment_id}.h5')
        outfile = h5py.File(out_h5_path, 'w')
        _populate_outfile(outfile, result, num_frames)

        if do_neuropil:
            neuropil_subtraction(outfile=outfile, roi_list=roi_list)

        gen_stim_cyc(outfile=outfile, pre=0, slag=0, dur_resp=dur_resp)

        unique_stims = result['unique_stims']
        n_stims      = len(unique_stims)
        if np.floor(len(result['stim_id']) / n_stims) > 2:
            cyc      = outfile['cyc'][:]
            n_trials = cyc.shape[2]

            resp_all     = np.zeros((num_cells, n_stims))
            resps_all    = np.zeros((num_cells, n_stims, n_trials))
            resp_err_all = np.zeros((num_cells, n_stims))
            for cc in range(num_cells):
                r, rs, re = compute_peak_resp(cyc[cc])
                resp_all[cc]     = r
                resps_all[cc]    = rs
                resp_err_all[cc] = re

            result['cyc']      = cyc
            result['resp']     = resp_all
            result['resps']    = resps_all
            result['resp_err'] = resp_err_all
        else:
            print('Too few trials per stimulus — skipping peak response computation.')

        # Per-stimulus average images (~30 frames post onset)
        n_avg_frames  = int(round(1.0 / frame_period))   # ~1 second
        stim_on_2p    = result['stim_on_2p_frame'].astype(int)
        stim_id_arr   = result['stim_id']
        stim_avg_imgs = np.zeros((n_stims, size_x, size_y))

        for si, sv in enumerate(tqdm(unique_stims,
                                      desc='Stim avg images', ncols=75)):
            trials = np.where(stim_id_arr == sv)[0]
            acc, cnt = np.zeros((size_x, size_y)), 0
            for ti in trials:
                onset = stim_on_2p[ti]
                if onset + n_avg_frames <= num_frames:
                    acc += np.mean(h[dat_name][onset:onset + n_avg_frames],
                                   axis=0)
                    cnt += 1
            if cnt:
                stim_avg_imgs[si] = acc / cnt

        result['stim_avg_images'] = stim_avg_imgs
        outfile.close()

    # -----------------------------------------------------------------------
    # Step 12 — 2P opto processing
    # -----------------------------------------------------------------------
    # Initialise opto keys to None
    for key in ('photostim_triggers', 'photostim_triggers_sec',
                'photostim_2p_frame',
                'markpoints_xy_norm', 'markpoints_xy_pix',
                'markpoints_condition_idx', 'markpoints_laser_power',
                'opto_stim_ids', 'opto_avg_images', 'opto_baseline_images',
                'opto_delta_images', 'opto_n_trials',
                'opto_grand_avg_image', 'opto_grand_baseline_image',
                'opto_grand_delta_image',
                'opto_blank_sec', 'opto_blank_frames',
                'opto_post_sec', 'opto_post_frames',
                'opto_pre_sec', 'opto_pre_frames'):
        result[key] = None

    if is_2p_opto and vrec is not None:
        vrec_sample_rate = result['vrec_sample_rate'] or 10000
        opto_ch = result['photostim_ch']

        # Detect photostim triggers
        print('Detecting photostimulation triggers...')
        ps_signal = medfilt(vrec[:, opto_ch], 51)
        ps_signal[ps_signal < 0] = 0
        photostim_triggers, _ = find_peaks(
            ps_signal, distance=1e4,
            height=(max(ps_signal) - max(ps_signal) * 0.9))

        photostim_triggers_sec = photostim_triggers / vrec_sample_rate
        result['photostim_triggers']     = photostim_triggers
        result['photostim_triggers_sec'] = photostim_triggers_sec

        # Map photostim triggers to 2P frames using frame_triggers
        ft = result.get('frame_triggers')
        if ft is None:
            # If spontaneous (no frame triggers from stim branch), read them now
            print('Reading frame triggers for 2P opto sync...')
            ft = read_xml_file(files['tseries_xml']) * 1e4
            ft = replace_missing_frame_triggers(ft)
            result['frame_triggers'] = ft

        photostim_2p_frame = np.array([
            np.argmin(np.abs(ps - ft))
            for ps in photostim_triggers
        ], dtype=int)
        result['photostim_2p_frame'] = photostim_2p_frame

        # MarkPoints coordinates
        if mp_data is not None:
            px_line  = xml_params['pixels_per_line'] or size_x
            px_frame = xml_params['lines_per_frame']  or size_y

            xy_norm_list = []
            cond_idx_list = []
            laser_powers  = []

            for ci, cond in enumerate(mp_data['conditions']):
                laser_powers.append(cond['uncaging_laser_power'])
                for pt in cond['points']:
                    xy_norm_list.append([pt['x_norm'], pt['y_norm']])
                    cond_idx_list.append(ci)

            xy_norm = np.array(xy_norm_list)
            xy_pix  = np.column_stack([
                xy_norm[:, 0] * px_line,
                xy_norm[:, 1] * px_frame,
            ])

            result['markpoints_xy_norm']        = xy_norm
            result['markpoints_xy_pix']         = xy_pix
            result['markpoints_condition_idx']   = np.array(cond_idx_list)
            result['markpoints_laser_power']     = np.array(laser_powers)

        # Per-stim-ID opto average images
        print('Computing per-stim-ID opto % change images...')
        opto_blank_frames = int(round(opto_blank_sec / frame_period))
        opto_pre_frames   = int(round(opto_pre_sec   / frame_period))
        opto_post_frames  = int(round(opto_post_sec  / frame_period))

        result.update({
            'opto_blank_sec':   opto_blank_sec,
            'opto_blank_frames': opto_blank_frames,
            'opto_post_sec':    opto_post_sec,
            'opto_post_frames': opto_post_frames,
            'opto_pre_sec':     opto_pre_sec,
            'opto_pre_frames':  opto_pre_frames,
        })

        # Determine stim_id per photostim event
        opto_stim_id = result.get('stim_id')
        if opto_stim_id is None or len(opto_stim_id) != len(photostim_triggers):
            # Fall back: assign each trigger an ID of 0 (single condition)
            print('[WARNING] Cannot match photostim triggers to stim IDs — '
                  'treating all as stim_id=0.')
            opto_stim_id = np.zeros(len(photostim_triggers))

        opto_unique_ids = np.unique(opto_stim_id)
        n_opto_ids      = len(opto_unique_ids)

        opto_avg_imgs      = np.zeros((n_opto_ids, size_x, size_y))
        opto_baseline_imgs = np.zeros((n_opto_ids, size_x, size_y))
        opto_n_trials_arr  = np.zeros(n_opto_ids, dtype=int)
        n_skipped          = 0

        for oi, sid in enumerate(tqdm(opto_unique_ids,
                                       desc='Opto avg images', ncols=75)):
            events = np.where(opto_stim_id == sid)[0]
            post_acc, pre_acc, cnt = (np.zeros((size_x, size_y)),
                                       np.zeros((size_x, size_y)), 0)
            for ev in events:
                f0 = photostim_2p_frame[ev]
                pre_start  = f0 - opto_pre_frames
                pre_stop   = f0
                post_start = f0 + opto_blank_frames
                post_stop  = f0 + opto_blank_frames + opto_post_frames

                if (pre_start < 0 or post_stop > num_frames):
                    n_skipped += 1
                    continue

                pre_window  = h[dat_name][pre_start:pre_stop]
                post_window = h[dat_name][post_start:post_stop]
                pre_acc  += np.mean(pre_window,  axis=0)
                post_acc += np.mean(post_window, axis=0)
                cnt += 1

            if cnt > 0:
                post_mean = post_acc / cnt
                pre_mean  = pre_acc  / cnt
                opto_avg_imgs[oi]      = post_mean
                opto_baseline_imgs[oi] = pre_mean
            opto_n_trials_arr[oi] = cnt

        if n_skipped:
            print(f'[WARNING] Skipped {n_skipped} edge photostim events '
                  f'(too close to recording boundaries).')

        # % change: (post / pre - 1) * 100
        with np.errstate(invalid='ignore', divide='ignore'):
            opto_delta_imgs = np.where(
                opto_baseline_imgs != 0,
                (opto_avg_imgs / opto_baseline_imgs - 1.0) * 100.0,
                0.0,
            )

        # Grand-average across all stim IDs (weighted by trial count)
        weights = opto_n_trials_arr.astype(float)
        if weights.sum() > 0:
            w = weights / weights.sum()
            opto_grand_avg      = np.tensordot(w, opto_avg_imgs,      axes=([0], [0]))
            opto_grand_baseline = np.tensordot(w, opto_baseline_imgs, axes=([0], [0]))
        else:
            opto_grand_avg      = np.zeros((size_x, size_y))
            opto_grand_baseline = np.zeros((size_x, size_y))

        with np.errstate(invalid='ignore', divide='ignore'):
            opto_grand_delta = np.where(
                opto_grand_baseline != 0,
                (opto_grand_avg / opto_grand_baseline - 1.0) * 100.0,
                0.0,
            )

        result['opto_stim_ids']            = opto_unique_ids
        result['opto_avg_images']          = opto_avg_imgs
        result['opto_baseline_images']     = opto_baseline_imgs
        result['opto_delta_images']        = opto_delta_imgs
        result['opto_n_trials']            = opto_n_trials_arr
        result['opto_grand_avg_image']     = opto_grand_avg       # (size_x, size_y)
        result['opto_grand_baseline_image']= opto_grand_baseline  # (size_x, size_y)
        result['opto_grand_delta_image']   = opto_grand_delta     # (size_x, size_y) % change

    # -----------------------------------------------------------------------
    # Close H5 movie
    # -----------------------------------------------------------------------
    h.close()

    # -----------------------------------------------------------------------
    # Step 13 — Visualization
    # -----------------------------------------------------------------------
    if do_plot:
        fig_path = os.path.join(output_dir, f'{experiment_id}_summary.png')
        plot_experiment_summary(result, save_path=fig_path)

    # -----------------------------------------------------------------------
    # Step 14 — Print summary
    # -----------------------------------------------------------------------
    print_experiment_summary(result)

    return result


# ===========================================================================
# Voltage channel auto-detection
# ===========================================================================

def detect_vrec_channel_layout(vrec, threshold=1.0):
    """
    Determine whether the voltage recording uses the 'old' layout
    (ch0=visual, ch1=photostim) or 'new' layout (ch1=visual, ch2=photostim).

    Strategy: the visual-trigger channel has evenly-spaced pulses spread
    across the whole experiment; the photostim channel has clustered bursts.
    We use the coefficient of variation (CV) of inter-event intervals on
    each candidate channel — lower CV → more regular → visual trigger.

    Parameters
    ----------
    vrec : np.ndarray
        Voltage recording array, shape (n_samples, n_channels).
    threshold : float
        Voltage threshold for detecting events (default 1.0 V).

    Returns
    -------
    dict with keys: visual_trigger_ch (int), photostim_ch (int), layout (str)
    """
    n_ch = vrec.shape[1]

    def _cv_of_iei(ch_idx):
        """Coefficient of variation of inter-event intervals on channel ch_idx."""
        sig = medfilt(vrec[:, ch_idx], 51)
        sig[sig < 0] = 0
        events, _ = find_peaks(np.diff(sig), distance=500, height=threshold * 0.5)
        if len(events) < 3:
            return np.inf   # can't compute CV — treat as photostim
        iei = np.diff(events).astype(float)
        return iei.std() / iei.mean() if iei.mean() > 0 else np.inf

    if n_ch < 2:
        # Only one channel: can't distinguish
        return {'visual_trigger_ch': 0, 'photostim_ch': None, 'layout': 'unknown'}

    cv0 = _cv_of_iei(0)
    cv1 = _cv_of_iei(1)

    if n_ch >= 3:
        # Decide between old (0/1) and new (1/2) layouts
        cv2 = _cv_of_iei(2)
        # Visual = channel with lowest CV among candidates {0, 1}
        # In "new" layout ch0 is unused/noise, so cv0 will be high
        if cv0 <= cv1:
            # ch0 is more regular → old layout
            layout = 'old'
            vis_ch  = 0
            opto_ch = 1
        else:
            # ch1 is more regular → new layout
            layout = 'new'
            vis_ch  = 1
            opto_ch = 2
    else:
        # Only 2 channels: assume old layout
        layout  = 'old'
        vis_ch  = 0
        opto_ch = 1 if n_ch > 1 else None

    print(f'Vrec channel layout detected: {layout} '
          f'(visual=ch{vis_ch}, photostim=ch{opto_ch})')

    return {
        'visual_trigger_ch': vis_ch,
        'photostim_ch':      opto_ch,
        'layout':            layout,
    }


# ===========================================================================
# Experiment summary printer
# ===========================================================================

def print_experiment_summary(result):
    """Print a human-readable summary of the processed experiment."""
    fp     = result.get('frame_period') or 0
    nf     = (result.get('dff').shape[0]
              if result.get('dff') is not None else '?')
    dur    = (nf * fp) if isinstance(nf, int) else '?'
    px_x   = result.get('pixels_per_line') or '?'
    px_y   = result.get('lines_per_frame') or '?'
    um_pp  = result.get('microns_per_pixel')
    obj    = result.get('objective_lens') or '?'
    na     = result.get('objective_na') or '?'
    zoom   = result.get('optical_zoom') or '?'

    n_rois = result.get('n_rois', 0)
    n_soma = int(np.sum(result['is_soma']))    if result.get('is_soma')    is not None else 0
    n_dend = int(np.sum(result['is_dendrite']))if result.get('is_dendrite')is not None else 0
    n_spin = int(np.sum(result['is_spine']))   if result.get('is_spine')   is not None else 0

    lines = [
        '=' * 44,
        ' EXPERIMENT SUMMARY',
        '=' * 44,
        f'Experiment : {result.get("experiment_id", "?")}',
        f'Date       : {result.get("date", "?")}',
        f'Data dir   : {result.get("data_dir", "?")}',
        '',
        '── Acquisition ──────────────────────────',
        f'  Frame period  : {fp:.5f} s  ({1/fp:.1f} Hz)' if fp else '  Frame period  : unknown',
        f'  Total frames  : {nf}',
        f'  Duration      : {dur:.1f} s' if isinstance(dur, float) else f'  Duration      : {dur}',
        f'  Frame size    : {px_x} × {px_y} px'
        + (f'  ({um_pp:.3f} µm/px)' if um_pp else ''),
        f'  Optical zoom  : {zoom}x',
        f'  Objective     : {obj}  NA={na}',
        '',
        '── ROIs ─────────────────────────────────',
        f'  Total ROIs    : {n_rois}',
        f'    Somas       : {n_soma}',
        f'    Dendrites   : {n_dend}',
        f'    Spines      : {n_spin}',
        '',
        '── Stimulus ─────────────────────────────',
        f'  is_spontaneous: {result.get("is_spontaneous")}',
    ]

    stim_id  = result.get('stim_id')
    stim_on  = result.get('stim_on')
    layout   = result.get('vrec_channel_layout', '?')
    vis_ch   = result.get('visual_trigger_ch', '?')
    opto_ch  = result.get('photostim_ch', '?')

    if stim_on is not None:
        lines.append(f'  Vrec layout   : {layout}  (vis=ch{vis_ch}, opto=ch{opto_ch})')
        lines.append(f'  Stim triggers : {len(stim_on)}')
        if stim_id is not None:
            match = '✓' if len(stim_on) == len(stim_id) else '✗ MISMATCH'
            lines.append(f'  Stim file rows: {len(stim_id)}  {match}')
            uniq = result.get('unique_stims')
            if uniq is not None:
                lines.append(f'  Unique stim IDs: {len(uniq)}')
    else:
        lines.append('  (no stimulus processing)')

    if result.get('is_2p_opto'):
        ps_trig = result.get('photostim_triggers')
        opto_ids = result.get('opto_stim_ids')
        opto_n  = result.get('opto_n_trials')
        mp_pwr  = result.get('markpoints_laser_power')
        mp_idx  = result.get('markpoints_condition_idx')
        lines += [
            '',
            '── 2P Optogenetics ───────────────────────',
            f'  is_2p_opto         : True',
        ]
        if ps_trig is not None:
            lines.append(f'  Photostim triggers : {len(ps_trig)}')
        if opto_ids is not None:
            lines.append(f'  Unique stim IDs    : {len(opto_ids)}')
        if opto_n is not None:
            lines.append(f'  Trials per ID      : {opto_n.tolist()}')
        if mp_pwr is not None and mp_idx is not None:
            n_groups = len(mp_pwr)
            for gi in range(n_groups):
                n_pts = int(np.sum(mp_idx == gi))
                lines.append(f'  Markpoints (group {gi+1}): '
                              f'{n_pts} targets  (power={mp_pwr[gi]:.0f} mW)')

    lines.append('=' * 44)
    print('\n'.join(lines))


# ===========================================================================
# Visualization
# ===========================================================================

def plot_experiment_summary(result, save_path=None):
    """
    Generate a multi-panel diagnostic figure and optionally save it.

    Panel 1 — ROI map (always)
    Panel 2 — ROI map + MarkPoints (if is_2p_opto)
    Panel 3 — Opto % change images per stim_id (if is_2p_opto)
    """
    is_2p_opto   = result.get('is_2p_opto', False)
    avg_image    = result.get('avg_image')
    mask2d       = result.get('mask2d')
    is_soma      = result.get('is_soma')
    is_dendrite  = result.get('is_dendrite')
    is_spine     = result.get('is_spine')
    opto_delta   = result.get('opto_delta_images')
    opto_ids     = result.get('opto_stim_ids')
    opto_n       = result.get('opto_n_trials')
    xy_pix       = result.get('markpoints_xy_pix')
    cond_idx     = result.get('markpoints_condition_idx')
    laser_pwr    = result.get('markpoints_laser_power')
    um_pp        = result.get('microns_per_pixel')
    size_x       = avg_image.shape[0] if avg_image is not None else 512
    size_y       = avg_image.shape[1] if avg_image is not None else 512

    n_opto_ids = len(opto_ids) if opto_ids is not None else 0
    n_cols_p3  = min(n_opto_ids, 5) if n_opto_ids > 0 else 0
    n_rows_p3  = math.ceil(n_opto_ids / n_cols_p3) if n_cols_p3 > 0 else 0

    # Decide figure layout
    n_top_panels = 2 if is_2p_opto else 1
    has_bottom   = is_2p_opto and n_opto_ids > 0

    fig_h = 6 * (1 + (n_rows_p3 if has_bottom else 0))
    fig   = plt.figure(figsize=(6 * n_top_panels, fig_h), constrained_layout=True)

    if has_bottom:
        gs_top = fig.add_gridspec(2, n_top_panels,
                                   height_ratios=[1, n_rows_p3 * 0.8])
        gs_bot = gs_top[1, :].subgridspec(n_rows_p3, n_cols_p3)
    else:
        gs_top = fig.add_gridspec(1, n_top_panels)

    # ------------------------------------------------------------------
    # Panel 1 — ROI map
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs_top[0, 0])
    if avg_image is not None:
        ax1.imshow(avg_image, cmap='gray', interpolation='nearest')

    roi_colors = {'soma': 'cyan', 'dendrite': 'yellow', 'spine': 'magenta'}
    legend_patches = []

    if mask2d is not None:
        for cc in range(mask2d.shape[0]):
            if is_soma is not None and is_soma[cc]:
                color = roi_colors['soma']
            elif is_dendrite is not None and is_dendrite[cc]:
                color = roi_colors['dendrite']
            elif is_spine is not None and is_spine[cc]:
                color = roi_colors['spine']
            else:
                color = 'lime'
            # Draw ROI outline as contour
            contour_overlay = mask2d[cc].astype(float)
            ax1.contour(contour_overlay, levels=[0.5], colors=[color],
                        linewidths=0.6, alpha=0.8)

        for label, col in roi_colors.items():
            legend_patches.append(mpatches.Patch(color=col, label=label.capitalize()))

    if um_pp and size_x:
        _draw_scale_bar(ax1, size_x, um_pp, bar_um=50)

    ax1.set_title('ROI Map')
    ax1.axis('off')
    if legend_patches:
        ax1.legend(handles=legend_patches, loc='lower right',
                   fontsize=6, framealpha=0.5)

    # ------------------------------------------------------------------
    # Panel 2 — ROI map + MarkPoints (2P opto only)
    # ------------------------------------------------------------------
    if is_2p_opto:
        ax2 = fig.add_subplot(gs_top[0, 1])
        if avg_image is not None:
            ax2.imshow(avg_image, cmap='gray', interpolation='nearest')
        # ROI outlines (same as panel 1)
        if mask2d is not None:
            for cc in range(mask2d.shape[0]):
                ax2.contour(mask2d[cc].astype(float), levels=[0.5],
                            colors=['gray'], linewidths=0.5, alpha=0.5)

        # Scatter MarkPoints
        if xy_pix is not None and cond_idx is not None:
            n_groups = (len(laser_pwr) if laser_pwr is not None
                        else int(cond_idx.max()) + 1)
            cmap_g   = plt.cm.get_cmap('tab10', n_groups)
            mp_patches = []
            for gi in range(n_groups):
                pts = xy_pix[cond_idx == gi]
                label = (f'Group {gi+1} ({laser_pwr[gi]:.0f} mW)'
                         if laser_pwr is not None else f'Group {gi+1}')
                ax2.scatter(pts[:, 0], pts[:, 1],
                            s=80, marker='x', linewidths=2,
                            color=cmap_g(gi), label=label, zorder=5)
                mp_patches.append(
                    mpatches.Patch(color=cmap_g(gi), label=label))
            ax2.legend(handles=mp_patches, loc='lower right',
                       fontsize=6, framealpha=0.6)

        ax2.set_title('ROI Map + MarkPoints')
        ax2.axis('off')

    # ------------------------------------------------------------------
    # Panel 3 — Opto % change images (2P opto only)
    # ------------------------------------------------------------------
    if has_bottom and opto_delta is not None:
        vlim = np.nanpercentile(np.abs(opto_delta), 99)
        for oi in range(n_opto_ids):
            row = oi // n_cols_p3
            col = oi %  n_cols_p3
            ax  = fig.add_subplot(gs_bot[row, col])
            im  = ax.imshow(opto_delta[oi], cmap='RdBu_r',
                            vmin=-vlim, vmax=vlim, interpolation='nearest')
            n_trials_str = (f', n={opto_n[oi]}'
                            if opto_n is not None else '')
            ax.set_title(f'stim_id={opto_ids[oi]:.0f}{n_trials_str}',
                         fontsize=8)
            ax.axis('off')
        # Single shared colorbar
        fig.colorbar(im, ax=fig.get_axes()[n_top_panels:],
                     label='% change', fraction=0.02, pad=0.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Summary figure saved: {save_path}')

    plt.show()


# ===========================================================================
# Internal helpers
# ===========================================================================

def _require(path, msg):
    """Assert that a file path is not None and the file exists."""
    if not path or not os.path.isfile(path):
        print(f'[ERROR] {msg}')
        sys.exit(1)


def _bruker_date_to_psychopy(experiment_id):
    """
    Extract the MMDDYYYY date from a TSeries experiment ID and convert it
    to the YYYY-MM-DD format used by the PsychoPy directory structure.

    e.g. 'TSeries-04022026-1315-003' → '2026-04-02'
    """
    # TSeries-MMDDYYYY-HHMM-###
    parts = experiment_id.split('-')
    if len(parts) >= 2 and len(parts[1]) == 8:
        mmddyyyy = parts[1]
        mm = mmddyyyy[0:2]
        dd = mmddyyyy[2:4]
        yyyy = mmddyyyy[4:8]
        return f'{yyyy}-{mm}-{dd}'
    # Fallback: return empty string; caller will warn
    return ''


def _populate_outfile(outfile, result, num_frames):
    """
    Write the minimal datasets that gen_stim_cyc / neuropil_subtraction
    expect to find in an open h5py.File.
    """
    fp = result.get('frame_period', 0.033)
    outfile.create_dataset('frame_period',     data=np.array([fp]))
    outfile.create_dataset('do_cascade',       data=np.array([False]))
    outfile.create_dataset('stim_file',        data=result.get('stim_file_num', -1))
    outfile.create_dataset('dff',              data=result['dff'])
    outfile.create_dataset('is_dendrite',      data=result['is_dendrite'])
    outfile.create_dataset('is_spine',         data=result['is_spine'])
    outfile.create_dataset('is_soma',          data=result['is_soma'])
    outfile.create_dataset('frame_triggers',   data=result['frame_triggers'])
    outfile.create_dataset('stim_on_2p_frame', data=result['stim_on_2p_frame'])
    outfile.create_dataset('stim_id',          data=result['stim_id'])
    outfile.create_dataset('unique_stims',     data=result['unique_stims'])
    outfile.create_dataset('mask2d',           data=result['mask2d'])
    outfile.create_dataset('neuropil_mask',    data=result['neuropil_mask'])
    if result.get('raw_neuropil') is not None:
        outfile.create_dataset('raw_neuropil', data=result['raw_neuropil'])
    if result.get('raw_traces') is not None:
        outfile.create_dataset('raw_cell_traces', data=result['raw_traces'])


def _draw_scale_bar(ax, size_px, um_per_px, bar_um=50):
    """Overlay a scale bar on an image axes."""
    bar_px   = bar_um / um_per_px
    x_start  = size_px * 0.05
    y_pos    = size_px * 0.93
    ax.plot([x_start, x_start + bar_px], [y_pos, y_pos],
            color='white', linewidth=2)
    ax.text(x_start + bar_px / 2, y_pos - size_px * 0.02,
            f'{bar_um} µm', color='white', ha='center', va='bottom',
            fontsize=7)
