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
import tifffile
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
DEFAULT_OUTPUT = '/mnt/bigdata/PROCESSED/'



# ===========================================================================
# Main pipeline function
# ===========================================================================

def process_experiment(
    date,
    file_num,
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
    do_vrec_diagnostic    = False,
    opto_offset_trigger   = True,
    chunk_size            = 1000,
    output_dir            = DEFAULT_OUTPUT,
):
    """
    Process a single Bruker TSeries experiment and return a dict of all
    processed variables.

    Parameters
    ----------
    date : str
        Acquisition date in MMDDYYYY format, e.g. '04022026'.
    file_num : int
        TSeries number, e.g. 3 (zero-padded to 003 internally).
    stim_file : int
        PsychoPy T-file number (e.g. 3 → T003.txt). -1 = no stim file.
    is_spontaneous : bool
        True if no visual stimulus triggers (spontaneous activity recording).
    is_2p_opto : bool
        True if dataset contains 2P photostimulation.
    use_inference : bool
        True = use inference.h5, False = use registered.h5.
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
    # Resolve the TSeries folder from date + file number
    bruker_base = os.path.join(BRUKER_ROOT, '')
    folder_list = get_target_folders_v2(bruker_base, date, file_num, 'TSeries')
    if len(folder_list) == 0:
        print(f'[ERROR] No TSeries folder found for date={date}, '
              f'file_num={file_num:03d} in {bruker_base}')
        sys.exit(1)
    if len(folder_list) > 1:
        print(f'[WARNING] Multiple folders matched — using first:\n'
              + '\n'.join(f'  {f}' for f in folder_list))
    data_dir      = folder_list[0]
    experiment_id = os.path.basename(data_dir)
    print(f'Data folder: {data_dir}')

    os.makedirs(output_dir, exist_ok=True)

    result = {
        'experiment_id': experiment_id,
        'data_dir':      data_dir,
        # ── Processing inputs (all parameters passed by the user) ─────────
        'date':               date,
        'file_num':           file_num,
        'stim_file_num':      stim_file,
        'is_spontaneous':     is_spontaneous,
        'is_2p_opto':         is_2p_opto,
        'use_inference':      use_inference,
        'do_neuropil':        do_neuropil,
        'do_plot':            do_plot,
        'do_vrec_diagnostic': do_vrec_diagnostic,
        'dur_resp':           dur_resp,
        'opto_post_sec':      opto_post_sec,
        'opto_pre_sec':       opto_pre_sec,
        'opto_blank_sec':     opto_blank_sec,
        'opto_offset_trigger':opto_offset_trigger,
        'chunk_size':         chunk_size,
        'output_dir':         output_dir,
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
    # Step 2b — Vrec diagnostic early exit
    # If do_vrec_diagnostic=True, load vrec + detect triggers, plot, and return
    # without running any of the expensive processing steps.
    # -----------------------------------------------------------------------
    if do_vrec_diagnostic:
        print('\n[do_vrec_diagnostic=True] Loading vrec and detecting triggers only...')
        vrec = None
        if files.get('vrec_csv'):
            vrec = genfromtxt_with_progress(files['vrec_csv'], delimiter=',', skip_header=1)
            vrec_sample_rate = (vrec_meta['sample_rate'] if vrec_meta else 10000)
            result['vrec']            = vrec
            result['vrec_sample_rate']= vrec_sample_rate
            ch_layout = detect_vrec_channel_layout(vrec)
            result['vrec_channel_layout'] = ch_layout['layout']
            result['photostim_ch']        = ch_layout['photostim_ch']
            # Visual triggers are irrelevant for spontaneous experiments
            vis_ch = None if is_spontaneous else ch_layout['visual_trigger_ch']
            result['visual_trigger_ch'] = vis_ch
            opto_ch = ch_layout['photostim_ch']

            # Visual triggers — rising-edge detection (positive pulses, not spontaneous)
            if vis_ch is not None:
                vis_signal = medfilt(vrec[:, vis_ch].astype(float), 51)
                vis_signal[vis_signal < 0] = 0
                vis_diff = np.diff(vis_signal)
                stim_on, _ = find_peaks(vis_diff, distance=1e3,
                                        height=(max(vis_diff) - max(vis_diff) * 0.9))
                stim_off, _ = find_peaks(-vis_diff, distance=1e3,
                                         height=(max(-vis_diff) - max(-vis_diff) * 0.9))
                result['stim_on']      = stim_on
                result['stim_on_sec']  = stim_on / vrec_sample_rate
                result['stim_off']     = stim_off
                result['stim_off_sec'] = stim_off / vrec_sample_rate

            # Photostim triggers — direct peak detection (positive pulses)
            if opto_ch is not None:
                ps_signal = medfilt(vrec[:, opto_ch].astype(float), 51)
                ps_signal[ps_signal < 0] = 0
                photostim_triggers, _ = find_peaks(ps_signal, distance=1e4,
                                                   height=(max(ps_signal) - max(ps_signal) * 0.9))
                result['photostim_triggers']     = photostim_triggers
                result['photostim_triggers_sec'] = photostim_triggers / vrec_sample_rate
        else:
            print('[WARNING] No vrec CSV found — cannot plot diagnostic.')

        # PsychoPy file (for stim_id colouring)
        if stim_file > -1:
            psychopy_date = _bruker_date_to_psychopy(experiment_id)
            psychopy_path = os.path.join(PSYCHOPY_ROOT, psychopy_date, f'T{stim_file:03d}.txt')
            if os.path.isfile(psychopy_path):
                psychopy_data = np.genfromtxt(psychopy_path)
                if psychopy_data.ndim == 1:
                    psychopy_data = psychopy_data[np.newaxis, :]
                if is_2p_opto and opto_offset_trigger:
                    psychopy_data = psychopy_data[1:]
                stim_id = psychopy_data[:, 0]
                result['stim_id']      = stim_id
                result['unique_stims'] = np.unique(stim_id)

        diag_path = os.path.join(output_dir, f'{experiment_id}_vrec_diagnostic.png')
        plot_vrec_diagnostic(result, save_path=diag_path)
        plt.show()
        for _k in ('vrec', 'vrec_channel_layout', 'visual_trigger_ch', 'photostim_ch'):
            result.pop(_k, None)
        return result

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
                 f'inference.h5 not found in {data_dir}')
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
    # Step 3b — MarkPoints pixel coordinates (computed now that image size is
    #            known; done here regardless of vrec availability so that
    #            spontaneous+opto experiments always populate these fields)
    # -----------------------------------------------------------------------
    for key in ('markpoints_xy_norm', 'markpoints_xy_pix',
                'markpoints_condition_idx', 'markpoints_laser_power',
                'markpoints_spiral_diameter_px',
                'roi_photostim_point', 'roi_photostim_group', 'roi_photostim_overlap',
                'markpoint_assigned_roi'):
        result[key] = None

    if mp_data is not None:
        px_line  = xml_params.get('pixels_per_line') or size_x
        px_frame = xml_params.get('lines_per_frame')  or size_y
        um_pp    = xml_params.get('microns_per_pixel') or 1.0

        xy_norm_list        = []
        cond_idx_list       = []
        laser_powers        = []
        spiral_diameters_px = []

        for ci, cond in enumerate(mp_data['conditions']):
            laser_powers.append(cond['uncaging_laser_power'])
            spiral_um = cond['galvo'].get('spiral_size_microns', 25.0)
            spiral_diameters_px.append(spiral_um / um_pp if um_pp else 25.0)
            for pt in cond['points']:
                xy_norm_list.append([pt['x_norm'], pt['y_norm']])
                cond_idx_list.append(ci)

        xy_norm = np.array(xy_norm_list)
        xy_pix  = np.column_stack([
            xy_norm[:, 0] * px_line,
            xy_norm[:, 1] * px_frame,
        ])

        result['markpoints_xy_norm']            = xy_norm
        result['markpoints_xy_pix']             = xy_pix
        result['markpoints_condition_idx']      = np.array(cond_idx_list)
        result['markpoints_laser_power']        = np.array(laser_powers)
        result['markpoints_spiral_diameter_px'] = np.array(spiral_diameters_px)

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
    # Step 6b — ROI-to-MarkPoint overlap (if 2P opto)
    # For each ROI find the markpoint whose stimulation circle has the greatest
    # fractional overlap with that ROI mask.  The circle radius comes from the
    # per-group spiral_diameter_px (index via cond_idx).
    # -----------------------------------------------------------------------
    roi_photostim_point   = np.full(num_cells, -1, dtype=int)
    roi_photostim_group   = np.full(num_cells, -1, dtype=int)
    roi_photostim_overlap = np.zeros(num_cells, dtype=float)

    mp_xy   = result.get('markpoints_xy_pix')
    mp_cidx = result.get('markpoints_condition_idx')
    mp_diam = result.get('markpoints_spiral_diameter_px')

    markpoint_assigned_roi = np.full(len(mp_xy) if mp_xy is not None else 0, -1, dtype=int)

    if mp_xy is not None and mp_cidx is not None and mp_diam is not None:
        n_points   = len(mp_xy)
        row_grid, col_grid = np.mgrid[0:size_x, 0:size_y]

        # Build full overlap matrix: overlap_matrix[pi, cc] = fraction of ROI cc
        # covered by the stimulation circle of markpoint pi.
        overlap_matrix = np.zeros((n_points, num_cells))
        for pi, (x_mp, y_mp) in enumerate(mp_xy):
            gi     = mp_cidx[pi]
            radius = mp_diam[gi] / 2.0
            circle = (col_grid - x_mp) ** 2 + (row_grid - y_mp) ** 2 <= radius ** 2
            for cc in range(num_cells):
                roi_area = mask2d[cc].sum()
                if roi_area == 0:
                    continue
                overlap_matrix[pi, cc] = float(
                    (circle & (mask2d[cc] > 0.5)).sum()) / roi_area

        # Winner-takes-all: each markpoint claims the single ROI with the
        # greatest overlap fraction.  If multiple markpoints claim the same ROI
        # (e.g. duplicate target locations in the XML) the one with the higher
        # overlap fraction wins.
        markpoint_assigned_roi = np.full(n_points, -1, dtype=int)
        for pi in range(n_points):
            row  = overlap_matrix[pi, :]
            best = int(np.argmax(row))
            if row[best] > 0:
                markpoint_assigned_roi[pi] = best

        # Derive per-ROI arrays (highest-overlap markpoint wins each ROI)
        roi_photostim_point = np.full(num_cells, -1, dtype=int)
        for pi in range(n_points):
            cc = markpoint_assigned_roi[pi]
            if cc >= 0:
                ov = overlap_matrix[pi, cc]
                if ov > roi_photostim_overlap[cc]:
                    roi_photostim_point[cc]   = pi
                    roi_photostim_group[cc]   = mp_cidx[pi]
                    roi_photostim_overlap[cc] = ov

    result['roi_photostim_point']   = roi_photostim_point    # (n_rois,) markpoint index, -1=none
    result['roi_photostim_group']   = roi_photostim_group    # (n_rois,) group index, -1=none
    result['roi_photostim_overlap'] = roi_photostim_overlap  # (n_rois,) overlap fraction 0-1
    result['markpoint_assigned_roi']= markpoint_assigned_roi # (n_points,) ROI index, -1=none

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
    for key in ('frame_triggers', 'frame_triggers_sec', 'vrec_sample_rate',
                'stim_on', 'stim_on_sec', 'stim_off', 'stim_off_sec',
                'stim_on_2p_frame', 'stim_id', 'unique_stims',
                'stim_properties', 'target_number', 'target_trial',
                'cyc', 'resp', 'resps', 'resp_err', 'stim_avg_images'):
        result[key] = None

    vrec    = None   # needed later for opto
    opto_ch = None   # set in whichever branch loads the vrec

    if not is_spontaneous:
        # --- Frame triggers from XML ---
        print('Reading 2P frame triggers from XML...')
        frame_triggers = read_xml_file(files['tseries_xml'])
        frame_triggers = frame_triggers * 1e4   # → 10 kHz samples
        if frame_triggers[0] > 340:
            print('[WARNING] First 2P frame trigger appears to be missing — interpolating.')
        frame_triggers = replace_missing_frame_triggers(frame_triggers)
        result['frame_triggers']     = frame_triggers
        result['frame_triggers_sec'] = frame_triggers / 1e4   # convert 10 kHz samples → seconds

        # --- Voltage recording ---
        _require(files['vrec_csv'],
                 f'VoltageRecording CSV not found in {data_dir}')
        print('Loading voltage recording...')
        vrec = genfromtxt_with_progress(files['vrec_csv'],
                                        delimiter=',', skip_header=1)
        vrec_sample_rate = (vrec_meta['sample_rate']
                            if vrec_meta else 10000)
        result['vrec_sample_rate'] = vrec_sample_rate

        # --- Auto-detect channel layout ---
        ch_layout = detect_vrec_channel_layout(vrec)
        vis_ch  = ch_layout['visual_trigger_ch']
        opto_ch = ch_layout['photostim_ch']

        if has_stim and vis_ch is not None:
            # --- Detect visual stimulus onsets from vrec ---
            # Triggers are positive pulses on Input 1 (col 2). Rising-edge
            # detection: median-filter → zero negatives → diff → find positive peaks.
            vis_signal = medfilt(vrec[:, vis_ch].astype(float), 51)
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

    # --- Read PsychoPy file whenever stim_file > -1, regardless of is_spontaneous ---
    if stim_file > -1:
        psychopy_date = _bruker_date_to_psychopy(experiment_id)
        psychopy_path = os.path.join(
            PSYCHOPY_ROOT, psychopy_date, f'T{stim_file:03d}.txt')
        if not os.path.isfile(psychopy_path):
            print(f'[WARNING] PsychoPy file not found: {psychopy_path}')
        else:
            print(f'Reading PsychoPy file: {psychopy_path}')
            psychopy_data = np.genfromtxt(psychopy_path)
            if psychopy_data.ndim == 1:
                psychopy_data = psychopy_data[np.newaxis, :]

            # Known acquisition bug: for 2P opto experiments the trigger
            # stream is offset by one row relative to the psychopy file.
            # Dropping row 0 realigns triggers → psychopy rows.
            if is_2p_opto and opto_offset_trigger:
                print(f'[opto_offset_trigger=True] Dropping first PsychoPy row '
                      f'({psychopy_data.shape[0]} → {psychopy_data.shape[0] - 1} rows).')
                psychopy_data = psychopy_data[1:]

            if not is_2p_opto:
                stim_id         = psychopy_data[:, 0]
                stim_properties = psychopy_data[:, 1:]
                result['stim_properties'] = stim_properties
            else:
                # 2P opto: columns are [target_number, target_trial, (stim_id), ...]
                if psychopy_data.shape[1] >= 3:
                    stim_id = psychopy_data[:, 2]
                else:
                    stim_id = psychopy_data[:, 0]
                result['target_number'] = psychopy_data[:, 0]
                result['target_trial']  = psychopy_data[:, 1]
                if psychopy_data.shape[1] > 3:
                    result['stim_properties'] = psychopy_data[:, 3:]

            result['stim_id']      = stim_id
            result['unique_stims'] = np.unique(stim_id)
            print(f'  {len(stim_id)} rows loaded, '
                  f'{len(np.unique(stim_id))} unique stim IDs: {np.unique(stim_id).tolist()}')

            # For non-spontaneous experiments: validate trigger count and map to frames
            if not is_spontaneous and result.get('stim_on') is not None:
                stim_on = result['stim_on']
                if len(stim_on) != len(stim_id):
                    print(f'[WARNING] Stim trigger mismatch: '
                          f'{len(stim_on)} detected vs {len(stim_id)} in stim file!')
                else:
                    print(f'Stim triggers: {len(stim_on)} ✓')

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
    # Step 11b — Load vrec for spontaneous + opto experiments
    #             (when is_spontaneous=True the vrec block above was skipped,
    #              but we still need the voltage recording to find photostim
    #              triggers.  Load it here if not already loaded.)
    # -----------------------------------------------------------------------
    if is_2p_opto and vrec is None and files.get('vrec_csv'):
        print('Loading voltage recording for opto trigger detection '
              '(spontaneous experiment)...')
        vrec = genfromtxt_with_progress(files['vrec_csv'],
                                        delimiter=',', skip_header=1)
        vrec_sample_rate = (vrec_meta['sample_rate'] if vrec_meta else 10000)
        result['vrec_sample_rate'] = vrec_sample_rate

        ch_layout = detect_vrec_channel_layout(vrec)
        opto_ch = ch_layout['photostim_ch']

    # -----------------------------------------------------------------------
    # Step 12 — 2P opto processing
    # -----------------------------------------------------------------------
    # Initialise opto keys to None (markpoints_* keys are initialised in Step 3b
    # and must NOT be reset here — they are populated regardless of vrec)
    for key in ('photostim_triggers', 'photostim_triggers_sec',
                'photostim_2p_frame',
                'opto_stim_ids', 'opto_avg_images', 'opto_baseline_image',
                'opto_delta_images', 'opto_trial_delta_images', 'opto_n_trials',
                'opto_grand_avg_image', 'opto_grand_baseline_image',
                'opto_grand_delta_image',
                'opto_blank_sec', 'opto_blank_frames',
                'opto_post_sec', 'opto_post_frames',
                'opto_pre_sec', 'opto_pre_frames',
                'cyc_photostim_only'):
        result[key] = None

    if is_2p_opto and vrec is not None and opto_ch is not None:
        vrec_sample_rate = result['vrec_sample_rate'] or 10000

        # Detect photostim triggers — direct peak detection on positive pulses
        print('Detecting photostimulation triggers...')
        ps_signal = medfilt(vrec[:, opto_ch].astype(float), 51)
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
            result['frame_triggers']     = ft
            result['frame_triggers_sec'] = ft / 1e4   # convert 10 kHz samples → seconds

        photostim_2p_frame = np.array([
            np.argmin(np.abs(ps - ft))
            for ps in photostim_triggers
        ], dtype=int)
        result['photostim_2p_frame'] = photostim_2p_frame

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

        # stim_id is populated whenever stim_file > -1 (see PsychoPy read above)
        opto_stim_id = result.get('stim_id')
        if opto_stim_id is None:
            print(f'[WARNING] No stim_id available (stim_file=-1?) — '
                  f'treating all {len(photostim_triggers)} triggers as stim_id=0.')
            opto_stim_id = np.zeros(len(photostim_triggers))
        elif len(opto_stim_id) != len(photostim_triggers):
            n_use = min(len(opto_stim_id), len(photostim_triggers))
            print(f'[WARNING] stim_id count ({len(opto_stim_id)}) != photostim trigger count '
                  f'({len(photostim_triggers)}) — using first {n_use} of each. '
                  f'Check trigger detection distance parameter.')
            opto_stim_id          = opto_stim_id[:n_use]
            photostim_triggers    = photostim_triggers[:n_use]
            photostim_2p_frame    = photostim_2p_frame[:n_use]

        opto_unique_ids = np.unique(opto_stim_id)
        n_opto_ids      = len(opto_unique_ids)

        # ── Pass 1: build a single global baseline from ALL pre-trigger windows ──
        # This shared baseline removes trial-to-trial baseline noise from the
        # delta images.  Events that are too close to the recording boundary are
        # excluded from both passes.
        print('Pass 1/2: accumulating global pre-trigger baseline...')
        global_pre_acc  = np.zeros((size_x, size_y))
        global_pre_cnt  = 0
        valid_event_set = set()

        for ev in range(len(photostim_2p_frame)):
            f0         = int(photostim_2p_frame[ev])
            pre_start  = f0 - opto_pre_frames
            post_stop  = f0 + opto_blank_frames + opto_post_frames
            if pre_start < 0 or post_stop > num_frames:
                continue
            valid_event_set.add(ev)
            global_pre_acc += np.mean(h[dat_name][pre_start:f0-1], axis=0)
            global_pre_cnt += 1

        if global_pre_cnt > 0:
            global_baseline = global_pre_acc / global_pre_cnt
        else:
            global_baseline = np.ones((size_x, size_y), dtype=float)
            print('[WARNING] No valid pre-trigger windows found; baseline set to 1.')

        n_skipped = len(photostim_2p_frame) - global_pre_cnt
        if n_skipped:
            print(f'[WARNING] Skipped {n_skipped} edge photostim events '
                  f'(too close to recording boundaries).')

        # ── Pass 2: accumulate post images per stim_id; delta uses global baseline ──
        print('Pass 2/2: computing per-stim-ID post averages...')
        opto_avg_imgs           = np.zeros((n_opto_ids, size_x, size_y))
        opto_n_trials_arr       = np.zeros(n_opto_ids, dtype=int)
        opto_trial_delta_images = []   # list[ndarray (n_trials, sx, sy)] per stim_id

        for oi, sid in enumerate(tqdm(opto_unique_ids,
                                       desc='Opto avg images', ncols=75)):
            events   = np.where(opto_stim_id == sid)[0]
            post_acc = np.zeros((size_x, size_y))
            trial_deltas = []

            for ev in events:
                if ev not in valid_event_set:
                    continue   # already counted as skipped in pass 1
                f0         = int(photostim_2p_frame[ev])
                post_start = f0 + opto_blank_frames
                post_stop  = f0 + opto_blank_frames + opto_post_frames
                post_img   = np.mean(h[dat_name][post_start:post_stop], axis=0)
                post_acc  += post_img

                # Per-trial delta: (post - global_baseline) / global_baseline
                with np.errstate(invalid='ignore', divide='ignore'):
                    trial_delta = np.where(
                        global_baseline != 0,
                        (post_img - global_baseline) / global_baseline,
                        0.0,
                    )
                trial_deltas.append(trial_delta.astype(np.float32))

            cnt = len(trial_deltas)
            if cnt > 0:
                opto_avg_imgs[oi] = post_acc / cnt
            opto_n_trials_arr[oi] = cnt
            opto_trial_delta_images.append(
                np.stack(trial_deltas)
                if trial_deltas else np.zeros((0, size_x, size_y), dtype=np.float32)
            )

        # Save per-trial images as TIFF stacks (one file per stim_id)
        print('Saving per-trial TIFF stacks...')
        for oi, sid in enumerate(opto_unique_ids):
            stack = opto_trial_delta_images[oi]
            if stack.shape[0] == 0:
                continue
            tif_path = os.path.join(
                output_dir,
                f'{experiment_id}_opto_group{int(sid)}_trials.tif')
            tifffile.imwrite(tif_path, stack, imagej=True)
            print(f'  Saved: {os.path.basename(tif_path)}  '
                  f'({stack.shape[0]} trials)')

        # Per-stim-ID % change: (post_avg - global_baseline) / global_baseline
        with np.errstate(invalid='ignore', divide='ignore'):
            opto_delta_imgs = np.where(
                global_baseline != 0,
                (opto_avg_imgs - global_baseline) / global_baseline,
                0.0,
            )

        # Grand-average post across all stim IDs (weighted by trial count)
        weights = opto_n_trials_arr.astype(float)
        if weights.sum() > 0:
            w = weights / weights.sum()
            opto_grand_avg = np.tensordot(w, opto_avg_imgs, axes=([0], [0]))
        else:
            opto_grand_avg = np.zeros((size_x, size_y))

        with np.errstate(invalid='ignore', divide='ignore'):
            opto_grand_delta = np.where(
                global_baseline != 0,
                (opto_grand_avg - global_baseline) / global_baseline,
                0.0,
            )

        result['opto_stim_ids']            = opto_unique_ids
        result['opto_avg_images']          = opto_avg_imgs
        result['opto_baseline_image']      = global_baseline          # single shared baseline
        result['opto_delta_images']        = opto_delta_imgs
        result['opto_trial_delta_images']  = opto_trial_delta_images  # list of (n_trials,sx,sy)
        result['opto_n_trials']            = opto_n_trials_arr
        result['opto_grand_avg_image']     = opto_grand_avg
        result['opto_grand_baseline_image']= global_baseline
        result['opto_grand_delta_image']   = opto_grand_delta

        # ── cyc_photostim_only: (n_cells, n_groups, max_trials) ─────────────
        # Per-trial, per-cell dF/F response = mean(post-blank) - mean(pre)
        # using opto_post_sec window.  NaN-padded where a group has fewer
        # trials than the maximum across groups.
        print('Computing cyc_photostim_only (cells × groups × trials)...')
        dff_arr = result['dff']   # (n_frames, n_cells)
        trial_resps = []          # one list per group, each entry shape (n_cells,)
        for oi, sid in enumerate(opto_unique_ids):
            events = np.where(opto_stim_id == sid)[0]
            group_trials = []
            for ev in events:
                if ev not in valid_event_set:
                    continue
                f0         = int(photostim_2p_frame[ev])
                pre_start  = f0 - opto_pre_frames
                post_start = f0 + opto_blank_frames
                post_stop  = f0 + opto_blank_frames + opto_post_frames
                baseline   = np.mean(dff_arr[pre_start:f0], axis=0)   # (n_cells,)
                response   = np.mean(dff_arr[post_start:post_stop], axis=0)
                group_trials.append(response - baseline)
            trial_resps.append(group_trials)

        max_trials = max((len(g) for g in trial_resps), default=0)
        cyc_ps = np.full((num_cells, n_opto_ids, max_trials), np.nan, dtype=np.float32)
        for oi, group_trials in enumerate(trial_resps):
            for ti, resp in enumerate(group_trials):
                cyc_ps[:, oi, ti] = resp
        result['cyc_photostim_only'] = cyc_ps

    # -----------------------------------------------------------------------
    # Close H5 movie
    # -----------------------------------------------------------------------
    h.close()

    # -----------------------------------------------------------------------
    # Step 13 — Save result dict to H5
    # -----------------------------------------------------------------------
    save_result_h5(result, output_dir)

    # -----------------------------------------------------------------------
    # Step 14 — Visualization
    # -----------------------------------------------------------------------
    if do_plot:
        fig1_path = os.path.join(output_dir, f'{experiment_id}_summary.png')
        plot_experiment_summary(result, save_path=fig1_path)
        if result.get('is_2p_opto') and result.get('opto_delta_images') is not None:
            fig2_path = os.path.join(output_dir, f'{experiment_id}_opto_images.png')
            plot_opto_images(result, save_path=fig2_path)

    if do_vrec_diagnostic:
        fig3_path = os.path.join(output_dir, f'{experiment_id}_vrec_diagnostic.png')
        plot_vrec_diagnostic(result, save_path=fig3_path)

    if do_plot or do_vrec_diagnostic:
        plt.show()   # display all figures simultaneously

    # -----------------------------------------------------------------------
    # Step 15 — Print summary
    # -----------------------------------------------------------------------
    print_experiment_summary(result)

    return result


# ===========================================================================
# Voltage channel auto-detection
# ===========================================================================

def detect_vrec_channel_layout(vrec, threshold=1.0):
    """
    Determine which vrec columns carry visual and photostimulation triggers.

    The vrec array columns are: [Time(ms), Input0, Input1, Input2, ...]
    Data channels start at column index 1.

    Fixed channel assignment (hardware convention):
      Input 0 (col 1) — reserved / other signals, ignored for trigger detection
      Input 1 (col 2) — visual stimulus trigger (always)
      Input 2+ (col 3+) — photostim trigger (first active channel found)

    Triggers are positive-going pulses. Rising-edge detection (diff) is used
    for visual stim; direct peak detection is used for photostim.

    Returned column indices are direct indices into the vrec array (i.e.
    col 1 = Input 0, col 2 = Input 1, col 3 = Input 2).

    Returns
    -------
    dict with keys: visual_trigger_ch (int or None), photostim_ch (int or None),
                    layout (str)
    """
    n_cols = vrec.shape[1]
    names  = {c: f'Input{c - 1}' for c in range(1, n_cols)}

    def _n_events(col_idx):
        """Count positive-going events on a channel."""
        sig = medfilt(vrec[:, col_idx].astype(float), 51)
        sig[sig < 0] = 0
        events, _ = find_peaks(sig, distance=500, height=threshold * 0.5)
        return len(events)

    # Count events on all data channels for diagnostic printout
    event_counts = {c: _n_events(c) for c in range(1, n_cols)}
    print('Vrec events per channel:', {names[c]: event_counts[c]
                                       for c in range(1, n_cols)})

    # Visual stim is always Input 1 (col 2)
    vis_ch = 2 if n_cols > 2 else None

    # Photostim: first active channel at col 3+ (Input 2+)
    opto_ch = None
    for c in range(3, n_cols):
        if event_counts.get(c, 0) >= 3:
            opto_ch = c
            break

    if vis_ch is not None and opto_ch is not None:
        layout = 'standard'
    elif opto_ch is not None:
        layout = 'spontaneous'
    else:
        layout = 'unknown'

    print(f'Vrec layout: {layout}  '
          f'(vis=col{vis_ch}/{names.get(vis_ch, "?")} '
          f'[Input 1 — fixed], '
          f'opto=col{opto_ch}/{names.get(opto_ch, "none")})')

    return {'visual_trigger_ch': vis_ch, 'photostim_ch': opto_ch, 'layout': layout}


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

    if stim_on is not None:
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
        ps_trig  = result.get('photostim_triggers')
        opto_ids = result.get('opto_stim_ids')
        opto_n   = result.get('opto_n_trials')
        mp_pwr   = result.get('markpoints_laser_power')
        mp_idx   = result.get('markpoints_condition_idx')
        xy_pix   = result.get('markpoints_xy_pix')
        xy_norm  = result.get('markpoints_xy_norm')
        diam_px  = result.get('markpoints_spiral_diameter_px')
        um_pp    = result.get('microns_per_pixel') or 1.0
        lines += [
            '',
            '── 2P Optogenetics ───────────────────────',
            f'  is_2p_opto         : True',
        ]
        # Trigger count vs stim file rows
        stim_id_arr = result.get('stim_id')
        n_stim_rows = len(stim_id_arr) if stim_id_arr is not None else None
        opto_offset = result.get('opto_offset_trigger', False)
        if ps_trig is not None:
            n_trig = len(ps_trig)
            if n_stim_rows is not None:
                match_sym = '✓' if n_trig == n_stim_rows else '✗ MISMATCH'
                lines.append(f'  Photostim triggers : {n_trig}  (stim file rows: {n_stim_rows})  {match_sym}')
                if n_trig != n_stim_rows:
                    lines.append(f'    → adjust trigger detection distance if mismatch is large')
            else:
                lines.append(f'  Photostim triggers : {n_trig}')
        else:
            lines.append('  Photostim triggers : NOT DETECTED (check vrec/channel layout)')
        if opto_ids is not None:
            lines.append(f'  Unique stim IDs    : {len(opto_ids)}  {opto_ids.tolist()}')
        if opto_n is not None:
            trial_str = opto_n.tolist()
            offset_note = '  (±1 trial expected due to opto_offset_trigger)' if opto_offset else ''
            lines.append(f'  Trials per ID      : {trial_str}{offset_note}')
        if mp_pwr is not None and mp_idx is not None:
            n_groups = len(mp_pwr)
            lines.append(f'  MarkPoints groups  : {n_groups}')
            for gi in range(n_groups):
                mask   = (mp_idx == gi)
                n_pts  = int(np.sum(mask))
                pwr    = mp_pwr[gi]
                d_px   = diam_px[gi] if diam_px is not None else None
                d_um   = d_px * um_pp if d_px is not None else None
                lines.append(f'  Group {gi+1}: {n_pts} target(s)  '
                              f'power={pwr:.0f} mW  '
                              + (f'diam={d_um:.1f} µm ({d_px:.1f} px)' if d_um else ''))
                if xy_pix is not None and xy_norm is not None:
                    pts_pix  = xy_pix[mask]
                    pts_norm = xy_norm[mask]
                    for pi, (pn, pp) in enumerate(zip(pts_norm, pts_pix)):
                        lines.append(f'    Spot {pi+1}: norm=({pn[0]:.3f}, {pn[1]:.3f})  '
                                     f'pix=({pp[0]:.1f}, {pp[1]:.1f})')
        else:
            lines.append('  MarkPoints         : NOT LOADED (check markpoints XML)')

    lines.append('=' * 44)
    print('\n'.join(lines))


# ===========================================================================
# Visualization
# ===========================================================================

def plot_experiment_summary(result, save_path=None):
    """
    Figure 1 — two panels side by side:
      Left  : mean projection + ROI outlines + MarkPoints circles (if 2P opto)
      Right : dF/F heatmap (cells × time) with colorbar
    """
    avg_image   = result.get('avg_image')
    mask2d      = result.get('mask2d')
    dff         = result.get('dff')
    is_soma     = result.get('is_soma')
    is_dendrite = result.get('is_dendrite')
    is_spine    = result.get('is_spine')
    is_2p_opto  = result.get('is_2p_opto', False)
    xy_pix      = result.get('markpoints_xy_pix')
    cond_idx    = result.get('markpoints_condition_idx')
    laser_pwr   = result.get('markpoints_laser_power')
    diam_px     = result.get('markpoints_spiral_diameter_px')
    um_pp       = result.get('microns_per_pixel')
    fp          = result.get('frame_period') or 0.033
    size_x      = avg_image.shape[0] if avg_image is not None else 512

    # Vivid palette shared by ROI labels and MarkPoints circles
    _OPTO_COLORS = [
        '#FF0000',  # red
        '#FF7700',  # orange
        '#FFD700',  # gold
        '#00FF00',  # lime
        '#00FFFF',  # cyan
        '#0077FF',  # dodger blue
        '#AA00FF',  # violet
        '#FF00AA',  # hot pink
        '#FF77FF',  # magenta-pink
        '#00FF88',  # spring green
    ]

    fig, (ax_roi, ax_dff) = plt.subplots(1, 2, figsize=(14, 6),
                                          constrained_layout=True)

    # ------------------------------------------------------------------
    # Left panel — mean projection + ROIs + MarkPoints
    # ------------------------------------------------------------------
    if avg_image is not None:
        vmax = avg_image.max() / 2   # half-max clamp so dim structures are visible
        ax_roi.imshow(avg_image, cmap='gray', interpolation='nearest', vmax=vmax)

    roi_colors    = {'soma': 'cyan', 'dendrite': 'yellow', 'spine': 'magenta'}
    legend_patches = []

    roi_ps_group = result.get('roi_photostim_group')  # (n_rois,) -1 = not stimulated

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
            ax_roi.contour(mask2d[cc].astype(float), levels=[0.5],
                           colors=[color], linewidths=0.6, alpha=0.8)

            # ROI number label at mask centroid
            pts = np.argwhere(mask2d[cc] > 0.5)
            if len(pts) > 0:
                cy, cx = pts.mean(axis=0)   # row → y, col → x
                # If photostimulated, use the group color; otherwise white
                gi = int(roi_ps_group[cc]) if roi_ps_group is not None else -1
                if gi >= 0:
                    txt_color = _OPTO_COLORS[gi % len(_OPTO_COLORS)]
                else:
                    txt_color = 'white'
                ax_roi.text(cx, cy, str(cc + 1),
                            ha='center', va='center',
                            fontsize=7, color=txt_color,
                            fontweight='bold',
                            clip_on=True)

        for label, col in roi_colors.items():
            legend_patches.append(mpatches.Patch(color=col,
                                                  label=label.capitalize()))

    # MarkPoints as filled, semi-transparent circles
    if is_2p_opto and xy_pix is not None and cond_idx is not None:
        n_groups  = (len(laser_pwr) if laser_pwr is not None
                     else int(cond_idx.max()) + 1)
        mp_patches = []
        for gi in range(n_groups):
            pts    = xy_pix[cond_idx == gi]
            radius = ((diam_px[gi] / 2) if diam_px is not None
                      else 12)                # fallback 12 px radius
            color  = _OPTO_COLORS[gi % len(_OPTO_COLORS)]
            label  = (f'Group {gi+1} ({laser_pwr[gi]:.0f} mW)'
                      if laser_pwr is not None else f'Group {gi+1}')
            for x, y in pts:
                circ = plt.Circle((x, y), radius=radius,
                                  color=color, alpha=0.45, zorder=5)
                ax_roi.add_patch(circ)
                # thin edge ring for visibility against bright background
                ring = plt.Circle((x, y), radius=radius,
                                  fill=False, edgecolor=color,
                                  linewidth=1.2, alpha=0.9, zorder=6)
                ax_roi.add_patch(ring)
            mp_patches.append(mpatches.Patch(color=color, alpha=0.7,
                                              label=label))
        legend_patches.extend(mp_patches)

    if um_pp and size_x:
        _draw_scale_bar(ax_roi, size_x, um_pp, bar_um=50)

    ax_roi.set_xlim(0, avg_image.shape[1] if avg_image is not None else 512)
    ax_roi.set_ylim(avg_image.shape[0] if avg_image is not None else 512, 0)
    ax_roi.set_title('Mean projection  |  ROIs' +
                     ('  |  MarkPoints' if is_2p_opto else ''))
    ax_roi.axis('off')
    if legend_patches:
        ax_roi.legend(handles=legend_patches, loc='lower right',
                      fontsize=7, framealpha=0.6)

    # ------------------------------------------------------------------
    # Right panel — dF/F time series (first 10 cells)
    # ------------------------------------------------------------------
    if dff is not None and dff.size > 0:
        n_frames, n_cells = dff.shape
        t_axis   = np.arange(n_frames) * fp          # seconds
        n_plot   = min(10, n_cells)

        # Vertical spacing: use 2× the 99th-percentile absolute dF/F value
        # so traces don't overlap under typical signal amplitudes.
        spacing = 2.0 * np.nanpercentile(np.abs(dff[:, :n_plot]), 99)
        spacing = max(spacing, 0.05)                  # floor so flat traces still separate

        # Collect handles for a compact legend
        trace_handles = []
        for cc in range(n_plot):
            offset = cc * spacing
            line,  = ax_dff.plot(t_axis, dff[:, cc] + offset,
                                 linewidth=0.6, color='black', alpha=0.75)
            ax_dff.text(t_axis[-1], offset,
                        f' {cc}', va='center', fontsize=6, color='black')
        trace_handles.append(mpatches.Patch(color='black', label='dF/F traces'))

        # Visual stimulus onsets — vertical dashed lines (blue)
        stim_on_sec = result.get('stim_on_sec')
        if stim_on_sec is not None and len(stim_on_sec):
            for t in stim_on_sec:
                ax_dff.axvline(t, color='royalblue', linewidth=0.5,
                               alpha=0.6, linestyle='--', zorder=2)
            trace_handles.append(mpatches.Patch(color='royalblue',
                                                label='Visual stim onset'))

        # Photostim triggers — vertical solid lines (red)
        ps_sec = result.get('photostim_triggers_sec')
        if ps_sec is not None and len(ps_sec):
            for t in ps_sec:
                ax_dff.axvline(t, color='tomato', linewidth=0.5,
                               alpha=0.5, linestyle='-', zorder=3)
            trace_handles.append(mpatches.Patch(color='tomato',
                                                label='Photostim trigger'))

        ax_dff.set_xlabel('Time (s)')
        ax_dff.set_ylabel('dF/F  (offset per cell)')
        ax_dff.set_title(f'dF/F traces — first {n_plot} cells')
        ax_dff.set_xlim(t_axis[0], t_axis[-1])
        ax_dff.set_yticks([])          # offsets have no absolute meaning
        if trace_handles:
            ax_dff.legend(handles=trace_handles, loc='upper right',
                          fontsize=7, framealpha=0.6)
    else:
        ax_dff.text(0.5, 0.5, 'dF/F not available',
                    ha='center', va='center', transform=ax_dff.transAxes)
        ax_dff.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Summary figure saved: {save_path}')


def plot_opto_images(result, save_path=None):
    """
    Figure 2 — per-stim-ID grand-average % change images.
    One subplot per unique opto stim_id. Shared red–blue colorscale,
    0 % = white.
    """
    opto_delta = result.get('opto_delta_images')
    opto_ids   = result.get('opto_stim_ids')
    opto_n     = result.get('opto_n_trials')

    if opto_delta is None or len(opto_delta) == 0:
        print('[plot_opto_images] No opto delta images to plot.')
        return

    n_ids   = len(opto_ids)
    n_cols  = min(n_ids, 5)
    n_rows  = math.ceil(n_ids / n_cols)

    # Shared symmetric colorscale across all panels
    # vlim = np.nanpercentile(np.abs(opto_delta), 99)
    # if vlim == 0:
    vlim = 0.5   # default to 50% change if data is flat; adjust as needed for visibility

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4.5 * n_rows),
                              constrained_layout=True,
                              squeeze=False)

    im = None
    for oi in range(n_ids):
        row = oi // n_cols
        col = oi %  n_cols
        ax  = axes[row, col]
        im  = ax.imshow(opto_delta[oi], cmap='bwr',
                        vmin=-vlim, vmax=vlim,
                        interpolation='nearest')
        n_str = f'  (n={opto_n[oi]})' if opto_n is not None else ''
        ax.set_title(f'Group ID {opto_ids[oi]:.0f}{n_str}', fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for oi in range(n_ids, n_rows * n_cols):
        axes[oi // n_cols, oi % n_cols].set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes, label='% change from baseline',
                     fraction=0.02, pad=0.02, shrink=0.6)

    fig.suptitle(f'{result.get("experiment_id", "")} — '
                 f'2P opto grand-average % change images', fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Opto image figure saved: {save_path}')


# ===========================================================================
# Vrec diagnostic plot
# ===========================================================================

def plot_vrec_diagnostic(result, save_path=None):
    """
    Plot vrec channels for the first 30 seconds (downsampled to ~1 kHz).

    Every channel shows:
      - Raw voltage trace
      - 2P frame trigger tick marks (gray '|') wherever a frame was acquired

    Additionally, channel-specific event overlays:
      - Visual-stim channel (vis_ch): downward triangles at stim onsets
      - Photostim channel (opto_ch): upward triangles coloured by stim_id

    Parameters
    ----------
    result : dict
        The pipeline result dict (must contain 'vrec' and related keys).
    save_path : str or None
        If given, save the figure to this path.
    """
    vrec             = result.get('vrec')
    vis_ch           = result.get('visual_trigger_ch')
    opto_ch          = result.get('photostim_ch')
    stim_id_arr      = result.get('stim_id')
    frame_triggers   = result.get('frame_triggers')   # may be None in early-exit mode
    experiment_id    = result.get('experiment_id', '')

    if vrec is None:
        print('[vrec_diagnostic] No vrec data available — skipping.')
        return

    vrec_sample_rate = result.get('vrec_sample_rate', 10000)

    # --- Window and downsample -----------------------------------------------
    # Save full-recording counts before trimming (used in title).
    stim_on_full        = result.get('stim_on')
    photostim_trig_full = result.get('photostim_triggers')
    n_vis_total  = len(stim_on_full)        if stim_on_full        is not None else 0
    n_opto_total = len(photostim_trig_full) if photostim_trig_full is not None else 0

    n_diag = min(30 * vrec_sample_rate, vrec.shape[0])
    step   = max(1, vrec_sample_rate // 1000)   # downsample to ~1 kHz
    vrec   = vrec[:n_diag:step, :]

    def _trim_triggers(trig_arr):
        """Map full-rate sample indices into the downsampled display space."""
        if trig_arr is None or len(trig_arr) == 0:
            return None
        # frame_triggers may be 2-D (1, N) from replace_missing_frame_triggers;
        # flatten defensively before comparison.
        arr  = np.asarray(trig_arr).ravel()
        mask = arr < n_diag
        return (arr[mask] // step).astype(int)

    stim_on         = _trim_triggers(stim_on_full)
    photostim_trig  = _trim_triggers(photostim_trig_full)
    frame_trig_disp = _trim_triggers(frame_triggers)

    # Trim stim_id_arr to the number of photostim triggers visible in the window
    stim_id_arr_win = None
    if stim_id_arr is not None and photostim_trig is not None:
        stim_id_arr_win = stim_id_arr[:len(photostim_trig)]

    # -------------------------------------------------------------------------
    n_samples, n_cols = vrec.shape
    data_cols = list(range(1, n_cols))   # skip col 0 = Time(ms)

    t_sec = vrec[:, 0] / 1000.0

    fig, ax = plt.subplots(1, 1, figsize=(18, 5), constrained_layout=True)

    _VIS_COLOR    = '#1E90FF'
    _FRAME_COLOR  = '#22CC44'
    _OPTO_COLORS  = [
        '#FF0000', '#FF7700', '#FFD700', '#00FF00', '#00FFFF',
        '#0077FF', '#AA00FF', '#FF00AA', '#FF77FF', '#00FF88',
    ]
    _TRACE_COLORS = ['#333333', '#555555', '#777777', '#999999']

    channel_names = {c: f'Input {c-1}' for c in data_cols}

    signals = [vrec[:, col] for col in data_cols]
    ranges  = [np.percentile(s, 99) - np.percentile(s, 1) for s in signals]
    spacing = max(ranges) * 1.4 if max(ranges) > 0 else 1.0
    offsets = {col: i * spacing for i, col in enumerate(data_cols)}

    legend_handles = []
    frame_legend_added = False

    for i, col in enumerate(data_cols):
        sig    = signals[i]
        offset = offsets[col]
        color  = _TRACE_COLORS[i % len(_TRACE_COLORS)]

        role  = ' [visual]' if col == vis_ch else (' [photostim]' if col == opto_ch else '')
        label = channel_names[col] + role

        ax.plot(t_sec, sig + offset, color=color, linewidth=0.5, alpha=0.85)

        ax.text(t_sec[0] - (t_sec[-1] * 0.005), offset, label,
                ha='right', va='center', fontsize=8, color=color)

        # 2P frame triggers — shown on every channel as green tick marks
        if frame_trig_disp is not None and len(frame_trig_disp):
            y_ft = vrec[frame_trig_disp, col] + offset
            lbl  = '2P frame' if not frame_legend_added else '_nolegend_'
            h = ax.scatter(t_sec[frame_trig_disp], y_ft,
                           marker='|', s=30, linewidths=0.8,
                           color=_FRAME_COLOR, alpha=0.5, zorder=4, label=lbl)
            if not frame_legend_added:
                legend_handles.append(h)
                frame_legend_added = True

        # Visual-stim channel: stim onset markers (downward triangles)
        if col == vis_ch and stim_on is not None and len(stim_on):
            t_t = t_sec[stim_on]
            y_t = vrec[stim_on, col] + offset
            h = ax.scatter(t_t, y_t, marker='v', s=60, color=_VIS_COLOR,
                           zorder=5, label='Visual stim onset')
            legend_handles.append(h)

        # Photostim channel: trigger markers coloured by stim_id
        if col == opto_ch and photostim_trig is not None and len(photostim_trig):
            if stim_id_arr_win is not None:
                unique_ids = np.unique(stim_id_arr_win)
                for ui, uid in enumerate(unique_ids):
                    sel = stim_id_arr_win == uid
                    c   = _OPTO_COLORS[ui % len(_OPTO_COLORS)]
                    t_t = t_sec[photostim_trig[sel]]
                    y_t = vrec[photostim_trig[sel], col] + offset
                    h = ax.scatter(t_t, y_t, marker='^', s=70, color=c,
                                   zorder=5, label=f'Photostim ID {uid:.0f}')
                    legend_handles.append(h)
            else:
                t_t = t_sec[photostim_trig]
                y_t = vrec[photostim_trig, col] + offset
                h = ax.scatter(t_t, y_t, marker='^', s=70,
                               color=_OPTO_COLORS[0], zorder=5,
                               label='Photostim trigger')
                legend_handles.append(h)

    ax.set_xlim(t_sec[0], t_sec[-1])
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_yticks([])
    ax.grid(axis='x', linewidth=0.4, alpha=0.4)
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc='upper right')

    n_rows  = len(stim_id_arr) if stim_id_arr is not None else '?'
    n_frames = len(np.asarray(frame_triggers).ravel()) if frame_triggers is not None else 'N/A'
    match   = '✓' if isinstance(n_rows, int) and n_opto_total == n_rows else '✗'
    ax.set_title(
        f'{experiment_id}  |  Vrec diagnostic  |  first 30 s (downsampled to 1 kHz)\n'
        f'2P frames: {n_frames}    '
        f'Visual triggers (total): {n_vis_total}    '
        f'Photostim triggers (total): {n_opto_total}    '
        f'Stim file rows: {n_rows}    '
        f'Match: {match}',
        fontsize=10
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Vrec diagnostic saved → {save_path}')


# ===========================================================================
# Internal helpers
# ===========================================================================

def save_result_h5(result, output_dir):
    """
    Save all numpy arrays and scalars from the result dict to an H5 file.
    Lists of arrays (e.g. opto_trial_delta_images) are saved as indexed
    sub-groups. Strings are saved as attributes. None values are skipped.
    """
    eid     = result.get('experiment_id', 'experiment')
    h5_path = os.path.join(output_dir, f'{eid}.h5')
    with h5py.File(h5_path, 'w') as f:
        _write_dict_to_h5(f, result)
    print(f'Result saved: {h5_path}')


def _write_dict_to_h5(group, d):
    """
    Recursively write a dict into an open h5py group.
    Every key is always written — None values become empty datasets
    marked with an attribute so callers know the field existed but
    was not populated for this experiment.
    """
    for key, val in d.items():
        if val is None:
            # Write an empty placeholder so the key is always present in H5
            ds = group.create_dataset(key, data=np.array([]))
            ds.attrs['is_none'] = True
        elif isinstance(val, np.ndarray):
            group.create_dataset(key, data=val)
        elif isinstance(val, dict):
            sub = group.require_group(key)
            _write_dict_to_h5(sub, val)
        elif isinstance(val, list):
            # e.g. opto_trial_delta_images: list of (n_trials, sx, sy)
            sub = group.require_group(key)
            for i, item in enumerate(val):
                if isinstance(item, np.ndarray):
                    sub.create_dataset(str(i), data=item)
        elif isinstance(val, bool):
            group.create_dataset(key, data=int(val))
        elif isinstance(val, (int, float)):
            group.create_dataset(key, data=val)
        elif isinstance(val, str):
            group.attrs[key] = val
        else:
            try:
                group.attrs[key] = str(val)
            except Exception:
                pass


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
