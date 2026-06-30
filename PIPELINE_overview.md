# Bruker 2P Pipeline — Overview

## Context

`bruker_pipeline.py` processes calcium imaging data from a Bruker two-photon microscope. It reads all acquisition parameters from XML, handles spontaneous and optogenetics experiments, and returns a structured Python dictionary of all processed variables.

---

## Usage

```python
from bruker_pipeline import process_experiment

result = process_experiment(
    date           = '06042026',   # MMDDYYYY
    file_num       = 3,            # TSeries number (e.g. 3 → folder ending -003)
    stim_file      = 3,            # PsychoPy T003.txt; -1 if no stimulus file
    is_2p_opto     = True,         # True if photostimulation present
    use_inference  = True,         # True = inference.h5, False = registered.h5
    do_neuropil    = False,
    dur_resp       = 2.5,          # response window (seconds)
    opto_post_sec  = 1.0,          # post-trigger averaging window (seconds)
    opto_pre_sec   = 0.5,          # pre-trigger dF/F baseline window (seconds)
    opto_blank_sec = 0.4,          # blanking period after trigger (seconds)
    do_plot        = True,
    do_vrec_diagnostic = False,    # True = early exit: load vrec, plot, return
)
```

When `stim_file = -1` the experiment is treated as spontaneous: vrec is still fully loaded and all channel events are detected; only stimulus-aligned matrices (`cyc`/`resp`) are skipped.

---

## File Structure

```
py-pipeline/
├── bruker_pipeline.py   # main entry point, returns dict
├── xml_utils.py         # XML parsing (TSeries, MarkPoints, VoltageRecording)
├── img_utils.py         # signal processing utilities
├── run_pipeline.py      # user-facing runner script
└── cleanup_raw_data.py  # GUI tool to delete raw files after processing
```

---

## Pipeline Flow

```
1.  get_target_folders_v2()    → resolve TSeries folder from date + file_num
2.  find_experiment_files()    → locate XML, H5, ROI zip, vrec CSV
3.  parse_tseries_xml()        → frame_period, zoom, objective, etc. → result['Bruker_Acq']
    parse_vrec_xml()           → sample_rate, channel info
3b. [if do_vrec_diagnostic]    → early exit: load vrec, detect all channel events,
                                 plot diagnostic, return
4.  parse_markpoints_xml()     → target coords, laser power → result['Bruker_Acq']
                                 (only if is_2p_opto=True)
5.  roifile.roiread()          → ROI classification (soma / dendrite / spine)
6.  Open registered.h5 / inference.h5
                               → avg_image (mean of first chunk_size frames),
                                 frame_times_sec
6b. MarkPoints → pixel coords  → roi_photostim_point/group/overlap,
                                 markpoint_assigned_roi
7.  gen_polyline_roi/in_polygon → mask2d
8.  Chunked trace extraction   → raw_traces, [raw_neuropil if do_neuropil]
9.  filter_baseline_dF_comp()  → dff, [dff_neuropil if do_neuropil]
                                 → result['params']['dff_window_frames/sec']
10. read_xml_file()            → frame_triggers (10 kHz samples, internal only)
                               → result['frame_triggers_sec']
    genfromtxt_with_progress() → vrec (hard error if CSV missing)
    detect_vrec_channel_layout() → vis_ch, opto_ch
    _detect_vrec_events()      → result['vrec_channel_events'] (all channels)
    map stim onsets            → result['stim_on_2p_frame']
    PsychoPy file (if stim_file > -1)
                               → stim_id, unique_stims, stim_properties,
                                 target_number, target_trial (opto only)
11. (if has_stim) gen_stim_cyc(), compute_peak_resp()
                               → cyc, resp, resps, resp_err
12. (if is_2p_opto)
    map photostim triggers     → result['photostim_2p_frame']
                               → result['photostim_triggers_sec']
    opto_baseline              → mean of frames at recording seconds 1–5
                                 (internal; not stored in result)
    NaN-blank dff at opto pulse windows → dff_nan (used below + by skewness)
    per-stim-ID post averages  → opto_delta_images  (% change vs baseline)
    per-trial post averages    → opto_trial_delta_images (first 5 trials/group;
                                 also saved as a float32 TIFF stack)
    per-trial dF/F matrix      → cyc_photostim_only (n_cells × n_groups × max_trials)
12b. Cell-quality skewness     → dff_skewness, is_good_cell (skew of dff_nan/dff
                                 per cell vs skewness_threshold)
13. save_result_h5()           → write result to {experiment_id}.h5
14. (if do_plot) plot_experiment_summary()  → {experiment_id}_summary.png
    (if is_2p_opto) plot_opto_images()      → {experiment_id}_opto_images.png
    (if do_vrec_diagnostic) plot_vrec_diagnostic() → {experiment_id}_vrec_diagnostic.png
15. print_experiment_summary()
16. return result dict
```

---

## Key Design Decisions

### Seconds-first timebase

All user-facing durations are in seconds. Frame counts are derived:
```python
opto_blank_frames = int(round(opto_blank_sec / frame_period))
```
This ensures correct behaviour across scanning rates (30 Hz, 7 Hz multi-ROI, etc.).

### VoltageRecording — always loaded

vrec is loaded for every experiment (hard error if CSV is missing). `_detect_vrec_events` runs on every channel and stores onsets in `vrec_channel_events`, keyed by column index. Named convenience variables (`stim_on_2p_frame`, `photostim_triggers_sec`) are derived from this.

### Opto % change images

Baseline is the mean of frames at recording **seconds 1–5** (hardcoded; avoids pre-trigger window computation):

```
[0s, 1s)  skipped   [1s, 5s)  opto_baseline   ...   [f+blank, f+blank+post)  response
```

`opto_delta_images = (post_avg − opto_baseline) / opto_baseline`

The 400 ms blanking window skips frames where the microscope shutter is closed during photostimulation.

### VoltageRecording channel layout

| Layout | Visual trigger col | Photostim col |
|--------|--------------------|---------------|
| Old    | 1                  | 2             |
| New    | 2                  | 3             |

**The layout is NOT auto-detected.** `detect_vrec_channel_layout` **hardcodes the NEW layout**
(`vis_ch = 2`, `opto_ch = 3`); all current acquisitions use it. Events are detected on *every*
column regardless (stored in `vrec_channel_events`); `vis_ch`/`opto_ch` only select the
named-variable outputs.

For the rare **OLD** recording (`vis=1`, `opto=2`), the channels must be hardcoded by hand in
`detect_vrec_channel_layout` (the "Visual stim is always Input 1" / "Photostim: hardcoded" lines).
To help spot these, when `is_2p_opto=True` the function prints a `[WARNING]` if the event counts
look like an OLD-layout session (the hardcoded photostim channel col 3 is near-silent while col 2
carries the photostim train) and names the line to edit.

---

## Output Dictionary — Complete Structure

```python
result = {

    # ── Identity ──────────────────────────────────────────────────────────
    'info': {
        'experiment_id': str,
        'data_dir':      str,
        'date':          str,    # from XML (MMDDYYYY)
        'file_num':      int,
        'output_dir':    str,
    },

    # ── Bruker acquisition metadata (from XML) ────────────────────────────
    'Bruker_Acq': {
        # TSeries XML
        'frame_period':      float,   # seconds per frame
        'optical_zoom':      float,
        'microns_per_pixel': float,
        'pixels_per_line':   int,
        'lines_per_frame':   int,
        'objective_lens':    str,
        'objective_mag':     float,
        'objective_na':      float,
        'laser_power':       dict,    # keyed by laser name
        'pmt_gain':          dict,
        'bit_depth':         int,
        # MarkPoints XML (only if is_2p_opto=True and XML found)
        'markpoints_xy_norm':            np.ndarray,  # (n_points, 2) normalized 0–1
        'markpoints_xy_pix':             np.ndarray,  # (n_points, 2) pixel coords
        'markpoints_condition_idx':      np.ndarray,  # (n_points,) group index
        'markpoints_laser_power':        np.ndarray,  # (n_groups,) mW
        'markpoints_spiral_diameter_px': np.ndarray,  # (n_groups,) pixels
    },

    # ── Processing parameters ─────────────────────────────────────────────
    'params': {
        'stim_file_num':      int,
        'is_spontaneous':     bool,   # derived: stim_file == -1
        'is_2p_opto':         bool,
        'use_inference':      bool,
        'do_neuropil':        bool,
        'do_plot':            bool,
        'do_vrec_diagnostic': bool,
        'dur_resp':           float,
        'opto_post_sec':      float,
        'opto_pre_sec':       float,
        'opto_blank_sec':     float,
        'opto_offset_trigger':bool,
        'chunk_size':         int,
        'skewness_threshold': float,  # dff_skewness cutoff for is_good_cell
        'dff_window_frames':  int,    # baseline window for dF/F computation
        'dff_window_sec':     float,
        # populated only if is_2p_opto=True:
        'opto_blank_frames':  int,
        'opto_post_frames':   int,
        'opto_pre_frames':    int,
    },

    # ── ROI metadata ──────────────────────────────────────────────────────
    'n_rois':      int,
    'is_soma':     np.ndarray,   # (n_rois,) bool
    'is_dendrite': np.ndarray,   # (n_rois,) bool
    'is_spine':    np.ndarray,   # (n_rois,) bool
    'mask2d':      np.ndarray,   # (n_rois, size_x, size_y)
    'dff_skewness':np.ndarray,   # (n_rois,) skewness of dff (dff_nan if opto)
    'is_good_cell':np.ndarray,   # (n_rois,) bool: dff_skewness >= skewness_threshold

    # ── Imaging ───────────────────────────────────────────────────────────
    'avg_image':      np.ndarray,   # (size_x, size_y) mean of first chunk_size frames
    'frame_times_sec':np.ndarray,   # (n_frames,) time axis in seconds

    # ── Fluorescence traces ───────────────────────────────────────────────
    'raw_traces':   np.ndarray,   # (n_frames, n_rois)
    'dff':          np.ndarray,   # (n_frames, n_rois)
    'dff_nan':      np.ndarray,   # (n_frames, n_rois) copy of dff with opto pulse
                                  # windows set to NaN; None unless is_2p_opto
    # only if do_neuropil=True:
    'raw_neuropil': np.ndarray,   # (n_frames,)
    'dff_neuropil': np.ndarray,   # (n_frames,)

    # ── VoltageRecording ──────────────────────────────────────────────────
    'vrec_sample_rate':    int,    # typically 10000 Hz
    'visual_trigger_ch':   int,    # vrec column index for visual triggers (None if absent)
    'photostim_ch':        int,    # vrec column index for photostim triggers (None if absent)
    'vrec_channel_events': dict,   # {col_idx: {'onsets': ndarray, 'onsets_sec': ndarray}}
                                   # populated for every non-time column

    # ── Timing ────────────────────────────────────────────────────────────
    'frame_triggers_sec':    np.ndarray,  # (n_frames,) 2P frame times in seconds (from XML)
    'stim_on_2p_frame':      np.ndarray,  # (n_events,) visual stim onset → 2P frame index
    'photostim_triggers_sec':np.ndarray,  # (n_triggers,) photostim times in seconds
    'photostim_2p_frame':    np.ndarray,  # (n_triggers,) photostim onset → 2P frame index

    # ── Stimulus / PsychoPy ───────────────────────────────────────────────
    # populated only if stim_file > -1:
    'stim_id':         np.ndarray,   # (n_trials,) stimulus identity
    'unique_stims':    np.ndarray,   # sorted unique stim IDs
    'stim_properties': np.ndarray,   # (n_trials, n_props) extra columns (non-opto)
    'target_number':   np.ndarray,   # (n_trials,) opto target number (opto only)
    'target_trial':    np.ndarray,   # (n_trials,) opto trial index (opto only)

    # ── Stimulus-aligned response analysis ───────────────────────────────
    # populated only if stim_file > -1 and stim_id is available:
    'cyc':          np.ndarray,   # (n_rois, n_stims, n_trials, resp_frames)
    'resp':         np.ndarray,   # (n_rois, n_stims) mean peak response
    'resps':        np.ndarray,   # (n_rois, n_stims, n_trials) per-trial peak
    'resp_err':     np.ndarray,   # (n_rois, n_stims) SEM

    # ── MarkPoints ROI mapping (opto only) ────────────────────────────────
    'roi_photostim_point':    np.ndarray,  # (n_rois,) markpoint index, -1=none
    'roi_photostim_group':    np.ndarray,  # (n_rois,) condition/group index, -1=none
    'roi_photostim_overlap':  np.ndarray,  # (n_rois,) fraction of ROI covered by spiral
    'markpoint_assigned_roi': np.ndarray,  # (n_points,) ROI index, -1=none

    # ── 2P opto images and responses (opto only) ─────────────────────────
    # populated only if is_2p_opto=True and opto_ch detected:
    'opto_unique_ids':    np.ndarray,  # sorted unique opto group IDs (= target_number)
    'opto_delta_images':  np.ndarray,  # (n_groups, size_x, size_y)
                                       # % change = (post_avg - baseline) / baseline
                                       # baseline = mean of recording seconds 1–5
    'opto_trial_delta_images': np.ndarray,  # (n_kept_trials, size_x, size_y) per-trial
                                       # % change, first 5 trials/group; also written as a
                                       # float32 TIFF stack ({experiment_id}_opto_trial_delta_images.tif)
    'cyc_photostim_only': np.ndarray,  # (n_rois, n_groups, max_trials) NaN-padded
                                       # value = mean(post) - mean(pre) dF/F per trial
}
```

---

## Verification

```python
# Quick sanity check after a full run
result = process_experiment(date='06042026', file_num=3, is_2p_opto=True, do_plot=True)

assert 'Bruker_Acq' in result and 'frame_period' in result['Bruker_Acq']
assert 'params' in result and 'is_spontaneous' in result['params']
assert 'info' in result and 'experiment_id' in result['info']

# Timing
fp = result['Bruker_Acq']['frame_period']
assert abs(result['frame_times_sec'][-1] - result['dff'].shape[0] * fp) < 1

# Opto
assert result['opto_delta_images'].ndim == 3   # (n_groups, H, W)
assert result['cyc_photostim_only'].ndim == 3  # (n_rois, n_groups, max_trials)

# H5 output
import h5py
with h5py.File('/mnt/bigdata/PROCESSED/TSeries-06042026-xxx-003.h5', 'r') as f:
    assert 'Bruker_Acq' in f
    assert 'params' in f
    assert 'info' in f
    assert 'dff' in f
```
