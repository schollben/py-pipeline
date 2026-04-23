# Bruker 2P Pipeline Refactor — Detailed Plan

## Context

The existing `getDFFv2.py` script processes calcium imaging data from a Bruker two-photon microscope but has several limitations:
- Key acquisition parameters (optical zoom, frame period) are hardcoded rather than read from the instrument XML
- No support for `is_spontaneous` experiments (no visual stimulus)
- MarkPoints photostimulation target coordinates are not extracted or plotted
- No per-stimulus-ID post-trigger image averaging for 2P opto experiments
- Output is written only to H5; no in-memory Python dictionary is returned
- All logic is in a single monolithic script

**Goal:** Refactor into a modular pipeline that reads all parameters from XML, handles spontaneous and opto experiments, generates 2P opto average images, and returns a complete Python dictionary of all processed variables.

---

## File Structure

```
py-pipeline/
├── bruker_pipeline.py       # NEW: main entry point, returns dict
├── xml_utils.py             # NEW: all XML parsing functions
├── img_utils.py             # KEEP: existing utilities (minimal changes)
├── getDFFv2.py              # KEEP: reference / legacy
├── get_dff.py               # KEEP: reference / legacy
└── getDFF_extract_dFF_only.py  # KEEP: legacy
```

---

## Usage

```python
from bruker_pipeline import process_experiment

result = process_experiment(
    data_dir       = '/mnt/bigdata/BRUKER/TSeries-04022026-1315-003',
    stim_file      = 3,           # PsychoPy T003.txt; -1 for none
    is_spontaneous = False,       # True if no visual stimulus
    is_2p_opto     = True,        # True if photostimulation present
    use_inference  = False,       # True = inference_results.h5
    do_neuropil    = False,
    dur_resp       = 2.5,         # response window (seconds)
    opto_post_sec  = 1.0,         # post-trigger averaging window (seconds)
    opto_pre_sec   = 0.5,         # pre-trigger baseline window (seconds)
    opto_blank_sec = 0.4,         # blanking period after trigger (seconds)
    do_plot        = True,        # generate and save summary figure
)
```

---

## Module 1: `xml_utils.py`

### `parse_tseries_xml(xml_path) -> dict`

Parses the main `{TSeries-name}.xml` file. Returns acquisition parameters
that were previously hardcoded:

| XML key | Dict key | Replaces |
|---------|----------|----------|
| `framePeriod` | `frame_period` | hardcoded `0.033` |
| `opticalZoom` | `optical_zoom` | hardcoded `12` |
| `micronsPerPixel` | `microns_per_pixel` | not present |
| `pixelsPerLine` | `pixels_per_line` | assumed 512 |
| `linesPerFrame` | `lines_per_frame` | assumed 512 |
| `objectiveLens` | `objective_lens` | not present |
| `objectiveLensMag` | `objective_mag` | not present |
| `objectiveLensNA` | `objective_na` | not present |
| `laserPower` | `laser_power` (dict) | not present |
| `pmtGain` | `pmt_gain` (dict) | not present |
| `bitDepth` | `bit_depth` | not present |

### `parse_markpoints_xml(xml_path) -> dict`

Parses `{TSeries-name}_Cycle#####_MarkPoints.xml`. Returns:
```python
{
    'iterations': int,
    'all_points_at_once': bool,
    'conditions': [
        {
            'uncaging_laser': str,
            'uncaging_laser_power': float,   # mW
            'repetitions': int,
            'galvo': {'initial_delay_ms', 'duration_ms', 'spiral_revolutions',
                      'spiral_size_microns'},
            'points': [{'index', 'x_norm', 'y_norm', 'z_norm', 'is_spiral',
                        'spiral_width_norm', 'spiral_height_norm'}]
        }
    ]
}
```
Coordinates are normalized 0–1. Pixel conversion:
```python
x_pix = x_norm * pixels_per_line
y_pix = y_norm * lines_per_frame
```

### `parse_vrec_xml(xml_path) -> dict`

Returns `sample_rate`, `n_samples`, `acquisition_time_ms`, and a list of
channel dicts (`index`, `name`, `gain`, `unit`, `enabled`).

### `find_experiment_files(data_dir) -> dict`

Scans the TSeries folder and returns paths for:
`tseries_xml`, `markpoints_xml`, `vrec_xml`, `vrec_csv`,
`registered_h5`, `inference_h5`, `roi_zip`.

---

## Module 2: `bruker_pipeline.py`

### Pipeline flow

```
1.   get_target_folders_v2()   → resolve TSeries folder from date + file_num
2.   find_experiment_files()   → locate all experiment files
3.   parse_tseries_xml()       → frame_period, optical_zoom, etc. from XML
     parse_vrec_xml()          → sample_rate, channel info
3b.  [if do_vrec_diagnostic]   → early exit: load vrec, detect triggers, plot, return
4.   parse_markpoints_xml()    → target X,Y coords (if is_2p_opto)
5.   roifile.roiread()         → ROI classification (soma / dendrite / spine)
6.   Open registered.h5 / inference_results.h5
                               → avg_image (first 100 frames), frame_times_sec
6b.  MarkPoints → pixel coords → markpoints_xy_pix, markpoints_condition_idx, etc.
     ROI–MarkPoint overlap     → roi_photostim_point/group/overlap, markpoint_assigned_roi
7.   gen_polyline_roi/in_polygon → mask2d, neuropil_mask
8.   Chunked trace extraction  → raw_traces, raw_neuropil
9.   filter_baseline_dF_comp() → dff, dff_neuropil, dff_window_frames, dff_window_sec
10.  BRANCH: is_spontaneous
     └─ False:
        read_xml_file()              → frame_triggers (in 10 kHz samples)
        genfromtxt_with_progress     → vrec
        detect_vrec_channel_layout() → visual_trigger_ch, photostim_ch, vrec_channel_layout
        find_peaks()                 → stim_on, stim_off (samples + seconds)
        PsychoPy file                → stim_id, unique_stims, stim_properties,
                                       target_number, target_trial (opto only)
        map stim onsets              → stim_on_2p_frame
11.  (if has_stim) gen_stim_cyc(), compute_peak_resp()
                               → cyc, resp, resps, resp_err, stim_avg_images
11b. (if is_2p_opto and spontaneous) load vrec for opto trigger detection
12.  (if is_2p_opto)
        detect photostim triggers    → photostim_triggers, photostim_triggers_sec
        map → photostim_2p_frame
        Pass 1: global pre-trigger baseline → opto_baseline_image
        Pass 2: per-stim-ID post averages   → opto_avg_images, opto_delta_images,
                                              opto_trial_delta_images (TIFF stacks saved)
        grand average                → opto_grand_avg_image, opto_grand_delta_image
        per-trial dF/F matrix        → cyc_photostim_only (n_rois × n_groups × max_trials)
13.  save_result_h5()          → write all arrays to {experiment_id}.h5
14.  (if do_plot) plot_experiment_summary() → {experiment_id}_summary.png
     (if is_2p_opto) plot_opto_images()     → {experiment_id}_opto_images.png
     (if do_vrec_diagnostic) plot_vrec_diagnostic() → {experiment_id}_vrec_diagnostic.png
15.  print_experiment_summary()
16.  return result dict
```

---

## Key Design Decisions

### Seconds-first timebase

All durations are specified in seconds. Frame counts are derived:
```python
opto_blank_frames = int(round(opto_blank_sec / frame_period))   # ~12 at 30 Hz, ~3 at 7 Hz
```
This ensures correct behaviour across different scanning rates (30 Hz, 7 Hz multi-ROI, etc.).

### Opto % change images

```
|← opto_pre_sec →|← blank (0.4s) →|← opto_post_sec →|
[f - pre, f)      [f, f+blank)      [f+blank, f+blank+post)
  ↑ baseline                          ↑ response (microscope back on)
```

`delta_img = (post_mean / pre_mean − 1) × 100`

The 400 ms blanking window skips frames where the microscope shutter is
closed during photostimulation.

### Voltage channel auto-detection

| Layout | Visual trigger | Photostim trigger |
|--------|---------------|-------------------|
| Old    | Channel 0      | Channel 1          |
| New    | Channel 1      | Channel 2          |

Detection uses the coefficient of variation of inter-event intervals: the
visual-trigger channel is more regular (lower CV) than the photostim channel.

---

## Output Dictionary — Complete Structure

```python
result = {
    # Identity
    'experiment_id': str,
    'data_dir': str,
    'date': str,               # from XML (MMDDYYYY)

    # Processing inputs (passed by user)
    'file_num': int,
    'stim_file_num': int,      # -1 = no stim file
    'is_spontaneous': bool,
    'is_2p_opto': bool,
    'use_inference': bool,
    'do_neuropil': bool,
    'do_plot': bool,
    'do_vrec_diagnostic': bool,
    'dur_resp': float,
    'opto_post_sec': float,
    'opto_pre_sec': float,
    'opto_blank_sec': float,
    'opto_offset_trigger': bool,
    'chunk_size': int,
    'output_dir': str,

    # Acquisition parameters (all from XML)
    'frame_period': float,
    'optical_zoom': float,
    'microns_per_pixel': float,
    'pixels_per_line': int,
    'lines_per_frame': int,
    'objective_lens': str,
    'objective_mag': float,
    'objective_na': float,
    'laser_power': dict,       # keyed by laser name
    'pmt_gain': dict,
    'bit_depth': int,

    # ROI data
    'n_rois': int,
    'is_dendrite': np.ndarray,     # (n_rois,) bool
    'is_spine': np.ndarray,        # (n_rois,) bool
    'is_soma': np.ndarray,         # (n_rois,) bool
    'mask2d': np.ndarray,          # (n_rois, size_x, size_y)
    'neuropil_mask': np.ndarray,   # (size_x, size_y)

    # Signals
    'raw_traces': np.ndarray,      # (n_frames, n_rois)
    'dff': np.ndarray,             # (n_frames, n_rois)
    'dff_window_frames': int,      # baseline window length in frames
    'dff_window_sec': float,       # baseline window length in seconds
    'raw_neuropil': np.ndarray,    # (n_frames,) — only if do_neuropil=True
    'dff_neuropil': np.ndarray,    # (n_frames,) — only if do_neuropil=True

    # Reference images
    'avg_image': np.ndarray,       # (size_x, size_y) mean of first 100 frames

    # Frame synchronization
    'frame_triggers': np.ndarray,  # (n_frames,) 2P frame onset times in vrec samples (10 kHz)
    'frame_times_sec': np.ndarray, # (n_frames,) frame onset times in seconds
    'vrec_sample_rate': int,       # typically 10000 Hz

    # Stimulus data (None if is_spontaneous or stim_file=-1)
    'stim_on': np.ndarray,         # (n_events,) vrec sample indices of visual stim onsets
    'stim_on_sec': np.ndarray,     # (n_events,) seconds
    'stim_off': np.ndarray,        # (n_events,) vrec sample indices of visual stim offsets
    'stim_off_sec': np.ndarray,    # (n_events,) seconds
    'stim_on_2p_frame': np.ndarray,# (n_events,) 2P frame index for each stim onset
    'stim_id': np.ndarray,         # (n_trials,) stimulus identity from PsychoPy
    'unique_stims': np.ndarray,    # sorted unique stim IDs
    'stim_properties': np.ndarray, # (n_trials, n_props) extra PsychoPy columns (non-opto only)
    'target_number': np.ndarray,   # (n_trials,) opto target number (2P opto only)
    'target_trial': np.ndarray,    # (n_trials,) opto trial index (2P opto only)
    'cyc': np.ndarray,             # (n_rois, n_stims, n_trials, resp_frames) trial-avg matrix
    'resp': np.ndarray,            # (n_rois, n_stims) mean peak response
    'resps': np.ndarray,           # (n_rois, n_stims, n_trials) per-trial peak response
    'resp_err': np.ndarray,        # (n_rois, n_stims) SEM of peak response
    'stim_avg_images': np.ndarray, # (n_stims, size_x, size_y) mean frame image per stim

    # 2P opto data (None if not is_2p_opto)
    'photostim_triggers': np.ndarray,       # (n_events,) vrec sample indices
    'photostim_triggers_sec': np.ndarray,   # (n_events,) seconds
    'photostim_2p_frame': np.ndarray,       # (n_events,) 2P frame index per trigger

    # MarkPoints targets (populated from XML regardless of vrec; None if no XML)
    'markpoints_xy_norm': np.ndarray,       # (n_points, 2) normalized 0–1 coords
    'markpoints_xy_pix': np.ndarray,        # (n_points, 2) pixel coords
    'markpoints_condition_idx': np.ndarray, # (n_points,) group index per point
    'markpoints_laser_power': np.ndarray,   # (n_conditions,) uncaging power in mW
    'markpoints_spiral_diameter_px': np.ndarray, # (n_conditions,) spiral diameter in pixels

    # ROI–MarkPoint overlap (winner-takes-all per point)
    'roi_photostim_point': np.ndarray,      # (n_rois,) markpoint index, -1=none
    'roi_photostim_group': np.ndarray,      # (n_rois,) condition/group index, -1=none
    'roi_photostim_overlap': np.ndarray,    # (n_rois,) fraction of ROI covered by spiral
    'markpoint_assigned_roi': np.ndarray,   # (n_points,) ROI index, -1=none

    # Opto images (None if not is_2p_opto)
    'opto_stim_ids': np.ndarray,            # unique stim IDs present in photostim triggers
    'opto_avg_images': np.ndarray,          # (n_ids, size_x, size_y) mean post-trigger image
    'opto_baseline_image': np.ndarray,      # (size_x, size_y) single global pre-trigger baseline
    'opto_delta_images': np.ndarray,        # (n_ids, size_x, size_y) % change = (post-baseline)/baseline
    'opto_trial_delta_images': list,        # list[ndarray (n_trials, size_x, size_y)] per stim_id
    'opto_n_trials': np.ndarray,            # (n_ids,) valid trial count per stim_id
    'opto_grand_avg_image': np.ndarray,     # (size_x, size_y) trial-weighted average post image
    'opto_grand_baseline_image': np.ndarray,# (size_x, size_y) same as opto_baseline_image
    'opto_grand_delta_image': np.ndarray,   # (size_x, size_y) grand-average % change image
    'opto_blank_sec': float,
    'opto_blank_frames': int,
    'opto_post_sec': float,
    'opto_post_frames': int,
    'opto_pre_sec': float,
    'opto_pre_frames': int,

    # Per-trial dF/F opto response matrix (None if not is_2p_opto)
    'cyc_photostim_only': np.ndarray,       # (n_rois, n_groups, max_trials) NaN-padded
                                             # value = mean(post) - mean(pre) dF/F per trial
}
```

---

## Verification

1. **XML parsing unit test:**
   ```python
   from xml_utils import parse_tseries_xml, parse_markpoints_xml
   p = parse_tseries_xml('/mnt/bigdata/BRUKER/TSeries-04022026-1315-003/'
                         'TSeries-04022026-1315-003.xml')
   assert abs(p['frame_period'] - 0.033474) < 1e-4
   assert p['optical_zoom'] == 2
   assert abs(p['microns_per_pixel'] - 1.11) < 0.01
   ```

2. **MarkPoints unit test:**
   ```python
   mp = parse_markpoints_xml('/mnt/bigdata/BRUKER/TSeries-04022026-1315-003/'
                              'TSeries-04022026-1315-003_Cycle00001_MarkPoints.xml')
   pts = mp['conditions'][0]['points']
   assert abs(pts[0]['x_norm'] - 0.731) < 0.01
   ```

3. **End-to-end:**
   ```python
   result = process_experiment(
       '/mnt/bigdata/BRUKER/TSeries-04022026-1315-003',
       stim_file=3, is_2p_opto=True, do_plot=True)
   assert result['opto_delta_images'].shape == (2, 512, 512)
   assert result['vrec_channel_layout'] in ('old', 'new')
   ```

4. **Seconds timebase sanity:**
   ```python
   assert abs(result['frame_times_sec'][-1] -
              result['dff'].shape[0] * result['frame_period']) < 1
   assert result['opto_blank_frames'] == int(round(0.4 / result['frame_period']))
   ```

5. **Regression against getDFFv2.py:**
   Run both on the same dataset and confirm `result['dff']` matches.
