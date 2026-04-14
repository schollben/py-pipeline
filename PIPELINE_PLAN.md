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
1.  find_experiment_files()
2.  parse_tseries_xml()         → frame_period, optical_zoom, etc. from XML
    parse_vrec_xml()            → sample_rate, channel info
3.  parse_markpoints_xml()      → target X,Y coords (if is_2p_opto)
4.  roifile.roiread()           → ROI classification
5.  Open registered.h5 / inference_results.h5
                                → avg_image (first 100 frames), frame_times_sec
6.  gen_polyline_roi/in_polygon → mask2d, neuropil_mask
7.  Chunked trace extraction    → raw_traces, raw_neuropil
8.  filter_baseline_dF_comp()  → dff, dff_neuropil
9.  BRANCH: is_spontaneous
    └─ False:
       read_xml_file()          → frame_triggers (in 10 kHz samples)
       genfromtxt_with_progress → vrec
       detect_vrec_channel_layout() → visual_trigger_ch, photostim_ch
       find_peaks()             → stim_on, stim_off (samples + seconds)
       PsychoPy file            → stim_id, target_number, target_trial
       map stim onsets → stim_on_2p_frame
10. (if has_stim) gen_stim_cyc(), compute_peak_resp(), stim_avg_images
11. (if is_2p_opto)
       detect photostim triggers → photostim_triggers_sec
       map → photostim_2p_frame
       convert markpoints → xy_pix
       compute opto % change images per stim_id (with blank offset)
12. (if do_plot) plot_experiment_summary() → saves PNG
13. print_experiment_summary()
14. return result dict
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
    'date': str,               # from XML

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

    # Flags
    'is_spontaneous': bool,
    'is_2p_opto': bool,
    'do_neuropil': bool,
    'do_plot': bool,

    # ROI data
    'n_rois': int,
    'is_dendrite': np.ndarray,     # (n_rois,) bool
    'is_spine': np.ndarray,
    'is_soma': np.ndarray,
    'mask2d': np.ndarray,          # (n_rois, size_x, size_y)
    'neuropil_mask': np.ndarray,   # (size_x, size_y)

    # Signals
    'raw_traces': np.ndarray,      # (n_frames, n_rois)
    'dff': np.ndarray,             # (n_frames, n_rois)
    'raw_neuropil': np.ndarray,    # (n_frames,) — optional
    'dff_neuropil': np.ndarray,    # (n_frames,) — optional

    # Reference images
    'avg_image': np.ndarray,       # (size_x, size_y)

    # Frame synchronization
    'frame_triggers': np.ndarray,  # (n_frames,) in vrec samples
    'frame_times_sec': np.ndarray, # (n_frames,) in seconds
    'vrec': np.ndarray,            # (n_samples, n_channels)
    'vrec_sample_rate': int,
    'vrec_channel_layout': str,    # 'old' or 'new'
    'visual_trigger_ch': int,
    'photostim_ch': int,

    # Stimulus data (None if is_spontaneous)
    'stim_on': np.ndarray,         # (n_events,) vrec sample indices
    'stim_on_sec': np.ndarray,     # (n_events,) seconds
    'stim_off': np.ndarray,
    'stim_off_sec': np.ndarray,
    'stim_on_2p_frame': np.ndarray,
    'stim_id': np.ndarray,
    'unique_stims': np.ndarray,
    'cyc': np.ndarray,             # (n_rois, n_stims, n_trials, resp_frames)
    'resp': np.ndarray,            # (n_rois, n_stims)
    'resps': np.ndarray,
    'resp_err': np.ndarray,
    'stim_avg_images': np.ndarray, # (n_stims, size_x, size_y)

    # 2P opto data (None if not is_2p_opto)
    'photostim_triggers': np.ndarray,      # (n_events,) vrec samples
    'photostim_triggers_sec': np.ndarray,
    'photostim_2p_frame': np.ndarray,
    'markpoints_xy_norm': np.ndarray,      # (n_points, 2) normalized 0-1
    'markpoints_xy_pix': np.ndarray,       # (n_points, 2) pixel coords
    'markpoints_condition_idx': np.ndarray,# (n_points,)
    'markpoints_laser_power': np.ndarray,  # (n_conditions,) mW
    'opto_stim_ids': np.ndarray,
    'opto_avg_images': np.ndarray,         # (n_ids, size_x, size_y) post-trigger mean
    'opto_baseline_images': np.ndarray,    # (n_ids, size_x, size_y) pre-trigger mean
    'opto_delta_images': np.ndarray,       # (n_ids, size_x, size_y) % change
    'opto_n_trials': np.ndarray,
    'opto_blank_sec': float,
    'opto_blank_frames': int,
    'opto_post_sec': float,
    'opto_post_frames': int,
    'opto_pre_sec': float,
    'opto_pre_frames': int,
    'target_number': np.ndarray,
    'target_trial': np.ndarray,
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
