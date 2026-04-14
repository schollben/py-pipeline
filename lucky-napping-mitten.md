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

## New File Structure

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

## Module 1: `xml_utils.py` (new)

### `parse_tseries_xml(xml_path: str) -> dict`

Parses the main `{TSeries-name}.xml` file in the data folder. Returns a dict with all acquisition parameters that are currently hardcoded.

**Reads from `PVStateShard` elements:**

| XML key | Dict key | Replaces hardcoded |
|---------|----------|--------------------|
| `framePeriod` | `frame_period` | `0.033` |
| `opticalZoom` | `optical_zoom` | `12` |
| `micronsPerPixel` (XAxis) | `microns_per_pixel` | not present |
| `pixelsPerLine` | `pixels_per_line` | assumed 512 |
| `linesPerFrame` | `lines_per_frame` | assumed 512 |
| `objectiveLens` | `objective_lens` | not present |
| `objectiveLensMag` | `objective_mag` | not present |
| `objectiveLensNA` | `objective_na` | not present |
| `laserPower` (each index) | `laser_power` (dict by name) | not present |
| `pmtGain` (each index) | `pmt_gain` (list) | not present |
| `bitDepth` | `bit_depth` | not present |

Also extracts from the `<PVScan>` root:
- `date` → `acquisition_date`

**Implementation note:** The XML uses repeated `<PVStateValue>` and `<SubindexedValues>` patterns. Parse with `xml.etree.ElementTree`. For subindexed values (e.g., laser power per laser), collect into a dict keyed by the `description` attribute.

---

### `parse_markpoints_xml(xml_path: str) -> dict`

Parses `{TSeries-name}_Cycle#####_MarkPoints.xml`. Returns coordinates and stimulation parameters for all defined targets.

**Returns:**
```python
{
    'iterations': int,              # total photostim iterations
    'all_points_at_once': bool,
    'conditions': [                  # one entry per PVMarkPointElement
        {
            'uncaging_laser': str,       # e.g. "Monaco"
            'uncaging_laser_power': float,
            'repetitions': int,
            'trigger_frequency': str,    # e.g. "FirstRepetition"
            'galvo': {
                'initial_delay_ms': float,
                'duration_ms': float,
                'spiral_revolutions': float,
                'spiral_size_microns': float,
            },
            'points': [
                {
                    'index': int,
                    'x_norm': float,   # normalized 0-1
                    'y_norm': float,
                    'z_norm': float,
                    'is_spiral': bool,
                }
            ]
        }
    ]
}
```

**Pixel coordinate conversion** (done in pipeline, not xml_utils):
```python
x_pix = x_norm * pixels_per_line
y_pix = y_norm * lines_per_frame
```

---

### `parse_vrec_xml(xml_path: str) -> dict`

Parses `{TSeries-name}_Cycle#####_VoltageRecording_001.xml`.

**Returns:**
```python
{
    'sample_rate': int,         # Hz (e.g. 10000)
    'n_samples': int,
    'acquisition_time_ms': float,
    'channels': [
        {'index': int, 'name': str, 'gain': float, 'unit': str, 'enabled': bool}
    ]
}
```

---

### `find_experiment_files(data_dir: str) -> dict`

Scans the TSeries folder and returns a dict of all relevant file paths:
```python
{
    'tseries_xml': str,
    'markpoints_xml': str or None,    # None if not 2P opto
    'vrec_xml': str or None,
    'vrec_csv': str or None,
    'registered_h5': str or None,
    'inference_h5': str or None,
    'roi_zip': str or None,
}
```

---

## Module 2: `bruker_pipeline.py` (new main)

### Main function signature

```python
def process_experiment(
    data_dir: str,
    stim_file: int = -1,             # T-file number in BRUKER_PSYCHOPY; -1 = no stim file
    is_spontaneous: bool = False,    # True = no visual stimulus triggers
    is_2p_opto: bool = False,        # True = has photostimulation
    use_inference: bool = False,     # True = use cascade inference_results.h5
    do_neuropil: bool = False,       # True = extract neuropil signal
    dur_resp: float = 2.5,           # response window duration (seconds)
    opto_post_sec: float = 1.0,      # seconds to average after blanking window
    opto_pre_sec: float = 0.5,       # seconds to average before trigger (baseline)
    opto_blank_sec: float = 0.4,     # blanking period after trigger onset (400ms)
    do_plot: bool = False,           # True = generate and save diagnostic figures
    chunk_size: int = 1000,          # memory chunk for trace extraction
    output_dir: str = '/mnt/processed/',
) -> dict:
```

### Pipeline flow

```
1. Resolve paths
   └── find_experiment_files(data_dir)

2. Parse XML metadata (ALWAYS)
   ├── parse_tseries_xml() → frame_period, optical_zoom, microns_per_pixel, etc.
   └── parse_vrec_xml()    → sample_rate, enabled channels

3. Parse MarkPoints XML (if is_2p_opto)
   └── parse_markpoints_xml() → target X,Y coordinates, laser power per condition

4. Load ROIs
   └── existing roifile logic from getDFFv2.py:120-71
       → is_dendrite, is_spine, is_soma arrays

5. Open calcium movie
   └── registered.h5 or inference_results.h5 (getDFFv2.py:73-88)
       → num_frames, size_x, size_y
       → avg_image (mean of first 100 frames)

6. Build cell masks
   └── existing gen_polyline_roi() / in_polygon() from img_utils.py
       → mask2d (n_cells, size_x, size_y)
       → neuropil_mask (size_x, size_y)

7. Extract raw traces  
   └── chunked loop (getDFFv2.py:113-129)
       → raw_traces (n_frames, n_cells)
       → raw_neuropil (n_frames,) if do_neuropil

8. Compute dF/F
   └── filter_baseline_dF_comp() from img_utils.py
       → dff (n_frames, n_cells)
       → dff_neuropil if do_neuropil

9. BRANCH: is_spontaneous
   ├── False → process stimulus triggers (step 10)
   └── True  → skip to step 12

10. Stimulus processing (if not is_spontaneous)
    ├── Read frame triggers from XML → frame_triggers (in vrec samples)
    ├── Load VoltageRecording CSV → vrec
    ├── Auto-detect channel layout → detect_vrec_channel_layout()
    │       → visual_trigger_ch, photostim_ch
    ├── Detect stim_on / stim_off from visual_trigger_ch
    ├── Convert all timing to seconds: stim_on_sec = stim_on / vrec_sample_rate
    ├── Map stim_on_sec → 2P frame index → stim_on_2p_frame
    └── If stim_file > -1: read PsychoPy file → stim_id, target_number, target_trial

11. Response analysis (if not is_spontaneous)
    ├── dur_resp_frames = int(round(dur_resp / frame_period))
    ├── gen_stim_cyc() with dur_resp_frames → cyc (n_cells, n_unique_stims, n_trials, resp_frames)
    ├── compute_peak_resp() → resp, resps, resp_err
    └── Per-stimulus average images → stim_avg_images (n_unique_stims, size_x, size_y)

12. 2P opto processing (if is_2p_opto)
    ├── Detect photostim triggers from photostim_ch → photostim_on_sec (seconds)
    ├── Map photostim_on_sec → 2P frame index → photostim_2p_frame
    ├── Convert markpoint coords to pixels: x_pix = x_norm * pixels_per_line
    ├── Read stim_file → stim_id per photostim event
    ├── Compute blank/pre/post frame counts from seconds + frame_period
    └── Generate opto % change average images per stim_id (see below)

13. Visualization (if do_plot=True)
    └── plot_experiment_summary(result) → saves PNG

14. Print experiment summary (always)
15. Compile output dict
16. Save to H5 (optional)
17. Return dict
```

---

## New Feature: 2P Opto Average Images

### Timebase: seconds throughout

All timing logic uses **seconds** as the primary unit. Frame indices are derived on-the-fly using `frame_period` (read from XML):

```python
frame_period = xml_params['frame_period']   # e.g. 0.033474 s at ~30 Hz, ~0.143 s at ~7 Hz

# Convert seconds → frame count
opto_blank_frames  = int(round(opto_blank_sec  / frame_period))   # ~12 at 30 Hz
opto_pre_frames    = int(round(opto_pre_sec    / frame_period))   # ~15 at 30 Hz
opto_post_frames   = int(round(opto_post_sec   / frame_period))   # ~30 at 30 Hz
dur_resp_frames    = int(round(dur_resp        / frame_period))   # ~75 at 30 Hz
```

This means the same parameter values (e.g. `opto_post_sec=1.0`) produce correct behavior whether the microscope is running at 30 Hz or 7 Hz — the frame count adjusts automatically.

All outputs stored in the dict include **both** the seconds parameter and the derived frame count so callers always know what was used.

---

### Logic

For each photostimulation trigger event `i` at frame `f_i`:

```
|<-- opto_pre_sec -->|<-- blank (0.4s) -->|<-- opto_post_sec -->|
[f_i - pre_frames,   f_i)                 [f_i + blank_frames,   f_i + blank_frames + post_frames)
   ↑ baseline window                             ↑ response window (microscope back on)
```

- **Baseline window:** `[f_i - opto_pre_frames, f_i)` — frames immediately before trigger
- **Blanking offset:** skip `opto_blank_frames` frames after trigger (microscope shutter effect)
- **Response window:** `[f_i + opto_blank_frames, f_i + opto_blank_frames + opto_post_frames)`

Group events by `stim_id`. For each unique stim_id:
1. Stack response windows → `(n_trials, opto_post_frames, size_x, size_y)`
2. Stack baseline windows → `(n_trials, opto_pre_frames, size_x, size_y)`
3. Trial-average both → `(opto_post_frames, size_x, size_y)` and `(opto_pre_frames, size_x, size_y)`
4. Collapse time axis (mean) → `post_img (size_x, size_y)` and `baseline_img (size_x, size_y)`
5. Compute **% change**: `delta_img = (post_img / baseline_img - 1) * 100`

**Store in output dict:**
```python
'opto_avg_images': np.ndarray,         # shape (n_unique_stim_ids, size_x, size_y) — post-trigger mean
'opto_baseline_images': np.ndarray,    # shape (n_unique_stim_ids, size_x, size_y) — pre-trigger mean
'opto_delta_images': np.ndarray,       # shape (n_unique_stim_ids, size_x, size_y) — % change
'opto_stim_ids': np.ndarray,           # stim_id labels for each image slice
'opto_n_trials': np.ndarray,           # number of trials averaged per stim_id
'opto_blank_sec': float,               # blanking period used
'opto_blank_frames': int,              # derived frame count
'opto_post_sec': float,
'opto_post_frames': int,
'opto_pre_sec': float,
'opto_pre_frames': int,
```

**Implementation notes:**
- Read frames directly from `registered.h5` or `inference_results.h5` (already open)
- Use photostim trigger indices mapped to 2P frame numbers
- Guard against edge triggers (events too close to start/end)
- Reuses the H5 file handle already open in step 5

---

## New Feature: Visualization (`do_plot=True`)

When `do_plot=True`, the pipeline calls `plot_experiment_summary(result)` which produces a multi-panel figure and saves it as `{experiment_id}_summary.png` in `output_dir`.

### Panel 1 — ROI Map (always shown)

- Background: `avg_image` (first 100 frames mean, displayed in grayscale)
- Overlaid: filled or outlined ROI masks from `mask2d`, color-coded by type:
  - Somas: one color
  - Dendrites: another
  - Spines: another
- Scale bar derived from `microns_per_pixel`

### Panel 2 — ROI Map + MarkPoints (only if `is_2p_opto=True`)

- Same background and ROI overlay as Panel 1
- Additionally overlays photostimulation target locations as scatter points
- Color-coded by **MarkPoints group/condition index** (each PVMarkPointElement is a group)
- Marker size scaled by spiral diameter (in pixels)
- Legend shows laser power per group

### Panel 3 — Post-photostim % change images (only if `is_2p_opto=True`)

- One subplot per unique `stim_id`
- Displays `opto_delta_images[i]` (% change image)
- Colormap: diverging (e.g. `RdBu_r`), centered at 0%
- Title: `stim_id={id}, n={n_trials} trials`
- Common colorbar across all subplots

**MarkPoints data in dict (for plotting):**
```python
'markpoints_xy_norm': np.ndarray,        # (n_points, 2) normalized 0-1
'markpoints_xy_pix': np.ndarray,         # (n_points, 2) pixel coordinates
'markpoints_condition_idx': np.ndarray,  # (n_points,) which group each point belongs to
'markpoints_laser_power': np.ndarray,    # (n_conditions,) mW per condition
```

---

## Output Dictionary — Complete Structure

```python
result = {
    # ── Identity ──────────────────────────────────────────────────
    'experiment_id': str,          # e.g. "TSeries-04022026-1315-003"
    'data_dir': str,
    'date': str,                   # from XML acquisition_date
    
    # ── Acquisition parameters (from XML, not hardcoded) ──────────
    'frame_period': float,         # seconds
    'optical_zoom': float,
    'microns_per_pixel': float,
    'pixels_per_line': int,
    'lines_per_frame': int,
    'objective_lens': str,
    'objective_mag': float,
    'objective_na': float,
    'laser_power': dict,           # keyed by laser name
    'pmt_gain': list,
    'bit_depth': int,
    
    # ── Flags ─────────────────────────────────────────────────────
    'is_spontaneous': bool,
    'is_2p_opto': bool,
    'do_neuropil': bool,
    'do_plot': bool,
    
    # ── ROI data ──────────────────────────────────────────────────
    'n_rois': int,
    'is_dendrite': np.ndarray,     # (n_rois,) bool
    'is_spine': np.ndarray,        # (n_rois,) bool
    'is_soma': np.ndarray,         # (n_rois,) bool
    'mask2d': np.ndarray,          # (n_rois, size_x, size_y)
    'neuropil_mask': np.ndarray,   # (size_x, size_y)
    
    # ── Signals ───────────────────────────────────────────────────
    'raw_traces': np.ndarray,      # (n_frames, n_rois)
    'dff': np.ndarray,             # (n_frames, n_rois)
    'raw_neuropil': np.ndarray,    # (n_frames,) — optional
    'dff_neuropil': np.ndarray,    # (n_frames,) — optional
    
    # ── Reference images ──────────────────────────────────────────
    'avg_image': np.ndarray,       # (size_x, size_y) mean of first 100 frames
    
    # ── Frame synchronization ─────────────────────────────────────
    'frame_triggers': np.ndarray,  # (n_frames,) in vrec samples
    'frame_times_sec': np.ndarray, # (n_frames,) in seconds
    'vrec': np.ndarray,            # (n_samples, n_channels) voltage recording
    'vrec_sample_rate': int,       # Hz
    'vrec_channel_layout': str,    # 'old' (ch0=vis, ch1=opto) or 'new' (ch1=vis, ch2=opto)
    'visual_trigger_ch': int,      # which vrec column has visual triggers
    'photostim_ch': int,           # which vrec column has photostim triggers
    
    # ── Stimulus data (populated if not is_spontaneous) ───────────
    'stim_on': np.ndarray,         # (n_events,) onset indices in vrec samples
    'stim_on_sec': np.ndarray,     # (n_events,) onset times in seconds
    'stim_off': np.ndarray,        # (n_events,) offset indices in vrec samples
    'stim_off_sec': np.ndarray,    # (n_events,) offset times in seconds
    'stim_on_2p_frame': np.ndarray,# (n_events,) onset in 2P frame indices
    'stim_id': np.ndarray,         # (n_events,) which stimulus
    'unique_stims': np.ndarray,    # (n_unique_stims,)
    'cyc': np.ndarray,             # (n_rois, n_unique_stims, n_trials, resp_frames)
    'resp': np.ndarray,            # (n_rois, n_unique_stims)
    'resps': np.ndarray,           # (n_rois, n_unique_stims, n_trials)
    'resp_err': np.ndarray,        # (n_rois, n_unique_stims)
    'stim_avg_images': np.ndarray, # (n_unique_stims, size_x, size_y)
    
    # ── 2P opto data (populated if is_2p_opto) ────────────────────
    'photostim_triggers': np.ndarray,      # (n_events,) in vrec samples
    'photostim_triggers_sec': np.ndarray,  # (n_events,) in seconds
    'photostim_2p_frame': np.ndarray,      # (n_events,) mapped to frame indices
    'markpoints_xy_norm': np.ndarray,      # (n_points, 2)
    'markpoints_xy_pix': np.ndarray,       # (n_points, 2)
    'markpoints_condition_idx': np.ndarray,# (n_points,)
    'markpoints_laser_power': np.ndarray,  # (n_conditions,)
    'opto_stim_ids': np.ndarray,           # (n_unique_stim_ids,)
    'opto_avg_images': np.ndarray,         # (n_unique_stim_ids, size_x, size_y)
    'opto_baseline_images': np.ndarray,    # (n_unique_stim_ids, size_x, size_y)
    'opto_delta_images': np.ndarray,       # (n_unique_stim_ids, size_x, size_y)
    'opto_n_trials': np.ndarray,           # (n_unique_stim_ids,)
    'target_number': np.ndarray,           # (n_events,)
    'target_trial': np.ndarray,            # (n_events,)
}
```

---

## New Feature: Experiment Summary Print

At the end of processing (before returning the dict), print a concise summary to stdout so the user can immediately sanity-check the experiment. Structured as labeled sections so issues are obvious at a glance:

```
========================================
 EXPERIMENT SUMMARY
========================================
Experiment : TSeries-04022026-1315-003
Date       : 4/2/2026
Data dir   : /mnt/bigdata/BRUKER/TSeries-04022026-1315-003

── Acquisition ──────────────────────────
  Frame period  : 0.03347 s  (29.9 Hz)
  Total frames  : 14540
  Duration      : 487.1 s
  Frame size    : 512 x 512 px  (1.11 µm/px)
  Optical zoom  : 2x
  Objective     : 16X Nikon  NA=0.8

── ROIs ─────────────────────────────────
  Total ROIs    : 42
    Somas       : 35
    Dendrites   : 5
    Spines      : 2

── Stimulus ─────────────────────────────
  is_spontaneous: False
  Stim file     : T003
  Vrec layout   : new  (vis=ch1, opto=ch2)
  Stim triggers : 120  (60 unique stim IDs)
  [WARNING: expected ~120 triggers based on stim file, got 120 ✓]

── 2P Optogenetics ───────────────────────
  is_2p_opto    : True
  Photostim groups   : 2
  Photostim triggers : 400
  Unique stim IDs    : 2
  Trials per ID      : [200, 200]
  Markpoints (group 1): 3 targets  (power=75 mW)
  Markpoints (group 2): 3 targets  (power=0 mW)
========================================
```

**Implementation:** `print_experiment_summary(result)` — reads from the completed dict. Called unconditionally. Warnings printed if:
- Number of detected stim triggers doesn't match stim_file row count
- Number of photostim triggers is not divisible by the number of stim groups
- Edge triggers were skipped during opto image averaging

---

## New Feature: Voltage Channel Auto-Detection

The voltage recording CSV has either 3 enabled channels. The assignment changed at some point in the recording setup:

| Layout | Visual trigger | Photostim trigger |
|--------|---------------|-------------------|
| Old    | Channel 0      | Channel 1          |
| New    | Channel 1      | Channel 2          |

### Detection strategy: `detect_vrec_channel_layout(vrec, vrec_xml_params) -> dict`

After loading the full voltage recording array:

1. Extract the enabled-channel list from `parse_vrec_xml()` (already in step 2)
2. For each candidate visual-trigger channel (ch0 and ch1):
   - Count threshold crossings (e.g. > 2V, using same logic as existing stim detection)
   - The visual trigger channel will have **regularly-spaced** pulses matching the expected stimulus count; the photostim channel will have clustered bursts
3. Heuristic: the visual trigger channel has a **higher inter-event interval variance ratio** (photostim triggers are clustered in bursts, visual triggers are more evenly spaced across the experiment)
4. If auto-detection is ambiguous, fall back to the "new" layout (ch1=visual, ch2=photostim) and emit a warning

**Returns:**
```python
{
    'visual_trigger_ch': int,    # 0 or 1
    'photostim_ch': int,         # 1 or 2 (or None if not is_2p_opto)
    'layout': str,               # 'old' | 'new' | 'ambiguous'
}
```

Store in output dict:
```python
'vrec_channel_layout': str,      # 'old' or 'new'
'visual_trigger_ch': int,
'photostim_ch': int,
```

---

## Changes to `img_utils.py`

Minimal changes — keep all existing functions. Add:

### `read_tseries_xml(xml_path)` → move here or keep in xml_utils.py
The existing `read_xml_file()` only reads frame triggers. The new `parse_tseries_xml()` is separate and goes in `xml_utils.py`.

---

## Handling `is_spontaneous`

```
is_spontaneous=True:
  - Skip steps 10 and 11 entirely
  - Skip VoltageRecording stim onset detection
  - stim_on, stim_off, stim_on_2p_frame, etc. → set to None or empty arrays
  - Still extract frame_triggers (needed for 2P opto sync if is_2p_opto=True)
  - Still process 2P opto triggers if is_2p_opto=True
  
is_spontaneous=False (default):
  - Requires stim_file >= 0 OR detectable stimulus onsets in vrec
  - Full stimulus cycle / response analysis
```

---

## Critical Files to Modify / Create

| File | Action | Key sections |
|------|--------|-------------|
| `xml_utils.py` | **CREATE** | All XML parsing functions |
| `bruker_pipeline.py` | **CREATE** | Main pipeline function, output dict |
| `img_utils.py` | **KEEP, minor additions** | Possibly add `find_experiment_files()` |
| `getDFFv2.py` | **KEEP for reference** | Source of logic to port |

---

## Reuse from Existing Code

These functions from `img_utils.py` should be called unchanged:

| Function | Used for |
|----------|----------|
| `gen_polyline_roi()` | Dendrite masks |
| `in_polygon()` | Soma/spine masks |
| `filter_baseline_dF_comp()` | dF/F computation |
| `read_xml_file()` | Frame trigger extraction |
| `replace_missing_frame_triggers()` | Frame trigger interpolation |
| `genfromtxt_with_progress()` | Loading voltage recording CSV |
| `neuropil_subtraction()` | Neuropil correction |
| `gen_stim_cyc()` | Response window extraction |
| `compute_peak_resp()` | F1/DC response amplitude |
| `get_target_folders_v2()` | Folder path resolution |

---

## Verification Plan

1. **Unit test XML parsing:**
   - Run `parse_tseries_xml()` on `/mnt/bigdata/BRUKER/TSeries-04022026-1315-003/TSeries-04022026-1315-003.xml`
   - Verify: `frame_period ≈ 0.033474`, `optical_zoom = 2`, `microns_per_pixel ≈ 1.11`
   - Run `parse_markpoints_xml()` on the MarkPoints XML
   - Verify: 3 points with X ≈ [0.731, 0.709, 0.248], Y ≈ [0.396, 0.648, 0.895]

2. **Voltage channel auto-detection:**
   - Load vrec CSV from example experiment
   - Run `detect_vrec_channel_layout()`, confirm it returns the correct layout
   - Verify `stim_on_sec` values are reasonable (evenly-spaced, ~30s ISI expected from PsychoPy script)

3. **Seconds timebase sanity check:**
   - Confirm `frame_times_sec[-1] ≈ num_frames * frame_period`
   - Confirm `opto_blank_frames == int(round(0.4 / frame_period))` ≈ 12 at 30 Hz
   - Simulate 7 Hz recording (frame_period ≈ 0.143): confirm `opto_blank_frames ≈ 3`

4. **End-to-end on example experiment:**
   - Call `process_experiment('/mnt/bigdata/BRUKER/TSeries-04022026-1315-003', stim_file=3, is_2p_opto=True, do_plot=True)`
   - Confirm returned dict has all expected keys
   - Confirm `opto_delta_images.shape == (n_stim_ids, 512, 512)`
   - Confirm `opto_delta_images` contains % change values (not raw or subtracted)
   - Open saved PNG and visually verify: MarkPoints land on expected FOV locations, % change images show plausible response

5. **Regression check with is_spontaneous=True:**
   - Call with `is_spontaneous=True, is_2p_opto=True`
   - Confirm `stim_on` is None/empty, opto images still generated

6. **Compare dF/F output to getDFFv2.py:**
   - Run both on same dataset
   - Confirm `result['dff']` matches output from old script
