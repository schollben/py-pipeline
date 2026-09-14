# Bruker 2P Calcium Imaging Pipeline

Python pipeline for processing two-photon calcium imaging data acquired with a Bruker microscope. Handles spontaneous activity recordings, visually-evoked experiments, and 2-photon optogenetic photostimulation experiments.

---

## Environment Setup

Requires Python 3.12 via conda.

```bash
conda env create -f environment.yml
conda activate pypipeline312
```

To update your environment after pulling changes:

```bash
conda env update -f environment.yml --prune
```

### VS Code

The `.vscode/settings.json` is included and points the Python interpreter to the `pypipeline312` conda env automatically.

---

## Data Directory Structure

The pipeline expects data on a shared drive mounted at `/mnt/bigdata/`:

```
/mnt/bigdata/
├── BRUKER/                  # Raw Bruker acquisition folders
│   └── TSeries-MMDDYYYY-HHMM-NNN/
│       ├── TSeries-*.xml        # Acquisition metadata
│       ├── MarkPoints-*.xml     # 2P photostim targets (opto experiments)
│       ├── VRecording-*.xml     # Voltage recording channels
│       ├── inference_results.h5 # Suite2p/inference ROI traces (optional)
│       └── *.ome.tif            # Raw image data
├── BRUKER_PSYCHOPY/         # PsychoPy stimulus files (T001.txt, T002.txt, ...)
└── PROCESSED/               # Pipeline outputs written here
```

Preprocessing reads from and writes to `/mnt/bigdata/`. Analysis only needs the resulting
`PROCESSED/*.h5`, so those files are often copied elsewhere (e.g. a local Dropbox folder) and
`SimpleAnalysis_SingleSession.py` pointed at that path instead.

---

## Usage

Edit the parameters at the top of `run_pipeline.py` and run it:

```bash
python run_pipeline.py
```

### Parameters in `run_pipeline.py`

| Parameter | Type | Description |
|---|---|---|
| `DATE` | `str` | Acquisition date, format `MMDDYYYY` |
| `FILE_NUM` | `int` | TSeries folder number (e.g. `3` → folder ending in `-003`) |
| `STIM_FILE` | `int` | PsychoPy file index (e.g. `2` → `T002.txt`). Set to `-1` if no stimulus (spontaneous) — vrec events are still detected, only stimulus-aligned `cyc`/`resp` are skipped. |
| `USE_INFERENCE` | `bool` | `True` to load ROI traces from `inference_results.h5`, `False` to use `registered.h5` |
| `DUR_RESP` | `float` | Response window duration in seconds for trial-averaged matrix (`cyc`) |
| `PRE_RESP` | `float` | Pre-stimulus window in seconds prepended to each `cyc` trial (`0` = none) |
| `IS_2P_OPTO` | `bool` | `True` if experiment includes 2P photostimulation |
| `OPTO_POST_SEC` | `float` | Seconds after blanking window to average (response image) |
| `OPTO_PRE_SEC` | `float` | Seconds before trigger onset to average (baseline image) |
| `DO_PLOT` | `bool` | `True` to generate and save a summary figure |

Additional `process_experiment` keyword arguments not exposed as constants in `run_pipeline.py`:

| Argument | Default | Description |
|---|---|---|
| `do_neuropil` | `False` | Extract and subtract neuropil signal |
| `do_vrec_diagnostic` | `False` | Plot first 60 s of voltage recording channels with detected triggers |
| `opto_offset_trigger` | `True` | Compensates for known 1-row offset bug in photostim trigger stream — keep `True` until fixed in acquisition software |
| `chunk_size` | `1000` | Frames processed at once during trace extraction |
| `output_dir` | `PROCESSED/` | Directory for output H5 and figure files |
| `skewness_threshold` | `1.0` | Skewness cutoff used to select cells |
| `n_plot_cells` | `15` | Number of cells drawn in the summary figure |

### Calling directly from Python

```python
from bruker_pipeline import process_experiment

result = process_experiment(
    date           = '07132025',
    file_num       = 3,
    stim_file      = 2,
    is_2p_opto     = True,
    use_inference  = True,
    do_neuropil    = False,
    dur_resp       = 2,
    pre_resp       = 1,
    opto_post_sec  = 1,
    opto_pre_sec   = 0.5,
    do_plot        = True,
)
```

`process_experiment` returns a dict containing all processed variables (traces, trial matrices, opto images, etc.).

For 2P opto experiments the result dict contains two dF/F arrays:
- `result['dff']` — original dF/F, unmodified (no NaN values)
- `result['dff_nan']` — copy of dff with NaN inserted at opto pulse windows (± 1 frame fudge around each blanked period); used by downstream opto analyses (`cyc_photostim_only`, plots)

MarkPoints group metadata is stored under `result['Bruker_Acq']`:
- `result['Bruker_Acq']['markpoints_group_info']` — `(n_conditions, 4)` float array. One row per XML MarkPoints Group (condition). Columns: `[condition_idx, unique_group_id, n_targets, dispersion_um]`. Conditions whose target coordinates are 100% overlapping (e.g. an 80 mW group paired with a 0 mW sham) share the same `unique_group_id`. `dispersion_um` is the std of pairwise distances between target centres in µm (0 for a single-target group).

---

## Single-Session Analysis

Preprocessing writes one `.h5` per session into `PROCESSED/`. The `analysis/` package
loads that file and provides the interactive analysis steps. `SimpleAnalysis_SingleSession.py`
is the driver, written as `# %%` cells to be stepped through in VS Code.

```python
from analysis import load_session, rebuild_cyc, compute_responses

dat = load_session('/path/to/PROCESSED/TSeries-07132025-1042-003.h5')
```

Typical order of operations:

1. `load_session` → a `Session` object (`dat`); `plot_avg_rois` to check the ROI mask.
2. `check_event_alignment` / `dropFirstEvents` — detect and remove a spurious first TTL pair.
3. `rebuild_cyc(dat, preStim=, postStim=, offsetFrames=)` — rebuild the trial matrix from raw
   dF/F with wider windows. `offsetFrames` corrects residual event-timing lead/lag.
4. `plot_stim_traces` to read baseline/peak windows off the plot, then
   `compute_responses(dat, baseline=, peak=)`.
5. `plot_tuning_curves`, `compute_selectivity` (gDSI/gOSI), `plot_preference_maps`.
6. Photostim: `describe_photostim_groups`, `plot_photostim_group_heatmaps`,
   `plot_photostim_target_traces`.
7. Influence: `influence_grand`, `influence_by_stim`, `influence_bootstrap`, and the
   matching `plot_influence_maps` / `plot_influence_by_contrast`.

### Known timing issues

- Some sessions contain a spurious first TTL pair — hence step 2.
- Event timing can lead the stimulus by ~15 frames (PMT shutter appears to open *before*
  stimulus onset, which is not physically possible); correct with `offsetFrames` for now.
- The photostim trigger stream is offset by 1 row relative to the PsychoPy file
  (`opto_offset_trigger=True` in preprocessing).

---

## File Overview

| File | Description |
|---|---|
| `run_pipeline.py` | Preprocessing entry point — set parameters here and run |
| `bruker_pipeline.py` | Main pipeline logic (`process_experiment`) |
| `xml_utils.py` | Parses Bruker XML files (TSeries, MarkPoints, VRecording) |
| `img_utils.py` | Image and signal processing utilities |
| `SimpleAnalysis_SingleSession.py` | Interactive single-session analysis driver (`# %%` cells) |
| `analysis/session.py` | `Session` object, `load_session`, event alignment, `rebuild_cyc` |
| `analysis/responses.py` | `compute_responses`, `plot_stim_traces` |
| `analysis/rois.py` | Average image with ROI overlay |
| `analysis/tuning.py` | Direction tuning curves, double-Gaussian fit |
| `analysis/selectivity.py` | gDSI / gOSI |
| `analysis/maps.py` | Direction / orientation preference maps |
| `analysis/photostim.py` | Photostim groups, target traces, influence measures |
| `PIPELINE_overview.md` | Detailed description of pipeline internals |
| `check_traces.py`, `diagnose_timing.py` | Diagnostic helper scripts |
| `cleanup_raw_data.py`, `sample_stack_from_h5_script.py` | Raw-data maintenance utilities |
| `getDFFv2.py` | Legacy monolithic script (reference only) |
| `get_dff.py` | Legacy dF/F extraction (reference only) |
| `getDFF_extract_dFF_only.py` | Legacy dF/F extraction (reference only) |
| `environment.yml` | Conda environment definition |
