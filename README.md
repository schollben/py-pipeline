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
| `STIM_FILE` | `int` | PsychoPy file index (e.g. `2` → `T002.txt`). Set to `-1` if no stimulus. |
| `IS_SPONTANEOUS` | `bool` | `True` if no visual stimulus (spontaneous activity only) |
| `USE_INFERENCE` | `bool` | `True` to load ROI traces from `inference_results.h5` |
| `DO_NEUROPIL` | `bool` | `True` to extract and subtract neuropil signal |
| `DUR_RESP` | `float` | Response window duration in seconds for trial-averaged matrix |
| `IS_2P_OPTO` | `bool` | `True` if experiment includes 2P photostimulation |
| `OPTO_POST_SEC` | `float` | Seconds after blanking window to average (response) |
| `OPTO_PRE_SEC` | `float` | Seconds before trigger onset to average (baseline) |
| `OPTO_BLANK_SEC` | `float` | Seconds immediately after trigger to exclude (shutter delay ~400 ms) |
| `DO_PLOT` | `bool` | `True` to generate and save a summary figure |
| `DO_VREC_DIAG` | `bool` | `True` to plot first 60 s of voltage recording channels with detected triggers |
| `OPTO_OFFSET` | `bool` | Compensates for known 1-row offset bug in photostim trigger stream — keep `True` until fixed in acquisition software |

### Calling directly from Python

```python
from bruker_pipeline import process_experiment

result = process_experiment(
    date           = '04022026',
    file_num       = 3,
    stim_file      = 2,
    is_spontaneous = False,
    is_2p_opto     = True,
    use_inference  = True,
    do_neuropil    = False,
    dur_resp       = 2,
    opto_post_sec  = 0.2,
    opto_pre_sec   = 0.2,
    opto_blank_sec = 0.5,
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

## File Overview

| File | Description |
|---|---|
| `run_pipeline.py` | Entry point — set parameters here and run |
| `bruker_pipeline.py` | Main pipeline logic (`process_experiment`) |
| `xml_utils.py` | Parses Bruker XML files (TSeries, MarkPoints, VRecording) |
| `img_utils.py` | Image and signal processing utilities |
| `getDFFv2.py` | Legacy monolithic script (reference only) |
| `get_dff.py` | Legacy dF/F extraction (reference only) |
| `getDFF_extract_dFF_only.py` | Legacy dF/F extraction (reference only) |
| `environment.yml` | Conda environment definition |
