# run_pipeline.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from bruker_pipeline import process_experiment

DATA_DIR = '/mnt/bigdata/BRUKER/TSeries-04022026-1315-003' # Full path to the TSeries folder for this experiment.
OUTPUT_DIR = '/mnt/processed/'
STIM_FILE = 3 # PsychoPy file, Set to -1 if there is no stimulus file.
IS_SPONTANEOUS = True # True  → experiment has NO visual stimulus (spontaneous activity only)
USE_INFERENCE = True # True  → use inference_results.h5
DO_NEUROPIL = False # True  → extract and subtract neuropil signal from cell traces.
DUR_RESP = 2.5 # Response window duration in seconds to build the trial-averaged response matrix (cyc)

IS_2P_OPTO = True # True  → experiment includes 2-photon photostimulation (optogenetics), Will read MarkPoints XML and compute opto % change images
OPTO_POST_SEC  = 0.5    # how long (sec) after the blanking window to average (response)
OPTO_PRE_SEC   = 0.5    # how long (sec) before trigger onset to average (baseline)
OPTO_BLANK_SEC = 0.4    # blanking period (sec) right after trigger (~400 ms shutter delay)

DO_PLOT = True # True → generate and save a summary figure (ROIs, MarkPoints, opto images).

if __name__ == '__main__':
    result = process_experiment(
        data_dir       = DATA_DIR,
        stim_file      = STIM_FILE,
        is_spontaneous = IS_SPONTANEOUS,
        is_2p_opto     = IS_2P_OPTO,
        use_inference  = USE_INFERENCE,
        do_neuropil    = DO_NEUROPIL,
        dur_resp       = DUR_RESP,
        opto_post_sec  = OPTO_POST_SEC,
        opto_pre_sec   = OPTO_PRE_SEC,
        opto_blank_sec = OPTO_BLANK_SEC,
        do_plot        = DO_PLOT,
        output_dir     = OUTPUT_DIR,
    )
