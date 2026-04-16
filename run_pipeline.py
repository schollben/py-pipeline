# run_pipeline.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from bruker_pipeline import process_experiment


DATE            = '04022026'    # acquisition date, format MMDDYYYY
FILE_NUM        = 3             # TSeries number (e.g. 3 → matches folder ending in -003)
STIM_FILE       = 2             # PsychoPy file, Set to -1 if there is no stimulus file.
IS_SPONTANEOUS  = True          # True  → experiment has NO visual stimulus (spontaneous activity only)
USE_INFERENCE   = True          # True  → use inference_results.h5
DO_NEUROPIL     = False         # True  → extract and subtract neuropil signal from cell traces.
DUR_RESP        = 2             # Response window duration in seconds to build the trial-averaged response matrix (cyc)

IS_2P_OPTO      = True          # True  → experiment includes 2-photon photostimulation (optogenetics), Will read MarkPoints XML and compute opto % change images
OPTO_POST_SEC   = 0.2           # seconds after the blanking window to average (response)
OPTO_PRE_SEC    = 0.2           # seconds before trigger onset to average (baseline)
OPTO_BLANK_SEC  = 0.4           # seconds right after trigger (~400 ms shutter delay)

DO_PLOT         = True          # True → generate and save a summary figure (ROIs, MarkPoints, opto images).
DO_VREC_DIAG    = False         # True → plot first 60 s of all vrec channels with detected triggers marked.

# Known acquisition bug: photostim trigger stream is offset by 1 row relative
# to the PsychoPy file. Keep True until the bug is fixed in the acquisition software.
OPTO_OFFSET     = True

if __name__ == '__main__':
    result = process_experiment(
        date                 = DATE,
        file_num             = FILE_NUM,
        stim_file            = STIM_FILE,
        is_spontaneous       = IS_SPONTANEOUS,
        is_2p_opto           = IS_2P_OPTO,
        use_inference        = USE_INFERENCE,
        do_neuropil          = DO_NEUROPIL,
        dur_resp             = DUR_RESP,
        opto_post_sec        = OPTO_POST_SEC,
        opto_pre_sec         = OPTO_PRE_SEC,
        opto_blank_sec       = OPTO_BLANK_SEC,
        do_plot              = DO_PLOT,
        do_vrec_diagnostic   = DO_VREC_DIAG,
        opto_offset_trigger  = OPTO_OFFSET,
    )
