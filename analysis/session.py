import os
from dataclasses import dataclass, field
import numpy as np
import h5py

_ARRAYS = ['avg_image', 'mask2d', 'dff', 'unique_stims', 'stim_id',
           'stim_properties', 'stim_on_2p_frame', 'cyc', 'resp', 'resp_err',
           'resps', 'is_soma', 'is_dendrite', 'is_spine', 'is_good_cell',
           'target_number', 'target_trial', 'roi_photostim_group',
           'roi_photostim_point', 'markpoint_assigned_roi', 'opto_unique_ids']

_BRUKER_ACQ_ARRAYS = ['markpoints_group_info', 'markpoints_laser_power',
                      'markpoints_condition_idx']

@dataclass
class Session:
    path: str
    exp_id: str
    n_rois: int
    frame_period: float
    dur_resp: float = None      # response window (s) used to build cyc; for onset fallback
    cyc_pre: int = None         # frames before onset in the cyc window (set by rebuild_cyc)
    avg_image: np.ndarray = None
    mask2d: np.ndarray = None
    dff: np.ndarray = None
    unique_stims: np.ndarray = None
    stim_id: np.ndarray = None
    stim_properties: np.ndarray = None
    stim_on_2p_frame: np.ndarray = None
    cyc: np.ndarray = None
    resp: np.ndarray = None
    resp_err: np.ndarray = None
    resps: np.ndarray = None
    is_soma: np.ndarray = None
    is_dendrite: np.ndarray = None
    is_spine: np.ndarray = None
    is_good_cell: np.ndarray = None
    # photostimulation (opto) — None when the session has no photostim data
    target_number: np.ndarray = None
    target_trial: np.ndarray = None
    roi_photostim_group: np.ndarray = None
    roi_photostim_point: np.ndarray = None
    markpoint_assigned_roi: np.ndarray = None
    opto_unique_ids: np.ndarray = None
    markpoints_group_info: np.ndarray = None
    markpoints_laser_power: np.ndarray = None
    markpoints_condition_idx: np.ndarray = None
    # derived (filled by tools)
    pref_dir: np.ndarray = None
    gdsi: np.ndarray = None
    gosi: np.ndarray = None
    fit_params: np.ndarray = None
    influence: dict = None
    _stim_table: np.ndarray = field(default=None, repr=False)

    @property
    def directions(self):
        return np.unique(self.stim_properties[:, 0])

    @property
    def contrasts(self):
        return np.unique(self.stim_properties[:, 1])

    @property
    def has_photostim(self):
        return (self.target_number is not None
                and self.markpoints_laser_power is not None)

    @property
    def stim_table(self):
        """(n_stims, 2) — each unique_stims[i] -> (direction, contrast)."""
        if self._stim_table is None:
            table = np.full((len(self.unique_stims), 2), np.nan)
            for i, sid in enumerate(self.unique_stims):
                rows = self.stim_properties[self.stim_id == sid]
                table[i] = rows[0]
            self._stim_table = table
        return self._stim_table


def load_session(path):
    exp_id = os.path.splitext(os.path.basename(path))[0]
    kw = {}
    with h5py.File(path, 'r') as f:
        for k in _ARRAYS:
            d = f[k][:] if k in f else None
            if d is not None and d.size == 0:
                d = None
            kw[k] = d
        n_rois = int(f['n_rois'][()])
        frame_period = float(f['Bruker_Acq']['frame_period'][()])
        dur_resp = (float(np.ravel(f['params']['dur_resp'][()])[0])
                    if 'params' in f and 'dur_resp' in f['params'] else None)
        for k in _BRUKER_ACQ_ARRAYS:
            d = f['Bruker_Acq'][k][:] if k in f['Bruker_Acq'] else None
            if d is not None and d.size == 0:
                d = None
            kw[k] = d
    if kw['stim_properties'] is None or kw['stim_id'] is None:
        raise ValueError(f'{exp_id}: no visual stimulus data in this session.')
    return Session(path=path, exp_id=exp_id, n_rois=n_rois,
                   frame_period=frame_period, dur_resp=dur_resp, **kw)


def cyc_blank_frames(s):
    """Indices of all-NaN frames in the cyc window (the photostim blanking gap).

    Empty for sessions without photostim blanking.
    """
    return np.where(np.all(np.isnan(s.cyc), axis=(0, 1, 2)))[0]


def cyc_onset(s):
    """Cyc-window frame index of visual stimulus onset.

    Prefers `s.cyc_pre - 1` when `rebuild_cyc` has set `cyc_pre` (the exact
    pre-onset frame count used to build the current `s.cyc`). Otherwise falls
    back to the pipeline convention: cyc width = pre-frames +
    round(dur_resp / frame_period), with onset at (pre-frames - 1). Returns 0 if
    neither is available. (Verified: 002 -> 14, 003 -> 29.)
    """
    if s.cyc_pre is not None:
        return s.cyc_pre - 1
    if not s.dur_resp:
        return 0
    return s.cyc.shape[3] - int(round(s.dur_resp / s.frame_period)) - 1


def rebuild_cyc(s, pre=1.5, post=2.5, blank=None):
    """Replace s.cyc with a raw, un-blanked cyc built directly from s.dff.

    `s.cyc` (as loaded) is built upstream from `dff_nan` (photostim-blanked).
    That NaN blank is applied per-trial at each *photostim* frame, but cyc is
    aligned to *visual onset* — and the photostim fires 0/3/4 frames after onset,
    jittering trial-to-trial. So after trial-averaging the all-NaN blank is
    narrower and mislocated vs. the true per-trial artifact. `s.dff` is
    un-blanked, so rebuilding cyc from it exposes the full trace: the true
    artifact extent, for choosing baseline / peak windows by eye.

    Replicates gen_stim_cyc's trial-fill (img_utils.py:231) but reads raw
    `s.dff` and guards both ends of the frame window (gen_stim_cyc only guards
    the upper end).

    pre, post : seconds before / after visual onset to include in the window.
    blank : optional (t0, t1) in seconds relative to onset. When given, NaNs
        that fixed window on the rebuilt cyc — for a clean, onset-locked blank
        instead of the jittered pipeline one. None (default): no blanking, the
        full raw trace is exposed.

    Sets and invalidates on `s`:
        s.cyc       <- the new (n_cells, n_stims, n_trials, width) array
        s.cyc_pre   <- pre-onset frame count (frames before onset in the window)
        s.dur_resp  <- post (kept consistent with the new window for
                       cyc_onset / cyc_response_windows fallbacks)
        s.resp = s.resps = s.resp_err = None   (stale: old cyc geometry)
        s.influence = None                      (stale: old cyc geometry)

    The original cyc remains in the h5 on disk; reload the session to restore it.

    Returns the new cyc.
    """
    fp = s.frame_period
    cyc_pre = int(round(pre / fp))
    stim_dur = int(round(post / fp))
    if cyc_pre < 1 or stim_dur < 1:
        raise ValueError(
            f'pre={pre}, post={post} must each be positive and >= one frame '
            f'({fp:.4f}s) to build a non-empty cyc window.')
    width = cyc_pre + stim_dur

    unique_stims = s.unique_stims
    stim_id = s.stim_id
    son = s.stim_on_2p_frame
    dff = s.dff
    n_cells = dff.shape[1]
    n_trials = int(max(np.sum(stim_id == u) for u in unique_stims))

    cyc = np.full((n_cells, len(unique_stims), n_trials, width), np.nan)
    trial_count = np.zeros(len(unique_stims), dtype=int)
    for ii in range(len(son)):
        ind = int(np.argwhere(unique_stims == stim_id[ii])[0][0])
        k = trial_count[ind]
        if k >= n_trials:
            trial_count[ind] += 1
            continue
        tt = np.arange(son[ii] - cyc_pre + 1, son[ii] + stim_dur + 1)
        if tt[0] >= 0 and tt[-1] < dff.shape[0]:
            cyc[:, ind, k, :] = dff[tt, :].T
        trial_count[ind] += 1

    if blank is not None:
        t0, t1 = blank
        f0 = int(np.clip(cyc_pre - 1 + round(t0 / fp), 0, width))
        f1 = int(np.clip(cyc_pre - 1 + round(t1 / fp), f0, width))
        cyc[..., f0:f1] = np.nan

    s.cyc = cyc
    s.cyc_pre = cyc_pre
    s.dur_resp = post
    s.resp = s.resps = s.resp_err = None
    s.influence = None
    print(f'{s.exp_id}: rebuilt cyc, width={width} frames '
          f'({width * fp:.3f}s = {cyc_pre} pre + {stim_dur} post), '
          f'n_trials={n_trials}')
    return cyc


def cyc_response_windows(s, baseline_guard_sec=0.5, post_sec=1.0,
                          baseline=None, peak=None):
    """Baseline / peak frame slices within the cyc window, artifact-aware.

    In 2p-opto sessions the cyc trace carries a large photostim/shutter artifact
    around visual onset: a negative dip that starts ~0.4 s *before* onset and a
    NaN-blanked core, followed by the genuine response. Measuring across it (as
    the pipeline's FFT-based `resp` does) is meaningless, so both windows are
    keyed off the empirically detected blank:

        baseline = clean pre-artifact frames  [0 : blank_start - guard]
        peak     = post-blank frames          [blank_end + 1 : + post_sec]

    `baseline_guard_sec` backs the baseline off the pre-onset dip. When no blank
    is present (non-opto session, or a `rebuild_cyc`-produced cyc, which is
    un-blanked by default) it falls back to the nominal onset derived from
    `cyc_onset` (`cyc_pre - 1` when set, else derived from `dur_resp`). In that
    branch no guard is applied, so the baseline spans the *whole* pre-onset
    region — on a rebuilt cyc that includes the exposed pre-onset photostim dip,
    so it is meant to keep state self-consistent, not to be the final
    measurement; pass explicit `baseline=`/`peak=` for that.

    baseline, peak : optional (t0, t1) in seconds relative to onset. When given,
        overrides the auto-derived window for that slice (converted via
        `cyc_onset` + `frame_period`). Independent partial override allowed
        (e.g. pass only `peak=`).

    Returns (base_slice, peak_slice).
    """
    width = s.cyc.shape[3]
    guard = int(round(baseline_guard_sec / s.frame_period))
    post = int(round(post_sec / s.frame_period))
    blank_frames = cyc_blank_frames(s)
    if len(blank_frames):
        b0, b1 = int(blank_frames[0]), int(blank_frames[-1])
        base_sl = slice(0, max(1, b0 - guard))
        peak_sl = slice(b1 + 1, min(width, b1 + 1 + post))
    else:
        # no blank -> no photostim artifact, so no guard needed: use the full
        # pre-onset region as baseline (a 0.5s guard would eat all of it when the
        # cyc has only ~0.5s of pre-frames).
        onset = cyc_onset(s)
        base_sl = slice(0, max(1, onset))
        peak_sl = slice(onset, min(width, onset + post))

    if baseline is not None:
        onset = cyc_onset(s)
        t0, t1 = baseline
        f0 = int(np.clip(onset + round(t0 / s.frame_period), 0, width))
        f1 = int(np.clip(onset + round(t1 / s.frame_period), f0 + 1, width))
        base_sl = slice(f0, f1)
    if peak is not None:
        onset = cyc_onset(s)
        t0, t1 = peak
        f0 = int(np.clip(onset + round(t0 / s.frame_period), 0, width))
        f1 = int(np.clip(onset + round(t1 / s.frame_period), f0 + 1, width))
        peak_sl = slice(f0, f1)
    return base_sl, peak_sl


def resp_grid(s, cell):
    """Return (resp, err) as (n_dir, n_contrast) grids for one cell."""
    dirs, cons = s.directions, s.contrasts
    table = s.stim_table
    resp = np.full((len(dirs), len(cons)), np.nan)
    err = np.full((len(dirs), len(cons)), np.nan)
    for i, (d, c) in enumerate(table):
        di = np.where(dirs == d)[0][0]
        ci = np.where(cons == c)[0][0]
        resp[di, ci] = s.resp[cell, i]
        err[di, ci] = s.resp_err[cell, i]
    return resp, err


def save_derived_to_h5(s):
    """Write pref_dir / gdsi / gosi back into the session's h5 (optional)."""
    with h5py.File(s.path, 'a') as f:
        for k in ['pref_dir', 'gdsi', 'gosi']:
            v = getattr(s, k)
            if v is None:
                continue
            if k in f:
                del f[k]
            f.create_dataset(k, data=v)
