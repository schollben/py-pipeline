import os
from dataclasses import dataclass, field
import numpy as np
import h5py

_ARRAYS = ['avg_image', 'mask2d', 'dff', 'unique_stims', 'stim_id',
           'stim_properties', 'stim_on_2p_frame', 'photostim_2p_frame', 'cyc',
           'resp', 'resp_err', 'resps', 'is_soma', 'is_dendrite', 'is_spine',
           'is_good_cell', 'target_number', 'target_trial', 'roi_photostim_group',
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
    cyc_offset: int = 0         # frame offset applied to the alignment reference
    avg_image: np.ndarray = None
    mask2d: np.ndarray = None
    dff: np.ndarray = None
    unique_stims: np.ndarray = None
    stim_id: np.ndarray = None
    stim_properties: np.ndarray = None
    stim_on_2p_frame: np.ndarray = None
    photostim_2p_frame: np.ndarray = None
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
    snr: np.ndarray = None
    pref_stim: np.ndarray = None
    gdsi: np.ndarray = None
    gosi: np.ndarray = None
    fit_params: np.ndarray = None
    influence: dict = None
    _stim_table: np.ndarray = field(default=None, repr=False)
    _events_dropped: bool = field(default=False, repr=False)

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


def check_event_alignment(s, n_show=5):
    """Diagnose whether the first visual/photostim TTL pair looks spurious.

    Some sessions carry a spurious first trigger on both the visual and
    photostim vrec channels — a known hardware artifact where the photostim
    trigger fires in lockstep with the very first visual trigger, well before
    the real trial train begins (the pipeline's own "drop erroneous first
    trigger" check, bruker_pipeline.py:624, tests the gap in raw vrec seconds
    and can fail to catch it). `target_number` is separately shifted by the
    pipeline's `opto_offset_trigger` psychopy row-drop (bruker_pipeline.py:674),
    which only ever applies to the photostim label columns — so when both
    conditions are present, `stim_id` (visual) is off by one against the
    untrimmed `stim_on_2p_frame`, while `target_number` (photostim) is already
    correctly aligned to the untrimmed `photostim_2p_frame`. See `dropFirstEvents`.

    The primary signal is the photostim lag, `photostim_2p_frame - stim_on_2p_frame`,
    tested against the session's OWN distribution rather than a fixed value. A real
    photostim trial has a characteristic positive lag (3-4 frames on the July-2025
    rig); the spurious event fires in lockstep, so its lag sits below every other
    trial's. A fixed `lag[0] == 0` test would be wrong: on the Nov-2024 sessions the
    lag is 0 on EVERY trial (that rig records the photostim at the visual frame), and
    nothing is spurious there.

    Prints, from the 2P frame-index arrays (`stim_on_2p_frame`,
    `photostim_2p_frame` — not the raw vrec/seconds triggers):
        - first `n_show` inter-event intervals for each stream
        - lag[0] vs the min/median of the remaining lags (flagged when below min)
        - first stim_on ITI vs the median of the rest (flagged if > 2x)
        - the length signature (len(pf) > len(son), len(tn) == len(stim_id) - 1)
        - target_number[0], informational only (1 = no psychopy row-offset applied,
          2 = applied); it corroborates but no longer drives the decision, since it
          assumes the group cycle starts at 1

    Returns True when dropFirstEvents(s) is recommended — the lag outlier plus at
    least one corroborating signal (anomalous first ITI, or the length signature).
    Returns False without printing photostim-specific lines when the session has no
    photostim data.
    """
    son = s.stim_on_2p_frame
    pf = s.photostim_2p_frame
    tn = s.target_number
    print(f'=== {s.exp_id}: event alignment check ===')

    son_iti = np.diff(son)
    print(f'  stim_on_2p_frame  first {n_show} ITIs (frames): {son_iti[:n_show]}')
    median_iti = np.median(son_iti[1:]) if len(son_iti) > 1 else np.median(son_iti)
    first_anomalous = bool(len(son_iti) and son_iti[0] > 2 * median_iti)
    print(f'  first stim_on ITI = {son_iti[0]} frames vs median {median_iti:.0f} '
          f'({"ANOMALOUS >2x" if first_anomalous else "normal"})')

    if not s.has_photostim or tn is None or not len(tn):
        print('  no photostim data in this session — skipping photostim checks.')
        return False

    if pf is not None and len(pf) > 1:
        pf_iti = np.diff(pf)
        print(f'  photostim_2p_frame first {n_show} ITIs (frames): {pf_iti[:n_show]}')

    # lag outlier: is the first pair in lockstep relative to this session's own lags?
    lag_outlier = False
    if pf is not None and len(pf) > 1 and len(son) > 1:
        n = min(len(son), len(pf))
        lag = pf[:n] - son[:n]
        rest = lag[1:]
        lag_outlier = bool(len(rest) and lag[0] < rest.min())
        print(f'  photostim - stim_on, first {n_show}: {lag[:n_show]}')
        print(f'  lag[0] = {lag[0]} vs min {rest.min()} / median {np.median(rest):.0f} '
              f'of the rest ({"LOCKSTEP OUTLIER" if lag_outlier else "normal"})')

    # length signature: the glitch adds a photostim TTL that psychopy never logged,
    # and the pipeline's row-drop already shortened target_number by one.
    pf_longer = pf is not None and len(pf) > len(son)
    tn_pre_dropped = len(tn) == len(s.stim_id) - 1
    print(f'  lengths: stim_id={len(s.stim_id)} stim_on={len(son)} '
          f'photostim={len(pf) if pf is not None else 0} target_number={len(tn)}'
          f'{"  [pf longer than stim_on]" if pf_longer else ""}'
          f'{"  [target_number pre-dropped]" if tn_pre_dropped else ""}')
    print(f'  target_number[0] = {tn[0]:g} (informational; '
          f'{"psychopy row-offset already applied" if tn[0] == 2 else "no offset applied"})')

    recommend = bool(lag_outlier and (first_anomalous or pf_longer or tn_pre_dropped))
    if recommend:
        print('  [!] WARNING: first TTL pair looks spurious (lockstep lag outlier) '
              '-- run dropFirstEvents(s) before rebuild_cyc().')
    return recommend


def dropFirstEvents(s):
    """Drop the spurious first visual+photostim TTL pair (see
    `check_event_alignment`) and re-align every per-presentation array to match,
    so no other function needs to know this happened.

    `stim_on_2p_frame[0]` and `photostim_2p_frame[0]` are dropped (both
    spurious). `stim_id`/`stim_properties` are truncated at the *end* — they
    were never shifted, so once the leading TTL is gone they are exactly one
    entry too long. That leaves `stim_id[i]` describing the presentation at the
    untrimmed `stim_on_2p_frame[i + 1]` (verified: across-stim tuning strength is
    0.042 under this pairing vs 0.027 under the alternative).

    `target_number`/`target_trial` must end up on that SAME presentation, because
    `cyc_trial_group` pairs `target_number[i]` with `stim_id[i]` to fill cyc trial
    slots — it never indexes `photostim_2p_frame`. Two cases:

      - The pipeline's `opto_offset_trigger` psychopy row-drop
        (bruker_pipeline.py:674) already removed `target_number[0]`, so the array
        arrives one short (`len(tn) == len(stim_id) - 1`). That row-drop aligned it
        to the UNTRIMMED presentations (`tn[i]` describes raw presentation `i`,
        verified on raw dff: real groups drive their targets 4-10x harder than sham
        under this pairing and the reverse under any other). Post-drop `stim_id[i]`
        describes raw presentation `i + 1`, so `target_number`/`target_trial` must
        lose one more leading entry to land on the same presentation.
      - No row-drop was applied (`len(tn) == len(stim_id)`): `target_number` still
        carries the spurious event's own entry, so drop that one instead — the same
        single leading drop, reached from a different starting length.

    Either way exactly one leading entry comes off `target_number`/`target_trial`
    here; what differs is only which upstream state we arrived from.

    Detection is by length, not by `target_number[0] == 2`, which would assume the
    group cycle always starts at 1.

    Mutates `s` in place. Raises if called twice on the same session.
    """
    if s._events_dropped:
        raise ValueError(f'{s.exp_id}: dropFirstEvents already applied.')
    n = len(s.stim_on_2p_frame) - 1
    n_stim_id = len(s.stim_id)
    s.stim_on_2p_frame = s.stim_on_2p_frame[1:]
    s.stim_id = s.stim_id[:n]
    if s.stim_properties is not None:
        s.stim_properties = s.stim_properties[:n]          # same rows dropped as stim_id

    note = ''
    if s.target_number is not None:
        pre_dropped = len(s.target_number) == n_stim_id - 1
        s.target_number = s.target_number[1:]
        if s.target_trial is not None:
            s.target_trial = s.target_trial[1:]
        note = ('; dropped target_number[0]/target_trial[0] '
                + ('(on top of the upstream psychopy row-drop)' if pre_dropped
                   else '(no upstream row-drop)'))
    if s.photostim_2p_frame is not None and s.target_number is not None:
        s.photostim_2p_frame = s.photostim_2p_frame[1:][:len(s.target_number)]
    s.unique_stims = np.unique(s.stim_id)
    s._events_dropped = True
    print(f'{s.exp_id}: dropped spurious first TTL pair; '
          f'{len(s.stim_on_2p_frame)} presentations remain{note}.')


def rebuild_cyc(s, preStim=1.5, postStim=2.5, blank=None, offsetFrames=0):
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

    preStim, postStim : seconds before / after visual onset to include in the
        window.
    blank : optional (t0, t1) in seconds relative to onset. When given, NaNs
        that fixed window on the rebuilt cyc — for a clean, onset-locked blank
        instead of the jittered pipeline one. None (default): no blanking, the
        full raw trace is exposed.
    offsetFrames : frames added to each trial's alignment reference, so a
        NEGATIVE value moves the collected window BACK (earlier) in time;
        offsetFrames=-15 collects each trial starting 15 frames before its
        visual TTL. Default 0 — the verified-correct alignment; this is an
        exploratory knob for testing alignment hypotheses or for a session whose
        triggers genuinely are offset, not a correction.

        For reference, measured on raw fluorescence (not dF/F) aligned to
        photostim_2p_frame: F begins declining at rel -15 as the PMT shutter
        closes, troughs exactly at rel 0 (the photostim frame), and recovers by
        rel +10. So the ~15-frame lead ahead of the pulse is physical (shutter),
        not a trigger error — the TTLs themselves are accurate, and preStim=1
        already opens the window well before the shutter with room for clean
        baseline.

        Note the offset moves the window, not the artifact: shifting earlier
        makes the artifact appear LATER in window coordinates.

    Sets and invalidates on `s`:
        s.cyc        <- the new (n_cells, n_stims, n_trials, width) array
        s.cyc_pre    <- pre-onset frame count (frames before onset in the window)
        s.cyc_offset <- offsetFrames actually applied
        s.dur_resp   <- postStim (kept consistent with the new window for
                        cyc_onset / cyc_response_windows fallbacks)
        s.resp = s.resps = s.resp_err = None   (stale: old cyc geometry)
        s.influence = None                      (stale: old cyc geometry)

    The original cyc remains in the h5 on disk; reload the session to restore it.

    Returns the new cyc.
    """
    fp = s.frame_period
    cyc_pre = int(round(preStim / fp))
    stim_dur = int(round(postStim / fp))
    if cyc_pre < 1 or stim_dur < 1:
        raise ValueError(
            f'preStim={preStim}, postStim={postStim} must each be positive and '
            f'>= one frame ({fp:.4f}s) to build a non-empty cyc window.')
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
        ref = son[ii] + offsetFrames
        tt = np.arange(ref - cyc_pre + 1, ref + stim_dur + 1)
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
    s.cyc_offset = offsetFrames
    s.dur_resp = postStim
    s.resp = s.resps = s.resp_err = None
    s.influence = None
    print(f'{s.exp_id}: rebuilt cyc, width={width} frames '
          f'({width * fp:.3f}s = {cyc_pre} preStim + {stim_dur} postStim), '
          f'n_trials={n_trials}'
          + (f', offsetFrames={offsetFrames}' if offsetFrames else ''))
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

    baseline, peak : optional (t0, t1) in seconds from the START of the cyc
        window (not from onset). t=0 is the first frame of cyc; the window runs
        to `cyc.shape[3] * frame_period`. For a cyc built with
        `rebuild_cyc(preStim=1, postStim=2)` the window spans 0 -> 3.0 s with
        visual onset at 1.0 s, so `baseline=(0, 0.5)` is the first half-second
        and `peak=(1.4, 2.2)` is 0.4-1.2 s after onset. When given, overrides
        the auto-derived window for that slice. Independent partial override
        allowed (e.g. pass only `peak=`). Raises ValueError if the requested
        span falls outside the cyc window or is empty.

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
        base_sl = _window_slice(s, baseline, width, 'baseline')
    if peak is not None:
        peak_sl = _window_slice(s, peak, width, 'peak')
    return base_sl, peak_sl


def _window_slice(s, span, width, name):
    """(t0, t1) seconds from cyc-window start -> frame slice, bounds-checked."""
    t0, t1 = span
    fp = s.frame_period
    dur = width * fp
    if t0 >= t1:
        raise ValueError(
            f'{name}={span}: t0 must be less than t1 (seconds from the start '
            f'of the cyc window).')
    if t0 < 0 or t1 > dur:
        raise ValueError(
            f'{name}={span} falls outside the cyc window, which spans '
            f'0 to {dur:.3f}s ({width} frames at {fp:.4f}s). Note these are '
            f'seconds from the START of the window, not from onset '
            f'(onset is at {cyc_onset(s) * fp:.3f}s).')
    f0 = int(round(t0 / fp))
    f1 = int(round(t1 / fp))
    if f1 <= f0:
        raise ValueError(
            f'{name}={span} is shorter than one frame ({fp:.4f}s) and selects '
            f'no data.')
    return slice(f0, min(f1, width))


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
