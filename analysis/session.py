import os
from dataclasses import dataclass, field
import numpy as np
import h5py

_ARRAYS = ['avg_image', 'mask2d', 'dff', 'unique_stims', 'stim_id',
           'stim_properties', 'cyc', 'resp', 'resp_err', 'resps',
           'is_soma', 'is_dendrite', 'is_spine', 'is_good_cell',
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
    avg_image: np.ndarray = None
    mask2d: np.ndarray = None
    dff: np.ndarray = None
    unique_stims: np.ndarray = None
    stim_id: np.ndarray = None
    stim_properties: np.ndarray = None
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
        for k in _BRUKER_ACQ_ARRAYS:
            d = f['Bruker_Acq'][k][:] if k in f['Bruker_Acq'] else None
            if d is not None and d.size == 0:
                d = None
            kw[k] = d
    if kw['stim_properties'] is None or kw['stim_id'] is None:
        raise ValueError(f'{exp_id}: no visual stimulus data in this session.')
    return Session(path=path, exp_id=exp_id, n_rois=n_rois,
                   frame_period=frame_period, **kw)


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
