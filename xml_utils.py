# xml_utils.py — XML parsing utilities for Bruker two-photon microscope data
#
# Functions:
#   parse_tseries_xml(xml_path)     → acquisition parameters dict
#   parse_markpoints_xml(xml_path)  → photostimulation targets dict
#   parse_vrec_xml(xml_path)        → voltage recording metadata dict
#   find_experiment_files(data_dir) → dict of relevant file paths

import os
import xml.etree.ElementTree as ET
from glob import glob


# ---------------------------------------------------------------------------
# parse_tseries_xml
# ---------------------------------------------------------------------------

def parse_tseries_xml(xml_path):
    """
    Parse the main TSeries XML file produced by PrairieView / Bruker.

    Parameters
    ----------
    xml_path : str
        Full path to the {TSeries-name}.xml file.

    Returns
    -------
    dict with keys:
        acquisition_date, frame_period, optical_zoom, microns_per_pixel,
        pixels_per_line, lines_per_frame, objective_lens, objective_mag,
        objective_na, laser_power, pmt_gain, bit_depth
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    params = {
        'acquisition_date': root.get('date', ''),
        'frame_period':      None,
        'optical_zoom':      None,
        'microns_per_pixel': None,
        'pixels_per_line':   None,
        'lines_per_frame':   None,
        'objective_lens':    '',
        'objective_mag':     None,
        'objective_na':      None,
        'laser_power':       {},   # keyed by laser description string
        'pmt_gain':          {},   # keyed by PMT description string
        'bit_depth':         None,
    }

    # Scalar string→float mappings for simple PVStateValue elements
    _scalar_map = {
        'framePeriod':      ('frame_period',      float),
        'opticalZoom':      ('optical_zoom',       float),
        'pixelsPerLine':    ('pixels_per_line',    int),
        'linesPerFrame':    ('lines_per_frame',    int),
        'objectiveLens':    ('objective_lens',     str),
        'objectiveLensMag': ('objective_mag',      float),
        'objectiveLensNA':  ('objective_na',       float),
        'bitDepth':         ('bit_depth',          int),
    }

    # Iterate over all PVStateValue elements anywhere in the tree
    for sv in root.iter('PVStateValue'):
        key = sv.get('key', '')

        # --- simple scalar values ---
        if key in _scalar_map:
            dest_key, cast = _scalar_map[key]
            val = sv.get('value', None)
            if val is not None:
                try:
                    params[dest_key] = cast(val)
                except (ValueError, TypeError):
                    pass

        # --- micronsPerPixel: take the XAxis subindex ---
        elif key == 'micronsPerPixel':
            for sub in sv.iter('SubindexedValue'):
                if sub.get('index', '') == '0':
                    try:
                        params['microns_per_pixel'] = float(sub.get('value'))
                    except (ValueError, TypeError):
                        pass

        # --- laserPower: collect all subindexed entries by description ---
        elif key == 'laserPower':
            for sub in sv.iter('SubindexedValue'):
                desc = sub.get('description', f"laser_{sub.get('index','?')}")
                try:
                    params['laser_power'][desc] = float(sub.get('value', 0))
                except (ValueError, TypeError):
                    pass

        # --- pmtGain: collect all subindexed entries by description ---
        elif key == 'pmtGain':
            for sub in sv.iter('SubindexedValue'):
                desc = sub.get('description', f"pmt_{sub.get('index','?')}")
                try:
                    params['pmt_gain'][desc] = float(sub.get('value', 0))
                except (ValueError, TypeError):
                    pass

    return params


# ---------------------------------------------------------------------------
# parse_markpoints_xml
# ---------------------------------------------------------------------------

def parse_markpoints_xml(xml_path):
    """
    Parse a Bruker MarkPoints XML file to extract photostimulation target
    coordinates and stimulation parameters.

    Parameters
    ----------
    xml_path : str
        Full path to the {TSeries-name}_Cycle#####_MarkPoints.xml file.

    Returns
    -------
    dict with keys:
        iterations (int)         — total number of photostim iterations
        all_points_at_once (bool)
        conditions (list of dict) — one dict per PVMarkPointElement, each with:
            uncaging_laser (str)
            uncaging_laser_power (float)   — mW
            repetitions (int)
            trigger_frequency (str)
            galvo (dict):
                initial_delay_ms (float)
                inter_point_delay_ms (float)
                duration_ms (float)
                spiral_revolutions (float)
                spiral_size_microns (float)
            points (list of dict):
                index (int)
                x_norm (float)   — normalized 0–1
                y_norm (float)
                z_norm (float)
                is_spiral (bool)
                spiral_width_norm (float)
                spiral_height_norm (float)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {
        'iterations':        int(root.get('Iterations', 0)),
        'all_points_at_once': root.get('AllPointsAtOnce', 'False').lower() == 'true',
        'conditions':        [],
    }

    for mp_elem in root.findall('PVMarkPointElement'):
        condition = {
            'uncaging_laser':       mp_elem.get('UncagingLaser', ''),
            'uncaging_laser_power': _safe_float(mp_elem.get('UncagingLaserPower', 0)),
            'repetitions':          int(mp_elem.get('Repetitions', 1)),
            'trigger_frequency':    mp_elem.get('TriggerFrequency', ''),
            'galvo':                {},
            'points':               [],
        }

        galvo_elem = mp_elem.find('PVGalvoPointElement')
        if galvo_elem is not None:
            # SpiralSizeInMicrons lives on each <Point>, not on the galvo element.
            # Read it from the first Point as the representative size for this
            # condition (all points in a group share the same spiral size).
            first_pt = galvo_elem.find('Point')
            spiral_size_um = (_safe_float(first_pt.get('SpiralSizeInMicrons', 0))
                              if first_pt is not None else 0.0)

            condition['galvo'] = {
                'initial_delay_ms':      _safe_float(galvo_elem.get('InitialDelay', 0)),
                'inter_point_delay_ms':  _safe_float(galvo_elem.get('InterPointDelay', 0)),
                'duration_ms':           _safe_float(galvo_elem.get('Duration', 0)),
                'spiral_revolutions':    _safe_float(galvo_elem.get('SpiralRevolutions', 0)),
                'spiral_size_microns':   spiral_size_um,
            }

            for pt in galvo_elem.findall('Point'):
                condition['points'].append({
                    'index':              int(pt.get('Index', 0)),
                    'x_norm':             _safe_float(pt.get('X', 0)),
                    'y_norm':             _safe_float(pt.get('Y', 0)),
                    'z_norm':             _safe_float(pt.get('Z', 0)),
                    'is_spiral':          pt.get('IsSpiral', 'False').lower() == 'true',
                    'spiral_width_norm':  _safe_float(pt.get('SpiralWidth', 0)),
                    'spiral_height_norm': _safe_float(pt.get('SpiralHeight', 0)),
                    'spiral_size_microns': _safe_float(pt.get('SpiralSizeInMicrons', 0)),
                })

        result['conditions'].append(condition)

    return result


# ---------------------------------------------------------------------------
# parse_vrec_xml
# ---------------------------------------------------------------------------

def parse_vrec_xml(xml_path):
    """
    Parse the Bruker VoltageRecording XML file to get DAQ metadata.

    Parameters
    ----------
    xml_path : str
        Full path to {TSeries-name}_Cycle#####_VoltageRecording_001.xml

    Returns
    -------
    dict with keys:
        sample_rate (int)       — Hz
        n_samples (int)
        acquisition_time_ms (float)
        channels (list of dict) — one per channel:
            index (int), name (str), gain (float), unit (str), enabled (bool)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    exp = root.find('Experiment')
    if exp is None:
        exp = root  # some versions omit the Experiment wrapper

    rate_raw = exp.get('Rate', exp.get('rate', '10000'))
    divider  = _safe_float(exp.get('Divider', exp.get('divider', '1'))) or 1.0
    # "Rate" is the underlying DAQ rate; effective recording rate = Rate / Divider
    # We want the raw sample rate for time conversion
    sample_rate = int(_safe_float(rate_raw) or 10000)

    n_samples = int(_safe_float(exp.get('SamplesAcquired',
                                         exp.get('samplesAcquired', '0'))) or 0)
    acq_time  = _safe_float(exp.get('AcquisitionTime',
                                     exp.get('acquisitionTime', '0'))) or 0.0

    channels = []
    for sig in root.iter('Signal'):
        enabled_val = sig.get('Enabled', sig.get('enabled', 'false'))
        channels.append({
            'index':   int(sig.get('Channel', sig.get('channel', '0'))),
            'name':    sig.get('Name',    sig.get('name',    '')),
            'gain':    _safe_float(sig.get('Gain', sig.get('gain', '1'))),
            'unit':    sig.get('Unit',    sig.get('unit',    'Volt')),
            'enabled': enabled_val.lower() == 'true',
        })

    return {
        'sample_rate':        sample_rate,
        'n_samples':          n_samples,
        'acquisition_time_ms': acq_time,
        'channels':           channels,
    }


# ---------------------------------------------------------------------------
# find_experiment_files
# ---------------------------------------------------------------------------

def find_experiment_files(data_dir):
    """
    Scan a TSeries data directory and return a dict of all relevant file paths.

    Parameters
    ----------
    data_dir : str
        Path to the TSeries folder, e.g.
        '/mnt/bigdata/BRUKER/TSeries-04022026-1315-003'

    Returns
    -------
    dict with keys (each value is a str path or None if not found):
        tseries_xml, markpoints_xml, vrec_xml, vrec_csv,
        registered_h5, inference_h5, roi_zip
    """
    def _first_or_none(pattern):
        hits = sorted(glob(os.path.join(data_dir, pattern)))
        return hits[0] if hits else None

    base = os.path.basename(data_dir.rstrip('/'))

    return {
        'tseries_xml':    _first_or_none(f'{base}.xml'),
        'markpoints_xml': _first_or_none('*MarkPoints*.xml'),
        'vrec_xml':       _first_or_none('*VoltageRecording*.xml'),
        'vrec_csv':       _first_or_none('*VoltageRecording*.csv'),
        'registered_h5':  _first_or_none('registered.h5'),
        'inference_h5':   _first_or_none('inference.h5'),
        'roi_zip':        _first_or_none('RoiSet.zip'),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.0):
    """Convert val to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
