# functions to pass along experiment information and hardcoded values
# getZoom - measured on scope to be recalculated
# getOffsetFrames - returns the number of frames to offset the stimulus timing for a given session 
# getWindow - returns the baseline and peak windows for a given session

def getOffsetFrames(session_name):

    offset_frames = {}

    offset_frames['TSeries-07132025-1042-003.h5'] = -15
    offset_frames['TSeries-07132025-1042-006.h5'] = -15
    offset_frames['TSeries-07132025-1042-009.h5'] = -15
    offset_frames['TSeries-07132025-1042-012.h5'] = -15

    offset_frames['TSeries-11032024-1313-001.h5'] = -(10 + 150) #lost first event??
    offset_frames['TSeries-11032024-1313-003.h5'] = -10 #lost first event?? 
    offset_frames['TSeries-11032024-1313-007.h5'] = -10 #lost first event?? 
    offset_frames['TSeries-11032024-1313-012.h5'] = -10

    offset_frames['TSeries-07212026-1350-003.h5'] = -15

    if session_name not in offset_frames:
        #default
        offset_frames[session_name] = 0

    return offset_frames.get(session_name, 0)


def removeROIs(session_name):

        roi = {}
        roi['TSeries-07132025-1042-003.h5'] = [54]
        roi['TSeries-11032024-1313-001.h5'] = []
        roi['TSeries-11032024-1313-003.h5'] = []

        if session_name not in roi:
            #default
            roi[session_name] = []

        return roi.get(session_name, [])


def getWindow(session_name):

    windows = {}
    windows['TSeries-11032024-1313-001.h5'] = ((0, 0.2), (0.9, 1.1)) #spontaneous
    windows['TSeries-11032024-1313-003.h5'] = ((0, 0.2), (1.15, 1.4))

    if session_name not in windows:
        #default
        windows[session_name] = ((0, 0.25), (1.15, 1.4))
        
    return windows.get(session_name, ((0, 0), (0, 0)))


def getZoom(optical_zoom):

    zoom = {}
    zoom[1] = 1136.7 
    zoom[1.5] = 757.8
    zoom[2] = 568.3

    return zoom.get(optical_zoom, None)


######################################################################
# note -  could turn this into a config file (yaml)
# example: 
# import yaml
#
# def load_session_config(path='analysis/session_windows.yml'):
#     with open(path) as f:
#         return yaml.safe_load(f)['sessions']
#
# cfg = load_session_config()
# baseline, peak = cfg[dat.exp_id]['baseline'], cfg[dat.exp_id]['peak']
# sessions:
#   TSeries-07132025-1042-003:
#     baseline: [0, 0.25]
#     peak: [1.15, 1.4]
#     notes: "offsetFrames=-15, good alignment"