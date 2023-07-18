import numpy as np
import pandas as pd
import multiprocessing as mp
import sys
import zmq
import time
import os
import json
from datetime import datetime as dt

from pandastim.textures import TextureBase

from pandastim.experiments.Minel_Exp import LocalSub, LocalMonitor, ClosedLoopStimuli, \
    port_provider, stytra_container, center_finder, pos_receiver, RadialSinCube, calibration_stimulus_wrapper, \
    calibration_fxn, LocalGrayTex, LocalGratingGrayTex, protocol_runner

tex_size = (1024, 1024)
thisrig = 'rig1'
zebrafish_id = 273
zebrafish_age = 8
baseline = True # pre drug treatment
# baseline = False # post drug treatment

calibrated = True
circle_radii = 3
triangle_size = 50
x_offset = 20
y_offset = -20

basic_gray = LocalGrayTex(texture_size=tex_size, brightness=127)
omr_grating = LocalGratingGrayTex(texture_size=tex_size)
exp_repeats = 1

if baseline: # pre drug treatment
    stim_types = ['s']
    angles = [0]
    vels = [0]
    textures = [basic_gray]
    omr_stim = ['forward', 'right', 'left', 'forward', 'right', 'backward', 'left', 'backward', 'forward',
                  'right', 'forward', 'backward', 'left', 'forward', 'backward', 'forward', 'left', 'right', 'forward',
                  'backward', 'forward', 'left', 'backward', 'forward', 'right', 'left', 'backward', 'forward', 'left',
                  'right', 'backward', 'right', 'left', 'backward', 'right', 'backward', 'right', 'left', 'right',
                  'left']
    stim_names = ['habituation'] + omr_stim
    durations = [300]
    stat_times = [0]
else: # post drug treatment
    stim_types = ['s']
    angles = [0]
    vels = [0]
    textures = [basic_gray]
    omr_stim = ['left', 'right', 'backward', 'left', 'forward', 'right', 'left', 'forward', 'backward', 'left', 'right',
                'left', 'right', 'backward', 'left', 'forward', 'right', 'left', 'backward', 'forward', 'backward',
                'forward', 'left', 'backward', 'forward', 'backward', 'forward', 'left', 'right', 'left', 'right',
                'backward', 'right', 'left', 'right', 'left', 'backward', 'left', 'right', 'left', 'forward', 'right',
                'forward', 'backward', 'forward', 'backward', 'right', 'forward', 'left', 'backward', 'right', 'forward',
                'right', 'forward', 'left', 'forward', 'right', 'left', 'forward', 'left', 'backward', 'left', 'forward',
                'backward', 'forward', 'right', 'backward', 'right', 'forward', 'backward', 'forward', 'right', 'backward',
                'forward', 'backward', 'left', 'backward', 'forward', 'backward', 'right', 'forward', 'backward', 'forward',
                'backward', 'right', 'left', 'forward', 'backward', 'left', 'right', 'forward', 'left', 'forward', 'right',
                'backward', 'right', 'left', 'backward', 'left', 'forward', 'right', 'left', 'right', 'forward', 'backward',
                'right', 'backward', 'right', 'backward', 'left', 'right', 'left', 'forward', 'right', 'left', 'right',
                'forward', 'backward', 'left', 'backward']
    stim_names = ['habituation'] + omr_stim
    durations = [5]
    stat_times = [0]

stim_types += ['s']*len(omr_stim)
vels += [-0.025]*len(omr_stim)
textures += [omr_grating]*len(omr_stim)
durations += [33]*len(omr_stim)
stat_times += [3]*len(omr_stim)

for stim in omr_stim:
    if stim == 'forward':
        angles.append(180)
    elif stim == 'backward':
        angles.append(0)
    elif stim == 'right':
        angles.append(270)
    elif stim == 'left':
        angles.append(90)

stim_dict = {'stim_type': stim_types, 'angle': angles, 'velocity': vels, 'texture': textures, 'stim_name' : stim_names,
             'duration': durations, 'stat_time': stat_times}
df = pd.DataFrame(stim_dict)
final = pd.concat([df] * exp_repeats, ignore_index=True)

with open(thisrig+".json", 'r') as json_file:
    params = json.load(json_file)

center_point = params['pandastim']['center']
mon = params['pandastim']['monitor']
offset_window = params['pandastim']['offset_window']
texture_sizes = params['pandastim']['texture_sizes']

cam_roi = params['stytra']['roi']
camera_rotation = params['stytra']['rotation']

stytraWindow = "Stytra" + str(params['generic']['rig'])

savingDir = os.path.join(r'C:\Users\Naumann_Lab\Data\DOI_minel', thisrig)

save_path = os.path.join(savingDir, str(dt.now().strftime("%m_%y__%H_%M") + '.txt'))

# because this all runs in diff processes we'll lump the pandas into a lil fxn
def pandas_stimuli(port1):

    def radial_sin(window_size):
        stack = []
        num_slices = 238
        phase_change = 0.08
        phase = 0
        for slice_num in range(num_slices):
            rad_slice = RadialSinCube(texture_size=window_size, phase=phase)
            stack.append(rad_slice)
            phase += phase_change
        return stack

    radial_sin_stack = radial_sin(texture_sizes)
    sub = LocalSub(topic='stim', port=port1)
    monitor = LocalMonitor(sub)
    cc = ClosedLoopStimuli(final, profile_on=False, radial_centering=True, fps=60, monitor=mon,
                           win_offset=offset_window,
                           save_path=save_path, fish_id=zebrafish_id, fish_age=zebrafish_age, window_size=texture_sizes,
                           center_pt=center_point, radial_centering_stack=radial_sin_stack, proj_fish=False)
    cc.run()

if __name__ == '__main__':

    centering_button = port_provider()
    pandas_socket = port_provider()
    go_socket = port_provider()
    timing_socket = port_provider()

    center_positions = mp.Queue()
    fish_positions = mp.Queue()
    calibration_dump = mp.Queue()

    stytra_main = mp.Process(target=stytra_container, args=(centering_button, go_socket, timing_socket, camera_rotation,
                                                            cam_roi, savingDir,))

    _protocol = mp.Process(target=protocol_runner, args=(final, pandas_socket, go_socket, timing_socket,
                                                          center_positions, fish_positions, center_point,))
    center_grabber = mp.Process(target=center_finder, args=(centering_button, center_positions))
    #
    if calibrated:
        pandas_stimulation = mp.Process(target=pandas_stimuli, args=(pandas_socket,))
    else:
        pandas_stimulation = mp.Process(target=calibration_stimulus_wrapper, args=(
        calibration_dump, mon, circle_radii, triangle_size, x_offset, y_offset, offset_window, texture_sizes))

    calibrator = mp.Process(target=calibration_fxn, args=(calibrated, centering_button, calibration_dump))
    fish_position_receiver = mp.Process(target=pos_receiver, args=(fish_positions,))

    _protocol.start()
    center_grabber.start()
    pandas_stimulation.start()
    calibrator.start()
    stytra_main.start()
    fish_position_receiver.start()

    #
    stytra_main.join()
    if not stytra_main.is_alive():
        center_grabber.terminate()
        _protocol.terminate()
        pandas_stimulation.terminate()
        stytra_main.terminate()
        calibrator.terminate()
        fish_position_receiver.terminate()
        calibrator.join()
        fish_position_receiver.join()
        center_grabber.join()
        _protocol.join()
        pandas_stimulation.join()
        sys.exit()