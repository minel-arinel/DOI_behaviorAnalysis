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

from pandastim.experiments.Matt_Exp import LocalSub, LocalMonitor, ClosedLoopStimuli, \
    port_provider, stytra_container,  center_finder, pos_receiver, RadialSinCube, calibration_stimulus_wrapper, calibration_fxn


tex_size = 1024
thisrig = 'rig1'
zebrafish_id = 25
zebrafish_age = 7

calibrated = True
circle_radii = 3
triangle_size = 50
x_offset = 20
y_offset = -20

class GrayTexture(TextureBase):
    """
    Gray
    """

    def __init__(self, texture_size=512, texture_name="gray", brightness=150):
        self.brightness = brightness
        super().__init__(texture_size=texture_size, texture_name=texture_name)

    def create_texture(self):
        circle_texture = np.ones((self.texture_size, self.texture_size)) * self.brightness

        output = np.uint8(circle_texture)

        return output

    def __str__(self):
        return f"{type(self).__name__} size:{self.texture_size} brightness:{self.brightness}"

basic_gray = GrayTexture(texture_size=tex_size, brightness=127)

monocular = 's'

#stimulus_durations = [300, 600] # pre drug treatment
stimulus_durations = [300, 10800] # post drug treatment
stationary_time = 0
exp_repeats = 1

stim_types = [monocular, monocular]
angles = [0,0]
vels = [0,0]
textures = [basic_gray, basic_gray]
stim_names = ['habituation', 'locomotion']

stim_dict = {'stim_type': stim_types, 'angle': angles, 'velocity': vels, 'texture': textures, 'stim_name' : stim_names}
df = pd.DataFrame(stim_dict)
df.loc[:, 'duration'] = stimulus_durations
df.loc[:, 'stationary_time'] = stationary_time

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

# i think this one calls it a different thing -- oops
final['stat_time'] = final.stationary_time.values


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

def minels_custom_protocol_runner(stimuli, pandas_port, go_number, time_sock, centering_dump, fish_dump, center=None, automated=False):
    if center is None:
        center = (663, 492)

    # when we get new centers or fish data, bring it in
    p_context = zmq.Context()
    p_socket = p_context.socket(zmq.PUB)
    p_socket.bind('tcp://*:' + str(pandas_port))
    stimulus_topic = 'stim'
    go_context = zmq.Context()
    go_socket = go_context.socket(zmq.REP)
    go_socket.bind('tcp://*:' + str(go_number))

    if not automated:
        t_context = zmq.Context()
        t_socket = t_context.socket(zmq.PUB)
        t_socket.bind('tcp://*:' + str(time_sock))
        time_topic = 'time'

        tmax = np.sum(stimuli['duration'].values)

    center = np.array(center)

    stim_time = 0
    last_sent = 0.1
    trials = len(stimuli) - 1


    # wait for handshake with stytra go button
    experiment_not_started = True
    if automated:
        experiment_not_started = False
    while experiment_not_started:
        msg = go_socket.recv_string()
        if msg:
            p_socket.send_string(stimulus_topic, zmq.SNDMORE)
            p_socket.send_pyobj(['GO'])
            experiment_not_started=False

    _time_0 = time.time()

    t0 = _time_0
    print('experiment started')
    stim_n = 0

    df = stimuli.copy()
    p_socket.send_string(stimulus_topic, zmq.SNDMORE)
    p_socket.send_pyobj(['next_stimulus'])
    experiment_finished = False
    # will be updated to be while trial <= max trials
    while not experiment_not_started and stim_n <= trials:


        try:
            current_length = df.loc[stim_n].duration + df.loc[stim_n].stationary_time
        except:
            experiment_finished = True
        t0 = time.time()

        p_socket.send_string(stimulus_topic, zmq.SNDMORE)
        p_socket.send_pyobj(df.loc[stim_n])

        while stim_n <= len(df) - 1 and not experiment_finished:
            if time.time() - t0 <= current_length:
                pass
            else:
                stim_n += 1

                try:
                    current_length = df.loc[stim_n].duration + df.loc[stim_n].stationary_time
                except:
                    experiment_finished=True
                t0 = time.time()
                p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                p_socket.send_pyobj(['next_stimulus'])


    if experiment_finished:
        print('Exp Done')
        p_socket.send_string(stimulus_topic, zmq.SNDMORE)
        p_socket.send_pyobj(['end_experiment'])
        t_socket.send_string(time_topic, zmq.SNDMORE)
        t_socket.send_pyobj([10 , 10])


if __name__ == '__main__':

    centering_button = port_provider()
    pandas_socket = port_provider()
    go_socket = port_provider()
    timing_socket = port_provider()

    center_positions = mp.Queue()
    fish_positions = mp.Queue()
    calibration_dump = mp.Queue()

    stytra_main = mp.Process(target=stytra_container, args=(centering_button, go_socket, timing_socket, camera_rotation,
                                                            cam_roi, savingDir))

    _protocol = mp.Process(target=minels_custom_protocol_runner, args=(final, pandas_socket, go_socket, timing_socket,
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