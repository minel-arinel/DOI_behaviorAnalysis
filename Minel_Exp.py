# panda imports
from panda3d.core import WindowProperties, Texture, TextureStage, ClockObject, PStatClient, CardMaker, TransformState, \
    Shader, PerspectiveLens, ColorBlendAttrib
from direct.showbase import ShowBaseGlobal, DirectObject
from direct.showbase.ShowBase import ShowBase
from direct.showbase.MessengerGlobal import messenger
from direct.gui.OnscreenText import OnscreenText
from pandastim import utils

# stytra imports
from stytra.stimulation.stimuli import Stimulus
from stytra import Protocol
from stytra.experiments.tracking_experiments import TrackingExperiment
from PyQt5.QtWidgets import QApplication

# other imports
from datetime import datetime
from scipy import ndimage

# etc imports
import multiprocessing as mp
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import zmq
import pygetwindow as gw
import qdarkstyle
import threading as tr
import sys
import time
import os
import pandas as pd
import smtplib, ssl

def updated_saving(file_path, fish_id, fish_age):
    """
    Initializes saving: saves texture classes and params for
    input-coupled stimulus classes.
    """

    if '\\' in file_path:
        file_path = file_path.replace('\\', '/')

    print(f"Saving data to {file_path}")
    filestream = open(file_path, "a")

    filestream.write(f"fish{fish_id}_{fish_age}dpf_{datetime.now()}")
    filestream.flush()
    return filestream


def final_saving(file_path):

    if '\\' in file_path:
        file_path = file_path.replace('\\', '/')

    with open(file_path) as file:
        contents = file.read()

    # separate the text file into the different stimulus lines and withdraw the stimulus dictionaries
    parsed = contents.split('\n')
    fish_details = parsed[0]
    stimulus_details = parsed[1:]

    # some tricky text splitting
    times = [i[:i.find('{')] for i in stimulus_details]
    stimulus_dicts = [eval(i[i.find('{'):]) for i in stimulus_details if 'stationary_end' not in i]

    # mostly a binocular gratings fix, need to stack the tuples into two separate columns
    for stim in range(len(stimulus_dicts)):
        for item in stimulus_dicts[stim].copy():
            try:
                if len(stimulus_dicts[stim][item]) > 1 and type(stimulus_dicts[stim][item]) is not str:
                    for i in range(len(stimulus_dicts[stim][item])):
                        name = item + '_' + str(i)
                        stimulus_dicts[stim][name] = stimulus_dicts[stim][item][i]
                    stimulus_dicts[stim].pop(item)
            except:
                pass

    # find stimuli that had stationary times and duplicate just them in the dataframe and set the stationary to 0 vel
    stim_df = pd.DataFrame(stimulus_dicts)
    # final_stims = stim_df.loc[(stim_df[stim_df.stat_time > 0].index.repeat(2)) | (
    #     stim_df[stim_df.stationary_time == 0].index.repeat(1))].reset_index(drop=True)
    # final_stims.loc[final_stims[final_stims.duplicated()].index - 1, 'velocity'] = 0
    # final_stims.loc[final_stims[(final_stims.velocity == 0) & (final_stims.stim_type == 'b')].index, 'velocity_0'] = 0
    # final_stims.loc[final_stims[(final_stims.velocity == 0) & (final_stims.stim_type == 'b')].index, 'velocity_1'] = 0
    final_stims = stim_df
    # interpret the times and set up an array to measure elapsed times across experiment
    ntime_array = []
    for i in range(len((times))):
        ntime_array.append(datetime.strptime(times[i].split(' ')[1], '%H:%M:%S.%f:'))
    time_array = []
    for i in range(len(ntime_array)):
        try:
            time_array.append((ntime_array[i + 1] - ntime_array[i]).total_seconds())
        except:
            pass
    aligned_times = np.cumsum(np.insert(time_array, 0, 0))

    # stick the times with the stimuli
    final_stims.loc[:, 'time'] = aligned_times

    # save a new file (don't overwrite an existing)
    fish_details = fish_details[:fish_details.rfind(' ')]

    val_offset = 0
    new_file = file_path[:file_path.rfind('/') + 1] + fish_details + '_' + str(val_offset) + '.h5'

    while os.path.exists(new_file):
        val_offset += 1
        new_file = file_path[:file_path.rfind('/') + 1] + fish_details + '_' + str(val_offset) + '.h5'

    # erase old file and place new dataframe there
    # os.remove(file_path)
    final_stims.to_hdf(new_file, key='df')
    print('file saved:', new_file)
    return


# small little script that returns a free PC port to set up ZMQ signals
def port_provider():
    c = zmq.Context()
    s = c.socket(zmq.SUB)
    rand_port = s.bind_to_random_port('tcp://*', min_port=5000, max_port=8000, max_tries=100)
    c.destroy()
    return rand_port


# this is the Stytra stimulus, which we're not presenting, but we're using to update time
class BlankUpdater(Stimulus):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sends a signal out to somewhere?
        self.external_starter = 0
        self.go = None

        # This takes in timing information from external protocol
        self.timing = None
        # initializes with some bogus duration, needs a buffer
        self.duration = 300

        # initializes the variables to measure the time
        self.sent_times = [0, 0]
        self.exp_max = 9999
        self.exp_elapsed = 0

        # this iterator makes it so we don't update every loop, only every so many loops
        self.iterator = 0
        self.timing_offset = 0
        self.fixed_duration = False

    # connects to stytra to get the internal experiment parameters from stytra
    def initialise_external(self, experiment):
        super().initialise_external(experiment)
        try:
            stims_socket = self._experiment.estimator.matt_go_socket()
            sending_context = zmq.Context()
            self.go = sending_context.socket(zmq.REQ)
            self.go.connect('tcp://localhost:' + str(stims_socket))
        except AttributeError:
            pass

        try:
            time_socket = self._experiment.estimator.matt_timing_socket()
            context = zmq.Context()
            self.timing = context.socket(zmq.SUB)
            self.timing.setsockopt(zmq.SUBSCRIBE, b'time')
            self.timing.connect(str("tcp://localhost:") + str(time_socket))
        except AttributeError:
            pass

    def update(self):
        # if condition met, update duration
        if self.external_starter == 0:
            self.go.send_string('True')
            self.external_starter = 1

        try:
            # check for a message, this will not block
            times_t = self.timing.recv_string(flags=zmq.NOBLOCK)
            self.sent_times = self.timing.recv_pyobj(flags=zmq.NOBLOCK)
            self.exp_max = self.sent_times[0]
            self.exp_elapsed = self.sent_times[1]

            if not self.fixed_duration:
                self.duration = np.float64(self.exp_max)
                self.fixed_duration = True

        except zmq.Again:
            pass

        # only update every 50 loop runs, this runs at ~30-40 Hz, hurts performance to do more often
        self.iterator += 1
        if self.iterator > 50:
            time_correction = self._elapsed - self.exp_elapsed - self.timing_offset

            if time_correction <= 0:
                time_correction = 0
            self.duration += time_correction
            self.timing_offset += time_correction
            self.iterator = 0


# blankest Stytra protocol.
class DummyStytra(Protocol):
    name = "dummy"

    def __init__(self,):
        super().__init__()

    def get_stim_sequence(self):
        return [BlankUpdater()]


# the physical function to put the above 2 classes together and run stytra, runs stytra as a pyqt application
def stytra_container(image_socket=5558, go_button_socket=5559, time_socket=6000, camera_rot=-2, roi=None, savingdir=None):
    if roi is None:
        roi = [262, 586, 1120, 1120]

    def fixer():
        time.sleep(4)
        gw.getWindowsWithTitle('Stytra stimulus display')[0].close()

    a = tr.Thread(target=fixer)
    a.start()

    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    protocol = DummyStytra()
    exp = TrackingExperiment(protocol=protocol, app=app, dir_save=savingdir,
                             tracking=dict(method='fish', embedded=False, estimator="position"),
                             camera=dict(type='spinnaker', min_framerate=155, rotation=camera_rot, roi=roi),
                             pusheen_sock=image_socket, go_sock=go_button_socket, time_sock=time_socket
                             )
    exp.start_experiment()
    app.exec_()
    a.join()


class LocalTextureBase:
    """
    Base class for stimuli: subclass this when making specific stimuli.
    You need to implement the create_texture() method, and any parameters
    needed for the texture function.
    """

    def __init__(self, texture_size=(512, 512), texture_name="stimulus"):
        self.texture_size = texture_size
        self.texture_name = texture_name
        # Create texture
        self.texture_array = self.create_texture()
        self.texture = Texture(self.texture_name)
        # Set texture formatting (greyscale or rgb have different settings)
        if self.texture_array.ndim == 2:
            self.texture.setup2dTexture(self.texture_size[0], self.texture_size[1],
                                        Texture.T_unsigned_byte,
                                        Texture.F_luminance)
            self.texture.setRamImageAs(self.texture_array, "L")
        elif self.texture_array.ndim == 3:
            self.texture.setup2dTexture(self.texture_size[0], self.texture_size[1],
                                        Texture.T_unsigned_byte,
                                        Texture.F_rgb8)
            self.texture.setRamImageAs(self.texture_array, "RGB")

    def create_texture(self):
        """
        Create 2d numpy array for stimulus: either nxmx1 (grayscale) or nxm x 3 (rgb)
        """
        pass

    def view(self):
        """
        Plot the texture using matplotlib. Useful for debugging.
        """
        plt.imshow(self.texture_array, vmin=0, vmax=255)
        if self.texture_array.ndim == 2:
            plt.set_cmap('gray')

        plt.title(self.texture_name)
        plt.gca().invert_yaxis()
        plt.show()

    def __str__(self):
        """
        Return the string you want print(Tex) to show, and to save to file
        when saving catalog of stimuli.
        """
        pass


class LocalTexFixed(ShowBase):
    def __init__(self, tex, fps=30, window_size=None, window_name="ShowTexStatic", profile_on=False,
                 monitor=1, win_offset=(0, 0)):
        super().__init__()

        self.scale = np.sqrt(8)

        self.tex = tex
        if window_size is None:
            self.window_size = self.tex.texture_size
        else:
            self.window_size = window_size

        self.texture_stage = TextureStage("texture_stage")
        self.window_name = window_name

        # Set frame rate (fps)
        ShowBaseGlobal.globalClock.setMode(ClockObject.MLimited)
        ShowBaseGlobal.globalClock.setFrameRate(fps)

        # Set up profiling if desired
        if profile_on:
            PStatClient.connect()  # this will only work if pstats is running: see readme
            ShowBaseGlobal.base.setFrameRateMeter(True)  # Show frame rate
            self.center_indicator = None

        # Window properties set up
        self.window_properties = WindowProperties()
        self.window_position = ((monitor * 1920) + (self.tex.texture_size[0] // 2) + win_offset[0], win_offset[1])
        self.window_size = self.tex.texture_size

        self.window_properties.setSize(self.window_size[0], self.window_size[1])
        self.window_properties.set_undecorated(True)
        self.window_properties.set_origin(self.window_position)
        self.window_properties.set_foreground(True)

        self.window_properties.setTitle(window_name)
        ShowBaseGlobal.base.win.requestProperties(self.window_properties)

        # Create scenegraph, attach stimulus to card.
        cm = CardMaker('card')
        cm.setFrameFullscreenQuad()
        self.card = self.aspect2d.attachNewNode(cm.generate())
        self.card.setScale(self.scale)
        self.card.setColor((1, 1, 1, 1))  # makes it bright when bright (default combination with card is add)
        self.card.setTexture(self.texture_stage, self.tex.texture)


class LocalGratingGrayTex(LocalTextureBase):
    """
    Grayscale 2d square wave (grating)
    """

    def __init__(self, texture_size=(512, 512), texture_name="grating_gray",
                 spatial_frequency=60):
        self.frequency = spatial_frequency
        super().__init__(texture_size=texture_size, texture_name=texture_name)

    def create_texture(self):
        x = np.linspace(0, 2 * np.pi, self.texture_size[0] + 1)
        y = np.linspace(0, 2 * np.pi, self.texture_size[1] + 1)
        X, Y = np.meshgrid(x[: self.texture_size[0]], y[: self.texture_size[1]])
        return utils.grating_byte(X, freq=self.frequency)

    def __str__(self):
        return f"{type(self).__name__} size:{self.texture_size} frequency:{self.frequency}"


class LocalGrayTex(LocalTextureBase):
    """
    Gray
    """

    def __init__(self, texture_size=(512, 512), texture_name="gray", brightness=150):
        self.brightness = brightness
        super().__init__(texture_size=texture_size, texture_name=texture_name)

    def create_texture(self):
        circle_texture = np.ones((self.texture_size[0], self.texture_size[1])) * self.brightness

        output = np.uint8(circle_texture)

        return output

    def __str__(self):
        return f"{type(self).__name__} size:{self.texture_size} brightness:{self.brightness}"


class BlankTex(LocalTextureBase):
    """
    Grayscale 2d square wave (grating)
    """

    def __init__(self, texture_size=(512, 512), texture_name="blank_tex"):
        super().__init__(texture_size=texture_size, texture_name=texture_name)

    def create_texture(self):
        tex = np.zeros(self.texture_size)
        return np.uint8(tex)

    def __str__(self):
        return f"{type(self).__name__} size:{self.texture_size} "


# the actual stimulus presented for calibration
class CalibrationTriangles(LocalTextureBase):
    """
    Filled circle: grayscale on grayscale with circle_radius, centered at circle_center
    with face color fg_intensity on background bg_intensity. Center position is in pixels
    from center of image.
    """

    def __init__(self, texture_size=(1024, 1024), texture_name="circs", tri_size=50,
                 circle_radius=7, x_off=500, y_off=0
                 ):

        self.texture_size = texture_size

        self.tri_size = tri_size
        self.x_offset = x_off
        self.y_offset = y_off

        self.radius = circle_radius

        self.midx = self.texture_size[0]//2
        self.midy = self.texture_size[1]//2

        self.pt1 = (int((self.midx + self.x_offset - (self.tri_size * math.sqrt(3)) // 2)),
                    int((self.midy + self.y_offset + self.tri_size // 2)))

        self.pt2 = (int((self.midx + self.x_offset + (self.tri_size * math.sqrt(3)) // 2)),
                    int((self.midy + self.y_offset - self.tri_size // 2)))

        self.pt3 = (int((self.midx + self.x_offset - (self.tri_size * math.sqrt(3)) // 2)),
                    int((self.midy + self.y_offset - self.tri_size // 2)))

        self.output_txt = None

        super().__init__(texture_size=texture_size, texture_name=texture_name)

    def create_texture(self):
        circle_texture = np.zeros((self.texture_size[1], self.texture_size[0]))

        cv2.circle(circle_texture, self.pt1, self.radius, 255, -1)
        cv2.circle(circle_texture, self.pt2, self.radius, 255, -1)
        cv2.circle(circle_texture, self.pt3, self.radius, 255, -1)

        output = np.uint8(circle_texture)

        return output

    def __str__(self):
        return f"{type(self).__name__} size:{self.texture_size} center:{self.midx, self.midy} radius:{self.radius}"

    def projct_coords(self):
        return [self.pt1, self.pt2, self.pt3]


# special exception for our calibration
class CalibrationException(Exception):
    """
    Blob detection for calibration failed
    """
    pass


def calibration_stimulus_wrapper(projected_pts_dump, mon=1, rad=7, tri=175,
                                 x_off=-150, y_off=-150, offset_window=(0, 50), tex_size=(1024, 1024)):

    triangle_circles = CalibrationTriangles(
        circle_radius=rad,
        tri_size=tri,
        x_off=x_off,
        y_off=y_off,
        texture_size=tex_size
    )
    # triangle_circles.view()
    projected_pts_dump.put(triangle_circles.projct_coords())
    circle_stim = LocalTexFixed(triangle_circles, monitor=mon, profile_on=False, window_name='calibrator_triangle',
                                win_offset=offset_window)
    circle_stim.run()


# used in centering and calibration. takes an img off stytra's camera feed
def img_receiver(socket, flags=0):
    string = socket.recv_string(flags=flags)
    msg_dict = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags)
    _img = np.frombuffer(bytes(memoryview(msg)), dtype=msg_dict['dtype'])
    img = _img.reshape(msg_dict['shape'])
    return np.array(img)


class StimulusCalibrator:
    def __init__(self, camera_img, proj_pts):

        self.camera_img = camera_img - 1
        self.projected_pts = np.array(proj_pts)
        self.projected_pts = self.projected_pts[np.argsort(self._find_angles(self.projected_pts)), :]
        self.camera_pts = self._find_triangle(self.camera_img)

    def transforms(self):
        x_proj = np.vstack([self.projected_pts.T, np.ones(3)])
        x_cam = np.vstack([self.camera_pts.T, np.ones(3)])
        proj_to_camera = self.camera_pts.T @ np.linalg.inv(x_proj)
        camera_to_proj = self.projected_pts.T @ np.linalg.inv(x_cam)

        print('cam coords:', self.camera_pts)
        print('projected in cam coords:', cv2.transform(np.reshape(self.projected_pts, (3, 1, 2)), proj_to_camera))
        return proj_to_camera, camera_to_proj

    def return_means(self):
        return np.mean(self.camera_pts, axis=0)

    @staticmethod
    def _find_triangle(image, blob_params=None):
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.maxThreshold = 255;
        if blob_params is None:
            blobdet = cv2.SimpleBlobDetector_create()
        else:
            blobdet = cv2.SimpleBlobDetector_create(blob_params)

        scaled_im = 255 - (image.astype(np.float32) * 255 / np.max(image)).astype(
            np.uint8
        )
        keypoints = blobdet.detect(scaled_im)
        if len(keypoints) != 3:
            raise CalibrationException("3 points for calibration not found")
        kps = np.array([k.pt for k in keypoints])

        # Find the angles between the points
        # and return the points sorted by the angles

        return kps[np.argsort(StimulusCalibrator._find_angles(kps)), :]

    @staticmethod
    def _find_angles(kps):
        angles = np.empty(3)
        for i, pt in enumerate(kps):
            pt_prev = kps[(i - 1) % 3]
            pt_next = kps[(i + 1) % 3]
            # angles are calculated from the dot product
            angles[i] = np.abs(
                np.arccos(
                    np.sum((pt_prev - pt) * (pt_next - pt)) / np.product(
                        [np.sqrt(np.sum((pt2 - pt) ** 2)) for pt2 in [pt_prev, pt_next]]
                    )
                )
            )
        return angles


def calibration_fxn(calibrate, input_socket, pt_dump, ):
    if not calibrate:
        while True:
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.SUBSCRIBE, b'calibration')
            socket.connect(str("tcp://localhost:") + str(input_socket))

            outputs = img_receiver(socket)
            img = outputs
            mywind = gw.getWindowsWithTitle('calibrator_triangle')[0]
            mywind.close()

            proj_pts = pt_dump.get()
            proj_to_camera, camera_to_proj = StimulusCalibrator(img, proj_pts).transforms()
            print('calibrated!')

            np.save('matt_calibration_params_cam2proj.npy', camera_to_proj)
            np.save('matt_calibration_params_proj2cam.npy', proj_to_camera)
            sys.exit()


class LocalSub:
    """
    Subscriber wrapper, not different from generic in utils, but allows for potential customization.
    This just creates a listener on a port specified
    """
    def __init__(self, port="1234", topic=""):
        """

        @param port: whichever port messages are being sent on
        @param topic: listener can be restricted to a certain message topic
        """

        self.port = port
        self.topic = topic
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect('tcp://localhost:' + str(self.port))
        self.socket.subscribe(self.topic)
        # print(self.port, self.topic)

    def kill(self):
        self.socket.close()
        self.context.term()


class LocalMonitor(DirectObject.DirectObject):
    """
    Again similar to generic monitor in utils.
    Feed into this a set up monitor (above class), this takes messages from zmq and converts to messages.
    Allows for potential processing, then sends messages which can be nicely accepted and processed within a stimulus
    """
    def __init__(self, subscriber):
        self.sub = subscriber

        self.run_thread = tr.Thread(target=self.run)

        self.run_thread.daemon = True
        self.run_thread.start()

    def run(self):
        # this is run on a separate thread so it can sit in a loop waiting to receive messages
        while True:
            topic = self.sub.socket.recv_string()
            data = self.sub.socket.recv_pyobj()
            # print(data)
            # this is a duplication at the moment, but provides an intermediate processing stage
            if data[0] == 'next_stimulus':
                messenger.send('next_stimulus')
            elif data[0] == 'centering':
                # print('MONITOR SENDING CENTERING')
                messenger.send('centering')
            elif data[0] == 'stat_time':
                messenger.send('stat_stim')

            elif data[0] == 'center':
                messenger.send('center_position', [data[1]])
            elif data[0] == 'live_center':
                messenger.send('adjust_center', [data[1]])
            elif data[0] == 'adjust_stim':
                messenger.send('live_thetas', [data[1]])
            elif data[0] == 'GO':
                messenger.send('begin_exp')
            elif data[0] == 'end_experiment':
                messenger.send('end_experiment')

    def kill(self):
        self.run_thread.join()


class RadialSinCube(LocalTextureBase):
    def __init__(self, texture_size=(1024, 1024), phase=0, period=32, texture_name='radial_sin_centering'):
        """
        Each run of this creates 1 frame of the radial_sin_centering stim, adjust phase to make full stack
        @param texture_size: should correspond to other textures being used
        @param phase: adjusts movement of waves
        @param period: adjust spacing of waves
        @param texture_name: name
        """
        self.texture_size = texture_size
        self.texture_name = texture_name
        self.phase = phase
        self.period = period

        self.texture_array = self.create_texture()

        self.texture = Texture(self.texture_name)

        # Set texture formatting (greyscale or rgb have different settings)

        self.texture.setup2dTexture(self.texture_size[0], self.texture_size[1],
                                    Texture.T_unsigned_byte,
                                    Texture.F_luminance)
        self.texture.setRamImageAs(self.texture_array, "L")

    def create_texture(self):
        x = np.linspace(-self.period*np.pi, self.period*np.pi, self.texture_size[0])
        y = np.linspace(-self.period*np.pi, self.period*np.pi, self.texture_size[1])
        return np.round((2*np.pi/self.period)*np.sin(np.sqrt(x[None, :]**2 +  y[:, None]**2)+self.phase)*127+127).astype(np.uint8)

    def __str__(self):
        return f"{type(self).__name__} size:{self.texture_size} period:{self.period}"


class LocalCircleGrayTex(LocalTextureBase):
    """
    Filled circle: grayscale on grayscale with circle_radius, centered at circle_center
    with face color fg_intensity on background bg_intensity. Center position is in pixels
    from center of image.
    """

    def __init__(self, texture_size=(512, 512), texture_name="gray_circle", circle_center=(0, 0),
                 circle_radius=5, bg_intensity=0, fg_intensity=255):
        self.center = circle_center
        self.radius = circle_radius
        self.bg_intensity = bg_intensity
        self.fg_intensity = fg_intensity
        super().__init__(texture_size=texture_size, texture_name=texture_name)

    def create_texture(self):
        min_int = np.min([self.fg_intensity, self.bg_intensity])
        max_int = np.max([self.fg_intensity, self.bg_intensity])
        if max_int > 255 or min_int < 0:
            raise ValueError('Circle intensity must lie in [0, 255]')
        x = np.linspace(-self.texture_size[0] / 2, self.texture_size[0] / 2, self.texture_size[0])
        y = np.linspace(-self.texture_size[1] / 2, self.texture_size[1] / 2, self.texture_size[1])
        X, Y = np.meshgrid(x, y)
        circle_texture = self.bg_intensity * np.ones((self.texture_size[0], self.texture_size[1]), dtype=np.uint8)
        circle_mask = (X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2 <= self.radius ** 2
        circle_texture[circle_mask] = self.fg_intensity
        return np.uint8(circle_texture)

    def __str__(self):
        part1 = f"{type(self).__name__} size:{self.texture_size} center:{self.center} "
        part2 = f"radius:{self.radius} bg:{self.bg_intensity} fg:{self.fg_intensity}"
        return part1 + part2


def center_finder(input_socket, out):
    while True:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b'centering')
        socket.connect(str("tcp://localhost:") + str(input_socket))
        # print(pusheen_receiver(socket))
        outputs = img_receiver(socket)

        img = outputs[:] - 3
        img[img < 0] = 0
        _img = np.array(img)

        def draw(event, x, y, flags, params):
            if event==1:
                cv2.line(_img, pt1=(x,y), pt2=(x,y), color=(255,255,255), thickness=3)
                cv2.destroyAllWindows()

        cv2.namedWindow('window')
        cv2.setMouseCallback('window', draw)
        mywind = gw.getWindowsWithTitle('window')[0]
        mywind.minimize()
        mywind.restore()
        mywind.maximize()
        cv2.imshow('window', _img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(_img)
        # print(maxloc[0], maxloc[1])
        new_center = np.array([maxloc[0], maxloc[1]])
        out.put(new_center)


def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi


def angle_mean(angles, axis=0):
    """Correct calculation of a mean of an array of angles
    """
    return np.arctan2(np.sum(np.sin(angles), axis), np.sum(np.cos(angles), axis))


def protocol_runner(stimuli, pandas_port, go_number, time_sock, centering_dump, fish_dump, center=None, automated=False):
    if center is None:
        center = (608, 608)

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

    def position_xformer(raw_pos, xy_flipped=False):
        calibrator = np.load('matt_calibration_params_cam2proj.npy')
        if xy_flipped:
            _x = 1
            _y = 0
        else:
            _x = 0
            _y = 1

        pos = (raw_pos[_x], raw_pos[_y])
        conv_pt = cv2.transform(np.reshape(pos, (1, 1, 2)), calibrator)[0][0]
        return conv_pt
        # x = -1*((conv_pt[0]/1024) - 0.5)
        # y = -1*((conv_pt[1]/1024) - 0.5)
        #
        # return x, y


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
    max_cutoff = 9999999999 # 7200 is two hours
    start_dst = 300
    curr_trial = -1
    last_message = None
    fish_info = [True]
    theta_diffs = []
    first_pass = True
    diff = [0, 0]
    xydiff = []
    fish_deltas = []
    convolved_thetas = []
    xy_coords = []
    fish_pres = []
    thetaSentValue = 0
    stat_times = stimuli.stat_time.values
    sent_theta = 5
    xy_out = [0,0]

    smoothing_filter = np.ones(15)/15
    # will be updated to be while trial <= max trials
    while not experiment_not_started and curr_trial < trials:


        # prevent memory buildup by keeping buffer to 10
        while len(fish_deltas) > 10:
            # fish_deltas = fish_deltas[1:]
            del fish_deltas[0]
        while len(xy_coords) > 10:
            # xy_coords = xy_coords[1:]
            del xy_coords[0]
        # while len(diff) > 3:
        #     # xy_coords = xy_coords[1:]
        #     del diff[0]

        while first_pass:
            while not fish_dump.empty():
                fish_dump.get()
            fish_info = [True]
            first_pass = False

        '''if stim_time != last_sent: # minel - commented out so that stytra ends on time even when fish is not present
            # print('time_left', tmax-stim_time)
            if not automated:
                t_socket.send_string(time_topic, zmq.SNDMORE)
                t_socket.send_pyobj([tmax, stim_time])
                last_sent = stim_time
            else:
                pass'''
    #
        # when we get new centers or fish data, bring it in
        if not centering_dump.empty():
            center = centering_dump.get()
            ncenter = [i for i in center]

            print('new center ', ncenter, 'calibrated center:', position_xformer(ncenter))
            p_socket.send_string(stimulus_topic, zmq.SNDMORE)
            p_socket.send_pyobj(['center', ncenter])
        if not fish_dump.empty():
            fish_info = fish_dump.get()
            m_t = fish_info[1]
            fish_info = fish_info[0]

        '''nofish = fish_info[0] # minel - commented out so that the fish does not have to be in the center to start the stimulus
        if nofish:
            message = 'centering'
            if last_message != message:
                p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                p_socket.send_pyobj([message])
                last_message = message'''

        try:
            fish_coords = fish_info[1]
            fish_deltas = [fish_info[2], fish_info[2]]
            xy_coords = [fish_coords]

            # send initial theta + xy
            p_socket.send_string(stimulus_topic, zmq.SNDMORE)
            p_socket.send_pyobj(['adjust_stim', fish_info[2]])
            p_socket.send_string(stimulus_topic, zmq.SNDMORE)
            p_socket.send_pyobj(['live_center', fish_coords])

        except IndexError:
            pass

        curr_trial += 1
        # start a stimulation and start a timer
        p_socket.send_string(stimulus_topic, zmq.SNDMORE)
        p_socket.send_pyobj(['next_stimulus'])
        stimulating = True
        t0 = time.time()
        stat_fixed = False
        loc_msg = 0

        while stimulating:
            # run a timer and keep some updated fish info
            elapsed = time.time()
#                 # fish runner for the if fish conditions
#                 # will need theta and xy list for smoothing (xy for % change)
            if elapsed - t0 <= stimuli['duration'].values[curr_trial] and stimuli.loc[curr_trial].texture.texture_name == "blank_tex":
                if loc_msg ==0:
                    print('blank texture proceeding for:', stimuli.loc[curr_trial].duration)
                    loc_msg+=1
                continue
            try:
                if elapsed- t0 >= stat_times[curr_trial] and not stat_fixed:
                    p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                    p_socket.send_pyobj(['stat_time'])
                    stat_fixed = True

            except IndexError:
                print('exp finished')
                p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                p_socket.send_pyobj(['end_experiment'])
                # sys.exit()

            if not fish_dump.empty():
                fish_info = fish_dump.get()
                m_t = fish_info[1]
                fish_info = fish_info[0]
                fish_pres.append(fish_info[0])

                if not fish_pres[-1]:
                    fish_deltas.append(fish_info[2])
                    xy_coords.append(fish_info[1])

            '''if elapsed - m_t >= 0.5: # minel - commented out because I want stimuli to continue even when Stytra cannot track
                stim_time += stimuli['duration'].values[curr_trial]
                p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                p_socket.send_pyobj(['centering'])
                p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                p_socket.send_pyobj(['centering'])
                stimulating = False
                # print('no fish, recentering')'''

                # prevent memory buildup by keeping buffer to 10
            # print(fish_deltas)
            if len(fish_pres) > 5:
                fish_pres = fish_pres[-3:]
            if len(fish_deltas) > 50:
                fish_deltas = fish_deltas[-25:]
                # del fish_deltas[-1]
            if len(xy_coords) > 5:
                xy_coords = xy_coords[-5:]
            if len(convolved_thetas) > 50:
                convolved_thetas = convolved_thetas[-30:]
            #     # del xy_coords[-1]
            # if len(diff) > 3:
            #     diff = diff[-3:]
                # del diff[0]
            # print(fish_pres)
            # if len(xy_coords)>1:
            #     xydiff.append(np.cumsum(abs(np.diff(xy_coords, axis=0)))[-1])
            # print(xydiff)

            try:
                convolved_thetas.append(np.convolve(fish_deltas, smoothing_filter, 'valid')[-1])
                thetaOut = reduce_to_pi(convolved_thetas[-1]) * 180 / np.pi
                if thetaOut >= 360:
                    thetaOut -= 360
                if thetaOut < 0:
                    thetaOut += 360

                if abs((thetaOut - thetaSentValue) / thetaOut) * 100 >= 8:
                    outputBool = True
                else:
                    outputBool = False
                if not np.isnan(thetaOut) and outputBool:
                    thetaSentValue = thetaOut
                    p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                    p_socket.send_pyobj(['adjust_stim', thetaOut])
            except:
                pass

                '''if sent_theta < -0.5:
                        if new_theta >= sent_theta * 1.1 and new_theta <= sent_theta * 0.9:
                            pass
                            # print('pass')
                        else:
                            sent_theta = new_theta
                            p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                            p_socket.send_pyobj(['adjust_stim', sent_theta])
                    elif sent_theta > 0.5:
                        if sent_theta * 1.1 >= new_theta >= sent_theta * 0.9:
                            pass
                            # print('pass')
                        else:
                            sent_theta = new_theta
                            p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                            p_socket.send_pyobj(['adjust_stim', sent_theta])
                    elif -0.5 <= sent_theta <= 0.5:
                        if sent_theta == 0:
                            sent_theta = 0.001
                        if abs((abs(sent_theta) - abs(new_theta))*180/np.pi) >= 15:
                            sent_theta = new_theta
                            p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                            p_socket.send_pyobj(['adjust_stim', sent_theta])'''


            #     p_socket.send_string(stimulus_topic, zmq.SNDMORE)
            #     p_socket.send_pyobj(['adjust_stim', theta_out])
            if elapsed - t0 >= stimuli['duration'].values[curr_trial]:
                stim_time += stimuli['duration'].values[curr_trial]
                #p_socket.send_string(stimulus_topic, zmq.SNDMORE) - minel - commented out, do not center between stimuli
                #p_socket.send_pyobj(['centering']) - minel - commented out, do not center between stimuli
                stimulating = False
                print('finished_stim') # minel - removed the print 'recentering'
                print(curr_trial, trials)
                if curr_trial >= trials:
                    print('Exp Done')
                    p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                    p_socket.send_pyobj(['end_experiment'])
                    t_socket.send_string(time_topic, zmq.SNDMORE)
                    t_socket.send_pyobj([10, 10])

                    # sys.exit()

            try:
                if len(xy_coords) == 1:
                    xy_out = xy_coords[0]
        #
                xy_avg = np.mean(xy_coords, axis=0)
                # x1 = abs(xy_avg - xy_coords[-1])[0] / xy_avg[0]
                # y1 = abs(xy_avg - xy_coords[-1])[1] / xy_avg[1]
                if xy_out[0]*0.9 < xy_avg[0] < xy_out[0]*1.1 and xy_out[1]*0.9 < xy_avg[1] < xy_out[1]*1.1:
                    pass
                else:
                    xy_out = xy_coords[-1]
                    message = xy_out
                    if last_message != message:
                        p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                        p_socket.send_pyobj(['live_center', message])
                        last_message = message
            except:
                pass

                    # theta_means = angle_mean(fish_deltas)
                    #                 print(etc)
                    #                 # theta_diffs.append(min((2 * np.pi) - abs(fish_deltas - last_delta), abs(fish_deltas - last_delta)))
                    #                 # last_delta = fish_deltas
                    #                 # print(theta_diffs[-5:], np.mean(theta_diffs[-10:]), last_delta)
                    #                 # theta_means = 0.1 ; fish_deltas = [0.1 ,0]
    # #                 # print(theta_means, angle_mean(fish_deltas[:-5]), (abs(angle_mean(fish_deltas[:-5])) - abs(theta_means)) / abs(theta_means))
    # #                 a1 = theta_means
    #                 a2 = fish_deltas[-1]
    # #                 # while len(fish_deltas) > 20:
    # #                 #     # fish_deltas = fish_deltas[1:]
    # #                 #     del fish_deltas[0]
    # #                 # print(a2, theta_means, min((2 * np.pi) - abs(theta_means - a2), abs(theta_means - a2)))
    #                 diff.append(min((2 * np.pi) - abs(theta_means - a2), abs(theta_means - a2)))
    #                 if np.mean(diff[-3:]) > 0.15:
    #                     # print('new angle, diff:', diff, 'old:', fish_deltas[-2], 'new:', fish_deltas[-1])
    #                     p_socket.send_string(stimulus_topic, zmq.SNDMORE)
    #                     p_socket.send_pyobj(['adjust_stim', a2])
    #                 # if diff*180/np.pi > 30 and some_hold != 1:
    #                 #     print(diff, a2, a1)
    #                 #     some_hold = 1
    #                 #     val_hold = a2
    #                 #
    #                 # elif some_hold == 1:
    #                 #     if diff * 180 / np.pi > 30:
    #                 #         print('refreshing', diff, a2, a1)
    #                 #         fish_deltas = [val_hold, val_hold, a2, a2]
    #                 #         some_hold = 0
    #                 # else:
    #                 #     pass
    #
    #
                    # if abs(abs(angle_mean(fish_deltas[:-5])) - abs(theta_means)) / abs(theta_means) < 0.15:
                    #     pass
                    # else:
                    #     pass
                    #     # print(fish_deltas)
                    #     theta_out = angle_mean(fish_deltas[:-3])
                    #     for i in range(2):
                    #         fish_deltas.append(angle_mean(fish_deltas[:-3]))
                    #
                    #     p_socket.send_string(stimulus_topic, zmq.SNDMORE)
                    #     p_socket.send_pyobj(['adjust_stim', theta_out])
                        # print('new theta:', theta_out)
    #
    #                 # print(stimulus, stim_dict[stimulus]['live'])
    #                 # theta_means = angle_mean(fish_deltas)
    #
    #                 # else:
    #                 #     message3 = 'centering'
    #                 #     p_socket.send_string(stimulus_topic, zmq.SNDMORE)
    #                 #     p_socket.send_pyobj([message3])
    #                 #     # print('centering')
    #                 # print(np.linalg.norm(fish_coords - center))
    #                 # print(fish_coords, center)
    #         else:
    #             message = 'centering'
    #             if last_message != message:
    #                 p_socket.send_string(stimulus_topic, zmq.SNDMORE)
    #                 p_socket.send_pyobj([message])
    #                 last_message = message
    #
    #     if curr_trial > trials:
    #         print('exp finished')
    #         p_socket.send_string(stimulus_topic, zmq.SNDMORE)
    #         p_socket.send_pyobj('0')
    #         # styt_wind = gw.getWindowsWithTitle('Stytra | Dummy')[0]
    #         # styt_wind.close()
    #         sys.exit()

def pos_receiver(fish_dump):
    time.sleep(8)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'fishies')
    path = 'fish_port.dat'
    with open(path) as file:
        for line in file:
            lst = line.split()
    socket.connect("tcp://localhost:" + str(lst[0]))
    while True:
        topic = socket.recv_string()
        msgs = socket.recv_pyobj()
        # print(msgs)
        # print(msgs[0])
        fish_dump.put([msgs, time.time()])


class ClosedLoopStimuli(ShowBase):
    """
    runs and updates stimuli based on messages
    """
    def __init__(self, stimuli, fps=60, radial_centering=False, profile_on=True, save_path=None, window_size=None,
                 window_name='ClosedLoop', monitor=1, win_offset=(0,0), fish_id=0, fish_age=6, center_pt=None,
                 radial_centering_stack=None, proj_fish=False, automated=False):
        super().__init__()

        # all the stimuli, including tex and params
        self.stimuli = stimuli

        self.proj_fish = proj_fish

        # setting initial stimulus
        self.curr_id = -1
        #self.current_stim = self.curr_params(self.curr_id)
        self.stimulus_initialized = False

        self.last_time = 0
        self.dots_made = False
        self.rotation_offset = 0

        # panda3d variables
        self.fps = fps
        self.profile_on = profile_on
        self.save_path = save_path
        self.disable_mouse()

        # framerate
        ShowBaseGlobal.globalClock.setMode(ClockObject.MLimited)
        ShowBaseGlobal.globalClock.setFrameRate(self.fps)

        # Window properties
        if window_size is None:
            self.window_size = self.current_stim['texture'].texture_size
        else:
            self.window_size = window_size

        self.window_properties = WindowProperties()
        self.window_properties.setSize(self.window_size[0], self.window_size[1])

        self.window_name = window_name
        self.window_position = ((monitor * 1920) + (self.window_size[0] // 2) + win_offset[0], win_offset[1])

        self.window_properties.set_undecorated(True)
        self.window_properties.set_origin(self.window_position)
        self.window_properties.set_foreground(True)
        self.set_title(self.window_name)
        ShowBaseGlobal.base.win.requestProperties(self.window_properties)

        # emailing part
        self.send_email = 'mdl.python@gmail.com'
        self.receive_email = 'minelarinel@gmail.com'
        self.email_port = 465
        self.email_context = ssl.create_default_context()

        self.radial_centering = radial_centering
        if self.radial_centering:
            if radial_centering_stack is not None:
                self.centering_stack = radial_centering_stack
            else:
                self.centering_stack = self.radial_sin(self.window_size)
            self.centering_index = 0
            self.centering_stack_size = len(self.centering_stack)
            self.center_card_created = False
            self.curr_txt = self.centering_stack[self.centering_index]
            self.current_stim = {'stim_type' : 'centering', 'velocity' : 0, 'angle' : 0, 'texture': self.curr_txt, 'stat_time':0}
            # self.current_stim = {'stim_type' : 's', 'velocity' : 0, 'angle' : 0, 'texture': self.centering_stim, 'stat_time':0}

        else:
            self.centering_stim = LocalCircleGrayTex(texture_size=self.window_size, circle_radius=20)
            self.current_stim = {'stim_type' : 's', 'stim_name' : 'centerdot', 'velocity' : 0, 'angle' : 0, 'texture': self.centering_stim, 'stat_time':0}
            self.center_card_created = False

        self._centering = True

        # set up saving
        if save_path:
            if '\\' in self.save_path:
                self.save_path = self.save_path.replace('\\', '/')

            val_offset = 0
            newpath = self.save_path
            while os.path.exists(newpath):
                val_offset += 1
                newpath = self.save_path[:self.save_path.rfind('/') + 1] + self.save_path[
                                                                           self.save_path.rfind('/') + 1:][:-4] \
                          + '_' + str(val_offset) + '.txt'

            self.save_path = newpath
            self.filestream = updated_saving(self.save_path, fish_id, fish_age)
        else:
            self.filestream = None

        try:
            self.calibrator = np.load('matt_calibration_params_cam2proj.npy')
        except:
            print('error loading calibration')
            pass

        self.strip_angle = 90

        if center_pt is None:
            center_pt = (663, 492)

        self.center = center_pt
        self.scale = np.sqrt(8)
        self.true_center_x, self.true_center_y = self.position_xformer(self.center)
        # print(self.position_xformer(self.center))


        self.fish_angle = 0
        self.center_x = self.true_center_x.copy()
        self.center_y = self.true_center_y.copy()
        self.bin_center_x = -1 * self.center_x * self.scale
        self.bin_center_y = -1 * self.center_y * self.scale

        # Set up profiling
        self.profile_on = profile_on
        if self.profile_on:
            PStatClient.connect()  # this will only work if pstats is running
            ShowBaseGlobal.base.setFrameRateMeter(True)  # Show frame rate

        self.accept('next_stimulus', self.advance_stimulus)
        self.accept('stat_stim', self.unset_stationary)
        self.accept('begin_exp', self.begin_move)
        self.accept('centering', self.centering_stimulus)
        self.accept('adjust_center', self.adjust_center, [])
        self.accept('center_position', self.center_pos_changes, [])
        self.accept('live_thetas', self.change_theta, [])
        self.accept('end_experiment', self.exp_end)

        self.set_stimulus(self.current_stim)

        self.automated = automated
        if self.automated:
            self.begin_move()

    def exp_end(self):
        # This is the end of the experiment
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', self.email_port, context=self.email_context) as server:
                server.login(self.send_email, self.return_pass())
                server.sendmail(self.send_email, self.receive_email, 'experiment finished rig 1')
        except:
            print('email sending failed')

        print('Exp Finished!')
        self.filestream.close()
        final_saving(self.save_path)
        # gw.getWindowsWithTitle(self.window_name)[0].close()
        sys.exit()

    def begin_move(self):
        self.taskMgr.add(self.move_textures, "move textures")

    def curr_params(self, curr_index):
        try:
            params = self.stimuli.loc[curr_index].copy()
        except KeyError:
            # This is the end of the experiment
            try:
                with smtplib.SMTP_SSL('smtp.gmail.com', self.email_port, context=self.email_context) as server:
                    server.login(self.send_email, self.return_pass())
                    server.sendmail(self.send_email, self.receive_email, 'experiment finished rig 1')
            except:
                print('email sending failed')

            print('not enough stimuli')
            final_saving(self.save_path)
            # gw.getWindowsWithTitle(self.window_name)[0].close()
            sys.exit()
        return params

    def unset_stationary(self):
        self.current_stim['velocity'] = self.curr_params(self.curr_id)['velocity']
        if self.filestream and self.current_stim['stat_time'] != 0:
            saved_stim = dict(self.curr_params(self.curr_id).copy())
            saved_stim.pop('texture')
            self.filestream.write("\n")
            self.filestream.write(f"{str(datetime.now())}: {self.curr_id} {saved_stim}")
            self.filestream.flush()

    def set_title(self, title):
        self.window_properties.setTitle(title)
        ShowBaseGlobal.base.win.requestProperties(self.window_properties)

    def set_stimulus(self, stim):
        if not self.stimulus_initialized:
            self.stimulus_initialized = True
        elif self.current_stim['stim_type'] == 'centering':
            self.clear_cards()

        self.current_stim = stim.copy()
        if self.current_stim['stat_time'] != 0:
            if self.current_stim['stim_type'] == 'b':
                self.current_stim['velocity'] = (0, 0)
            else:
                self.current_stim['velocity'] = 0
        # self.center_x = self.true_center_x
        # self.center_y = self.true_center_y
        # print(self.current_stim['angle'])
        if self.current_stim['stim_type'] != 'centering':
            print('showing:', self.current_stim['stim_name'])
        self.create_texture_stages()
        self.create_cards()
        self.set_texture_stages()
        self.set_transforms()

        if self.filestream:
            saved_stim = dict(self.current_stim.copy())
            saved_stim.pop('texture')
            self.filestream.write("\n")
            self.filestream.write(f"{str(datetime.now())}: {self.curr_id} {saved_stim}")
            self.filestream.flush()

    def center_pos_changes(self, data):
        self.true_center_x, self.true_center_y = self.position_xformer(data)
        print('center is:', self.true_center_x, self.true_center_y)
        self.center_x = self.true_center_x.copy()
        self.center_y = self.true_center_y.copy()
        self.bin_center_x = -1 * self.center_x * self.scale
        self.bin_center_y = -1 * self.center_y * self.scale
        self.set_transforms()

    def change_theta(self, data):
        # print(data)
        # data = data * 180/np.pi

        # self.strip_angle = self.reduce_to_pi(data + self.rotation_offset)
        # self.fish_angle = self.reduce_to_pi(data)
        # print(data)
        self.strip_angle = data + self.rotation_offset
        self.fish_angle = data
        self.set_transforms()

        self.set_transforms()
        # print('changed theta to', self.strip_angle)

    def adjust_center(self, data):

        self.center_x, self.center_y = self.position_xformer(data)

        self.bin_center_x = -1 * self.center_x * self.scale
        self.bin_center_y = -1 * self.center_y * self.scale
        # print('adjusted xy', data, self.center_x, self.center_y, self.bin_center_x, self.bin_center_y)
        self.set_transforms()

    def position_xformer(self, raw_pos, xy_flipped=False):

        if xy_flipped:
            _x = 1
            _y = 0
        else:
            _x = 0
            _y = 1

        pos = (raw_pos[_x], raw_pos[_y])
        conv_pt = cv2.transform(np.reshape(pos, (1, 1, 2)), self.calibrator)[0][0]

        x = -1*((conv_pt[0]/self.window_size[0]) - 0.5)
        y = -1*((conv_pt[1]/self.window_size[1]) - 0.5)

        return x, y

    def centering_stimulus(self):
        # print('centering in pandas')
        self._centering = True
        if self.radial_centering:
            self.curr_txt = self.centering_stack[self.centering_index]
            self.clear_cards()
            self.current_stim = {'stim_type' : 'centering', 'angle': 0, 'velocity':0, 'texture': self.curr_txt, 'stat_time':0}
            self.set_stimulus(self.current_stim)
        else:
            self.clear_cards()
            self.current_stim = {'stim_type' : 's', 'velocity' : 0, 'angle' : 0, 'texture': self.centering_stim, 'stat_time':0}
            self.set_stimulus(self.current_stim)
        # print(self.center_x, self.center_y)

        if self.filestream:
            saved_stim = self.current_stim.copy()
            saved_stim.pop('texture')
            self.filestream.write("\n")
            self.filestream.write(f"{str(datetime.now())}: {self.curr_id} {saved_stim}")
            self.filestream.flush()

    def move_textures(self, task):
        # moving the stimuli
        # print(self.current_stim)
        if self.current_stim['stim_type'] == 'b':
            left_tex_position = -task.time * self.current_stim['velocity'][0]  # negative b/c texture stage
            right_tex_position = -task.time * self.current_stim['velocity'][1]
            try:
                self.left_card.setTexPos(self.left_texture_stage, left_tex_position, 0, 0)
                self.right_card.setTexPos(self.right_texture_stage, right_tex_position, 0, 0)
            except Exception as e:
                print('error on move_texture_b')

        elif self.current_stim['stim_type'] == 's':
            if self.current_stim['velocity'] == 0:
                pass
            else:
                new_position = -task.time*self.current_stim['velocity']
                # Sometimes setting position fails when the texture stage isn't fully set
                try:
                    self.card.setTexPos(self.texture_stage, new_position, 0, 0) #u, v, w
                except Exception as e:
                    print('error on move_texture_s')

        elif self.current_stim['stim_type'] == 'rdk' and self.dots_made:
            dt = task.time - self.last_time
            self.last_time = task.time

            # because this isnt the 2D card, lets set up a lens to see it
            self.lens = PerspectiveLens()
            self.lens.setFov(90, 90)
            self.lens.setNearFar(0.001, 1000)
            self.lens.setAspectRatio(1)
            self.cam.node().setLens(self.lens)

            # ???
            random_vector = np.random.randint(100, size=10000)
            self.coherent_change_vector_ind = np.where(random_vector < self.current_stim['coherence'])

            #######
            # Continously update the dot stimulus
            #####
            self.dots_position[0, :, 0][self.coherent_change_vector_ind] += \
                np.cos(self.current_stim['angle'] * np.pi / 180) * self.current_stim['velocity'] * dt

            self.dots_position[0, :, 1][self.coherent_change_vector_ind] += \
                np.sin(self.current_stim['angle'] * np.pi / 180) * self.current_stim['velocity'] * dt

            # Randomly redraw dot with a short lifetime
            k = np.random.random(10000)
            if self.current_stim['lifetime'] == 0:
                ind = np.where(k >= 0)[0]
            else:
                ind = np.where(k < dt / self.current_stim['lifetime'])[0]

            self.dots_position[0, :, 0][ind] = 2 * np.random.random(len(ind)).astype(np.float32) - 1  # x
            self.dots_position[0, :, 1][ind] = 2 * np.random.random(len(ind)).astype(np.float32) - 1  # y
            self.dots_position[0, :, 2] = np.ones(10000) * self.current_stim['brightness']

            # Wrap them
            self.dots_position[0, :, 0] = (self.dots_position[0, :, 0] + 1) % 2 - 1
            self.dots_position[0, :, 1] = (self.dots_position[0, :, 1] + 1) % 2 - 1

            memoryview(self.dummytex.modify_ram_image())[:] = self.dots_position.tobytes()

        elif self.current_stim['stim_type'] == 'centering' and self.radial_centering:
            # this value is modifiable to change speed of radial sine
            if task.time > 1.75:
                self.clear_cards()
                #print('showing centering index', self.centering_index)
                self.current_stim['texture'] = self.centering_stack[self.centering_index]
                # self.centering_stack[self.centering_index].view()
                self.set_stimulus(self.current_stim)
                self.centering_index += 1
                if self.centering_index == self.centering_stack_size:
                    self.centering_index = 0
        return task.cont

    def advance_stimulus(self):
        self._centering = False
        try:
            self.curr_id += 1
            self.clear_cards()
            self.current_stim = self.curr_params(self.curr_id)
            self.set_stimulus(self.current_stim)
        except IndexError:
            self.filestream.close()
            final_saving(self.save_path)
            sys.exit()

    def create_texture_stages(self):
        """
        Create the texture stages: these are basically textures that you can apply
        to cards (sometimes mulitple textures at the same time -- is useful with
        masks).
        For more on texture stages:
        https://docs.panda3d.org/1.10/python/programming/texturing/multitexture-introduction
        """
        # Binocular cards
        if self.current_stim['stim_type'] == 'b':
            # TEXTURE STAGES FOR LEFT CARD
            # Texture itself
            self.left_texture_stage = TextureStage('left_texture_stage')
            # Mask
            self.left_mask = Texture("left_mask_texture")
            self.left_mask.setup2dTexture(self.current_stim['texture'].texture_size[0],
                                          self.current_stim['texture'].texture_size[1],
                                          Texture.T_unsigned_byte, Texture.F_luminance)
            self.left_mask_stage = TextureStage('left_mask_array')

            # TEXTURE STAGES FOR RIGHT CARD
            self.right_texture_stage = TextureStage('right_texture_stage')
            # Mask
            self.right_mask = Texture("right_mask_texture")
            self.right_mask.setup2dTexture(self.current_stim['texture'].texture_size[0],
                                           self.current_stim['texture'].texture_size[1],
                                           Texture.T_unsigned_byte, Texture.F_luminance)
            self.right_mask_stage = TextureStage('right_mask_stage')

        # monocular cards
        elif self.current_stim['stim_type'] == 's':
            self.texture_stage = TextureStage("texture_stage")

        # random dots are special cards because they are actually full panda3d models with a special lens  to appear 2D
        # NOT the 2D card based textures the others are based on
        elif self.current_stim['stim_type'] == 'rdk':
            self.dot_motion_coherence_shader = [
                """ #version 140
                    uniform sampler2D p3d_Texture0;
                    uniform mat4 p3d_ModelViewProjectionMatrix;
                    in vec4 p3d_Vertex;
                    in vec2 p3d_MultiTexCoord0;
                    uniform int number_of_dots;
                    uniform float size_of_dots;
                    uniform float radius;
                    out float dot_color;
                    void main(void) {
                        vec4 newvertex;
                        float dot_i;
                        float dot_x, dot_y;
                        float maxi = 10000.0;
                        vec4 dot_properties;
                        dot_i = float(p3d_Vertex[1]);
                        dot_properties = texture2D(p3d_Texture0, vec2(dot_i/maxi, 0.0));
                        dot_x = dot_properties[2];
                        dot_y = dot_properties[1];
                        dot_color = dot_properties[0];
                        newvertex = p3d_Vertex;
                        if (dot_x*dot_x + dot_y*dot_y > radius*radius || dot_i > number_of_dots) { // only plot a certain number of dots in a circle
                            newvertex[0] = 0.0;
                            newvertex[1] = 0.0;
                            newvertex[2] = 0.0;
                        } else {
                            newvertex[0] = p3d_Vertex[0]*size_of_dots+dot_x;
                            newvertex[1] = 0.75;
                            newvertex[2] = p3d_Vertex[2]*size_of_dots+dot_y;
                        }
                        gl_Position = p3d_ModelViewProjectionMatrix * newvertex;
                    }
                """,

                """ #version 140
                    in float dot_color;
                    //out vec4 gl_FragColor;
                    void main() {
                        gl_FragColor = vec4(dot_color, dot_color, dot_color, 1);
                    }
                """
            ]
            self.compiled_dot_motion_shader = Shader.make(Shader.SLGLSL, self.dot_motion_coherence_shader[0],
                                                          self.dot_motion_coherence_shader[1])

            self.circles = self.loader.loadModel('circles.bam')

            self.dummytex = Texture("dummy texture")  # this doesn't have an associated texture (as above)
            self.dummytex.setup2dTexture(10000, 1, Texture.T_float, Texture.FRgb32)
            self.dummytex.setMagfilter(Texture.FTNearest)

            tex = TextureStage("dummy followup")
            tex.setSort(-100)  # ???

            self.circles.setTexture(tex, self.dummytex)
            self.circles.setShader(self.compiled_dot_motion_shader)

        elif self.current_stim['stim_type'] == 'centering':
            self.texture_stage = TextureStage('texture_stage')

    def return_pass(self):
        import pandastim.experiments.matt as matt
        return matt.password

    def create_cards(self):
        """
        Create cards: these are panda3d objects that are required for displaying textures.
        You can't just have a disembodied texture. In pandastim (at least for now) we are
        only showing 2d projections of textures, so we use cards.
        """
        cardmaker = CardMaker("stimcard")
        cardmaker.setFrameFullscreenQuad()

        # Binocular cards
        if self.current_stim['stim_type'] == 'b':
            self.setBackgroundColor((0, 0, 0, 1))  # without this the cards will appear washed out
            self.left_card = self.aspect2d.attachNewNode(cardmaker.generate())
            self.left_card.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.M_add))  # otherwise only right card shows

            self.right_card = self.aspect2d.attachNewNode(cardmaker.generate())
            self.right_card.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.M_add))

        # Tex card
        elif self.current_stim['stim_type'] == 's':
            self.card = self.aspect2d.attachNewNode(cardmaker.generate())
            self.card.setColor((1, 1, 1, 1))
            self.card.setScale(self.scale)

        elif self.current_stim['stim_type'] == 'centering':
             self.card = self.aspect2d.attachNewNode(cardmaker.generate())
             self.card.setColor((1, 1, 1, 1))  # ?
             # self.setBackgroundColor((0, 0, 0, 1))
             self.card.setScale(self.scale)
             self.center_card_created = True

        # attach model to card w/ the rdk stimulus
        elif self.current_stim['stim_type'] == 'rdk':
            self.card = self.render.attachNewNode('dumb node')
            self.circles.reparentTo(self.card)
            self.circles.setShaderInput("number_of_dots", int(self.current_stim['number']))
            self.circles.setShaderInput("size_of_dots", self.current_stim['size'])
            self.circles.setShaderInput("radius", self.current_stim['window'])
            self.setBackgroundColor(0, 0, 0, 1)

    def set_texture_stages(self):
        """
        Add texture stages to cards
        """
        if self.current_stim['stim_type'] == 'b':

            # self.mask_position_uv = (self.bin_center_x, self.bin_center_y)

            # CREATE MASK ARRAYS
            self.left_mask_array = 255 * np.ones((self.current_stim['texture'].texture_size[0],
                                                  self.current_stim['texture'].texture_size[1]), dtype=np.uint8)
            self.left_mask_array[:, (self.current_stim['texture'].texture_size[1] // 2)
                                 - self.current_stim['center_width'] // 2:] = 0

            self.right_mask_array = 255 * np.ones((self.current_stim['texture'].texture_size[0],
                                                   self.current_stim['texture'].texture_size[1]), dtype=np.uint8)
            self.right_mask_array[:,
            : (self.current_stim['texture'].texture_size[1] // 2) + self.current_stim['center_width'] // 2] = 0

            if self.proj_fish:

                half_tex = self.current_stim['texture'].texture_size[1] // 2
                ls = [-4, 1, 0, -3]
                rs = [-1, 4, 0, 3]
                set_val = 200

                self.left_mask_array[506:515, 511:512]= 120
                self.right_mask_array[506:515, 512:513]= 120
                self.left_mask_array[514:516, 510:512]= 255
                self.right_mask_array[514:516, 512:514]= 255
                # self.left_mask_array[half_tex + ls[0] : half_tex + ls[1], half_tex + ls[2] : half_tex + ls[3]] = set_val
                # self.right_mask_array[half_tex+ rs[0] : half_tex + rs[1], half_tex + rs[2]:half_tex + rs[3]] = set_val

                #self.left_mask_array[half_tex + ls[0] : half_tex + ls[1], half_tex + ls[2] : half_tex + ls[3]]
                # self.right_mask_array[]

            # ADD TEXTURE STAGES TO CARDS
            self.left_mask.setRamImage(self.left_mask_array)
            self.left_card.setTexture(self.left_texture_stage, self.current_stim['texture'].texture)
            self.left_card.setTexture(self.left_mask_stage, self.left_mask)

            # Multiply the texture stages together
            self.left_mask_stage.setCombineRgb(TextureStage.CMModulate,
                                               TextureStage.CSTexture,
                                               TextureStage.COSrcColor,
                                               TextureStage.CSPrevious,
                                               TextureStage.COSrcColor)
            self.right_mask.setRamImage(self.right_mask_array)
            self.right_card.setTexture(self.right_texture_stage, self.current_stim['texture'].texture)
            self.right_card.setTexture(self.right_mask_stage, self.right_mask)

            # Multiply the texture stages together
            self.right_mask_stage.setCombineRgb(TextureStage.CMModulate,
                                                TextureStage.CSTexture,
                                                TextureStage.COSrcColor,
                                                TextureStage.CSPrevious,
                                                TextureStage.COSrcColor)

        elif self.current_stim['stim_type'] == 's':
            self.card.setTexture(self.texture_stage, self.current_stim['texture'].texture)

        elif self.current_stim['stim_type'] == 'centering':
            self.card.setTexture(self.texture_stage, self.current_stim['texture'].texture)

    def set_transforms(self):
        """
        Set up the transforms to apply to textures/cards (e.g., rotations/scales)
        This is different from the framewise movement handled by the task manager
        """
        if self.current_stim['stim_type'] == 'b':
            self.mask_transform = self.trs_transform()

            # self.left_angle = self.reduce_to_pi(self.fish_angle+self.current_stim['angle'][0])
            # self.right_angle = self.reduce_to_pi(self.fish_angle+self.current_stim['angle'][1])
            self.left_angle = self.strip_angle + self.current_stim['angle'][0] + self.rotation_offset - 90
            self.right_angle = self.strip_angle + self.current_stim['angle'][1] + self.rotation_offset - 90

            self.left_card.setTexTransform(self.left_mask_stage, self.mask_transform)
            self.right_card.setTexTransform(self.right_mask_stage, self.mask_transform)
            # Left texture
            self.left_card.setTexScale(self.left_texture_stage, 1 / self.scale)
            self.left_card.setTexRotate(self.left_texture_stage, self.left_angle)

            # Right texture
            self.right_card.setTexScale(self.right_texture_stage, 1 / self.scale)
            self.right_card.setTexRotate(self.right_texture_stage, self.right_angle)

        elif self.current_stim['stim_type'] == 's' and not self._centering:
            self.card.setTexRotate(self.texture_stage, self.current_stim['angle'] + self.fish_angle - 90)
            self.card.setTexPos(self.texture_stage,  self.center_x, self.center_y, 0)

        elif self.current_stim['stim_type'] == 's' and  self._centering:
            self.card.setTexPos(self.texture_stage,  self.true_center_x, self.true_center_y, 0)

        elif self.current_stim['stim_type'] == 'centering':
            self.card.setTexPos(self.texture_stage,  self.true_center_x, self.true_center_y, 0)

        elif self.current_stim['stim_type'] == 'rdk':
            self.dots_position = np.empty((1, 10000, 3)).astype(np.float32)
            self.dots_position[0, :, 0] = 2 * np.random.random(10000).astype(np.float32) - 1  # x
            self.dots_position[0, :, 1] = 2 * np.random.random(10000).astype(np.float32) - 1  # y
            self.dots_position[0, :, 2] = np.ones(10000) * self.current_stim['brightness']
            self.dots_made = True
            self.card.setTexPos(self.texture_stage,  self.center_x, self.center_y, 0)




    def clear_cards(self):
        """
        Clear cards when new stimulus: stim-class sensitive
        """
        if self.center_card_created:
            # print('detached')
            self.card.detachNode()
            self.center_card_created = False

        elif self.current_stim['stim_type'] == 'b':
            self.left_card.detachNode()
            self.right_card.detachNode()
            # if self.profile_on:
            # self.center_indicator.detachNode()

        elif self.current_stim['stim_type'] == 's':
            self.card.detachNode()

        elif self.current_stim['stim_type'] == 'rdk':
            self.card.detachNode()

    def trs_transform(self):
        """
        trs = translate-rotate-scale transform for mask stage
        panda3d developer rdb contributed to this code
        """
        # self.mask_position_uv = (self.center_x, self.center_y)

        # # pos = 0.5 + self.mask_position_uv[0], 0.5 + self.mask_position_uv[1]
        # center_shift = TransformState.make_pos2d((self.mask_position_uv[0], self.mask_position_uv[1]))
        # scale = TransformState.make_scale2d(1 / self.scale)
        # rotate = TransformState.make_rotate2d(self.strip_angle)s
        # translate = TransformState.make_pos2d((0.5, 0.5))
        # return translate.compose(scale.compose(center_shift))
        if self.current_stim['stim_type'] == 'b':
            self.mask_position_uv = (self.bin_center_x, self.bin_center_y)
        else:
            self.mask_position_uv = (self.center_x, self.center_y)

        # print(self.curr_params)

        pos = 0.5 + self.mask_position_uv[0], 0.5 + self.mask_position_uv[1]
        center_shift = TransformState.make_pos2d((-pos[0], -pos[1]))
        scale = TransformState.make_scale2d(1 / self.scale)
        rotate = TransformState.make_rotate2d(self.strip_angle)
        translate = TransformState.make_pos2d((0.5, 0.5))

        return translate.compose(rotate.compose(scale.compose(center_shift)))

    @staticmethod
    def reduce_to_pi(ar):
        # ar = ar * np.pi/180
        return (np.mod(ar + np.pi, np.pi * 2) - np.pi)*180/np.pi

    @staticmethod
    def radial_sin(window_size):
        stack = []
        num_slices = 190
        phase_change = 0.1
        phase = 0
        for slice_num in range(num_slices):
            rad_slice = RadialSinCube(texture_size=window_size, phase=phase)
            stack.append(rad_slice)
            phase += phase_change
        return stack




# q = mp.Queue()
# calibration_stimulus_wrapper(q, mon=0)
