import numpy as np
import subprocess as sp
from .progress_bar import update_progress
import re
import os
from sys import platform


def ffmpeg_image_list_to_video(video_filename, image_list, fps=30, ffopts=[],
                               verbose=True):
    """Fast save of a list of images using only numpy and ffmpeg into a video

    Streans directly the the images to ffmpeg.
        

    Parameters
    ----------
    video_filename: string

    image_list: list of images

    fps: int, default is 30

    ffopts: array, default is []

    verbose: bool, default is True
    """
    n_images = len(image_list)
    for index, image in enumerate(image_list):
        if not index:
            cmd =['ffmpeg', '-y', '-s', '{0}x{1}'.format(image.shape[1], image.shape[0]),
                  '-r', str(fps),
                  '-an',
                  '-c:v', 'rawvideo', '-f', 'rawvideo',
                  '-pix_fmt', 'rgb24',
                  '-i', '-'] + ffopts + [video_filename]
            pipe = sp.Popen(cmd, stdin=sp.PIPE)
        pipe.stdin.write(image.tostring())

        if verbose:
            update_progress(percent=index/(n_images - 1),
                            title='Creating the video: ',
                            text=' Processing frame {}/{}'.format(index+1, n_images))

    pipe.stdin.close()
    pipe.wait()



def matplotlib_image_list_to_video(video_filename, image_list,
                                   dpi=220, fps=24,
                                   frame_numbers=False,
                                   verbose=True):
    """ Uses matplotlib.animation.writers['ffmpeg'] to save a list of images as a video

    Parameters
    ----------
    video_filename: string
        absolute path to the target video
        (e.g. : '/temp/video.mp4')

    image_list
        list like object of images

    dpi: int, default is 220

    fps: int, default is 15

    frame_numbers: bool, default is False
        if True, displays the frame numbers in the video

    verbose: int
        level of verbosity
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    
    FFMpegWriter = manimation.writers['ffmpeg']
    
    metadata = dict(title='Created with pyfeel.',
                    artist='Matplotlib',
                    comment='v.1.0')
    
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    n_images = len(image_list)
    
    # Create the figure to update
    image = image_list[0]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    if len(image.shape) == 3:
        ax_img = ax.imshow(image)
    else:
        ax_img = ax.imshow(image, cmap=plt.cm.Greys_r)
    if frame_numbers:
        ax_text = ax.text(0.01, 0.8, 'Frame 0',
                          fontsize=50, 
                          color='black',
                          alpha=0.8)

    with writer.saving(fig, video_filename, dpi):
        for index, image in enumerate(image_list):
            ax_img.set_array(image)
            if frame_numbers:
                ax_text.set_text('Frame {}'.format(index))
            writer.grab_frame()


def image_list_to_animation(image_list, dpi=200, fps=15, video_filename=None):
    """ Uses matplotlib.animation to create an animation from a list of images

    Optionally saves it using matplotlib.animation.writers['ffmpeg']

    Parameters
    ----------
    image_list: ndarray list 
        list like object of images

    dpi: int, default is 220

    fps: int, default is 15

    video_filename: string, optional
        if specified, the annimation is saved in a video
        absolute path to the target video
        (e.g. : '/temp/video.mp4')

    frame_numbers: bool, default is False
        if True, displays the frame numbers in the video

    Returns
    -------
    animation (matplotlib.animation)
        call plt.show() to see it ;)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import numpy as np
    import matplotlib.pyplot as plt
    image = image_list[0]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    if len(image.shape) == 3:
        ax_img = ax.imshow(image)
    else:
        ax_img = ax.imshow(image, cmap=plt.cm.Greys_r)
    if frame_numbers:
        ax_text = ax.text(0.01, 0.8, 'Frame 0',
                          fontsize=50, 
                          color='black',
                          alpha=0.8)

    def update_figure(index):
        image = image_list[index]
        ax_img.set_array(image)
        ax_scatter.set_offsets(shape)
        if frame_numbers:
            ax_text.set_text('Frame {}'.format(index))

        return ax_img, ax_scatter

    ani = animation.FuncAnimation(fig, update_figure, frames=len(image_list), blit=False)

    if video_filename is not None:
        metadata = dict(title='Created with pyfeel.',
                        artist='Matplotlib',
                        comment='v.1.0')
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        ani.save(video_filename, writer=writer, dpi=dpi, extra_args=['-vcodec', 'libx264'])

    return ani


class FFMpegVideoReader():
    """Reads a video through a pipe using ffmpeg.

    Parameters
    ----------
    video_filename : `Path`
        Absolute path to the video
    """
    def __init__(self, video_filename):
        try:
            infos = video_infos_ffprobe(video_filename)
        except:
            infos = video_infos_ffmpeg(video_filename)
        self.duration = infos['duration']
        self.width = infos['width']
        self.height = infos['height']
        self.n_frames = infos['n_frames']
        self.fps = infos['fps']
        self.video_filename = video_filename
        self._pipe = None

        # contains the index of the last read frame
        # the index is updated in _open_pipe, _read_one_frame and _trash_frames
        self.index = -1

    def _shutdown_pipe(self):
        if self._pipe is not None:
            if self._pipe.stdout:
                self._pipe.stdout.close()
            if self._pipe.stderr:
                self._pipe.stderr.close()
            if self._pipe.stdin:
                self._pipe.stdin.close()
        self._pipe = None

    def __del__(self):
        r"""
        Close the pipe if open.
        """
        self._shutdown_pipe()

    def __len__(self):
        return self.n_frames

    def _open_pipe(self, frame=None):
        r"""
        Open a pipe at the time just before the specified frame
        Parameters
        ----------
        frame : `int`, optional
            If ``None``, pipe opened from the beginning of the video
            otherwise, pipe opened at the time corresponding to that frame
        Note
        ----
        Since v.2.1 of ffmpeg, this is frame-accurate
        """
        if frame is not None and frame > 0:
            time = str(frame / float(self.fps))
            command = ['ffmpeg',
                       '-ss', time,
                       '-i', str(self.video_filename),
                       '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24',
                       '-vcodec', 'rawvideo', '-']
        else:
            command = ['ffmpeg',
                       '-i', str(self.video_filename),
                       '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24',
                       '-vcodec', 'rawvideo', '-']
            frame = 0

        self._shutdown_pipe()
        self._pipe = sp.Popen(command, stdout=sp.PIPE, stdin=sp.PIPE,
                              stderr=sp.PIPE,
                              bufsize=10**8)
        # We have not yet read the specified frame
        self.index = frame - 1

    def __iter__(self):
        r"""
        Iterate through all frames of the video in order
        Only opens the pipe once at the beginning
        """
        self.index = 0
        for index in range(self.n_frames):
            yield self[index]

    def __getitem__(self, index):
        r"""
        Get a specific frame from the video
        """
        # If the user is reading consecutive frames, or a frame later in the
        # video, do not reopen a pipe
        if self._pipe is None or self._pipe.poll() is not None or index <= self.index:
            self._open_pipe(frame=index)
        else:
            to_trash = index - self.index - 1
            if to_trash > 0:
                self._trash_frames(to_trash)

        return self._read_one_frame()

    def _trash_frames(self, n_frames):
        r"""
        Reads and trashes the data corresponding to ``n_frames``
        """
        _ = self._pipe.stdout.read(self.height*self.width*3*n_frames)
        self._pipe.stdout.flush()
        self.index += n_frames

    def _read_one_frame(self):
        r"""
        Reads one frame from the opened ``self._pipe`` and converts it to
        a numpy array
        Returns
        -------
        image : :map:`Image`
            Image of shape ``(self.height, self.width, 3)``
        """
        raw_data = self._pipe.stdout.read(self.height*self.width*3)
        frame = np.fromstring(raw_data, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        self._pipe.stdout.flush()
        self.index += 1

        return frame


def video_infos_ffmpeg(video_filename):
    """ Parses the information from a video using ffmpeg
    Uses subprocesses to get the information through a pipe

    Parameters
    ----------
    video_filename: string
        absolute path to the video file which information to extract

    Returns
    -------
    infos: dict
        keys are width, height (size of the frames)
        duration (duration of the video in seconds)
        n_frames
    """
    # Read information using ffmpeg
    command = ['ffmpeg', '-i', video_filename, '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    pipe.stdout.readlines()
    raw_infos = pipe.stderr.read().decode()
    pipe.terminate()

    # parse the information
    # Note: we use '\d+\.?\d*' so we can match both int and float for the fps
    video_info = re.search(r"Video:.*(?P<width> \d+)x(?P<height>\d+).*(?P<fps> \d+\.?\d*) fps",
                           raw_infos, re.DOTALL).groupdict()
    time = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),",
                     raw_infos, re.DOTALL).groupdict()

    # Get the duration in seconds and convert size to ints
    hours = float(time['hours'])
    minutes = float(time['minutes'])
    seconds = float(time['seconds'])
    duration = 60*60*hours + 60*minutes + seconds

    fps = round(float(video_info['fps']))
    n_frames = round(duration*fps)
    width = int(video_info['width'])
    height = int(video_info['height'])

    # Create the resulting dictionnary
    infos = {'duration':duration, 'width':width, 'height':height, 'n_frames':n_frames, 'fps':fps}

    return infos


def video_infos_ffprobe(video_filename):
    """ Parses the information from a video using ffprobe
    Uses subprocesses to get the information through a pipe

    Parameters
    ----------
    video_filename: string
        absolute path to the video file which information to extract

    Returns
    -------
    infos: dict
        keys are width, height (size of the frames)
        duration (duration of the video in seconds)
        n_frames
    """
    p = sp.Popen(
            ['ffprobe', '-show_format', '-show_streams', video_filename],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )
    # Store all the information in a dictionnary
    result = dict(streams=[])
    stream = {}
    is_stream = is_format = False
    for line in p.stdout.readlines():
        #result.append(line)
        line = line.decode().strip()
        if line == '[STREAM]':
            is_stream = True
            continue
        if line == '[/STREAM]':
            is_stream = False
            result['streams'] = stream
            stream = {}
            continue
        if line == '[FORMAT]':
            is_format = True
            continue
        if line == '[/FORMAT]':
            break
        tokens = line.split('=')
        if is_stream: stream[tokens[0]] = tokens[1]
        if is_format: result[tokens[0]] = tokens[1]

    # Keep only the relevant parts
    width = int(result['streams']['width'])
    height = int(result['streams']['height'])
    n_frames = int(result['streams']['nb_frames'])
    duration = float(result['streams']['duration'])
    fps = float(eval(result['streams']['avg_frame_rate']))

    # Create the resulting dictionnary
    infos = {'duration':duration, 'width':width, 'height':height, 'n_frames':n_frames, 'fps':fps}

    return infos


class FFMpegWebcamReader():
    """ Uses ffmpeg to read images from a webcam

        Work in progress

    Parameters
    ----------
    video_filename: string
        absolute path to the video
    """
    def __init__(self, fps=30):
        self.fps = str(fps)
        if platform.startswith("linux"):
            self.f = 'video4linux2'
        elif platform.startswith("darwin"):
            self.f = 'avfoundation'
        elif platform.startswith("win"):
            self.f = 'vfwcap'

    def get_frame(self):
        """ Reads one frame from the webcam using ffmpeg by saving into a temporary file
            loading it, and deleting it (slow...)

        Returns
        -------
        image: ndarray
        """
        target = './temp_ffmeg_pyfeel.jpeg'
        command = ['ffmpeg',
                   '-f', self.f,
                   '-r', self.fps,
                   '-pix_fmt', 'rgb0',
                   '-i', '0',
                   '-vframes', '1',
                   target]
        sp.call(command)
        image = read_image(target, to_gray=self.image_to_gray)
        os.remove(target)
        return image
        
    def get_video_filename(self, n_frames=200, video_filename='./temp_ffmeg_pyfeel.mp4'):
        """ Reads n_frames from the webcam and saves them in video_filename.
        
        Returns
        -------
        string: filename in which the frames were saved
        """
        command = ['ffmpeg',
                   '-y',
                   '-f', self.f,
                   '-r', self.fps,
                   '-pix_fmt', 'rgb0',
                   '-i', '0',
                   '-vframes', str(n_frames),
                   '-an', video_filename]
        sp.call(command)
        return video_filename


