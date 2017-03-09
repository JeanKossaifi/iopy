import sys
from time import time

def time_to_string(seconds):
    """Returns time as dd:hh:mm:ss omitting null values
    
    Parameters
    ----------
    seconds: int
        time in seconds

    Returns
    -------
    string: formatted time
    """
    formatted_time = ''
    seconds = round(seconds)
    mm, ss = divmod(seconds, 60)
    if mm > 0:
        if mm > 60:
            hh, mm = divmod(mm, 60)
            if hh > 24:
                dd, hh = divmod(hh, 24)
                formatted_time += '{}d, '.format(dd)
            formatted_time += '{}h, '.format(hh)
        formatted_time += '{}mn, '.format(mm)
    formatted_time += '{}s'.format(ss)
    return formatted_time


def update_progress(percent, title='Progress: ', text='',
                    bar_length=50, bar_char='█'):
    """ prints / updates a progress_bar
    
    Parameters
    ----------
    percent: float between 0 and 1

    title: string, default is 'Progress: '
        text to display just before the bar

    text: string, default is ''
        text to display after the bar

    bar_length: int, default is 50
        size of the progress_bar
    """
    bar_length = int(bar_length)
    if isinstance(percent, int):
        percent = float(percent)
    if not isinstance(percent, float):
        percent = 0
        text = "Error: percent var must be float"

    if percent < 0:
        percent = 0
        text = "Starting..."
    elif percent >= 1:
        percent = 1
        text += "Done."
    block = int(round(bar_length*percent))
    text = "\r{0} [{1}] {2}% {3}".format( 
            title,
            bar_char*block + "-"*(bar_length-block),
            round(percent*100, 2),
            text)
    sys.stdout.write(text)
    sys.stdout.flush()


class ProgressBar():
    def __init__(self, bar_length=25):
        self.bar_length = bar_length

    def start(self, title='Starting: ', text=''):
        update_progress(0, title=title, text=text, bar_length=self.bar_length)
        self.n_estimations = 0
        self.current_estimation = None
        self.previous_percent = 0
        self.previous_time = time()

    def update(self, percent, text=''):
        """
        Note
        ----
        estimation is the estimated time per percent
        """
        if percent > 1:
            percent = 1
        percent_updated = percent - self.previous_percent
        if percent_updated != 0:
            t = time()
            estimation = (t - self.previous_time)/percent_updated
            if self.current_estimation is not None:
                estimation = (self.current_estimation*self.n_estimations + estimation)/(self.n_estimations + 1)
            remaining = (1 - percent)*estimation
            self.n_estimations += 1
            self.current_estimation = estimation
            self.previous_time = t
            title = '{} to go: '.format(time_to_string(remaining))
        elif self.current_estimation is not None:
            estimation = self.current_estimation
            remaining = (1 - percent)*estimation
            title = '{} to go: '.format(time_to_string(remaining))
        else:
            remaining = '...'
            title = 'Estimating remaining time.'

        update_progress(percent, title=title, text=text,
                        bar_length=self.bar_length)
        self.previous_percent = percent
