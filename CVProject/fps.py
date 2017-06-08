# import the necessary packages
import datetime


class FPS:
    # frame processing throughput rate

    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._now = None
        self._numFrames = 0

    @property
    def NumFrame(self):
        return self._numFrames

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
        self._now = datetime.datetime.now()

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._now - self._start).total_seconds()

    def reset(self):
        self.__init__()
        self.start()

    def approx_compute(self):
        # compute the (approximate) frames per second
        self.update()
        return self._numFrames / self.elapsed()
