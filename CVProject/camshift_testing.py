import numpy as np
import cv2
from fps import FPS
from WebcamVideoStream import WebcamVideoStream
from rectselector import RectSelector
from SuspendTracking import SuspendTracking
from localbinarypatterns import LocalBinaryPatterns as LBP


class App(object):

    def __init__(self):
        # self.cam = cv2.VideoCapture(0)
        # self.cam.set(3, 320)
        # self.cam.set(4, 240)
        self.cam = WebcamVideoStream(src=0, resolution=(640, 480)).start()
        self.fps = FPS().start()

        ret, self.frame = self.cam.read()

        self.suspend_tracking = SuspendTracking(teta=3)

        self.height, self.width = self.frame.shape[:2]
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,
                                                                          3))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,
                                                                           7))

        cv2.namedWindow('camshift')
        self.obj_select = RectSelector('camshift', self.onmouse)

        radius = 3
        n_point = 8 * radius
        self.lbpDesc = LBP(n_point, radius)

        self.HSV_CHANNELS = (
            (24, [0, 180], "hue"),  # Hue
            (8, [0, 256], "sat"),  # Saturation
            (8, [0, 256], "val")  # Value
        )

        self.show_backproj = False
        self.track_window = None
        self.histHSV = []
        self.track_box = None

    def onmouse(self, rect):
        xmin, ymin, xmax, ymax = rect
        hsvRoi = self.hsv[ymin:ymax, xmin:xmax]

        self.calcHSVhist(hsvRoi)
        self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)
        self.init_suspend(hsvRoi)
        self.fps.reset()

    def init_suspend(self, hsvRoi):
        track_window_condition = self.track_window and self.track_window[
            2] > 0 and self.track_window[3] > 0
        if track_window_condition:
            self.camshift_algorithm()
            self.suspend_tracking.init(hsvRoi)

    def calcHSVhist(self, hsvRoi):
        self.histHSV = []
        for channel, param in enumerate(self.HSV_CHANNELS):
            # Init HSV histogram
            hist = cv2.calcHist([hsvRoi], [channel], None, [param[0]],
                                param[1])
            hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            self.histHSV.append(hist)
            # Show hist of each channel separately
            self.show_hist(hist, param[2])

    def calcBackProjection(self):
        ch_prob = []
        ch_back_proj_prob = []
        # back_proj_prob = np.ones(shape=(self.height, self.width), dtype=np.uint8) * 255
        # back_proj_prob = np.zeros(shape=(self.height, self.width), dtype=np.uint8)

        for channel, param in enumerate(self.HSV_CHANNELS):
            prob = cv2.calcBackProject([self.hsv], [channel],
                                       self.histHSV[channel], param[1], 1)
            cv2.imshow('Back projection ' + str(param[2]), prob)
            # ret, prob = cv2.threshold(prob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ret, prob = cv2.threshold(prob, 70, 255, cv2.THRESH_BINARY)
            cv2.imshow('Back projection thresh ' + str(param[2]), prob)
            # prob = cv2.morphologyEx(prob, cv2.MORPH_ERODE, self.kernel_erode, iterations=2)
            # prob = cv2.morphologyEx(prob, cv2.MORPH_DILATE, self.kernel_dilate, iterations=3)
            # back_proj_prob = cv2.bitwise_and(back_proj_prob, prob)
            # back_proj_prob = cv2.addWeighted(back_proj_prob, 0.4, prob, 0.6, 0)
            ch_prob.append(prob)

        ch_back_proj_prob.append(
            cv2.addWeighted(ch_prob[0], 0.6, ch_prob[1], 0.4, 0))

        ch_back_proj_prob.append(
            cv2.addWeighted(ch_prob[0], 0.6, ch_prob[2], 0.4, 0))

        back_proj_prob = cv2.bitwise_and(ch_back_proj_prob[0],
                                         ch_back_proj_prob[1])
        ret, back_proj_prob = cv2.threshold(back_proj_prob, 150, 255,
                                            cv2.THRESH_BINARY)

        back_proj_prob = cv2.morphologyEx(
            back_proj_prob, cv2.MORPH_ERODE, self.kernel_erode, iterations=1)
        back_proj_prob = cv2.morphologyEx(
            back_proj_prob, cv2.MORPH_DILATE, self.kernel_erode, iterations=2)

        return back_proj_prob

    @staticmethod
    def show_hist(hist, channel='None'):
        bin_count = hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(hist[i])
            if str(channel) == 'hue':
                cv2.rectangle(img, (i * bin_w + 2, 255),
                              ((i + 1) * bin_w - 2, 255 - h), (int(
                                  180.0 * i / bin_count), 255, 255), -1)
            elif str(channel) == 'sat':
                cv2.rectangle(img, (i * bin_w + 2, 255),
                              ((i + 1) * bin_w - 2, 255 - h), (180, int(
                                  255.0 * i / bin_count), 255), -1)
            elif str(channel) == 'val':
                cv2.rectangle(img, (i * bin_w + 2, 255),
                              ((i + 1) * bin_w - 2, 255 - h), (180, 255, int(
                                  255.0 * i / bin_count)), -1)
            else:
                cv2.rectangle(img, (i * bin_w + 2, 255), (
                    (i + 1) * bin_w - 2, 255 - h), (180, 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist ' + str(channel), img)

    def camshift_algorithm(self):
        prob = self.calcBackProjection()
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.track_box, self.track_window = cv2.CamShift(
            prob, self.track_window, term_crit)

        if self.show_backproj:
            cv2.imshow("Back Projection", prob[..., np.newaxis])
        else:
            cv2.destroyWindow("Back Projection")

    def run(self):
        scaling_factor = 0.5
        while True:
            if not self.obj_select.dragging:
                ret, self.frame = self.cam.read()
                self.frame = cv2.resize(
                    self.frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                # blur_frame = cv2.GaussianBlur(self.frame, (21,21), 0)
                self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            if ret:
                vis = self.frame.copy()

            track_window_condition = self.track_window and self.track_window[
                2] > 0 and self.track_window[3] > 0

            if track_window_condition and not self.suspend_tracking.is_suspend(self.hsv,
                                                                               self.track_box):
                self.camshift_algorithm()

                try:
                    cv2.ellipse(vis, self.track_box, (0, 0, 255), 2)
                    pts = cv2.boxPoints(self.track_box)
                    pts = np.int0(pts)
                    cv2.polylines(vis, [pts], True, 255, 2)
                except:
                    print(self.track_box)
            else:
                cv2.putText(vis, 'Target Lost', (10, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,
                                                          255), 1, cv2.LINE_AA)

            # frame processing throughput rate
            fps = self.fps.approx_compute()
            # print("FPS: {:.3f}".format(fps))
            cv2.putText(vis, 'FPS {:.3f}'.format(fps), (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                        1, cv2.LINE_AA)

            self.obj_select.draw(vis)
            cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()


if __name__ == '__main__':
    App().run()
