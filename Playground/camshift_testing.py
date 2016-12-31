import numpy as np
import cv2
from Playground.rectselector import RectSelector


class App(object):
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        ret, self.frame = self.cam.read()

        self.height, self.width = self.frame.shape[:2]

        cv2.namedWindow('camshift')
        self.obj_select = RectSelector('camshift', self.onmouse)

        self.HSV_CHANNELS = (
            (8, [0, 180], "hue"),  # Hue
            (8, [0, 256], "sat"),  # Saturation
            (8, [0, 256], "val")  # Value
        )

        self.show_backproj = False
        self.track_window = None

    def onmouse(self, rect):
        self.histHSV = []
        xmin, ymin, xmax, ymax = rect
        hsvRoi = self.hsv[ymin:ymax, xmin:xmax]
        lbpRoi = self.frame[ymin:ymax, xmin:xmax]
        lbpRoi = cv2.cvtColor(lbpRoi, cv2.COLOR_BGR2GRAY)
        self.calcHSVhist(hsvRoi)
        self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def calcHSVhist(self, hsvRoi):
        for channel, param in enumerate(self.HSV_CHANNELS):
            # Init HSV histogram
            if len(self.histHSV) != len(self.HSV_CHANNELS):
                hist = cv2.calcHist([hsvRoi], [channel], None, [param[0]], param[1])
                hist = cv2.normalize(hist, hist, 0, 256, cv2.NORM_MINMAX)
                self.histHSV.append(hist)
                # Show hist of each channel separately
                self.show_hist(hist, param[2])

    def calcBackProjection(self):
        back_proj_prob = np.ones(shape=(self.height, self.width), dtype=np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # kernel = np.ones((3, 3), np.uint8)
        for channel, param in enumerate(self.HSV_CHANNELS):
            prob = cv2.calcBackProject([self.hsv], [channel], self.histHSV[channel], param[1], 1)
            ret, prob = cv2.threshold(prob, 30, 255, 0)
            prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel, iterations=2)
            back_proj_prob &= prob

        return back_proj_prob

    @staticmethod
    def show_hist(hist, channel):
        bin_count = hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(hist[i])
            if str(channel) == 'hue':
                cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                              (int(180.0 * i / bin_count), 255, 255), -1)
            elif str(channel) == 'sat':
                cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                              (180, int(255.0 * i / bin_count), 255), -1)
            else: # str(channel) == 'val'
                cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                              (180, 255, int(255.0 * i / bin_count)), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist ' + str(channel), img)

    def run(self):
        while True:
            if not self.obj_select.dragging:
                ret, self.frame = self.cam.read()
                self.hsv = cv2.medianBlur(self.frame, 11)
                self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            if ret:
                vis = self.frame.copy()

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                prob = self.calcBackProjection()
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[..., np.newaxis]
                try:
                    cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                except:
                    print(track_box)

            self.obj_select.draw(vis)
            cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()


if __name__ == '__main__':
    App().run()
