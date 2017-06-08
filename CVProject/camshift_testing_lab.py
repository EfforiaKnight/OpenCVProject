import numpy as np
import cv2
from CVProject.fps import FPS
from CVProject.WebcamVideoStream import WebcamVideoStream
from CVProject.rectselector import RectSelector
from CVProject.ColorDistance import LABDistance
from CVProject.localbinarypatterns import LocalBinaryPatterns
from CVProject.AdaptiveThreshold import AdaptiveThreshold
from CVProject.SuddenChanges import SuddenChanges

class App(object):
    def __init__(self):
        # self.cam = cv2.VideoCapture(0)
        # self.cam.set(3, 320)
        # self.cam.set(4, 240)
        self.cam = WebcamVideoStream(src=0, resolution=(640, 480)).start()
        self.fps = FPS().start()

        ret, self.frame = self.cam.read()

        self.conf = {
            'ColorFrameNum': 7,
            'LBPFrameNum': 7,
            'MaxFrameDiffClr': 15,
            'MaxLBPFrameUpdate': 30,
            'L_Weight': 0.3,
            'A_Weight': 0.7,
            'B_Weight': 0.7
        }

        self.ColorCheck = AdaptiveThreshold(teta=3, max_lost_cnt=1)
        self.LBPCheck = AdaptiveThreshold(teta=2, max_lost_cnt=1)

        self.XYSuddenChange = SuddenChanges(maxlen=20, N_vars=2, var_type="X, Y", threshold=35)
        self.AreaSuddenChange = SuddenChanges(maxlen=10, N_vars=1, var_type="Area", threshold=4500)
        self.sudden_change = False

        self.ColorDistance = LABDistance()
        self.LBPDistance = LocalBinaryPatterns(numPoints=8, radius=2, update_prev_hist=self.conf['MaxLBPFrameUpdate'])

        self.isLost = False
        self.isLBPLost = False

        self.height, self.width = self.frame.shape[:2]

        self.kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        cv2.namedWindow('camshift')

        self.obj_select = RectSelector('camshift', self.onmouse)
        self.LAB_CHANNELS = (
            (24, [0, 256], "light"),  # L
            (24, [0, 256], "a"),  # a
            (24, [0, 256], "b")  # b
        )

        self.show_backproj = False
        self.track_window = None
        self.histLAB = []
        self.track_box = None
        self.last_frame_number = 0

    def onmouse(self, rect):
        xmin, ymin, xmax, ymax = rect
        labRoi = self.lab[ymin:ymax, xmin:xmax]
        bgrRoi = self.frame[ymin:ymax, xmin:xmax]

        self.calcLABhist(labRoi)
        self.ColorDistance.init(bgrRoi)

        self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)
        # self.init_suspend(labRoi)

        '''
        Initialized all track lost conditions and fps counter
        '''
        self.isLost = False
        self.isLBPLost = False
        self.sudden_change = False
        self.XYSuddenChange.ClearQue()
        self.AreaSuddenChange.ClearQue()
        self.fps.reset()

    def calcLABhist(self, labRoi):
        self.histLAB = []
        for channel, param in enumerate(self.LAB_CHANNELS):
            # Init LAB histogram
            hist = cv2.calcHist([labRoi], [channel], None, [param[0]],
                                param[1])
            hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            self.histLAB.append(hist)
            # Show hist of each channel separately
            # self.show_hist(hist, param[2])

    def calcBackProjection(self):
        ch_prob = []
        ch_back_proj_prob = []

        for channel, param in enumerate(self.LAB_CHANNELS):
            prob = cv2.calcBackProject([self.lab], [channel],
                                       self.histLAB[channel], param[1], 1)
            cv2.imshow('Back projection ' + str(param[2]), prob)
            ret, prob = cv2.threshold(prob, 70, 255, cv2.THRESH_BINARY)
            cv2.imshow('Back projection thresh ' + str(param[2]), prob)
            # prob = cv2.morphologyEx(prob, cv2.MORPH_ERODE, self.kernel_e, iterations=1)
            # prob = cv2.morphologyEx(prob, cv2.MORPH_DILATE, self.kernel, iterations=1)
            prov = cv2.morphologyEx(prob, cv2.MORPH_CLOSE, self.kernel_e, iterations=2)
            ch_prob.append(prob)

        ch_back_proj_prob.append(
            cv2.addWeighted(ch_prob[0], self.conf['L_Weight'], ch_prob[1], self.conf['A_Weight'], 0))

        ch_back_proj_prob.append(
            cv2.addWeighted(ch_prob[0], self.conf['L_Weight'], ch_prob[2], self.conf['B_Weight'], 0))

        back_proj_prob = cv2.bitwise_and(ch_back_proj_prob[0],
                                         ch_back_proj_prob[1])
        ret, back_proj_prob = cv2.threshold(back_proj_prob, 150, 255,
                                            cv2.THRESH_BINARY)

        back_proj_prob = cv2.morphologyEx(
            back_proj_prob, cv2.MORPH_ERODE, self.kernel_e, iterations=1)
        back_proj_prob = cv2.morphologyEx(
            back_proj_prob, cv2.MORPH_DILATE, self.kernel_e, iterations=2)

        return back_proj_prob

    def show_hist(self, hist, channel='None'):
        bin_count = hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(hist[i])
            if str(channel) == 'light':
                cv2.rectangle(img, (i * bin_w + 2, 255),
                              ((i + 1) * bin_w - 2, 255 - h), (int(
                        255.0 * i / bin_count), 255, 255), -1)
            elif str(channel) == 'a':
                cv2.rectangle(img, (i * bin_w + 2, 255),
                              ((i + 1) * bin_w - 2, 255 - h), (255, int(
                        255.0 * i / bin_count), 255), -1)
            elif str(channel) == 'b':
                cv2.rectangle(img, (i * bin_w + 2, 255),
                              ((i + 1) * bin_w - 2, 255 - h), (255, 255, int(
                        255.0 * i / bin_count)), -1)
            else:
                cv2.rectangle(img, (i * bin_w + 2, 255), (
                    (i + 1) * bin_w - 2, 255 - h), (255, 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
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

    def check_if_object_lost(self):
        '''
        Method checking if the color or LBP tracking lost every # of frames.
        Lost condition performed by histogram distance compare.
        :return: Update self.last_frame_number to the last frame number the tracking lost the object
        '''
        if self.fps.NumFrame % self.conf['ColorFrameNum'] == 0 and not self.isLost:
            color_distance = self.ColorDistance.update(self.frame, self.track_box)
            self.isLost = self.ColorCheck.target_lost(color_distance)
            print("[INFO] Color track is lost:  '{}'\n".format(self.isLost))
            if self.isLost:
                self.last_frame_number = self.fps.NumFrame

        if self.fps.NumFrame % self.conf['LBPFrameNum'] == 0 and not self.isLBPLost:
            LBP_distance = self.LBPDistance.update(self.gray, self.track_window)
            self.isLBPLost = self.LBPCheck.target_lost(LBP_distance)
            print("[INFO] LBP track is lost:  '{}'\n".format(self.isLBPLost))
            if self.isLBPLost:
                self.last_frame_number = self.fps.NumFrame

    def run(self):
        scaling_factor = 0.5

        while True:
            if not self.obj_select.dragging:
                ret, self.frame = self.cam.read()
                self.frame = cv2.resize(
                    self.frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                self.lab = cv2.cvtColor(self.frame, cv2.COLOR_BGR2LAB)
                # self.lab = cv2.GaussianBlur(self.lab, (3,3), 0)
                kernel = np.ones((5, 5), np.float32) / 25
                self.lab = cv2.filter2D(self.lab, -1, kernel)
                self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            if ret:
                vis = self.frame.copy()

            track_window_condition = (self.track_window and
                                      (self.track_window[2] > 0) and
                                      (self.track_window[3] > 0))

            target_lost = self.isLost and self.isLBPLost

            # Main proccess flow
            if track_window_condition:
                if not target_lost and not self.sudden_change:
                    # Apply CamShift algorithm and get new track_box
                    self.camshift_algorithm()

                    # Check if target object lost and return last frame number the tracking lost it
                    self.check_if_object_lost()

                    if self.fps.NumFrame % 4 == 0:
                        (x, y) = self.track_box[0]
                        self.sudden_change = self.XYSuddenChange.CheckChange(x, y)

                        (width, height) = self.track_box[1]
                        self.sudden_change = self.AreaSuddenChange.CheckChange(width * height)


                    if self.fps.NumFrame - self.last_frame_number >= self.conf['MaxFrameDiffClr']:
                        self.isLBPLost = False
                        self.isLost = False

                    try:
                        cv2.ellipse(vis, self.track_box, (0, 0, 255), 2)
                        pts = cv2.boxPoints(self.track_box)
                        pts = np.int0(pts)
                        cv2.polylines(vis, [pts], True, 255, 2)
                    except:
                        print(self.track_box)
                else:
                    cv2.putText(vis, 'Target Lost', (10, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                1, cv2.LINE_AA)
                    # print("[INFO] Starting recovery proccess")

            elif not track_window_condition:
                cv2.putText(vis, 'Mark area of the object', (10, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            1, cv2.LINE_AA)

            # frame processing throughput rate
            fps = self.fps.approx_compute()
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
