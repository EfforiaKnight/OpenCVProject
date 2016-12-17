#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.com/research/track/camshift.html

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys

from collections import deque
import numpy as np
import cv2


class App(object):
    def __init__(self, video_src, deque_len):
        self.cam = cv2.VideoCapture(video_src)
        ret, self.frame = self.cam.read()
        self.high = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv2.namedWindow('camshift')
        cv2.setMouseCallback('camshift', self.onmouse)

        self.trail_buffer = deque_len
        self.trail_pts = deque(maxlen=deque_len)
        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None

        # set up kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.03
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)

        self.lastCenter = (0, 0)

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                          (int(180.0 * i / bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def draw_trails(self, vis):
        # loop over the set of tracked points
        for i in range(1, len(self.trail_pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.trail_pts[i - 1] is None or self.trail_pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(self.trail_buffer / float(i + 1)) * 2.5)
            cv2.line(vis, self.trail_pts[i - 1], self.trail_pts[i], (0, 0, 255), thickness)

    def run(self):
        while True:
            ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            # self.frame = cv2.GaussianBlur(self.frame, (11, 11), 0)
            # kernel = np.array([[-2, -1, 0],
            #                    [-1, 1, 1],
            #                    [0, 1, 2]])
            # cv2.filter2D(self.frame, -1, kernel, self.frame)
            self.frame = cv2.medianBlur(self.frame, 5)
            cv2.imshow("blur", self.frame)
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            # cv2.imshow("mask", mask)

            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None

                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask

                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

                pts = np.int0(cv2.boxPoints(track_box))
                ((x, y), radius) = cv2.minEnclosingCircle(pts)
                # M = cv2.moments(pts)
                x, y = int(x), int(y)
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                self.kalman.correct(np.array([np.float32(x), np.float32(y)], np.float32))
                prediction = self.kalman.predict()[:2][:2]

                if norm_sqrt(prediction, self.lastCenter) > 350 or cv2.contourArea(pts) < 10:
                    # Target Lost
                    cv2.putText(vis, "Target lost", (int(self.width / 2), int(self.high / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    # draw the circle and centroid on the frame, then update the list of tracked points
                    cv2.putText(vis, "[{:.1f},{:.1f}] R={:.1f}".format(int(prediction[0]), int(prediction[1]), radius),
                                (19, self.high - 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(vis, "[{:.1f},{:.1f}] R={:.1f}".format(int(prediction[0]), int(prediction[1]), radius),
                                (20, self.high - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

                    cv2.circle(vis, (x, y), int(radius), (0, 255, 255), 2)
                    cv2.circle(vis, (int(prediction[0]), int(prediction[1])), int(radius), (0, 255, 0), 2)

                    cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

                    # check if borders was trespassed
                    # trespass_borders(frame=frame, curr_pos=x, left_border=left_border, right_border=right_border)

                    # update the points queue
                    self.trail_pts.appendleft((int(x), int(y)))
                    self.lastCenter = prediction
                if self.show_backproj:
                    vis[:] = prob[..., np.newaxis]

                    # try:
                    #     cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                    # except:
                    #     print(track_box)
            self.draw_trails(vis)
            cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()


def norm_sqrt(vect1, vect2):
    return cv2.norm((int(vect1[0] - vect2[0]), int(vect1[1] - vect2[1])), cv2.NORM_L2)


if __name__ == '__main__':
    import sys

    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    deque_len = 20
    print(__doc__)
    App(video_src, deque_len).run()
