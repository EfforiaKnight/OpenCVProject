#!/usr/bin/env python

# import the necessary packages
from WebcamVideoStream import WebcamVideoStream
import numpy as np
import argparse
import atexit
from collections import deque

import cv2
import numpy as np


def get_arguments():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--thread", action="store_true", help="use threaded video capture method")
    ap.add_argument("-c", "--color", action="store_true", help="change color with opencv Trackbars")
    ap.add_argument("-b", "--buffer", type=int, default=20, help="max buffer size")
    args = ap.parse_args()
    return args


def cleanup(camera):
    print("...program exits")
    camera.release()
    cv2.destroyAllWindows()


def callback(value):
    pass


def setup_trackbars():
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in "HSV":
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

            # cv2.createTrackbar("MORPH Shape", "Trackbars", 0, 2, callback)
            # cv2.createTrackbar("kernel_m", "Trackbars", 0, 20, callback)
            # cv2.createTrackbar("kernel_n", "Trackbars", 0, 20, callback)


def get_trackbar_values():
    values = []

    for i in ["MIN", "MAX"]:
        for j in "HSV":
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values


class BallTracking:
    def __init__(self, deque_len, clr_lower, clr_upper):
        self.clr_upper = clr_upper
        self.clr_lower = clr_lower
        self.trail_buffer = deque_len
        self.pts = deque(maxlen=deque_len)
        self.mask = None

    def apply_masks(self, frame, track_bar_color):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # morph_indx = cv2.getTrackbarPos("MORPH Shape", "Trackbars")
        # m = cv2.getTrackbarPos("kernel_m", "Trackbars")
        # n = cv2.getTrackbarPos("kernel_n", "Trackbars")
        #
        # shapes = ["MORPH_ELLIPSE", "MORPH_RECT", "MORPH_CROSS"]

        # morph_shape = getattr(cv2, shapes[morph_indx])
        # kernel_close_morph = cv2.getStructuringElement(morph_shape, (2*m+1, 2*n+1))
        # kernel_erosion = cv2.getStructuringElement(morph_shape, (2*m+1, 2*n+1))

        # morphology kernels
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        if track_bar_color:
            v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values()
            self.clr_lower = (v1_min, v2_min, v3_min)
            self.clr_upper = (v1_max, v2_max, v3_max)

        # apply inRange filter then erode to lower noise and then closing (erosion then dilation)
        self.mask = cv2.inRange(hsv, self.clr_lower, self.clr_upper)
        cv2.imshow('inRange', self.mask)
        # self.mask = cv2.erode(self.mask, kernel_erosion, iterations=1)
        # cv2.imshow('erode', self.mask)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel_open)
        cv2.imshow('opening', self.mask)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel_close)
        cv2.imshow('closing', self.mask)

    def find_contours(self, frame, left_border, right_border):
        cnts = cv2.findContours(self.mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            except ZeroDivisionError:
                return

            # only proceed if the radius meets a minimum size
            if radius > 17:
                # draw the circle and centroid on the frame, then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # check if borders was trespassed
                trespass_borders(frame=frame, curr_pos=x, left_border=left_border, right_border=right_border)

                # update the points queue
                self.pts.appendleft(center)
            else:
                self.pts.append(None)

    def draw_trails(self, frame):
        # loop over the set of tracked points
        for i in range(1, len(self.pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(self.trail_buffer / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)


def draw_borders(frame, high=0, width=0):
    # calculate left and right border
    left_border = int(width * 0.35)
    right_border = int(width * 0.65)

    # draw border indications
    cv2.line(frame, (left_border, 0), (left_border, high), (230, 230, 230))
    cv2.line(frame, (right_border, 0), (right_border, high), (230, 230, 230))

    return left_border, right_border


def trespass_borders(frame, curr_pos, left_border, right_border):
    if int(curr_pos) < left_border:
        cv2.putText(frame, 'Left', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (230, 230, 230), 1, cv2.LINE_AA)
    elif int(curr_pos) > right_border:
        cv2.putText(frame, 'Right', (450, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (230, 230, 230), 1, cv2.LINE_AA)


def main():
    # get arguments
    args = get_arguments()

    if args.color:
        setup_trackbars()

    # grab a reference to the webcam
    camera = cv2.VideoCapture(0)
    high = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # create register to cleanup on exit
    atexit.register(cleanup, camera)

    # define the lower and upper boundaries of the "red" ball in the HSV color space, then initialize the
    # list of tracked points and run_once parameter for roi
    redLower = (160, 60, 0)
    redUpper = (180, 180, 255)

    track_object = BallTracking(args.buffer, redLower, redUpper)

    # keep looping
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        # draw borders indications and return border values
        left_border, right_border = draw_borders(frame=frame, high=high, width=width)

        # apply binary and morphological filters
        track_object.apply_masks(frame=frame, track_bar_color=args.color)

        # find contours in the mask and initialize the current (x, y) center of the ball
        track_object.find_contours(frame=frame, left_border=left_border, right_border=right_border)

        # draw trails of the ball on main frame
        track_object.draw_trails(frame=frame)

        # show the frame to our screen
        cv2.imshow("Frame", frame)

        # write frame to "output.avi"
        # out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break


if __name__ == '__main__':
    main()
