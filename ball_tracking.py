# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=20, help="max buffer size")
args = vars(ap.parse_args())
font = cv2.FONT_HERSHEY_SIMPLEX
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
redLower = (170, 90, 0)
redUpper = (185, 180, 255)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# initiate Resolution of image var
(high, width) = None, None
roi_taken = False

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    if roi_taken is not True:
        (high, width, _) = frame.shape
        roi_taken = True

    left_border = int(width * 0.35)
    right_border = int(width * 0.65)

    # draw border indications
    cv2.line(frame, (left_border, 0), (left_border, high), (230, 230, 230))
    cv2.line(frame, (right_border, 0), (right_border, high), (230, 230, 230))

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    # frame = imutils.resize(frame, width=600)
    # blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # blurred = cv2.medianBlur(frame, 9, 0)
    # cv2.imshow('blurred', blurred)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    # kernel_test = np.matrix(([[1, 1, 0], [0, 0, 0], [0, 1, 1]]), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # kernel_er = np.ones((3, 3), np.uint8)

    mask = cv2.inRange(hsv, redLower, redUpper)
    cv2.imshow('inRange', mask)
    mask = cv2.erode(mask, kernel, iterations=1)
    cv2.imshow('erode', mask)
    # mask = cv2.dilate(mask, kernel, iterations=3)
    # cv2.imshow('dilate', mask)
    # None is a matrix 3x3 rectangle shape
    # mask = cv2.dilate(mask, None, iterations=3)
    # cv2.imshow('final', mask)
    # mask = cv2.erode(mask, None, iterations=1)
    # cv2.imshow('final', mask)
    # mask = cv2.dilate(mask, None, iterations=4)
    # cv2.imshow('final', mask)

    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_er)
    # cv2.imshow('opening', mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    cv2.imshow('closing', mask)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except ZeroDivisionError:
            continue

        # only proceed if the radius meets a minimum size
        if radius > 17:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            if int(x) < left_border:
                cv2.putText(frame, 'Left', (50, 130), font, 2, (230, 230, 230), 1, cv2.LINE_AA)
            elif int(x) > right_border:
                cv2.putText(frame, 'Right', (450, 130), font, 2, (230, 230, 230), 1, cv2.LINE_AA)

            # update the points queue
            pts.appendleft(center)
        else:
            pts.append(None)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
