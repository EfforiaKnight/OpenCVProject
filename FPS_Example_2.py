#!/usr/bin/env python

import cv2
import time

if __name__ == '__main__':

    # Start default camera
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    while True:
        ret, frame = video.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video
    video.release()
    cv2.destroyAllWindows()