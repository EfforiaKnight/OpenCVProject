import cv2
import numpy as np


class LABDistance:

    def __init__(self):
        self.objectHist = None

    def init(self, objectROI):
        hue_frame = cv2.cvtColor(objectROI, cv2.COLOR_BGR2HSV)
        objectHist = cv2.calcHist([hue_frame], [0], None, [
            24], [0, 256])

        # cv2.normalize(objectHist, objectHist)
        # self.objectHist = objectHist.flatten()
        self.objectHist = objectHist

    def update(self, frame, track_box):
        mask = self.mask_roi(frame, track_box)
        hue_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        candidateHist = cv2.calcHist([hue_frame], [0], mask, [
            24], [0, 256])

        # Normalize canidate histogram
        # cv2.normalize(candidateHist, candidateHist)
        # candidateHist = candidateHist.flatten()

        # Compare object and candidate histograms
        bh_distance = cv2.compareHist(
            self.objectHist, candidateHist, cv2.HISTCMP_BHATTACHARYYA)
        print("[INFO] bh Hue distance is {:.5f}".format(bh_distance))
        return bh_distance

    @staticmethod
    def mask_roi(frame, track_box):
        '''
        Mask the object in the frame. Return masked object
        '''
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.ellipse(mask, track_box, 255, -1)
        cv2.imshow("Masked Roi", mask)
        return mask
