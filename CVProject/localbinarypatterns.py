from skimage.feature import local_binary_pattern
import numpy as np
import cv2


class LocalBinaryPatterns:

    def __init__(self, numPoints, radius, update_prev_hist=10):
        self.numPoints = numPoints
        self.radius = radius
        self.update_prev_hist = update_prev_hist
        self.cnt = 0
        self.initLBP = False

    def update(self, image, track_window):
        xmin, ymin, xmax, ymax = (track_window[0], track_window[1],
                                  track_window[2] + track_window[0],
                                  track_window[3] + track_window[1])
        object_window = image[ymin:ymax, xmin:xmax]
        if self.initLBP is False:
            self.prevHist = self.describe(object_window)
            self.prevHist = self.prevHist.flatten()
            self.initLBP = True
            return 0
        else:
            newHist = self.describe(object_window)
            newHist = newHist.flatten()

            if len(self.prevHist) != len(newHist):
                print("[WARN] Different LBP Hist lenght")
                return 1  # Due to issue in LBP Histogram different size

            bh_distance = cv2.compareHist(
                newHist, self.prevHist, cv2.HISTCMP_BHATTACHARYYA)
            print("[INFO] bh LBP distance is {:.5f}".format(bh_distance))
            if (self.cnt % self.update_prev_hist) == 0:
                self.prevHist = newHist
            self.cnt += 1
            return bh_distance

    def describe(self, image, mask=None):
        lbp = local_binary_pattern(
            image, self.numPoints, self.radius, method="uniform")
        lbp = np.asarray(lbp, dtype=np.uint8)
        n_bins = int(lbp.max() + 1)
        hist = cv2.calcHist(
            [lbp], [0], mask, [n_bins], [0, n_bins])
        # cv2.normalize(hist, hist)
        # hist = hist.flatten()

        # cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        # hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), normed=True)
        # hist = np.asarray(hist, dtype=np.float32)
        return hist
