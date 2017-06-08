import cv2
import numpy as np


class SuspendTracking:

    def __init__(self, teta):
        self.teta = teta  # Scalar for std influence
        self.cnt = 1  # Number of elapse frame for mean and std measures

        self.objectHist = None

        self.mean = 0
        self.std = 0

    def init(self, hsvROI):
        self.objectHist = cv2.calcHist([hsvROI], [0], None, [24], [0, 180])
        self.show_hist(self.objectHist, "Object")

    def calc_candidate_hist(self, hsv_frame, track_box):
        mask = self.mask_roi(hsv_frame, track_box)
        candidate_hist = cv2.calcHist([hsv_frame], [0], mask, [24], [0, 180])
        self.show_hist(candidate_hist, "Candidate")
        return candidate_hist

    def is_suspend(self, hsv_frame, track_box):
        candidateHist = self.calc_candidate_hist(hsv_frame, track_box)
        bh_distance = cv2.compareHist(
            self.objectHist, candidateHist, cv2.HISTCMP_BHATTACHARYYA)
        prev_mean = self.mean
        prev_std = self.std

        adaptive_threshold = self.calc_adaptive_threshold(
            prev_mean, prev_std, self.teta)

        if bh_distance >= adaptive_threshold and adaptive_threshold != 0 or adaptive_threshold >= 1:
            print("bh={:.4f} > adptv_thresh={:.4f}, statement is {:.4f}".format(bh_distance, prev_mean + self.teta * prev_std,
                                                                                bh_distance > prev_mean + self.teta * prev_std))
            print("Suspending tracking, target lost")
            self.cnt = 1
            self.mean = 0
            self.std = 0
            return True
        else:
            print("bh={:.4f} and adaptive threshold = {:.4f}".format(
                bh_distance, adaptive_threshold))
            self.mean = self.calc_new_mean(bh_distance, prev_mean, self.cnt)
            self.std = self.calc_new_std(
                bh_distance, prev_mean, self.mean, prev_std, self.cnt)
            self.cnt += 1
            return False

    @staticmethod
    def calc_new_mean(distance, prev_mean, cnt):
        mean = prev_mean + (distance - prev_mean) / cnt
        return mean

    @staticmethod
    def calc_new_std(distance, prev_mean, new_mean, prev_std, cnt, eps=1e-11):
        std = (cnt - 2) * (prev_std ** 2) + \
            (distance - prev_mean) * (distance - new_mean)
        std = np.sqrt(std / (cnt - 1 + eps))
        return std

    @staticmethod
    def calc_adaptive_threshold(mean, std, teta):
        threshold = mean + teta * std
        return threshold

    # Mask the object in the frame. Return masked object
    @staticmethod
    def mask_roi(frame, track_box):
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.ellipse(mask, track_box, 255, -1)
        cv2.imshow("Masked Roi", mask)
        return mask

    @staticmethod
    def show_hist(hist, type):
        bin_count = hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(hist[i])
            cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                          (int(180.0 * i / bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow(str(type) + ' Hue Hist', img)
