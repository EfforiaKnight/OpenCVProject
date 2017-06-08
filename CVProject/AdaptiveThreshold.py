import numpy as np


class AdaptiveThreshold:

    def __init__(self, teta=3, max_lost_cnt=3):
        self.teta = teta  # Scalar for std influence
        self.cnt = 1  # Number of elapse frame for mean and std measures
        self.max_lost_cnt = max_lost_cnt

        self.mean = 0
        self.std = 0
        self.lost_cnt = 0

    def target_lost(self, distance):
        prev_mean = self.mean
        prev_std = self.std

        adaptive_threshold = self.calc_adaptive_threshold(
            prev_mean, prev_std, self.teta)

        print("[INFO] Adaptive Threshold = {:.5f}".format(adaptive_threshold))

        if (((distance >= adaptive_threshold) and (adaptive_threshold != 0) or
                (adaptive_threshold >= 1)) and (self.cnt >= 3)):
            if self.lost_cnt == self.max_lost_cnt:
                print("======================================================")
                print("Suspending tracking, target lost {} times, Distance > Threshold".format(
                    self.lost_cnt))
                print("======================================================")
                self.cnt = 1
                self.mean = 0
                self.std = 0
                self.lost_cnt = 0
                return True
            else:
                self.lost_cnt += 1
        else:
            self.lost_cnt = 0  # Target found, reset lost counter
        self.mean = self.calc_new_mean(distance, prev_mean, self.cnt)
        self.std = self.calc_new_std(
            distance, prev_mean, self.mean, prev_std, self.cnt)
        self.cnt += 1
        return False

    @staticmethod
    def calc_new_mean(distance, prev_mean, cnt):
        mean = prev_mean + (distance - prev_mean) / cnt
        return mean

    @staticmethod
    def calc_new_std(distance, prev_mean, new_mean, prev_std, cnt, eps=1e-11):
        '''
        http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
        '''
        std = (cnt - 2) * (prev_std ** 2) + \
            (distance - prev_mean) * (distance - new_mean)
        std = np.sqrt(std / (cnt - 1 + eps))
        return std

    @staticmethod
    def calc_adaptive_threshold(mean, std, teta):
        threshold = mean + teta * std
        return threshold
