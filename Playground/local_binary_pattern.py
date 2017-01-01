import cv2
from Playground.localbinarypatterns import LocalBinaryPatterns
from Playground.rectselector import RectSelector
import numpy as np
from scipy import stats


class LBP:
    def __init__(self):
        cv2.namedWindow("gray")
        self.cap = cv2.VideoCapture(0)
        self.object_hist = None
        self.rect = RectSelector("gray", self.ondrag)
        # LBP settings
        self.object_described = None
        radius = 2
        n_point = 8 * radius
        self.desc = LocalBinaryPatterns(n_point, radius)

    def ondrag(self, rect):
        x0, y0, x1, y1 = rect
        self.roi = rect
        self.object_hist, self.object_lbp = self.desc.describe(self.gray_frame[y0:y1, x0:x1])
        self.object_described = True
        # cv2.imshow("object", self.object_lbp)

    def run(self):
        while True:
            if not self.rect.dragging or frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break

            self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.object_described:
                x0, y0, x1, y1 = self.roi
                hist, lbp = self.desc.describe(self.gray_frame[y0:y1, x0:x1])
                score = kullback_leibler_divergence(self.object_hist, hist)
                # print("object hist: {}".format(self.object_hist))
                # print("hist: {}".format(hist))
                print(score)
                cv2.imshow("lbp", lbp)
            self.rect.draw(self.gray_frame, (255, 255, 255))
            cv2.imshow("gray", self.gray_frame)



            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break


def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d

def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

if __name__ == '__main__':
    LBP().run()
    cv2.destroyAllWindows()
