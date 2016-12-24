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
        radius = 3
        n_point = 8 * radius
        self.desc = LocalBinaryPatterns(n_point, radius)

    def ondrag(self, rect):
        x0, y0, x1, y1 = rect
        roi = self.object_frame[y0:y1, x0:x1]
        self.object_hist, self.object_lbp = self.desc.describe(roi)
        # cv2.imshow("object", self.object_lbp)

    def run(self):
        while True:
            if not self.rect.dragging or frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break

            self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.object_frame = self.gray_frame.copy()
            hist, lbp = self.desc.describe(self.gray_frame)
            if self.object_hist:

                print(score)
            self.rect.draw(self.gray_frame, (255, 255, 255))
            cv2.imshow("gray", self.gray_frame)
            # cv2.imshow("lbp", lbp)


            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break


def kullback_leibler_divergence(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, (p - q) * np.log10(p / q), 0))

if __name__ == '__main__':
    LBP().run()
    cv2.destroyAllWindows()
