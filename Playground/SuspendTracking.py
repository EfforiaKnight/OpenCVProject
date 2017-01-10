import cv2
import numpy as np
from Playground.rectselector import RectSelector
from skimage.feature import local_binary_pattern as lbp


def onmouse(rect):
    xmin, ymin, xmax, ymax = rect
    global roi, image
    roi = image[ymin:ymax, xmin:xmax]

def show_hist(hist, type):
    bin_count = hist.shape[0]
    bin_w = 24
    img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
    for i in range(bin_count):
        h = int(hist[i])
        cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('hist ' + str(type), img)

image = cv2.imread("/home/efforia/PycharmProjects/OpenCVProject/images/grant.jpg")
cv2.imshow("image", image)

target_flag = False
candidate_flag = False
target_hist = None
candidate_hist = None

t = 1
mean = 0
mean_lbp = 0
sd = 0
sd_lbp = 0
eps = 1e-11
teta = 3

roi = None
ROI_selector = RectSelector("image", onmouse)

while True:
    if not ROI_selector.dragging:
        cv2.imshow("image", image)
        if roi is not None:
            cv2.imshow("roi", roi)
            hsv_roi = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2HSV)
            gray_roi = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)

            if target_flag:
                target_flag = False
                cv2.imshow("target roi", roi)
                target_hist = cv2.calcHist([hsv_roi], [0], None, [24], [0, 180])
                show_hist(target_hist, "target")

                target_lbp = lbp(gray_roi, 24, 3, 'uniform')
                target_lbp = np.asarray(target_lbp, dtype=np.uint8)
                n_bins = int(target_lbp.max() + 1)
                target_lbp_hist = cv2.calcHist([target_lbp], [0], None, [n_bins], [0, n_bins])
                # target_lbp_hist, _ = np.histogram(target_lbp, bins=n_bins, range=(0, n_bins), normed=True)
                # target_lbp_hist = np.asarray(target_lbp_hist, dtype=np.float32)
            elif candidate_flag:
                candidate_flag = False
                candidate_hist = cv2.calcHist([hsv_roi], [0], None, [24], [0, 180])
                show_hist(candidate_hist, "candidate")

                candidate_lbp = lbp(gray_roi, 24, 3, 'uniform')
                candidate_lbp = np.asarray(candidate_lbp, dtype=np.uint8)
                n_bins = int(candidate_lbp.max() + 1)
                candidate_lbp_hist = cv2.calcHist([candidate_lbp], [0], None, [n_bins], [0, n_bins])
                # candidate_lbp_hist, _ = np.histogram(candidate_lbp, bins=n_bins, range=(0, n_bins), normed=True)
                # candidate_lbp_hist = np.asarray(candidate_lbp_hist, dtype=np.float32)
    else:
        rect_image = image.copy()
        ROI_selector.draw(rect_image)
        cv2.imshow("image", rect_image)

    if (type(target_hist) and type(candidate_hist)) is not type(None):
        bh_distance = cv2.compareHist(target_hist, candidate_hist, cv2.HISTCMP_BHATTACHARYYA)

        mean_prev = mean
        sd_prev =sd
        mean = mean + (bh_distance-mean)/t
        sd = (t-2)*sd**2 + (bh_distance-mean_prev)*(bh_distance-mean)
        sd /= t-1 + eps
        sd = np.sqrt(sd)

        print("bh={}, mean_bh={}, sd={}".format(bh_distance, mean, sd))
        print("bh={} > adptv_thresh={}, statement is {}".format(bh_distance, mean_prev + teta * sd_prev, bh_distance > mean_prev + teta * sd_prev))
        print("-" * 60)

        bh_lbp_distance = cv2.compareHist(target_lbp_hist, candidate_lbp_hist, cv2.HISTCMP_BHATTACHARYYA)

        mean_lbp_prev = mean_lbp
        sd_lbp_prev =sd_lbp
        mean_lbp = mean_lbp + (bh_lbp_distance-mean_lbp)/t
        sd_lbp = (t-2)*sd_lbp**2 + (bh_lbp_distance-mean_lbp_prev)*(bh_lbp_distance-mean_lbp)
        sd_lbp /= t-1 + eps
        sd_lbp = np.sqrt(sd_lbp)

        print("bh_lbp={}, mean_bh_lbp={}, sd_lbp={}".format(bh_lbp_distance, mean_lbp, sd_lbp))
        print("bh_lbp={} > adptv_thresh_lbp={}, statement is {}".format(bh_lbp_distance, mean_lbp_prev + teta * sd_lbp_prev, bh_lbp_distance > mean_lbp_prev + 2 * sd_lbp_prev))
        print("-" * 60)

        t += 1
        candidate_hist = None

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
    elif ch == ord('t'):
        target_flag = True
    elif ch == ord('c'):
        candidate_flag = True

cv2.destroyAllWindows()

