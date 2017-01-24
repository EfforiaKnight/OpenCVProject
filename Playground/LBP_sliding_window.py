import cv2
import numpy as np
from Playground.rectselector import RectSelector
from skimage.feature import local_binary_pattern as lbp
from timeit import default_timer as timer

def onmouse(rect):
    xmin, ymin, xmax, ymax = rect
    global roi, image, lbp_roi
    roi = image[ymin:ymax, xmin:xmax]
    lbp_roi = lbp_image[ymin:ymax, xmin:xmax]

def show_hist(hist, type):
    bin_count = hist.shape[0]
    bin_w = 24
    img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
    for i in range(bin_count):
        h = int(hist[i])
        cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('hist ' + str(type), img)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


image = cv2.imread("/home/efforia/PycharmProjects/OpenCVProject/images/grant.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", image)

target_flag = False
target_lbp_hist = None
finish_sliding = False

radius = 2
n_points = radius * 8

roi = None
ROI_selector = RectSelector("image", onmouse)
lbp_image = lbp(image, n_points, radius, 'uniform')

while True:
    if not ROI_selector.dragging:
        cv2.imshow("image", image)
        if roi is not None:
            cv2.imshow("roi", roi)

            if target_flag:
                lbp_count = 0
                attempt = 0
                stepSize = 32
                clone = image.copy()
                target_flag = False
                finish_sliding = False
                cv2.imshow("target roi", roi)

                # target_lbp = lbp(roi, n_points, radius, 'uniform')
                target_lbp = np.asarray(lbp_roi, dtype=np.uint8)
                n_bins = int(target_lbp.max() + 1)
                target_lbp_hist = cv2.calcHist([target_lbp], [0], None, [n_bins], [0, n_bins])
                # cv2.normalize(target_lbp_hist, target_lbp_hist, 0, 1, cv2.NORM_MINMAX)
                # target_lbp_hist, _ = np.histogram(target_lbp, bins=n_bins, range=(0, n_bins), normed=True)
                # target_lbp_hist = np.asarray(target_lbp_hist, dtype=np.float32)

    else:
        rect_image = image.copy()
        ROI_selector.draw(rect_image)
        cv2.imshow("image", rect_image)

    if type(target_lbp_hist) is not type(None) and finish_sliding is False:
        start = timer()
        winH, winW = roi.shape[:2]
        bh_distance_list = []
        win_coordinates_list = []

        for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # candidate_lbp = lbp(window, n_points, radius, 'uniform')
            candidate_lbp = np.asarray(lbp_image[y:y+winH, x:x+winW], dtype=np.uint8)

            n_bins = int(candidate_lbp.max() + 1)
            candidate_lbp_hist = cv2.calcHist([candidate_lbp], [0], None, [n_bins], [0, n_bins])

            bh_distance = cv2.compareHist(target_lbp_hist, candidate_lbp_hist, cv2.HISTCMP_BHATTACHARYYA)
            if bh_distance < 0.03:
                lbp_count += 1
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 255, 255), 2)
            if lbp_count >= 3:
                finish_sliding = True
                break

            clone_rect = image.copy()
            cv2.rectangle(clone_rect, (x, y), (x + winW, y + winH), (255, 255, 255), 2)
            cv2.imshow("Sliding", clone_rect)
            cv2.waitKey(1)

        if lbp_count <= 3 and attempt < 2:
            print("Sliding failed\nTime {}\nAttempt #{}, stepSize={}".format(timer() - start, attempt, stepSize))
            stepSize = int(stepSize / 2)
            attempt += 1
        else:
            finish_sliding = True
            print("Sliding succeeded after {}\nDid {} attempts, last stepSize={}".format(timer() - start, attempt, stepSize))
        cv2.imshow("LBP regions", clone)
        # bh_distance_list, win_coordinates_list = zip(*sorted(zip(bh_distance_list, win_coordinates_list)))
        # for i in range(5):
        #     (x, y) = win_coordinates_list[i]
        #     win = image[y:y+winH, x:x+winW]
        #     print("bh distance #{} is {}".format(i, bh_distance_list[i]))
        #     cv2.imshow("Best match #"+str(i), win)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
    elif ch == ord('t'):
        target_flag = True

cv2.destroyAllWindows()

