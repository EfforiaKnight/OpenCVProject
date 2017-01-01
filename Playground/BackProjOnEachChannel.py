import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

HSV_CHANNELS = (
    (8, [0, 180], "hue"),  # Hue
    (6, [60, 256], "sat"),  # Saturation
    (4, [32, 256], "val")  # Value
)

hist_hsv = []

bgrImage = cv2.imread("grant.jpg")
hsvImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2HSV)

# fig, ax = plt.subplots(3, 1)
height, width = hsvImage.shape[:2]
back_proj = np.ones(shape=(height, width), dtype=np.uint8) * 255
cx, cy = int(width / 2), int(height / 2)
(x0, y0, x1, y1) = (cx, cy - 50, cx + 15, cy + 120)
cv2.rectangle(bgrImage, (x0, y0), (x1, y1), (255, 255, 255), 1)
hsvRoi = hsvImage[y0:y1, x0:x1]

start = time.clock()
for channel, param in enumerate(HSV_CHANNELS):
    hist = cv2.calcHist([hsvRoi], [channel], None, [param[0]], param[1])
    hist = cv2.normalize(hist, hist, 0, 256, cv2.NORM_MINMAX)

    prob = cv2.calcBackProject([hsvImage], [channel], hist, param[1], 1)
    ret, prob = cv2.threshold(prob, 0, 255, 0)
    # disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # cv2.filter2D(prob, -1, disc, prob)
    cv2.imshow("back proj " + str(param[2]), prob)
    # ax = plt.subplot(3, 1, channel + 1)
    # ax.set_xlim([0, param[0]])
    # ax.set_ylim([0, 255])
    # ax.set_title(str(param[2]))
    # ax.plot(hist)
    back_proj &= prob
    hist_hsv.append(hist)
# disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# cv2.filter2D(back_proj, -1, disc, back_proj)
print(len(hist_hsv))
cv2.imshow("back projection", back_proj)
three_channel_back = time.clock() - start
# fig.tight_layout()
# fig.show()

hist = cv2.calcHist([hsvRoi], [0, 1], None, [8, 2], [0, 180, 60, 256])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
prob = cv2.calcBackProject([hsvImage], [0, 1], hist, [0, 180, 60, 256], 1)
ret, prob = cv2.threshold(prob, 0, 255, 0)
# disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# cv2.filter2D(prob, -1, disc, prob)
cv2.imshow("opencv backproj", prob)

opencv_back = time.clock() - start

print("3 channel = {} \nOpencv = {}".format(three_channel_back, opencv_back))

cv2.imshow("color", bgrImage)
cv2.waitKey()
cv2.destroyAllWindows()
