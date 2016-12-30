import cv2
import numpy as np
import matplotlib.pyplot as plt

HSV_CHANNELS = (
    (8, [0, 180], "hue"),  # Hue
    (2, [0, 256], "sat"),  # Saturation
    (4, [0, 256], "val")  # Value
)

hist_hsv = []

bgrImage = cv2.imread("C:\MyProject\OpenCVProject\Playground\grant.jpg")
hsvImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2HSV)

# fig, ax = plt.subplots(3, 1)
height, width = hsvImage.shape[:2]
back_proj = np.ones(shape=(height, width), dtype=np.uint8) * 255
cx, cy = int(width / 2), int(height / 2)
(x0, y0, x1, y1) = (cx, cy - 50, cx + 15, cy + 120)
cv2.rectangle(bgrImage, (x0, y0), (x1, y1), (255, 255, 255), 1)
hsvRoi = hsvImage[y0:y1, x0:x1]
for channel, param in enumerate(HSV_CHANNELS):
    hist = cv2.calcHist([hsvRoi.copy()], [channel], None, [param[0]], param[1])
    hist = cv2.normalize(hist, hist, 0, 256, cv2.NORM_MINMAX)

    prob = cv2.calcBackProject([hsvImage.copy()], [channel], hist, param[1], 1)
    ret, prob = cv2.threshold(prob, 240, 255, 0)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(prob, -1, disc, prob)
    cv2.imshow("back proj " + str(param[2]), prob)
    # ax = plt.subplot(3, 1, channel + 1)
    # ax.set_xlim([0, param[0]])
    # ax.set_ylim([0, 255])
    # ax.set_title(str(param[2]))
    # ax.plot(hist)
    back_proj = cv2.bitwise_and(back_proj, back_proj, mask=prob)
    hist_hsv.append(hist)

masked = bgrImage & back_proj[:, :, np.newaxis]
cv2.imshow("masked", masked)
cv2.imshow("back projection", back_proj)
cv2.imshow("color", bgrImage)
# fig.tight_layout()
# fig.show()

hist = cv2.calcHist([hsvRoi], [0, 1, 2], None, [8, 2, 4], [0, 180, 0, 256, 0, 256])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
prob = cv2.calcBackProject([hsvImage], [0, 1, 2], hist, [0, 180, 0, 256, 0, 256], 1)
# ret, prob = cv2.threshold(prob, 240, 255, 0)
# disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# cv2.filter2D(prob, -1, disc, prob)
cv2.imshow("opencv backproj", prob)

cv2.waitKey()
cv2.destroyAllWindows()
