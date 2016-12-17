import numpy as np
import cv2
from matplotlib import pyplot as plt

im = cv2.imread('box.jpg', cv2.IMREAD_COLOR)
im2 = cv2.imread('japao.png', cv2.IMREAD_COLOR)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imgray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# thresh = cv2.bitwise_not(imgray)
ret, thresh = cv2.threshold(imgray, 220, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
ret, thresh2 = cv2.threshold(imgray2, 220, 255, cv2.THRESH_BINARY_INV)
contours2 = cv2.findContours(thresh2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
cnt = contours[0]
cnt2 = contours2[0]
img = cv2.drawContours(im.copy(), [cnt], -1, (0, 255, 0), 2, cv2.LINE_AA)
img2 = cv2.drawContours(im2.copy(), [cnt2], -1, (0, 255, 0), 2, cv2.LINE_AA)
approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
approx_img = cv2.drawContours(im.copy(), [approx], -1, (0, 255, 255), 2, cv2.LINE_AA)
hull = cv2.convexHull(cnt)
hull_img = cv2.drawContours(im.copy(), [hull], -1, (0, 255, 0), 2, cv2.LINE_AA)
k = cv2.isContourConvex(hull)
print(k)
x, y, w, h = cv2.boundingRect(cnt)
img_rect = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
match = cv2.matchShapes(cnt, cnt2, 3, 0.0)
print(match)

mean_val = np.int0(cv2.mean(im2, mask=thresh2))
print(mean_val[:3])
mean_hsv = cv2.cvtColor(np.uint8([[mean_val[:3]]]), cv2.COLOR_BGR2HSV)
print(mean_hsv)
x, y = im2.shape[:2]
cut_off = im2[int(x/2), int(y/2)]
print("This is cut_off", cut_off)
std = np.std(im2, axis=2)
print(std, std.shape)
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
print(pixelpoints)

cv2.imshow("im2", im2)
# cv2.imshow("im", im)
cv2.imshow("tresh2", thresh2)
# cv2.imshow("res2", img2)
# # cv2.imshow("imgray", imgray)
# # cv2.imshow("thresh", thresh)
# cv2.imshow("res", img)
# cv2.imshow("approx", approx_img)
# cv2.imshow("hull_img", hull_img)
# cv2.imshow("img_rect", img_rect)

cv2.waitKey()
cv2.destroyAllWindows()
