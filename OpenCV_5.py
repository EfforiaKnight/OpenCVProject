import numpy as np
import cv2

img1 = cv2.imread('pic1.jpg')
img2 = cv2.imread('pic2.jpg')
logo = cv2.imread('logo.png')
# add = img1 + img2
# add = cv2.add(img1, img2)
# weighted = cv2.addWeighted(img1, 0.4, img2, 0.9, 0)

rows, cols, channels = logo.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
logo_fg = cv2.bitwise_and(logo, logo, mask=mask)

dst = cv2.add(img1_bg, logo_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.imshow('img2gray', img2gray)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('img1_bg', img1_bg)
cv2.imshow('logo_fg', logo_fg)
cv2.imshow('mask', mask)
cv2.imshow('dst', dst)

# cv2.imshow('mask', mask)
# cv2.imshow('weighted', weighted)
cv2.waitKey()
cv2.destroyAllWindows()