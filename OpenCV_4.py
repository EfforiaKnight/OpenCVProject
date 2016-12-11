import numpy as np
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

img[55,55] = [255,255,255]
px = img[55, 55]

# Region of image
img[100:150, 100:150] = [255,255,255]

face = img[10:90, 150:230]
img[0:80, 0:80] = face

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
