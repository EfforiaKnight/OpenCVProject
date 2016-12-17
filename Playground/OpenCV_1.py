import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('image', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.plot([50, 100], [100, 150], 'w', linewidth=10)
plt.show()