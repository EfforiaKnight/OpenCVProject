import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

# cv2.line(img, (0, 0), (150, 150), (200, 200, 200), 4)
cv2.rectangle(img, (90, 10), (230, 150), (200, 200, 200), 4)
cv2.circle(img, (100, 100), 50, (20, 20, 20), -1)

pts = np.array([[10,5], [20,30], [70,20], [50,10], [40, 30]], np.int32)
# pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], isClosed=True, color=(20,20,20), thickness=10)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Welcome', (0,130), font, 2, (230, 230, 230), 1, cv2.LINE_AA)

cv2.imshow('image', img)

cv2.waitKey()
cv2.destroyAllWindows()