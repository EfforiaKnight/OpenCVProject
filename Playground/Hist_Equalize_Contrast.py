import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    _, frame = cap.read()

    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    eq_yuv_frame = yuv_frame.copy()
    eq_yuv_frame[:, :, 0] = cv2.equalizeHist(eq_yuv_frame[:, :, 0])

    clh_yuv_frame = yuv_frame.copy()
    clh_yuv_frame[:, :, 0] = clahe.apply(clh_yuv_frame[:, :, 0])

    eq_frame = cv2.cvtColor(eq_yuv_frame, cv2.COLOR_YUV2BGR)
    clh_frame = cv2.cvtColor(clh_yuv_frame, cv2.COLOR_YUV2BGR)

    # eq_frame = cv2.blur(eq_frame, (5, 5))
    # clh_frame = cv2.blur(clh_frame, (5, 5))

    cv2.imshow('frame', frame)
    cv2.imshow('eq_frame', eq_frame)
    cv2.imshow('clh_frame', clh_frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
