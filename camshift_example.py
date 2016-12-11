import numpy as np
import cv2

frame = None
roiPts = []
inputMode = False

# Getting the camera reference
cap = cv2.VideoCapture(0)


# Callback function to get the ROI by clicking into four points
def click_and_crop(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("image", frame)


# Attaching the callback into the video window
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
roiBox = None

# Main loop
while (1):
    ret, frame = cap.read()

    if roiBox is not None:
        # Making the frame into HSV and backproject the HSV frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply meanshift to get the new location
        ret, roiBox = cv2.CamShift(dst, roiBox, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, 255, 2)

        # Draw the center
        cx = (pts[0][0] + pts[1][0]) / 2
        cy = (pts[0][1] + pts[2][1]) / 2
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), 2)
        # cv2.imshow('img2',frame)

    # handle if the 'i' key is pressed, then go into ROI
    # selection mode
    cv2.imshow("image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("i") and len(roiPts) < 4:
        # indicate that we are in input mode and clone the
        # frame
        inputMode = True
        orig = frame.copy()

        # keep looping until 4 reference ROI points have
        # been selected; press any key to exit ROI selction
        # mode once 4 points have been selected
        while len(roiPts) < 4:
            cv2.imshow("image", frame)
            cv2.waitKey(0)

        # determine the top-left and bottom-right points
        roiPts = np.array(roiPts)
        s = roiPts.sum(axis=1)
        tl = roiPts[np.argmin(s)]
        br = roiPts[np.argmax(s)]

        # grab the ROI for the bounding box and convert it
        # to the HSV color space
        roi = orig[tl[1]:br[1], tl[0]:br[0]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # compute a HSV histogram for the ROI and store the
        # bounding box
        roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        roiBox = (tl[0], tl[1], br[0], br[1])

    # k = cv2.waitKey(60) & 0xff
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
