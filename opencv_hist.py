# import the necessary packages
import numpy as np
from matplotlib import pyplot as plt
import cv2


class RGBHistogram:
    def __init__(self, bins):
        # store the number of bins the histogram will use
        self.bins = bins

    def describe(self, image):
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        hist_flat = cv2.normalize(hist, hist.copy())
        print(hist_flat)
        # return out 3D histogram as a flattened array
        return hist_flat.flatten()


BGR_hist = RGBHistogram([8, 8, 8])
image = cv2.imread('blue_red.png', cv2.IMREAD_COLOR)

cv2.imshow('image', image)

hist_flat = BGR_hist.describe(image)
print(hist_flat)

plt.title("Histogram_flat")
plt.plot(hist_flat)
plt.xlabel("# Bin")
plt.ylabel("% pixel")
plt.xlim([0, 512])

chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure(2)
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []

# loop over the image channels
for (chan, color) in zip(chans, colors):
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)

    # plot the histogram
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
print(np.array(features).flatten().shape[0])
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
cv2.distanceTransformWithLabels()
