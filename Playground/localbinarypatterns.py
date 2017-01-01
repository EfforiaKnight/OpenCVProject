from skimage import feature
import matplotlib.pyplot as plt
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        n_bins = lbp.max() + 1
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), normed=True)
        hist = np.asarray(hist, dtype=np.float32)
        return hist, lbp
