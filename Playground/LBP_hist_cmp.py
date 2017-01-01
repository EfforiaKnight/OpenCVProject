import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins), facecolor='0.5')


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def match(lbp_obj, window):
    best_score = 10
    lbp = local_binary_pattern(window, n_points, radius, METHOD)
    n_bins = lbp.max() + 1
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    ref_hist, _ = np.histogram(lbp_obj, normed=True, bins=n_bins, range=(0, n_bins))
    score = kullback_leibler_divergence(hist, ref_hist)
    print("Score is {}".format(score))
    return score

METHOD = 'uniform'
plt.rcParams['font.size'] = 9

# settings for LBP
radius = 2
n_points = 8 * radius

obj = cv2.imread("/home/efforia/PycharmProjects/OpenCVProject/images/grant.jpg", 0)

lbp_obj = local_binary_pattern(obj, n_points, radius, METHOD)

window = cv2.imread("/home/efforia/PycharmProjects/OpenCVProject/images/jurassicpark.jpg", 0)

score = match(lbp_obj, window)

# fig, ax = plt.figure()
# plt.gray()
#
# ax.imshow(image)
# ax.axis('off')
# hist(ax, )
# ax.set_ylabel('Percentage')

plt.show()
