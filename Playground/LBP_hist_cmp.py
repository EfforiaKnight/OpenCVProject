import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.transform import rotate


def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins), facecolor='0.5')


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d

def match(lbp_obj, window):
    lbp = local_binary_pattern(window, n_points, radius, METHOD)
    n_bins = lbp.max() + 1
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    ref_hist, _ = np.histogram(lbp_obj, normed=True, bins=n_bins, range=(0, n_bins))
    score_kl = kullback_leibler_divergence(hist, ref_hist)
    score_chi = chi2_distance(hist, ref_hist)
    score_bh = cv2.compareHist(np.asarray(hist, dtype=np.float32), np.asarray(ref_hist, dtype=np.float32), cv2.HISTCMP_BHATTACHARYYA)
    score_kl2 = cv2.compareHist(np.asarray(hist, dtype=np.float32), np.asarray(ref_hist, dtype=np.float32), cv2.HISTCMP_KL_DIV)
    print("Score kl is {}\nScore chi is {}\nScore bh is {}\nScore kl2 is {}".format(score_kl, score_chi, score_bh, score_kl2))
    return score_kl, score_chi


METHOD = 'uniform'
plt.rcParams['font.size'] = 9

# settings for LBP
radius = 2
n_points = 8 * radius
xmin, ymin, xmax, ymax = (300, 120, 380, 220)

obj = cv2.imread("/home/efforia/PycharmProjects/OpenCVProject/images/grant.jpg", 0)
obj_cut = obj[ymin:ymax, xmin:xmax].copy()
cv2.rectangle(obj, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
cv2.imshow("object", obj)
cv2.imshow("cut object", obj_cut)

lbp_obj = local_binary_pattern(obj, n_points, radius, METHOD)
cv2.imshow("LBP object", lbp_obj)

window = cv2.imread("/home/efforia/PycharmProjects/OpenCVProject/images/grant.jpg", 0)

window = window[ymin-80:ymax-80, xmin-80:xmax-80]
window = rotate(window, angle=180, resize=False)

cv2.imshow("windows", window)

score = match(lbp_obj, window)

# fig, ax = plt.figure()
# plt.gray()
#
# ax.imshow(image)
# ax.axis('off')
# hist(ax, )
# ax.set_ylabel('Percentage')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
