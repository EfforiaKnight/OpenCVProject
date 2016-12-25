from skimage import feature
import numpy as np
import scipy.io as sio
import cv2


def describe(image):
    # Get the image size
    shape = image.shape

    # divide the image into 10x10 regions
    wid = int(shape[0] / 10)
    hi = int(shape[1] / 10)

    # histogram initialization
    hist = []

    # Use uniform LBP operator in each region
    for i in range(10):
        for j in range(10):
            temp = image[i * wid:i * wid + 10, j * hi:j * hi + 10]
            lbp = feature.local_binary_pattern(temp, 8, 1, method="nri_uniform")
            (sub_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 60))
            # sub_hist.astype("uint8")
            hist.extend(sub_hist)

    return hist


if __name__ == '__main__':
    path = open('/Users/tone/Documents/NYU/Computer Vision/Project/lfw/imageList.txt', 'r')
    imageList = path.readlines()

    data = []

    for imagePath in imageList:
        imagePath = imagePath.rstrip('\n')
        if imagePath == '':
            break
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = describe(gray)
        data.append(hist)

    sio.savemat("data.mat", {'lbp_data': data})
