import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def bilinear_interpolation(x, y, img):
    x1, y1 = int(r), int(c)
    x2, y2 = math.ceil(r), math.ceil(c)

    r1 = (x2 - x) / (x2 - x1) * get_pixel_else_0(img, x1, y1) + (x - x1) / (x2 - x1) * get_pixel_else_0(img, x2, y1)
    r2 = (x2 - x) / (x2 - x1) * get_pixel_else_0(img, x1, y2) + (x - x1) / (x2 - x1) * get_pixel_else_0(img, x2, y2)

    return (y2 - y) / (y2 - y1) * r1 + (y - y1) / (y2 - y1) * r2    

def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(image, idx, idy):
    if idx < int(len(image)) - 1 and idy < len(image[0]):
        return image[idx,idy]
    else:
        return 0

def find_variations(pixel_values):
    prev = pixel_values[-1]
    t = 0
    for p in range(0, len(pixel_values)):
        cur = pixel_values[p]
        if cur != prev:
            t += 1
        prev = cur
    return t

img = cv2.imread('/home/efforia/PycharmProjects/OpenCVProject/images/grant.jpg', 0)
transformed_img = cv2.imread('/home/efforia/PycharmProjects/OpenCVProject/images/grant.jpg', 0)
unassigned = []
pixel_values = set()

P = 8 # number of pixels
R = 1 # radius 

variating_blocks = 0

for x in range(0, len(img)):
    for y in range(0, len(img[0])):
        center        = img[x,y]
        pixels = []
        for point in range(1, P + 1):
            r = x + R * math.cos(2 * math.pi * point / P)
            c = y - R * math.sin(2 * math.pi * point / P)
            if r < 0 or c < 0:
                pixels.append(0)
                continue            
            if int(r) == r:
                if int(c) != c:
                    c1 = int(c)
                    c2 = math.ceil(c)
                    w1 = (c2 - c) / (c2 - c1)
                    w2 = (c - c1) / (c2 - c1)
                                    
                    pixels.append(int((w1 * get_pixel_else_0(img, int(r), int(c)) + \
                                   w2 * get_pixel_else_0(img, int(r), math.ceil(c))) / (w1 + w2)))
                else:
                    pixels.append(get_pixel_else_0(img, int(r), int(c)))
            elif int(c) == c:
                r1 = int(r)
                r2 = math.ceil(r)
                w1 = (r2 - r) / (r2 - r1)
                w2 = (r - r1) / (r2 - r1)                
                pixels.append((w1 * get_pixel_else_0(img, int(r), int(c)) + \
                               w2 * get_pixel_else_0(img, math.ceil(r), int(c))) / (w1 + w2))
            else:
                pixels.append(bilinear_interpolation(r, c, img))


        values = thresholded(center, pixels)
        variations = find_variations(values)
        if variations <= 2:
            res = 0
            variating_blocks += 1
            for a in range(0, len(values)):
                res += values[a] * 2 ** a
            transformed_img.itemset((x,y), res)
            pixel_values.add(res)
        else:
            unassigned.append((x,y))
    print(x)

unassigned_value = len(pixel_values)
pixel_values = sorted(pixel_values)
no_of_pixel_values = len(pixel_values)
trans_p1_u2 = {}
for p in range(0, len(pixel_values)):
    trans_p1_u2[pixel_values[p]] = p

for r in range(0, len(transformed_img)):
    for c in range(0, len(transformed_img[0])):
        if (r,c) in unassigned:
            transformed_img.itemset((r,c), unassigned_value)
        else:
            p1 = transformed_img[r,c]
            transformed_img.itemset((r,c), trans_p1_u2[p1])
    

cv2.imshow('image', img)
cv2.imshow('thresholded image', transformed_img)

hist,bins = np.histogram(transformed_img.flatten(),no_of_pixel_values + 1,[0,no_of_pixel_values])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.show()
plt.hist(transformed_img.flatten(),no_of_pixel_values,[0,no_of_pixel_values], color = 'b')
plt.xlim([0,no_of_pixel_values])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
