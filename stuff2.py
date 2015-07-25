import sys

import numpy as np
import cv2


img_file = 'sample_images/0_cup.jpg'
# img_file = 'sample_images/1_cup_dark2.jpg'
# Load an color image in grayscale
img = cv2.imread(img_file, cv2.IMREAD_COLOR)
img2 = cv2.bilateralFilter(img,9,75,75)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img3 = [[[2*r,g,b] for r,g,b in column] for column in img]

lower = np.array([0,0,50], dtype = "uint8")
upper = np.array([40,40,255], dtype = "uint8")

# b,g,r = cv2.split(img)  # np.array(img, dtype = np.float32))
# g2 = g*0.5
# b2 = b*0.5
# r = (r - g2) - b2

# img3 = np.array(cv2.merge((b2,g2,r)), dtype = np.uint8)

# http://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
mask = cv2.inRange(img, lower, upper)
mask_inv = cv2.bitwise_not(mask)
img3 = cv2.bitwise_and(img, img, mask=mask)

image = np.zeros(img3.shape, np.uint8)
image[:] = (255, 255, 0)

img3 = cv2.bitwise_or(img3, image, mask=mask_inv)

cv2.imshow('image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()






# ret, r = cv2.threshold(r, 40, 255, cv2.THRESH_BINARY)
# ret, g = cv2.threshold(g, 40, 255, cv2.THRESH_BINARY)
# ret, b = cv2.threshold(b, 40, 255, cv2.THRESH_BINARY)
# r = cv2.bitwise_and(r,g,mask = r)
# mask_inv = cv2.bitwise_not(mask)
