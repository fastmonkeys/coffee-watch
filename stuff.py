import sys

import numpy as np
import cv2


def get_aabb(img):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30

    # cv2.imshow('image', 255-img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    detector = cv2.SimpleBlobDetector(params)
    keypoints = detector.detect(255 - img)
    if len(keypoints) == 1:
        return ()
    return detector


def get_aabb(img):
    template = cv2.imread('sample_images/sample.png', cv2.IMREAD_COLOR)
    # import pudb;pu.db
    w, h = template.shape[1], template.shape[0]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 127, 2)
    return top_left, bottom_right
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



lower = np.array([0, 0, 50], dtype="uint8")
upper = np.array([40, 40, 255], dtype="uint8")


for img_file in sys.argv[1:]:
    # Load an color image in grayscale
    print "Opening", img_file
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)

    mask = cv2.inRange(img, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    img3 = cv2.bitwise_and(img, img, mask=mask)

    image = np.zeros(img3.shape, np.uint8)
    image[:] = (255, 255, 255)

    img3 = cv2.bitwise_or(img3, image, mask=mask_inv)

    aabb = get_aabb(img3)

    our_file = 'temp/' + img_file.split('/')[1].split('.')[0] + '.png'
    print "Writing", our_file
    print cv2.imwrite(our_file, img3)
