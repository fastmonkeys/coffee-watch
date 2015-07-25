import sys

import numpy as np
import cv2


def extract_red_color_as_new_image(img):
    lower = np.array([0, 0, 50], dtype="uint8")
    upper = np.array([40, 40, 255], dtype="uint8")

    mask = cv2.inRange(img, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    img3 = cv2.bitwise_and(img, img, mask=mask)

    image = np.zeros(img3.shape, np.uint8)
    image[:] = (255, 255, 255)

    img3 = cv2.bitwise_or(img3, image, mask=mask_inv)
    return img3


def get_coffee_maker_aabb(img):
    template = cv2.imread('sample_images/sample.png', cv2.IMREAD_COLOR)
    w, h = template.shape[1], template.shape[0]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right


def extract_coffee_pot_as_new_image(img, top_left, bottom_right):
    offset = (13, 30)
    offset = (offset[0] + top_left[0], offset[1] + top_left[1])
    size = (50, 50)
    img2 = img[offset[1]:offset[1]+size[1], offset[0]:offset[0]+size[0]]
    return img2


for img_file in sys.argv[1:]:
    # Load an color image in grayscale
    print "Opening", img_file
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)

    img3 = extract_red_color_as_new_image(img)
    top_left, bottom_right = get_coffee_maker_aabb(img3)

    img = extract_coffee_pot_as_new_image(img, top_left, bottom_right)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    our_file = 'temp/' + img_file.split('/')[1].split('.')[0] + '.png'
    print "Writing", our_file
    print cv2.imwrite(our_file, img)
