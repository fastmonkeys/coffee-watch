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


def extract_black_color_as_new_image(img):
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([40, 40, 40], dtype="uint8")

    mask = cv2.inRange(img, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    img3 = cv2.bitwise_and(img, img, mask=mask)

    image = np.zeros(img3.shape, np.uint8)
    image[:] = (255, 255, 255)

    img3 = cv2.bitwise_or(img3, image, mask=mask_inv)
    return img3


def extract_coffee_color_as_new_image(img):
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([35, 35, 35], dtype="uint8")

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


def get_coffee_pot_location(img, top_left, bottom_right):
    offset = (12, 25)
    offset = (offset[0] + top_left[0], offset[1] + top_left[1])
    size = (64, 60)
    top_left = (offset[0], offset[1])
    bottom_right = (offset[0]+size[0], offset[1]+size[1])
    return top_left, bottom_right


def get_sub_image(img, top_left, bottom_right):
    return img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def get_center_pole_location(img):
    pole_image = cv2.imread('sample_images/sample2.png', cv2.IMREAD_COLOR)
    w, h, d = pole_image.shape
    STRIP_SIDES = 0
    # img = cv2.resize(img, (50, 1), interpolation=cv2.INTER_AREA)

    # img = img[:, STRIP_SIDES:-STRIP_SIDES-1]
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.imshow('image', pole_image)
    # cv2.waitKey(0)
    # res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = extract_black_color_as_new_image(img)
    res = cv2.matchTemplate(img, pole_image, cv2.TM_CCOEFF)
    # cv2.imshow('image', res)
    # # cv2.imshow('image', cv2.resize(res, (0,0), fy=100, fx=1, interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    # print cv2.imwrite('temp/debug_source.png', img)
    # print cv2.imwrite('temp/debug_template.png', pole_image)
    # print cv2.imwrite('temp/debug_output.png', res)


    # import pudb;pu.db

    top_left = (top_left[0] + STRIP_SIDES + w/2, top_left[1])

    return top_left


def get_coffee_level(img, position, name):
    def measure_coffee_level(line):
        # import pudb;pu.db
        return sum(1 for l in line if not all(l[0]))
    cv2.imwrite('temp2/%s_debug_source.png' % name, img)
    img = extract_coffee_color_as_new_image(img)
    cv2.imwrite('temp2/%s_debug_output.png' % name, img)

    X_OFFSETS = [4, -4, 8, -8]
    Y_OFFSET = 6
    Y_MAX = 45

    lines = [
        img[
            position[1] + Y_OFFSET:position[1] + Y_MAX,
            position[0] - x_offset:position[0] - x_offset+1
        ] for x_offset in X_OFFSETS
    ]
    cv2.imwrite('temp2/%s_debug_output2.png' % name, lines[0])
    return [measure_coffee_level(line) for line in lines]


for img_file in sys.argv[1:]:
    # Load an color image in grayscale
    # print "Opening", img_file
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)

    img3 = extract_red_color_as_new_image(img)
    top_left, bottom_right = get_coffee_maker_aabb(img3)

    pot_tl, pot_br = get_coffee_pot_location(
        img,
        top_left,
        bottom_right
    )

    img_pot = get_sub_image(img, pot_tl, pot_br)

    # img_pot = cv2.Canny(img_pot,100,200)

    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
    cv2.rectangle(img, pot_tl, pot_br, (200, 200, 200), 2)

    pos = get_center_pole_location(img_pot)
    # print "Pot pos %r %r" % pos

    print "%s: %s" % (get_coffee_level(img_pot, pos, img_file.split('/')[1].split('.')[0]), img_file)

    # cv2.imshow('image', img_pot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('image', cv2.resize(img3, (0,0), fy=100, fx=1, interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    center_tl = pot_tl[0] + pos[0], pot_tl[1] + pos[1]
    center_br = pot_tl[0] + pos[0], pot_tl[1] + pos[1] + 50

    cv2.rectangle(img, center_tl, center_br, (0, 0, 255), 2)


    our_file = 'temp/' + img_file.split('/')[1].split('.')[0] + '.png'
    # print "Writing", our_file
    cv2.imwrite(our_file, img)
