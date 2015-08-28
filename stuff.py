import sys
from math import ceil

import numpy as np
import cv2


COFFEE_MAKES_COLOR_RANGES = [
    ((0, 80), (0, 80), (50, 255)),
    ((0, 100), (0, 100), (50, 255)),
]

COFFEE_POT_TOP_COLOR_RANGES = [
    ((0, 60), (0, 60), (0, 60)),
    ((0, 100), (0, 100), (0, 100)),
    ((0, 120), (0, 120), (0, 120)),
]


def extract_color_as_new_image(img, limit):
    lower_limit = [limit[i][0] for i in range(3)]
    upper_limit = [limit[i][1] for i in range(3)]

    lower = np.array(lower_limit, dtype="uint8")
    upper = np.array(upper_limit, dtype="uint8")

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
    return top_left, bottom_right, max_val


def get_coffee_pot_location_offset(img, top_left, bottom_right):
    offset = (30, 12)
    offset = (offset[0] + top_left[0], offset[1] + top_left[1])
    size = (42, 36)
    top_left = (offset[0], offset[1])
    bottom_right = (offset[0]+size[0], offset[1]+size[1])
    return top_left, bottom_right


def get_sub_image(img, top_left, bottom_right):
    return img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def get_coffee_pot_aabb(img_pot, pot_tl, pot_br, sample_img):
    pot_w = pot_br[0] - pot_tl[0]
    pos_x = max(pot_tl[0] - 2*pot_w, 0)

    img[pot_tl[1]:pot_br[1], pos_x:pos_x + pot_w] = img_pot
    img_pot_det = get_sub_image(img_pot, (0, 0), (pot_br[0] - pot_tl[0], 14))
    res = cv2.matchTemplate(img_pot_det, sample_img, cv2.TM_CCOEFF)
    # cv2.imwrite('temp2/img_pot_res.png', res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    real_top_left = (
        pot_tl[0] + top_left[0],
        pot_tl[1] + top_left[1]
    )
    h, w, d = sample_img.shape

    return (
        real_top_left[0] + w/2,
        real_top_left[1]
    ), max_val


def get_coffee_pot_location(img, pot_tl, pot_br, img_file):
    pole_image = cv2.imread('sample_images/sample2.png', cv2.IMREAD_COLOR)
    img_pot_org = get_sub_image(img, pot_tl, pot_br)
    cv2.imwrite('temp2/coffee_pot_org.png', img_pot_org)

    places = []

    for i, limits in enumerate(COFFEE_POT_TOP_COLOR_RANGES):
        img_pot = extract_color_as_new_image(img_pot_org, limits)
        cv2.imwrite('temp2/coffee_pot_bw_%d.png' % i, img_pot)
        top_left, max_val = get_coffee_pot_aabb(img_pot, pot_tl, pot_br, pole_image)
        img_normal = img.copy()
        bottom_right = (top_left[0], top_left[0] + 30)
        cv2.rectangle(img_normal, top_left, bottom_right, (255, 0, 0), 2)
        cv2.imwrite('temp2/coffee_maker_pos_%d.png' % i, img_normal)
        places.append((max_val, top_left))

    max_val, real_top_left = sorted(places, key=lambda x: x[0])[-1]
    max_val = int(max_val / 10000)

    CENTER_POLE_MATCH_THRESHOLD = 500
    print "Pot match: %d/%d" % (max_val, CENTER_POLE_MATCH_THRESHOLD)

    ok = max_val > CENTER_POLE_MATCH_THRESHOLD
    if ok:
        our_file = 'temp/ok_pot_' + img_file.split('/')[1].split('.')[0] + '.png'
        cv2.imwrite(our_file, img_pot_org)
    else:
        our_file = 'temp/fail_pot_' + img_file.split('/')[1].split('.')[0] + '_bw.png'
        cv2.imwrite(our_file, img_pot)
        our_file = 'temp/fail_pot_' + img_file.split('/')[1].split('.')[0] + '.png'
        cv2.imwrite(our_file, img_pot_org)

    # pole_image2 = cv2.imread('sample_images/sample3.png', cv2.IMREAD_COLOR)
    # h2, w2, d2 = pole_image2.shape

    # # img_pot2 = cv2.Canny(img_pot_org, 200, 500)
    # img_pot2 = img_pot_org

    # # cv2.imwrite('temp2/img_pot2.png', img_pot2)
    # img_src = get_sub_image(
    #     img_pot2,
    #     (max_loc[0] + w/2 - ceil(w2/2.0), max_loc[1]),
    #     (max_loc[0] + w/2 + ceil(w2/2.0), max_loc[1] + h2)
    # )

    # # cv2.rectangle(
    # #     img,
    # #     (real_top_left[0] + w/2 - w2/2, real_top_left[1]),
    # #     (real_top_left[0] + w/2 + w2/2, real_top_left[1] + h2),
    # #     (0, 0, 255),
    # #     2
    # # )
    # import pudb;pu.db
    # res = cv2.matchTemplate(img_src, pole_image2, cv2.TM_CCOEFF)
    # min_val, max_val2, min_loc, max_loc = cv2.minMaxLoc(res)

    # cv2.imshow('image', cv2.resize(pole_image2, (0,0), fy=10, fx=10, interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)
    # cv2.imshow('image', cv2.resize(img_pot, (0,0), fy=10, fx=10, interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)
    # cv2.imshow('image', cv2.resize(img_src, (0,0), fy=10, fx=10, interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # import pudb;pu.db

    # print "initial: %s second: %s" % (max_val, max_val2)

    return (
        real_top_left,
        ok #and
        # (max_val2 > CENTER_POLE_MATCH2_THRESHOLD)
    )


def get_coffee_level(img, position, name):
    IMG_X_OFFSET = -30
    IMG_W = 10

    X_RANGE = 10
    Y_OFFSET = 4
    Y_MAX = 29
    Y_DELTA = Y_MAX - Y_OFFSET

    position = (position[0], position[1] + Y_OFFSET)

    img_bw = get_sub_image(
        img,
        (position[0] - X_RANGE, position[1]),
        (position[0] + X_RANGE, position[1] + Y_DELTA)
    )
    # print img_bw.shape

    our_file = 'temp2/coffee_avg_src.png'
    cv2.imwrite(our_file, img_bw)

    # img_asd = cv2.resize(img_bw, (1,Y_MAX), interpolation=cv2.INTER_AREA)
    # print img_asd.shape
    img_asd = cv2.reduce(img_bw, 1, cv2.cv.CV_REDUCE_AVG)
    # print img_asd.shape
    img_asd2 = cv2.resize(
        img_asd,
        (IMG_W, Y_DELTA),
        interpolation=cv2.INTER_AREA
    )

    COFFEE_SKIP_HEIGHT = 3
    levels = [img_asd[i][0][0] for i in range(img_bw.shape[0])]
    threshold = sum(levels[0:COFFEE_SKIP_HEIGHT]) / COFFEE_SKIP_HEIGHT - 50

    for i, v in enumerate(levels):
        if i < 3:
            continue
        # print "i:%d %r" % (i, v)
        if v < threshold:
            break
    result = Y_DELTA - i
    print "Coffee: %d/%d" % (result, Y_MAX - COFFEE_SKIP_HEIGHT)

    y_top = position[1]
    y_middle = position[1] + Y_DELTA - result
    y_bottom = position[1] + Y_DELTA
    cv2.rectangle(
        img,
        (position[0] + IMG_X_OFFSET-1, y_top-1),
        (position[0] + IMG_X_OFFSET + IMG_W+1, y_bottom+1),
        (255, 255, 255),
        -1
    )
    cv2.rectangle(
        img,
        (position[0] + IMG_X_OFFSET, y_top),
        (position[0] + IMG_X_OFFSET + IMG_W, y_middle),
        (255, 255, 0),
        -1
    )
    cv2.rectangle(
        img,
        (position[0] + IMG_X_OFFSET, y_middle),
        (position[0] + IMG_X_OFFSET + IMG_W, y_bottom),
        (0, 0, 0),
        -1
    )

    img[
        y_top:y_top+Y_DELTA,
        position[0] + IMG_X_OFFSET - IMG_W:position[0] + IMG_X_OFFSET - IMG_W + 10
    ] = img_asd2

    return int(result * 100 / Y_MAX)


def process_image(img, img_file):
    img = get_sub_image(img, (520, 200), (720, 350))

    places = []

    for i, limits in enumerate(COFFEE_MAKES_COLOR_RANGES):
        img3 = extract_color_as_new_image(img, limits)
        cv2.imwrite('temp2/coffee_maker_bw_%d.png' % i, img3)
        top_left, bottom_right, max_val = get_coffee_maker_aabb(img3)
        img_normal = img.copy()
        cv2.rectangle(img_normal, top_left, bottom_right, (255, 0, 0), 2)
        cv2.imwrite('temp2/coffee_maker_pos_%d.png' % i, img_normal)
        places.append((max_val, top_left, bottom_right))

    _, top_left, bottom_right = sorted(places, key=lambda x: x[0])[-1]

    pot_tl, pot_br = get_coffee_pot_location_offset(
        img,
        top_left,
        bottom_right
    )

    # img_pot = cv2.Canny(img_pot,100,200)


    pos, match = get_coffee_pot_location(img, pot_tl, pot_br, img_file)
    # print "Pot pos %r %r" % pos
    center_tl = pos[0], pos[1]
    center_br = pos[0], pos[1] + 30

    if match:
        result = get_coffee_level(img, pos, img_file.split('/')[1].split('.')[0])
        print "%s: %s" % (result, img_file)
        cv2.rectangle(img, center_tl, center_br, (255, 255, 255), 2)
    else:
        cv2.rectangle(img, center_tl, center_br, (0, 0, 255), 2)
        print "No pot: %s" % img_file

    # print "GFROM %s %s to %s %s" % (center_tl[0],center_tl[1], center_br[0],center_br[1])

    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
    cv2.rectangle(img, pot_tl, pot_br, (200, 200, 200), 2)

    # cv2.imshow('image', img_pot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('image', cv2.resize(img3, (0,0), fy=100, fx=1, interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # img = get_sub_image(img, top_left, bottom_right)
    # img = cv2.resize(img, (0,0), fy=10, fx=10, interpolation=cv2.INTER_NEAREST)
    return img


for img_file in sys.argv[1:]:
    # Load an color image in grayscale
    # print "Opening", img_file
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = process_image(img, img_file)
    our_file = 'temp/' + img_file.split('/')[1].split('.')[0] + '.png'
    # print "Writing", our_file
    cv2.imwrite(our_file, img)
