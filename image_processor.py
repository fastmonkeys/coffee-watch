import sys

import numpy as np
import cv2


COFFEE_MAKES_COLOR_RANGES = [
    ((0, 80), (0, 80), (50, 255)),
    ((0, 100), (0, 100), (50, 255)),
    ((0, 20), (0, 20), (20, 100)),
]

COFFEE_POT_TOP_COLOR_RANGES = [
    ((0, 60), (0, 60), (0, 60)),
    ((0, 100), (0, 100), (0, 100)),
    ((0, 120), (0, 120), (0, 120)),
    ((0, 40), (0, 40), (0, 40)),
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
    img_pot_det = get_sub_image(img_pot, (0, 0), (pot_br[0] - pot_tl[0], 14))
    res = cv2.matchTemplate(img_pot_det, sample_img, cv2.TM_CCOEFF)

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

    CENTER_POLE_MATCH_THRESHOLD = 550
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

    return real_top_left, ok


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

    our_file = 'temp2/coffee_avg_src.png'
    cv2.imwrite(our_file, img_bw)

    # img_coffee_level = cv2.resize(img_bw, (1,Y_MAX), interpolation=cv2.INTER_AREA)
    img_coffee_level = cv2.reduce(img_bw, 1, cv2.cv.CV_REDUCE_AVG)
    img_coffee_level_big = cv2.resize(
        img_coffee_level,
        (IMG_W, Y_DELTA),
        interpolation=cv2.INTER_AREA
    )

    COFFEE_SKIP_HEIGHT = 3
    levels = [img_coffee_level[i][0][0] for i in range(img_bw.shape[0])]
    threshold = sum(levels[0:COFFEE_SKIP_HEIGHT]) / COFFEE_SKIP_HEIGHT * 0.65

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
    ] = img_coffee_level_big

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

    pos, match = get_coffee_pot_location(img, pot_tl, pot_br, img_file)
    center_tl = pos[0], pos[1]
    center_br = pos[0], pos[1] + 30

    result = None
    if match:
        result = get_coffee_level(
            img,
            pos,
            img_file.split('/')[1].split('.')[0]
        )
        print "%s: %s" % (result, img_file)
        cv2.rectangle(img, center_tl, center_br, (255, 255, 255), 2)
    else:
        cv2.rectangle(img, center_tl, center_br, (0, 0, 255), 2)
        print "No pot: %s" % img_file

    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
    cv2.rectangle(img, pot_tl, pot_br, (200, 200, 200), 2)

    return img, result


for img_file in sys.argv[1:]:
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img, value = process_image(img, img_file)
    our_file = 'temp/' + img_file.split('/')[1].split('.')[0] + '.png'
    cv2.imwrite(our_file, img)
