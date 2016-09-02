import sys

import numpy as np

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

debugging = False

COFFEE_MAKER_MASK_FILE = 'sample_images/coffee_maker_sample_mask.png'

COFFEE_MAKES_COLOR_RANGES = [
    ((0, 80), (0, 80), (0, 80)),
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


def match_template(img, template):
    w, h = template.shape[1], template.shape[0]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right, max_val


def get_coffee_maker_aabb(img):
    template = cv2.imread(COFFEE_MAKER_MASK_FILE, cv2.IMREAD_COLOR)
    return match_template(img, template)


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


def get_coffee_level(img):
    Y_MAX = img.shape[0]

    img_coffee_level = cv2.reduce(img, 1, cv2.cv.CV_REDUCE_AVG)
    debug(img_coffee_level, 'coffee_pot_reduced')

    levels = [
        img_coffee_level[i][0][0] for i in range(img_coffee_level.shape[0])
    ]
    COFFEE_SKIP_HEIGHT = 1 # TODO: How much we have certain black to measure blackness threshold ***********
    threshold = sum(levels[0:COFFEE_SKIP_HEIGHT]) / COFFEE_SKIP_HEIGHT * 0.80

    for i, v in enumerate(levels):
        # print "i:%d %r" % (i, v)
        if v < threshold:
            break
    result = Y_MAX - i
    print("Coffee: %d/%d" % (result, Y_MAX - COFFEE_SKIP_HEIGHT))

    return int(result * 100 / Y_MAX), img_coffee_level


def process_image(img, img_file):
    places = []

    for i, limits in enumerate(COFFEE_MAKES_COLOR_RANGES):
        img3 = extract_color_as_new_image(img, limits)
        debug(img3, 'coffee_maker_bw_%d' % i)
        top_left, bottom_right, max_val = get_coffee_maker_aabb(img3)
        img_normal = img.copy()
        cv2.rectangle(img_normal, top_left, bottom_right, (255, 0, 0), 2)
        debug(img_normal, 'coffee_maker_pos_%d' % i)
        places.append((max_val, top_left, bottom_right))

    score, top_left, bottom_right = sorted(places, key=lambda x: x[0])[-1]

    img = get_sub_image(img, top_left, bottom_right)

    print("Coffee maker mask score: %s" % score)
    debug(img, 'coffee_maker_pos')

    pot_tl, pot_br = (70, 90), (120, 170)
    pot_header_tl, pot_header_br = (0, 0), (50, 20)

    img_pot_with_top = get_sub_image(img, pot_tl, pot_br)

    h, w, d = img_pot_with_top.shape

    img_pot_confirmer = get_sub_image(
        img_pot_with_top.copy(),
        pot_header_tl,
        pot_header_br
    )
    img_pot = get_sub_image(
        img_pot_with_top.copy(),
        (0, pot_header_br[1]),
        (w, h)
    )

    cv2.rectangle(img_pot_with_top, pot_header_tl, pot_header_br, (255, 0, 0), 2)

    debug(img_pot_with_top, 'coffee_pot_pos')

    # Why this isn't working??? *****************************************************************************
    black_template = np.zeros(img_pot_confirmer.shape, np.uint8)

    _, _, score = match_template(img_pot_confirmer, black_template)

    print("Pot confirmer mask score: %s" % score)

    result = None
    result, coffee_level_image = get_coffee_level(img_pot)
    if result is not None:
        print("%s: %s" % (result, img_file))
    else:
        print("No pot: %s" % img_file)

    cv2.rectangle(img, pot_tl, pot_br, (200, 200, 200), 2)

    h, w, _ = img.shape
    COFFEE_IMG_WIDTH = 10
    coffee_img_height = coffee_level_image.shape[0]

    coffee_level_image = cv2.resize(
        coffee_level_image,
        (COFFEE_IMG_WIDTH, coffee_img_height),
        interpolation=cv2.INTER_AREA
    )

    img[
        pot_br[1] - coffee_img_height:pot_br[1],
        w-COFFEE_IMG_WIDTH:w
    ] = coffee_level_image

    cv2.rectangle(
        img,
        (w-2*COFFEE_IMG_WIDTH, pot_br[1] - result*coffee_img_height/100),
        (w-COFFEE_IMG_WIDTH, pot_br[1]),
        (0, 0, 0),
        -1
    )

    cv2.rectangle(
        img,
        (w-2*COFFEE_IMG_WIDTH, pot_br[1] - coffee_img_height),
        (w-COFFEE_IMG_WIDTH, pot_br[1] - result*coffee_img_height/100),
        (0, 0, 255),
        -1
    )

    return img, result


def debug(img, step):
    if debugging:
        filename = 'debug_output/%s.png' % step
        cv2.imwrite(filename, img)
        print('Wrote %s' % filename)


def rotate_image(img, degrees):
    rows, cols, _ = img.shape
    return cv2.warpAffine(
        img,
        cv2.getRotationMatrix2D((cols/2, rows/2), degrees, 1),
        (cols, rows)
    )


def preprocess_image(img):
    img = rotate_image(img, -90)
    img = get_sub_image(img, (650, 130), (960, 350))
    return img


if __name__ == "__main__":
    start = 1
    args = [a for a in sys.argv[1:]]
    if '-d' in args:
        debugging = True
        args = [a for a in args if a != '-d']

    for img_file in args:
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = preprocess_image(img)
        debug(img, 'after_preprocess')
        img, value = process_image(img, img_file)

        our_file = 'debug_output/' + img_file.split('/')[1].split('.')[0] + '.png'
        cv2.imwrite(our_file, img)
