from skimage import io
from skimage.filters import gaussian, laplace, median
# from skimage.morphology import disk
from skimage import transform as tf
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def print_runtime(descr=""):
    global start_time
    print(descr, "took", datetime.now() - start_time)
    start_time = datetime.now()


def read_file(relpath):
    board = io.imread(relpath)
    return board


def extract_color(im, col, tol, sup=None):
    ix = np.sum((np.fabs(im - col * np.ones_like(im)) <= tol), 2) == 3
    cCube = np.zeros_like(im[:, :, 0])
    cCube[ix] = 1
    if sup is not None:
        cCube[sup] = 0
    cCube = cCube.astype('uint8')
    return cCube


def clean_white_board(bb, wb):
    # optimisation of wb: only count white pixels surrounded by blue ones,
    # i.e. discard all pixels of the present wb until a blue one is reached (from all four directions)
    for x in range(bb.shape[0]):
        blue_pos = np.nonzero(bb[x, :])[0]
        if blue_pos.size >= 1:
            wb[x, 0:blue_pos[0]] = 0  # from left
            wb[x, blue_pos[-1]:bb.shape[1]] = 0  # from right
        else:
            wb[x, :] = 0  # completely delete the lines from wb in which there is no blue at all

    for y in range(bb.shape[1]):
        blue_pos = np.nonzero(bb[:, y])[0]
        if blue_pos.size >= 1:
            wb[0:blue_pos[0], y] = 0  # from top
            wb[blue_pos[-1]:bb.shape[0], y] = 0  # from bottom
        else:
            wb[:, y] = 0  # completely delete the lines from wb in which there is no blue at all
            wb[:, y] = 0  # completely delete the lines from wb in which there is no blue at all

    # in order to account for the board stand, do the same for br->tl and bl->tr directions
    # take lines of 45 degree angle to avoid Bresenham algorithm etc.
    for offset in range(bb.shape[1] - bb.shape[0]):  # bl->tr
        for x in range(bb.shape[0]):
            if bb[bb.shape[0] - 1 - x, x] == 1:
                break
            else:
                wb[bb.shape[0] - 1 - x, x + offset] = 0

    for offset in range(bb.shape[1] - bb.shape[0]):  # br->tl
        for x in range(bb.shape[0]):
            if bb[bb.shape[0] - x - 1, x] == 1:
                break
            else:
                wb[bb.shape[0] - 1 - x, bb.shape[1] - 1 - x - offset] = 0
    return wb


def treshold_brightness(x, thresh):
    x[x < thresh] = 0
    x[x >= thresh] = 1
    return x


def find_corners(b, margin=40, dimensions=[350, 300]):
    # mirror board
    bM = np.fliplr(b)
    topleft = [b.shape[0], b.shape[1]]
    for x in range(b.shape[0]):
        for y in range(b.shape[1]):
            if b[x, y] > 0 and x + y < topleft[0] + topleft[1]:
                topleft[0] = x
                topleft[1] = y

    botright = [0, 0]
    for x in range(b.shape[0]):
        for y in range(b.shape[1]):
            if b[x, y] > 0 and x + y > botright[0] + botright[1]:
                botright[0] = x
                botright[1] = y

    # bottom-left and top-right can be obtained in the same way from the board which is mirrored on the y-axis:
    topright = [bM.shape[0], bM.shape[1]]
    for x in range(bM.shape[0]):
        for y in range(bM.shape[1]):
            if bM[x, y] > 0 and x + y < topright[0] + topright[1]:
                topright[0] = x
                topright[1] = y
    # invert mirroring
    topright[1] = bM.shape[1] - topright[1]

    botleft = [0, 0]
    for x in range(bM.shape[0]):
        for y in range(bM.shape[1]):
            if bM[x, y] > 0 and x + y > botleft[0] + botleft[1]:
                botleft[0] = x
                botleft[1] = y
    # invert mirroring
    botleft[1] = bM.shape[1] - botleft[1]

    src = np.array([[margin, margin], [margin, dimensions[1]], [dimensions[0], dimensions[1]], [dimensions[0], margin]])
    dst = np.array([topleft[::-1], botleft[::-1], botright[::-1], topright[::-1]])
    return src, dst, dimensions, margin


def generate_discrete_boardconfig(b1, b2, num_cols, num_rows, treshold=1e-3):
    # set up grid to extract colour information
    grid = np.array([[50 * (i + 1), 50 * (num_rows - j)] for i in range(num_cols) for j in range(num_rows)])
    grid = np.fliplr(grid)

    # extract colour information
    discrete_board = np.ones([num_rows, num_cols])
    is_type1 = np.reshape(b1[grid[:, 0], grid[:, 1]] > treshold, (6, 7), order='F')
    is_type2 = np.reshape(b2[grid[:, 0], grid[:, 1]] > treshold, (6, 7), order='F')

    discrete_board[is_type1] = 0
    discrete_board[is_type2] = 2
    return discrete_board


def is_valid(boardconfig):
    for row in range(6 - 1):
        for col in range(7):
            if boardconfig[row, col] == 1 and boardconfig[row + 1, col] != 1:
                return False
    return True


# def convert_board_datastructure(b, in1, in2):
#     return bC


def process_board(image_path):
    global start_time
    board = read_file(image_path)

    # keep track of script runtime for different tasks
    start_time = datetime.now()

    board_yellow = extract_color(board, yellow, tolerance)
    board_red = extract_color(board, red, tolerance, board_yellow)
    board_white = extract_color(board, white, tolerance, board_yellow + board_red)
    board_blue = extract_color(board, blue, tolerance / 2, board_yellow + board_red + board_white)

    board_white = clean_white_board(board_blue, board_white)

    # add Gaussian smoothing
    board_yellow = gaussian(board_yellow, sigma=2)
    board_red = gaussian(board_red, sigma=2)
    board_white = gaussian(board_white, sigma=2)

    # cut away low values below 95% brightness (i.e. increase contrast)
    board_total = board_yellow + board_red + board_white
    thresh = np.percentile(board_total, 95)

    for board_current in [board_yellow, board_red, board_white]:
        board_current = treshold_brightness(board_current, thresh)

    board_total = board_yellow + board_red + board_white
    # board_mirrored = np.fliplr(board_total)

    src, dst, dimensions, margin = find_corners(board_total)

    transf = tf.ProjectiveTransform()
    transf.estimate(src, dst)
    board_total_warped = tf.warp(board_total, transf, output_shape=(dimensions[1] + margin, dimensions[0] + margin))
    board_yellow_warped = tf.warp(board_yellow, transf, output_shape=(dimensions[1] + margin, dimensions[0] + margin))
    board_red_warped = tf.warp(board_red, transf, output_shape=(dimensions[1] + margin, dimensions[0] + margin))

    # add an extra Gaussian smoothing in case the spots don't end up perfectly on the grid points:
    board_yellow_warped_smoothed = gaussian(board_yellow_warped, sigma=4)
    board_red_warped_smoothed = gaussian(board_red_warped, sigma=4)

    boardconfig = generate_discrete_boardconfig(board_yellow_warped_smoothed, board_red_warped_smoothed, BOARD_COLS, BOARD_ROWS)

    if is_valid(boardconfig):
        print_runtime(descr="Total Run")
        print("Valid game configuration extracted:")
        # # draw board and original photograph for comparison:
        # colormap = ListedColormap(['gold', 'white', 'red'])
        # fig = plt.figure(figsize=(17, 5))
        # plt1 = fig.add_subplot(121)
        # plt1.imshow(board)
        # plt2 = fig.add_subplot(122)
        # plt2.matshow(boardconfig, cmap=colormap)
        # plt.gca().invert_yaxis()
        # plt.show()
        return boardconfig
    else:
        print("No valid game board could be extracted. Please retake photograph.")
        return None


# reference colours
yellow = [193, 155, 17]
red = [156, 10, 25]
white = [150, 148, 138]
blue = [12, 11, 69]
tolerance = 50

# board configuration
BOARD_COLS = 7
BOARD_ROWS = 6

image_folder = '../board_images/board'
image_type = '.jpg'

for i in range(19):
    pathToImage = image_folder + str(i + 1) + image_type
    print(pathToImage)
    board_config = process_board(pathToImage)