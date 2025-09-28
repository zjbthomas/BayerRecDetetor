import numpy as np
import cv2 as cv
import time
'''
Our implementation code of the demosaicing algorithm is heavily based on
the [`demosaicing_CFA_Bayer_Malvar2004`](https://github.com/colour-science/colour-demosaicing/blob/develop/colour_demosaicing/bayer/demosaicing/malvar2004.py)
function in the open-source project [colour-demosaicing](https://github.com/colour-science/colour-demosaicing).
We made some changes to the code in this open-source project to make it work on values in integers in the range of [0-255] and to fix compatibility
issues. The demosaicing algorithm is proposed by H.S. Malvar, Li-wei He, and R. Cutler [1].
'''

def bayer_mask(shape, pattern):
    # return RGB mask given pattern, taken from colour_demosaicing
    # init dict object channels contain "r", "g", and "b" with same input shape
    channels = {channel: np.zeros((shape[0], shape[1]), dtype="bool") for channel in "rgb"}

    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        # not a good implementation, but channels[channel][y::2][x::2]=1 not work
        for col in channels[channel][y::2]:
            col[x::2] = 1

    return tuple(channels.values())


def mosaic(rgbimg, pattern):
    # mosaic transforms an RGB image to a cfa image following a specific bayer pattern.
    # rgb should be a 2D numpy array representing an image with three channel. the pattern should be one "rggb", "bggr", "grbg", "gbrg".
    R_m, G_m, B_m = bayer_mask(rgbimg.shape, pattern)

    return rgbimg[:, :, 0] * R_m + rgbimg[:, :, 1] * G_m + rgbimg[:, :, 2] * B_m


def demosaic(cfaimg, pattern):
    # Given a cfa image, demosaic it using the Malvar et al. demosaicing algorithm and return the resulting RGB image.
    # cfaimage should be a 2D numpy array representing an image with one channel. the pattern should be one "rggb", "bggr", "grbg", "gbrg".
    R_m, G_m, B_m = bayer_mask(cfaimg.shape, pattern)
    cfaimg = cfaimg.astype(np.float32)

    GR_GB = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0],
        ], np.float32
    ) / 8

    Rg_RB_Bg_BR = np.array(
        [
            [0, 0, 0.5, 0, 0],
            [0, -1, 0, -1, 0],
            [-1, 4, 5, 4, -1],
            [0, -1, 0, -1, 0],
            [0, 0, 0.5, 0, 0],
        ], np.float32
    ) / 8

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = np.array(
        [
            [0, 0, -1.5, 0, 0],
            [0, 2, 0, 2, 0],
            [-1.5, 0, 6, 0, -1.5],
            [0, 2, 0, 2, 0],
            [0, 0, -1.5, 0, 0],
        ], np.float32
    ) / 8

    R = cfaimg * R_m
    G = cfaimg * G_m
    B = cfaimg * B_m

    # free up RAM
    del G_m

    # calculate bilinear G value at all R and B locations
    G = np.where(np.logical_or(R_m == 1, B_m == 1), cv.filter2D(cfaimg, -1, GR_GB), G)

    RBg_RBBR = cv.filter2D(cfaimg, -1, Rg_RB_Bg_BR)
    RBg_BRRB = cv.filter2D(cfaimg, -1, Rg_BR_Bg_RB)
    RBgr_BBRR = cv.filter2D(cfaimg, -1, Rb_BB_Br_RR)

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
    # Red columns.
    R_c = np.any(R_m == 1, axis=0)[None] * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)
    # Blue columns
    B_c = np.any(B_m == 1, axis=0)[None] * np.ones(B.shape)

    del R_m, B_m

    R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

    # remove values out of bonds (0-255) and use BGR arrangement for imwrite
    return np.stack((np.clip(R, 0, 255), np.clip(G, 0, 255), np.clip(B, 0, 255)), axis=2, dtype=np.uint8, casting="unsafe")



if __name__ == "__main__":
    start = time.time()
    print("opencv implementation")
    # TODO iterate sensor alignment
    pattern = "bggr"

    # convert opencv default BGR arrangement to RGB
    rgbimg = cv.imread("../resources/images/1.tif")[:, :, [2, 1, 0]]
    print("imread\t {:.3f}s".format(time.time() - start))

    cfaimg = mosaic(rgbimg, pattern)
    print("mosaic\t {:.3f}s".format(time.time() - start))

    # newimg in BGR arrangement for imwrite
    newimg = demosaic(cfaimg, pattern)
    print("demosaic {:.3f}s".format(time.time() - start))

    cv.imwrite("new-cv.tif", newimg)
    print("imwrite\t {:.3f}s".format(time.time() - start))
