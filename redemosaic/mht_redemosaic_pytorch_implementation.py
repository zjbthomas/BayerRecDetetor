import cv2 as cv
import numpy
import torch
import time
'''
Our implementation code of the demosaicing algorithm is heavily based on
the [`demosaicing_CFA_Bayer_Malvar2004`](https://github.com/colour-science/colour-demosaicing/blob/develop/colour_demosaicing/bayer/demosaicing/malvar2004.py)
function in the open-source project [colour-demosaicing](https://github.com/colour-science/colour-demosaicing).
We made some changes to the code in this open-source project to make it work on values in integers in the range of [0-255] and to fix compatibility
issues. The demosaicing algorithm is proposed by H.S. Malvar, Li-wei He, and R. Cutler [1].
We made PyTorch-based implementations based on the implementations in the colour-demosaicing library,
so we can use PyTorch for batch processing. The PyTorch adaptation works well in batch processing.
'''

def bayer_mask(size, pattern):
    # return RGB mask given pattern, taken from colour_demosaicing
    # init dict object channels contain "r", "g", and "b" with same input shape
    channels = {channel: torch.zeros([size[0], size[1]], dtype=torch.bool, device=device) for channel in "rgb"}

    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        # not a good implementation, but channels[channel][y::2][x::2]=1 not work
        for col in channels[channel][y::2]:
            col[x::2] = 1

    return tuple(channels.values())


def mosaic(rgbimg, pattern):
    # mosaic transforms an RGB image to a cfa image following a specific bayer pattern.
    # rgb should be a 2D PyTorch tensor with 3 channels representing an image with three channel. the pattern should be one "rggb", "bggr", "grbg", "gbrg".
    R_m, G_m, B_m = bayer_mask(rgbimg.size(), pattern)

    return rgbimg[:, :, 0] * R_m + rgbimg[:, :, 1] * G_m + rgbimg[:, :, 2] * B_m


def demosaic(cfaimg, pattern):
    # Given a cfa image, demosaic it using the Malvar et al. demosaicing algorithm and return the resulting RGB image.
    # cfaimage should be a 2D PyTorch tensor representing an image with one channel.The pattern should be one "rggb", "bggr", "grbg", "gbrg".
    R_m, G_m, B_m = bayer_mask(cfaimg.size(), pattern)

    # [None, None, ...] equivalent to .unsqueeze(0).unsqueeze(0)
    cfaimg = cfaimg.float()[None, None, ...]

    GR_GB = torch.tensor(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0],
        ], dtype=torch.float32, device=device
    ) / 8

    Rg_RB_Bg_BR = torch.tensor(
        [
            [0, 0, 0.5, 0, 0],
            [0, -1, 0, -1, 0],
            [-1, 4, 5, 4, -1],
            [0, -1, 0, -1, 0],
            [0, 0, 0.5, 0, 0],
        ], dtype=torch.float32, device=device
    ) / 8

    Rg_BR_Bg_RB = torch.t(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = torch.tensor(
        [
            [0, 0, -1.5, 0, 0],
            [0, 2, 0, 2, 0],
            [-1.5, 0, 6, 0, -1.5],
            [0, 2, 0, 2, 0],
            [0, 0, -1.5, 0, 0],
        ], dtype=torch.float32, device=device
    ) / 8

    R = cfaimg * R_m[None, None, ...]
    G = cfaimg * G_m[None, None, ...]
    B = cfaimg * B_m[None, None, ...]

    # free up (V)RAM
    del G_m

    # calculate bilinear G value at all R and B locations
    G = torch.where(torch.logical_or(R_m == 1, B_m == 1), torch.conv2d(torch.nn.ReflectionPad2d(2)(cfaimg), GR_GB[None, None, ...]), G)

    RBg_RBBR = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfaimg), Rg_RB_Bg_BR[None, None, ...])
    RBg_BRRB = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfaimg), Rg_BR_Bg_RB[None, None, ...])
    RBgr_BBRR = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfaimg), Rb_BB_Br_RR[None, None, ...])

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR

    # Red rows.
    R_r = torch.t(torch.any(R_m == 1, axis=1)[None]) * torch.ones(R.size(), device=device)
    # Red columns.
    R_c = torch.any(R_m == 1, axis=0)[None] * torch.ones(R.size(), device=device)
    # Blue rows.
    B_r = torch.t(torch.any(B_m == 1, axis=1)[None]) * torch.ones(B.size(), device=device)
    # Blue columns
    B_c = torch.any(B_m == 1, axis=0)[None] * torch.ones(B.size(), device=device)

    del R_m, B_m

    R = torch.where(torch.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = torch.where(torch.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = torch.where(torch.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = torch.where(torch.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = torch.where(torch.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = torch.where(torch.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

    # remove values out of bonds (0-255), and assembly BGR arrangement for opencv imwrite
    return torch.stack([torch.clamp(B.squeeze([0, 1]), 0, 255),
                        torch.clamp(G.squeeze([0, 1]), 0, 255),
                        torch.clamp(R.squeeze([0, 1]), 0, 255)], 2).type(torch.uint8)


def batch_redemosaic_malvar(
        rgbimg: torch.Tensor,
        bayer_patterns
) -> torch.Tensor:
    """
    The function creates redemosaiced images of B Bayer patterns given a rgb image. This function uses the Malvar et al. demosaicing algorithm.
    Each pattern in bayer_patterns should be one "rggb", "bggr", "grbg", "gbrg".
    The demosaicing results of different Bayer patterns are stacked together and only one tensor is returned. The stacking order is the same of as the Bayer patterns given in bayer_patterns.
    Input: (H, W, 3) with channels in the RGB order.
    Return: (B, H, W, 3)
    """
    assert isinstance(rgbimg, torch.Tensor)
    #for bayer_pattern in bayer_patterns:
    #    assert bayer_pattern in ["gbrg", "grbg", "bggr", "rggb"]
    device = rgbimg.device
    H, W, _ = rgbimg.size()

    GR_GB = torch.tensor([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0]
    ], dtype=torch.float32, device=device) / 8

    Rg_RB_Bg_BR = torch.tensor([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0]
    ], dtype=torch.float32, device=device) / 8

    Rg_BR_Bg_RB = torch.t(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = torch.tensor([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0]
    ], dtype=torch.float32, device=device) / 8

    basic_masks = torch.tensor([
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]]
    ], dtype=torch.bool, device=device).repeat(1, (H + 1) // 2, (W + 1) // 2)[:, :-1 if H % 2 == 1 else None, :-1 if W % 2 == 1 else None]

    rgbmasks_bayerpatterns = torch.zeros([len(bayer_patterns), 3, H, W], dtype=torch.bool, device=device)
    rgb_bayerpatterns = torch.zeros_like(rgbmasks_bayerpatterns, dtype=torch.float32, device=device)
    for rgbmasks, rgb, bayer_pattern in zip(rgbmasks_bayerpatterns, rgb_bayerpatterns, bayer_patterns):
        rgbmasks[0] = basic_masks[bayer_pattern.find("r")]
        rgb[0] = rgbimg[:, :, 0] * rgbmasks[0]
        rgbmasks[1] = (basic_masks[bayer_pattern.find("g")] + basic_masks[bayer_pattern.rfind("g")])
        rgb[1] = rgbimg[:, :, 1] * rgbmasks[1]
        rgbmasks[2] = basic_masks[bayer_pattern.find("b")]
        rgb[2] = rgbimg[:, :, 2] * rgbmasks[2]
    cfa_bayerpatterns = torch.sum(rgb_bayerpatterns, dim=1, keepdim=True)

    del basic_masks

    rgb_bayerpatterns[:, 1, :, :] = torch.where(
        torch.logical_or(rgbmasks_bayerpatterns[:, 0, :, :] == 1, rgbmasks_bayerpatterns[:, 2, :, :] == 1),
        torch.conv2d(torch.nn.ReflectionPad2d(2)(cfa_bayerpatterns), GR_GB[None, None, ...]).squeeze(1),
        rgb_bayerpatterns[:, 1, :, :]
    )

    RBg_RBBR = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfa_bayerpatterns), Rg_RB_Bg_BR[None, None, ...]).squeeze(1)
    RBg_BRRB = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfa_bayerpatterns), Rg_BR_Bg_RB[None, None, ...]).squeeze(1)
    RBgr_BBRR = torch.conv2d(torch.nn.ReflectionPad2d(2)(cfa_bayerpatterns), Rb_BB_Br_RR[None, None, ...]).squeeze(1)

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR, cfa_bayerpatterns

    R_row = torch.any(rgbmasks_bayerpatterns[:, 0, :, :] == 1, axis=2).unsqueeze(2).expand(-1, -1, W)
    R_col = torch.any(rgbmasks_bayerpatterns[:, 0, :, :] == 1, axis=1).unsqueeze(1).expand(-1, H, -1)

    B_row = torch.any(rgbmasks_bayerpatterns[:, 2, :, :] == 1, axis=2).unsqueeze(2).expand(-1, -1, W)
    B_col = torch.any(rgbmasks_bayerpatterns[:, 2, :, :] == 1, axis=1).unsqueeze(1).expand(-1, H, -1)

    rgb_bayerpatterns[:, 0, :, :] = torch.where(torch.logical_and(R_row == 1, B_col == 1), RBg_RBBR, rgb_bayerpatterns[:, 0, :, :])
    rgb_bayerpatterns[:, 0, :, :] = torch.where(torch.logical_and(B_row == 1, R_col == 1), RBg_BRRB, rgb_bayerpatterns[:, 0, :, :])

    rgb_bayerpatterns[:, 2, :, :] = torch.where(torch.logical_and(B_row == 1, R_col == 1), RBg_RBBR, rgb_bayerpatterns[:, 2, :, :])
    rgb_bayerpatterns[:, 2, :, :] = torch.where(torch.logical_and(R_row == 1, B_col == 1), RBg_BRRB, rgb_bayerpatterns[:, 2, :, :])

    rgb_bayerpatterns[:, 0, :, :] = torch.where(torch.logical_and(B_row == 1, B_col == 1), RBgr_BBRR, rgb_bayerpatterns[:, 0, :, :])
    rgb_bayerpatterns[:, 2, :, :] = torch.where(torch.logical_and(R_row == 1, R_col == 1), RBgr_BBRR, rgb_bayerpatterns[:, 2, :, :])

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_row, R_col, B_row, B_col, rgbmasks_bayerpatterns

    return torch.stack([torch.clamp(rgb_bayerpatterns[:, 0, :, :], 0, 255),
                        torch.clamp(rgb_bayerpatterns[:, 1, :, :], 0, 255),
                        torch.clamp(rgb_bayerpatterns[:, 2, :, :], 0, 255)], 3).type(torch.uint8)


def batch_redemosaic_malvar_numpy_adapter(
        rgbimg: numpy.ndarray,
        bayer_patterns,
        cuda_id
) -> numpy.ndarray:
    '''
    This is an adapter to convert numpy array image to pytorch tensors and pass it to the batch_redemosaic_malvar function.
    The rgbimg should be of shape (H, W, 3) with channels in the RGB order.
     It also converts the result of demosaicing back to numpy array and return it.
     Each pattern in bayer_patterns should be one "rggb", "bggr", "grbg", "gbrg".
     The demosaicing results of different Bayer patterns are stacked together and only on numpy array of shape (B, H, W, 3) is returned. The stacking order is the same of as the Bayer patterns given in bayer_patterns.
     This function should be a parameter of the redemosaic_directory function is redemosaic_directory.py. See comments there for more details.
    '''
    device = torch.device("cuda:" + str(cuda_id) if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
    else "cpu")
    rgbimg_tensor = torch.from_numpy(rgbimg).to(device)
    results = batch_redemosaic_malvar(rgbimg=rgbimg_tensor, bayer_patterns=bayer_patterns)
    return results.cpu().detach().numpy()


if __name__ == "__main__":
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
    else "cpu")
    print(f"torch use {device}")
    # TODO iterate sensor alignment
    pattern = "bggr"

    # torch can't read or write tif, use opencv helper
    # convert opencv default BGR arrangement to RGB
    rgbimg = torch.tensor(cv.imread("../resources/images/1.tif")[:, :, [2, 1, 0]], device=device)
    print("imread\t {:.3f}s".format(time.time() - start))

    cfaimg = mosaic(rgbimg, pattern)
    print("mosaic\t {:.3f}s".format(time.time() - start))

    # newimg in BGR arrangement for imwrite
    newimg = demosaic(cfaimg, pattern)
    print("demosaic {:.3f}s".format(time.time() - start))

    cv.imwrite("new-torch.tif", newimg.cpu().detach().numpy())
    print("imwrite\t {:.3f}s".format(time.time() - start))
