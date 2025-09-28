import numpy
import torch

'''
Our implementation of the bilinear demosaicing algorithm is heavily based on
the [`demosaicing_CFA_Bayer_bilinear`](https://github.com/colour-science/colour-demosaicing/blob/develop/colour_demosaicing/bayer/demosaicing/bilinear.py)
function in the open-source project [colour-demosaicing](https://github.com/colour-science/colour-demosaicing). 
We made PyTorch-based implementations based on the implementations in the colour-demosaicing library,
so we can use PyTorch for batch processing. The PyTorch adaptation works well in batch processing.
'''
def redemosaic_bilinear(
        rgbimg: torch.Tensor,
        bayer_patterns
) -> torch.Tensor:
    """
    The function creates redemosaiced images of B Bayer patterns given an RGB image. This function uses bilinear demosaicing algorithm.
    Each pattern in bayer_patterns should be one "rggb", "bggr", "grbg", "gbrg".
    The demosaicing results of different Bayer patterns are stacked together and only on tensor is returned. The stacking order is the same of as the Bayer patterns given in bayer_patterns.
    Input: (H, W, 3)
    Return: (B, H, W, 3)
    """
    assert isinstance(rgbimg, torch.Tensor)
    for bayer_pattern in bayer_patterns:
        assert bayer_pattern in ["gbrg", "grbg", "bggr", "rggb"]
    device = rgbimg.device
    H, W, _ = rgbimg.size()

    H_G = torch.tensor([
        [0, 1, 0],
        [1, 4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=device) / 4

    H_RB = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=torch.float32, device=device) / 4

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

    del basic_masks, rgbmasks_bayerpatterns

    rgb_bayerpatterns[:, 0, :, :] = torch.conv2d(torch.nn.ReflectionPad2d(1)(rgb_bayerpatterns[:, 0, :, :]).unsqueeze(1),
                                                 H_RB[None, None, ...]).squeeze(1)
    rgb_bayerpatterns[:, 1, :, :] = torch.conv2d(torch.nn.ReflectionPad2d(1)(rgb_bayerpatterns[:, 1, :, :]).unsqueeze(1),
                                                 H_G[None, None, ...]).squeeze(1)
    rgb_bayerpatterns[:, 2, :, :] = torch.conv2d(torch.nn.ReflectionPad2d(1)(rgb_bayerpatterns[:, 2, :, :]).unsqueeze(1),
                                                 H_RB[None, None, ...]).squeeze(1)

    return torch.stack([torch.clamp(rgb_bayerpatterns[:, 0, :, :], 0, 255),
                        torch.clamp(rgb_bayerpatterns[:, 1, :, :], 0, 255),
                        torch.clamp(rgb_bayerpatterns[:, 2, :, :], 0, 255)], 3).type(torch.uint8)


def batch_redemosaic_bilinear_numpy_adapter(
        rgbimg: numpy.ndarray,
        bayer_patterns
) -> numpy.ndarray:
    '''
    This is an adapter to convert numpy array image to pytorch tensors and pass it to the redemosaic_bilinear function.
     The rgbimg should be of shape (H, W, 3) with channels in the RGB order.
     It also converts the result of demosaicing back to numpy array and return it.
     Each pattern in bayer_patterns should be one "rggb", "bggr", "grbg", "gbrg".
     The demosaicing results of different Bayer patterns are stacked together and only on numpy array of shape (B, H, W, 3) is returned. The stacking order is the same of as the Bayer patterns given in bayer_patterns.
     This function should be used as a parameter to the redemosaic_directory function in redemosaic_directory.py. See the comments there for more details.
     '''
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
    else "cpu")
    rgbimg_tensor = torch.from_numpy(rgbimg).to(device)
    results = redemosaic_bilinear(rgbimg=rgbimg_tensor, bayer_patterns=bayer_patterns)
    return results.cpu().detach().numpy()
