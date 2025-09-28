import cv2
import numpy as np

# The two functions here should be used as parameters for the function redemosaic_directory in redemosaic_directory.py. See the comment of that function for more details.
def vng_demosaic(cfa_img: np.ndarray, pattern: str):
    'Demosaic a given cfa image with the VNG demosaicing algorithm, using the Bayer pattern specified.'
    codes = {
        "rggb": cv2.COLOR_BayerBG2RGB_VNG,
        "grbg": cv2.COLOR_BayerGB2RGB_VNG,
        "bggr": cv2.COLOR_BayerRG2RGB_VNG,
        "gbrg": cv2.COLOR_BayerGR2RGB_VNG,
    }
    return cv2.cvtColor(src=cfa_img, code=codes[pattern])


def ea_demosaic(cfa_img: np.ndarray, pattern: str):
    'Demosaic a given cfa image with the EA demosaicing algorithm, using the Bayer pattern specified.'
    codes = {
        "rggb": cv2.COLOR_BayerBG2RGB_EA,
        "grbg": cv2.COLOR_BayerGB2RGB_EA,
        "bggr": cv2.COLOR_BayerRG2RGB_EA,
        "gbrg": cv2.COLOR_BayerGR2RGB_EA,
    }
    return cv2.cvtColor(src=cfa_img, code=codes[pattern])
