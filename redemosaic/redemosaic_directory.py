import multiprocessing
import os
from typing import Union, Callable, Literal

import cv2
import numpy as np
from tqdm import tqdm

from redemosaic.batch_redemosaic_bilinear import batch_redemosaic_bilinear_numpy_adapter
from redemosaic.mht_redemosaic_opencv_implementation import demosaic, mosaic
from redemosaic.mht_redemosaic_pytorch_implementation import batch_redemosaic_malvar_numpy_adapter
from redemosaic.opencv_demosaic_algorithms import vng_demosaic, ea_demosaic
from utils import imreadRGB, imwriteRGB, get_all_images_in_directory


def get_bayer_patterns():
    return ["rggb", "bggr", "grbg", "gbrg"]


def split_filename_and_read_img(img_filename: str,
                                img_full_path: str, ):
    '''
    image_filename is only the filename of an image. img_full_path is the path to the folder storing the image plus the filename.
    This function returns the numpy image read from the location specified by img_full_path, the image filename without file type postfix, and the image filename postfix specifying the image type.
    We split the file name so it is easier to compute the file name of the redemosaiced image.
    '''
    split_img_filename = img_filename.split(".")
    assert len(split_img_filename) == 2, "Invalid image file name: " + img_filename
    img_filename_no_postfix = split_img_filename[0]
    img_filename_postfix = split_img_filename[1]
    return imreadRGB(img_full_path), img_filename_no_postfix, img_filename_postfix


def get_demosaiced_img_filename(img_filename_no_postfix, img_filename_postfix, pattern):
    'Given the parts of the original image name, calculate the name for a redemosaiced version of the image.'
    return f"{img_filename_no_postfix}_redemosaiced_{pattern}.{img_filename_postfix}"


def write_redemosaiced_images(
        img_filename: str,
        img_full_path: str,
        dst_directory_path: Union[str, bytes, os.PathLike],
        demosaic_func,
):
    # Read an image specified by img_full_path; redemosaic it with four Bayer patterns and store
    # the redemosaic results to the folder of dst_directory_path.
    assert os.path.exists(dst_directory_path) and os.path.isdir(dst_directory_path), "The dst_directory_path is invalid."

    try:
        img, img_filename_no_postfix, img_filename_postfix = split_filename_and_read_img(img_filename, img_full_path)
    except Exception as e:
        print(f"Failed to load image {img_filename}: {e}")
        return

    for pattern in get_bayer_patterns():
        cfaimg = mosaic(img, pattern)
        redemosaiced_img = demosaic_func(cfaimg, pattern)
        imwriteRGB(
            os.path.join(dst_directory_path, get_demosaiced_img_filename(img_filename_no_postfix, img_filename_postfix, pattern)),
            redemosaiced_img
        )


def write_redemosaiced_img_wrapper(args):
    '''Wrapper function for multi-processing.
    args should be a tuple containing (img_filename, img_file_fullpath, result_storing_directory_path, demosaic_func)
    '''
    filename, filepath, dst_directory_path, demosaic_func = args
    write_redemosaiced_images(filename, filepath, dst_directory_path, demosaic_func)