import json
import multiprocessing
import os
from typing import Literal, Union

import numpy as np
import tqdm

from image_similarity_metrics.image_similarity_metrics import Metric, PSNR, SSIM, batch_psnr_numpy_adapter, batch_ssim_numpy_adapter
from redemosaic.redemosaic_directory import get_all_images_in_directory, get_bayer_patterns
from utils import imreadRGB, imwriteRGB
from redemosaic.mht_redemosaic_opencv_implementation import mosaic, demosaic

metrics = [PSNR, SSIM]

def read_redemosaiced_img(img_filename: str, pattern: str, redemosaiced_img_dir: str):
    '''Given a name of the original image file, and a Bayer pattern, read the redemosaiced image from the folder of redemosaiced_img_dir.
    The read image is given in numpy array. The pattern parameter must be one of ["rggb", "bggr", "grbg", "gbrg"].
    Expects the redemosaiced image in redemosaiced_img_dir named by the get_demosaiced_img_filename function.
    '''

    split_img_filename = img_filename.split(".")
    assert len(split_img_filename) == 2, "Invalid image file name: " + img_filename
    img_filename_no_postfix = split_img_filename[0]
    img_filename_postfix = split_img_filename[1]
    redemosaiced_img_path = os.path.join(redemosaiced_img_dir, f"{img_filename_no_postfix}_redemosaiced_{pattern}.{img_filename_postfix}")
    demosaiced_img = imreadRGB(redemosaiced_img_path)
    return demosaiced_img


def single_img_single_pattern_pipeline(
        rgb_img: np.ndarray,
        pattern: Literal["rggb", "bggr", "grbg", "gbrg"],
        redemosaiced_img_dir: Union[str, bytes, os.PathLike],
        img_filename: str = None,
) -> dict:
    '''
    For a single image and a specific Bayer pattern, compare the average PSNR and SSIM difference between the original image and the redemosaiced image.
    The dict returned is something like this {"PSNR":1.0,"SSIM":2.0}.
    Will try to look for a precomputed redemosaiced image in the folder of redemosaiced_img_dir if this argument is given. img_filename is used to get the redemosaiced image name based on the original image name.
    '''
    assert pattern in get_bayer_patterns(), 'The pattern parameter must be one of ["rggb", "bggr", "grbg", "gbrg"].'
    if redemosaiced_img_dir is not None and img_filename is not None:
        demosaiced_img = read_redemosaiced_img(img_filename=img_filename, pattern=pattern, redemosaiced_img_dir=redemosaiced_img_dir)
    else:
        mosaiced_img = mosaic(rgb_img, pattern)
        demosaiced_img = demosaic(mosaiced_img, pattern)
    return {
        metric.metric_name(): metric.evaluate(rgb_img, demosaiced_img) for metric in metrics
    }


def compile_single_img_patterns_results(patterns_results: dict, metrics: [Metric]) -> dict:
    '''
    pattern_results should be a dictionary like {"rggb": {"DeltaE2000": 0.49998870491981506}, "bggr": {"DeltaE2000": 0.4919033646583557}, "grbg": {"DeltaE2000": 0.6200681328773499}, "gbrg": {"DeltaE2000": 0.622094988822937}}
    This function picks the best comparison results and the corresponding Bayer pattern from four redemosaiced images' comparison results.
    metrics specifies the image comparison metrics whose results we should try to compile. Each item in the list should be a class object extending the Metric abstract base class.
    The returned dictionary has two fields: all_results and best_results. all_results is just the passed-in patterns_results. best_results compiles the best results and their corresponding Bayer pattern used for redemosaicing.
    The best_results dict should be like {"DeltaE2000": {"best_result": 0.4919033646583557, "best_pattern": "bggr"}} where the keys are the metric names in the metrics argument.
    '''
    patterns = get_bayer_patterns()
    best_results = {
        metric.metric_name():
            {
                "best_result": float("-inf") if metric.higher_is_better() else float("inf"), # initialize default values which must be replaced later.
                "best_pattern": "",
            }
        for metric in metrics
    }
    all_results = {pattern: {} for pattern in patterns}

    is_first_pattern = True

    for pattern in patterns:
        result = patterns_results[pattern]
        for metric in metrics:
            metric_result = result[metric.metric_name()]
            all_results[pattern][metric.metric_name()] = metric_result

            if (is_first_pattern):
                best_results[metric.metric_name()] = {
                        "best_result": metric_result,
                        "best_pattern": pattern
                    }
                
                is_first_pattern = False

                continue

            if metric.higher_is_better():
                if best_results[metric.metric_name()]["best_result"] < metric_result:
                    best_results[metric.metric_name()] = {
                        "best_result": metric_result,
                        "best_pattern": pattern
                    }
            else:
                if best_results[metric.metric_name()]["best_result"] > metric_result:
                    best_results[metric.metric_name()] = {
                        "best_result": metric_result,
                        "best_pattern": pattern
                    }
    return {
        "all_results": all_results,
        "best_results": best_results
    }


def single_img_all_patterns_pipeline(
        rgb_img: np.ndarray,
        redemosaiced_img_dir: Union[str, bytes, os.PathLike],
        img_filename: str = None,
) -> dict:
    '''
    for an rgb image, calculate the PSNR and SSIM difference between the original image and the four redemosaiced images.
    the original image's file name should be given so we can look for its redemosaiced images in redemosaiced_img_dir.
    the returned dictionary should look like {
      "rggb": {
        "PSNR": 36.539815234715675,
        "SSIM": 0.9309087918152702
      },
      "bggr": {
        "PSNR": 36.571228781977034,
        "SSIM": 0.9309760606619485
      },
      "grbg": {
        "PSNR": 40.87411067611139,
        "SSIM": 0.9744703795550419
      },
      "gbrg": {
        "PSNR": 40.73910791589714,
        "SSIM": 0.9733848375710222
      }
    }
    '''
    patterns = get_bayer_patterns()
    patterns_results = {
        pattern: single_img_single_pattern_pipeline(
            rgb_img,
            pattern,
            redemosaiced_img_dir=redemosaiced_img_dir,
            img_filename=img_filename
        ) for pattern in patterns
    }
    return compile_single_img_patterns_results(patterns_results, metrics)


def compute_for_single_img(args):
    '''wrapper function for parallel multiprocessing.'''
    img_file_info, redemosaiced_img_directory = args # each args should be a ((image_filename, image_full_path), redemosaiced_image_directory) tuple.
    img_filename, img_path = img_file_info
    try:
        img = imreadRGB(img_path)
    except Exception as e:
        print(f"Failed to load image {img_filename}: {e}")
        return img_filename, {}
    img_result = single_img_all_patterns_pipeline(img, redemosaiced_img_dir=redemosaiced_img_directory, img_filename=img_filename)
    return img_filename, img_result