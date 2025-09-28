import json
import multiprocessing
import os
from typing import Union, Literal
import math

import cv2
import numpy as np
import skimage.color
from tqdm import tqdm

from image_similarity_metrics.image_similarity_metrics import Metric
from image_similarity_metrics.psnr_ssim_all_img_directory import compile_single_img_patterns_results, read_redemosaiced_img
from redemosaic.mht_redemosaic_opencv_implementation import mosaic, demosaic
from redemosaic.redemosaic_directory import get_bayer_patterns
from utils import imreadRGB, get_all_images_in_directory
import torch

'''
This image comparison metric (Delta E) implementations are based on the implementations 
(https://github.com/scikit-image/scikit-image/blob/v0.23.1/skimage/color/delta_e.py#L152-L283) in scikit-image[3] library. Again the modifications are made, so 
we can use PyTorch for batch processing. The PyTorch adaptation works well in batch processing.
'''
def batch_deltae_2000(
        preds: torch.Tensor,
        target: torch.Tensor,
        kL: float = 1,
        kC: float = 1,
        kH: float = 1
) -> torch.Tensor:
    """
    The function calculates mean deltaE across the Lab of rgb images between B predictions and the target.
    This function is modified from the implementation of scikit-image:
    https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.deltaE_ciede2000
    https://github.com/scikit-image/scikit-image/blob/v0.23.1/skimage/color/delta_e.py#L152-L283
    Note: input only accepts rgb images, and rgb2lab conversion is done within the function.

    Input: preds(B, H, W, 3) and target(H, W, 3)

    Return: deltaE(B)
    """
    device = preds.device

    target = skimage.color.rgb2lab(target.cpu().numpy())

    results = []
    for p in preds:
        results.append(np.mean(skimage.color.deltaE_ciede2000(target, skimage.color.rgb2lab(p.cpu().numpy()), kL=1, kC=1, kH=1)))

    return torch.tensor(results, device = device)


def batch_deltae_adapter(
        preds: np.ndarray,
        target: np.ndarray,
) -> np.ndarray:
    '''
    Adapter function for batch_deltae_2000, so it works on numpy arrays. See the comments of batch_deltae_2000 for shape of input and output numpy arrays.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pt_preds = torch.from_numpy(preds).to(device)
    pt_target = torch.from_numpy(target).to(device)
    results = batch_deltae_2000(preds=pt_preds, target=pt_target)
    return results.cpu().detach().numpy().astype("float64")


class DeltaE2000(Metric):

    @staticmethod
    def evaluate(image1: np.ndarray, image2: np.ndarray) -> float:
        return delta_e_cie_2000(image1, image2)

    @staticmethod
    def metric_name() -> str:
        return "DeltaE2000"

    @staticmethod
    def higher_is_better() -> bool:
        return False


deltae_metric = [DeltaE2000]


def delta_e_cie_2000(rgbimg1, rgbimg2) -> float:
    # This code is based Sam Mason's Stack overflow answer
    # https://stackoverflow.com/a/57227800
    '''
    Returns the average Delta E 2000 difference of two images.
    Two images should be RGB images in numpy arrays.
    '''
    image_lab1 = skimage.color.rgb2lab(rgbimg1)
    image_lab2 = skimage.color.rgb2lab(rgbimg2)
    delta_e = skimage.color.deltaE_ciede2000(lab1=image_lab1, lab2=image_lab2, kL=1, kC=1, kH=1, channel_axis=-1)
    return np.mean(delta_e)


def single_img_single_pattern_deltaE2000_pipeline(
        rgb_img: np.ndarray,
        pattern: Literal["rggb", "bggr", "grbg", "gbrg"],
        redemosaiced_img_dir,
        img_filename: str = None,
) -> dict:
    '''
    For a single image and a specific Bayer pattern, compare the average Delta E difference between the original image and the redemosaiced image.
    The dict returned is something like this {"DeltaE2000":1.0}.
    Will try to look for a precomputed redemosaiced image in the folder of redemosaiced_img_dir if this argument is given. img_filename is used to get the redemosaiced image name based on the original image name.
    '''
    assert pattern in get_bayer_patterns(), 'The pattern parameter must be one of ["rggb", "bggr", "grbg", "gbrg"].'
    if redemosaiced_img_dir is not None and img_filename is not None:
        demosaiced_img = read_redemosaiced_img(img_filename=img_filename, pattern=pattern, redemosaiced_img_dir=redemosaiced_img_dir)
    else:
        mosaiced_img = mosaic(rgb_img, pattern)
        demosaiced_img = demosaic(mosaiced_img, pattern)
    return {
        metric.metric_name(): metric.evaluate(rgb_img, demosaiced_img) for metric in deltae_metric
    }


def single_img_deltaE2000_all_patterns_pipeline(
        rgb_img: np.ndarray,
        redemosaiced_img_dir,
        img_filename: str = None,
) -> dict:
    '''
    for an rgb image, calculate the Delta E average difference between the original image and the four redemosaiced images.
    the original image's file name should be given so we can look for its redemosaiced images in redemosaiced_img_dir.
    the returned dictionary should look like {"rggb": {"DeltaE2000": 0.6528810858726501}, "bggr": {"DeltaE2000": 0.6480197906494141}, "grbg": {"DeltaE2000": 0.5732150673866272}, "gbrg": {"DeltaE2000": 0.5659797787666321}}
    '''
    patterns = get_bayer_patterns()
    patterns_results = {
        pattern: single_img_single_pattern_deltaE2000_pipeline(
            rgb_img,
            pattern,
            redemosaiced_img_dir=redemosaiced_img_dir,
            img_filename=img_filename
        ) for pattern in patterns
    }
    return compile_single_img_patterns_results(patterns_results, metrics=deltae_metric)


def compute_deltaE2000_for_single_img(args):
    'A wrapper function for parallel multiprocessing.'
    img_file_info, redemosaiced_img_directory = args # each args should be a ((image_filename, image_full_path), redemosaiced_image_directory) tuple.
    img_filename, img_path = img_file_info
    try:
        img = imreadRGB(img_path)
    except Exception as e:
        print(f"Failed to load image {img_filename}: {e}")
        return img_filename, {}
    img_result = single_img_deltaE2000_all_patterns_pipeline(img, redemosaiced_img_dir=redemosaiced_img_directory, img_filename=img_filename)
    return img_filename, img_result


def batch_deltae_single_img(
        img_path: Union[str, bytes, os.PathLike],
        redemosaiced_img_dir,
        img_filename: str = None,
) -> dict:
    '''
    Use PyTorch to calculate delta e values in batch processing. img_path points to the original image and redemosaiced_img_dir should contain the four
    precomputed redemosaiced image. img_filename is img_path without the qualifying folder path.
    Must have redemosaiced images precomputed and stored
    in redemosaiced_img_dir before this function is called.
    '''
    # TODO: Batch redemosaic image on the fly.
    assert redemosaiced_img_dir is not None and img_filename is not None, "Only support batch processing with already demosaiced image for now."
    try:
        original_img = imreadRGB(img_path)
    except Exception as e:
        print(f"Failed to load image {img_filename}: {e}")
        return {}
    demosaiced_imgs = [read_redemosaiced_img(img_filename=img_filename, pattern=pattern, redemosaiced_img_dir=redemosaiced_img_dir) for pattern in
                       get_bayer_patterns()]
    demosaiced_imgs = np.stack(demosaiced_imgs, axis=0) # Stack for batch processing using PyTorch. Will be converted to PyTorch tensors later.
    deltae_results = batch_deltae_adapter(preds=demosaiced_imgs, target=original_img)
    img_results = {}
    for pattern_idx, pattern in enumerate(get_bayer_patterns()):
        img_results[pattern] = {
            "DeltaE2000": deltae_results[pattern_idx],
        }
    return compile_single_img_patterns_results(img_results, deltae_metric)


def deltae_all_imgs_in_directory_pipeline(
        directory_path: Union[str, bytes, os.PathLike],
        json_dump_directory_path: Union[str, bytes, os.PathLike] = ".",
        json_dump_filename: Union[str, bytes, os.PathLike] = "deltae2000_results.json",
        redemosaiced_img_dir = None,
        batch_deltae: bool = True,
):
    '''
    For all original images in the folder of directory_path, we calculate Delta E difference with its four redemosaiced images in redemosaiced_img_dir,
    The results are stored in JSON files with name json_dump_filename in json_dump_directory_path. If batch_deltae is set to true, we use PyTorch based
    Delta E implementation for batch processing.
    Each entry in the result JSON should be like "r000da54ft.tif": {"all_results": {"rggb": {"DeltaE2000": 0.49998870491981506}, "bggr": {"DeltaE2000": 0.4919033646583557}, "grbg": {"DeltaE2000": 0.6200681328773499}, "gbrg": {"DeltaE2000": 0.622094988822937}}, "best_results": {"DeltaE2000": {"best_result": 0.4919033646583557, "best_pattern": "bggr"}}}
    Batch processing can only be done if you have precomputed redemosaiced images. If you do not want to use batch processing and do not provide a directory storing precomputed redemosaiced images, redemosacing is done on the fly.
    '''
    print(f"Trying the proposed method on all images in the directory {directory_path}.")
    img_file_infos = get_all_images_in_directory(directory_path)
    all_results = {}
    args = [(img_file_info, redemosaiced_img_dir) for img_file_info in img_file_infos] # For parallel processing, see comments of compute_deltaE2000_for_single_img
    if not batch_deltae: # If we do not use batch processing, use four processes to parallel process the delta e calculations.
        num_process = 4
        print(f"{num_process} processes are used in this run.")
        with multiprocessing.Pool(processes=num_process) as pool:
            result_iterator = pool.imap_unordered(compute_deltaE2000_for_single_img, args)
            # Iterate through results as they become available
            for i, result in tqdm(enumerate(result_iterator), total=len(img_file_infos)):
                if len(result[1]) != 0:
                    all_results[result[0]] = result[1]
    else:
        for arg in tqdm(args, total=len(args)): # Use PyTorch for batch processing.
            img_filename = arg[0][0]
            all_results[img_filename] = batch_deltae_single_img(
                img_path=arg[0][1],
                redemosaiced_img_dir=redemosaiced_img_dir,
                img_filename=img_filename,
            )

    with open(os.path.join(json_dump_directory_path, json_dump_filename), "w") as handle:
        json.dump(all_results, handle)


if __name__ == '__main__':
    genimage_types = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_5", "VQDM", 'wukong']
    for demosaic_method in ["bilinear", "vng", "ea"]:
        for genimage_type in tqdm(genimage_types, total=len(genimage_types)):
            task = [
                f"E:\\EECE541_dataset\\GenImage\\{genimage_type}",
                f"E:\\EECE541_dataset\\GenImage\\{demosaic_method}_redemosaiced_{genimage_type}",
                f"deltae2000_genimage_{genimage_type}_{demosaic_method}_results.json"
            ]
            original_img_dir = task[0]
            redemosaiced_img_dir = task[1]
            json_dump_file_name = task[2]
            json_dump_dir = "../results"
            deltae_all_imgs_in_directory_pipeline(
                directory_path=original_img_dir,
                json_dump_directory_path=json_dump_dir,
                json_dump_filename=json_dump_file_name,
                redemosaiced_img_dir=redemosaiced_img_dir,
                batch_deltae=True
            )
