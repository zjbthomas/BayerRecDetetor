import os
from typing import Union

import cv2
import numpy as np


def imreadRGB(path: str, flag: int = cv2.IMREAD_COLOR):
    '''
    Given an image path, read an image as numpy array. Set the flag to determine if we should read a colorful image or grayscale image.
    '''
    img = cv2.imread(path)
    if flag == cv2.IMREAD_COLOR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwriteRGB(filename: str, img: np.ndarray):
    '''
    Given an image path, write a numpy array as an image
    '''
    assert len(img.shape) == 3 and img.shape[-1] == 3, \
        (f"The img parameter must has 3 dimensions and the last dimension must be 3 representing red, green, blue colors."
         f"The given img parameter has shape {img.shape}")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def convert_tiff_to_jpeg_opencv(tiff_file_path, jpeg_file_path, quality=90):
    '''
    Given a tiff image path, read the image and write it as a JPEG image with a specific compression quality. This is used for compressing tiff images.
    In practice tiff_file_path may specify an image of any file format, not necessarily a tiff image.
    '''
    img = cv2.imread(tiff_file_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    cv2.imwrite(jpeg_file_path, img, encode_param)


def get_all_images_in_directory(directory_path: Union[str, bytes, os.PathLike]):
    # Return a list of tuples, whose structures follow (filename, full file path), of all the tiff or png images in a directory.
    # Only detects images whose file name postfix is .tif, .tiff, .png, .jpeg, .jpg.
    directory = os.fsencode(directory_path)
    img_paths = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if (filename.lower().endswith(".tif") or
                filename.lower().endswith(".tiff") or
                filename.lower().endswith(".png") or
                filename.lower().endswith(".jpeg") or
                filename.lower().endswith(".jpg")):
            img_paths.append((filename, os.path.join(directory_path, filename)))
    return img_paths


def combine_genimage_results(demosaic_algo: str, results_dir: Union[str, bytes, os.PathLike] = "results"):
    '''
    Given a demosaic algorithm, we combine comparison results of all GenImage subsets into a single file containing all the results.
    All existing comparison results of each subset should be in the results_dir.
    For example, after this function is called, on python/results with bilinear as the demosaicing algorithm, all files whose name follows the pattern
    deltae2000_genimage_{genimage_subset_name}_bilinear_results.json in the python/results folder will be combined into a JSON result file whose name is
    deltae2000_genimage_combined_bilinear_results.json. Works for the JSON files storing results of PSNR, SSIM, Delta E, VMAF, VMAF 4K.
    '''
    assert demosaic_algo in ["vng", "malvar", "bilinear", "ea", ""]
    genimage_types = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_5", "VQDM", 'wukong']
    metric_file_prefix = {
        "PSNR": 'psnr_ssim',
        "SSIM": 'psnr_ssim',
        "vmaf_v0.6.1": "vmaf",
        "vmaf_4k_v0.6.1": "vmaf",
        "DeltaE2000": 'deltae2000'
    }
    import json
    for metric in metric_file_prefix:
        prefix = metric_file_prefix[metric]
        combind_dict = {}
        for genimage_type in genimage_types:
            result_path = f"{results_dir}/{prefix}_genimage_{genimage_type}_{demosaic_algo}_results.json" if demosaic_algo not in [
                "malvar" or ""] else f"{results_dir}/{prefix}_genimage_{genimage_type}_results.json"
            with open(result_path, "r") as handle:
                results = json.load(handle)
                for k, v in results.items():
                    combind_dict[k] = v
        with open(f"{results_dir}/{prefix}_genimage_combined_{demosaic_algo}_results.json" if demosaic_algo not in [
            "malvar" or ""] else f"results/{prefix}_genimage_combined_results.json", "w") as handle:
            json.dump(combind_dict, handle)
