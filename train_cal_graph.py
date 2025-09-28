import os
from typing import Union, Callable, Literal
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import textwrap
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
import concurrent

import torch

from redemosaic.redemosaic_directory import split_filename_and_read_img, get_bayer_patterns
from redemosaic.mht_redemosaic_pytorch_implementation import batch_redemosaic_malvar_numpy_adapter

#from utils import get_all_images_in_directory

from image_similarity_metrics.image_similarity_metrics import PSNR, SSIM
from image_similarity_metrics.psnr_ssim_all_img_directory import batch_psnr_numpy_adapter, compile_single_img_patterns_results, batch_ssim_numpy_adapter
from image_similarity_metrics.delta_e import batch_deltae_adapter, DeltaE2000

from results.process_results.calculate_cross_bayer_pattern_diff_var import diff_variance_analysis_single_image

NUM_GPUS = 4

## Redomosaic the images
def get_all_images_in_directory_temp(iut_paths_file: Union[str, bytes, os.PathLike]):
    # Return a list of tuples, whose structures follow (filename, full file path), of all the tiff or png images in a directory.
    # Only detects images whose file name postfix is .tif, .tiff, .png, .jpeg, .jpg.
    img_paths = []

    with open(iut_paths_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            parts = l.rstrip().split(' ')
            iut_path = parts[0]

            filename = os.path.basename(iut_path)

            img_paths.append((filename, iut_path))

    return img_paths

metrics = [PSNR, SSIM, DeltaE2000]

def rec_loss_single_img(original_img, demosaiced_imgs) -> dict:
    '''
    Use PyTorch to calculate PSNR and SSIM values in batch processing. img_path points to the original image and redemosaiced_img_dir should contain the four
    precomputed redemosaiced image. img_filename is img_path without the qualifying folder path. Must have redemosaiced images precomputed and stored
    in redemosaiced_img_dir before this function is called.
    '''
    # TODO: Batch redemosaic image on the fly.
    demosaiced_imgs = np.stack(demosaiced_imgs, axis=0) # For batch processing. Will be converted to PyTorch tensors.
    psnr_results = batch_psnr_numpy_adapter(preds=demosaiced_imgs, target=original_img)
    ssim_results = batch_ssim_numpy_adapter(preds=demosaiced_imgs, target=original_img)
    deltae_results = batch_deltae_adapter(preds=demosaiced_imgs, target=original_img)
    img_results = {}
    for pattern_idx, pattern in enumerate(get_bayer_patterns()):
        img_results[pattern] = {
            "PSNR": psnr_results[pattern_idx],
            "SSIM": ssim_results[pattern_idx],
            "DeltaE2000": deltae_results[pattern_idx]
        }
    return compile_single_img_patterns_results(img_results, metrics)

def diff_variance_analysis_all_images_in_json_file(json_results):
    '''
    For a JSON file containing image comparison results specified by json_path,
    for each dictionary storing image comparison results between the original images and the
    redemosaiced images, add a new sub-dictionary called "analysis_results" that record the min-max differences, and the variance of the four image comparison results.
    It looks like this "analysis_results": {"DeltaE2000": {"min_max_diff": 0.0522923469543457, "variance": 0.0004258269550945215}}}
    '''
    for image_file_name in json_results.keys():
        image_redemosaic_result = json_results[image_file_name]
        analyzed_results = diff_variance_analysis_single_image(image_redemosaic_result)
        json_results[image_file_name] = analyzed_results

    return json_results

def handle_single_image(item, index, batch_redemosaic_func):
    img_filename, img_filepath = item
    try:
        img, img_filename_no_postfix, img_filename_postfix = split_filename_and_read_img(img_filename, img_filepath)
    except Exception as e:
        print(f"Failed to load image {img_filename}: {e}")
        return

    patterns = get_bayer_patterns()
    redemosaic_results = batch_redemosaic_func(img, patterns, index)

    demosaiced_imgs = []

    for pattern_idx, pattern in enumerate(patterns):
        cur_result = redemosaic_results[pattern_idx]
        
        demosaiced_imgs.append(cur_result)

    ## Comparing the Original Images against the Redemosaiced Images
    return img_filename, rec_loss_single_img(img, demosaiced_imgs)
    
def redemosaic_directory(
        source_directory_path: Union[str, bytes, os.PathLike],
        batch_redemosaic_func
):
    '''
    Apply the redemosaic process to all the images in the folder of source_directory_path.
    The redemosaic results are stored in the folder of dst_directory_path.
    demosaic_func specifies the demosaic function to be used. By default, it uses the demosaic function in mht_redemosaic_opencv_implementation.py that performs Malvar et al. demosaicing. This function
    can also be the two functions in the `opencv_demosaic_algorithms.py` file.
    batch_redemosaic_func specifies a function that can be used for batch demosaicing. If it is set, the function specified by demosaic_func is ignored.
    batch_redemosaic_func should be set to one of "batch_redemosaic_malvar_numpy_adapter" and "batch_redemosaic_bilinear_numpy_adapter".
    demosaic_func should be set to one of "demosaic", "vng_demosaic", and "ea_demosaic". vng demosaicing and ea demosaicing have no batch implementation. Bilinear demosaicing only has a batch implementation.
    '''
    print(
        f"Doing a directory-level redemosaicing. "
        f"The source directory path is {source_directory_path}"
    )

    img_file_infos = get_all_images_in_directory_temp(source_directory_path)
    args = [(img_file_info[0], img_file_info[1]) for img_file_info in img_file_infos] # For parallel processing

    print(f"Using PyTorch-based batch redemosaicing for this execution. The function name is {batch_redemosaic_func.__qualname__}.")

    all_results = {}

    with concurrent.futures.ProcessPoolExecutor(NUM_GPUS) as executor:
        futures = [executor.submit(handle_single_image, arg, ix % NUM_GPUS, batch_redemosaic_func) for ix, arg in enumerate(args)]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(args), mininterval=1800):
            img_filename, result = f.result()
            all_results[img_filename] = result

    return diff_variance_analysis_all_images_in_json_file(all_results)

def read_comparison_std(name, json, metric_name) -> np.ndarray:
    '''

    :param json_file_path: the path to a JSON file storing image comparison results. e.g. python/results/deltae2000_raise1000_ea_results.json
    :param metric_name: The metric name for which we want to read the standard deviations of the four image comparison results of all images recorded in the result JSON file.
    :return: a numpy array recording the standard deviations of the four image comparison results of the metric specified by metric_name of all the images recorded in the result JSON file.
    '''
    analysis_results = {filename: json[filename]["analysis_results"][metric_name] for filename in json}
    variance_values = np.asarray([sqrt(analysis_results[img]["variance"]) for img in analysis_results])

    # assert not np.isnan(np.sum(variance_values))
    variance_values = variance_values[~np.isnan(variance_values)]

    # write to file
    with open(name + '_' + metric_name + '.txt', 'w') as f:
        for filename in json:
            f.write(filename + ',' + str(sqrt(json[filename]["analysis_results"][metric_name]["variance"])) + '\n')

    return variance_values

def plot_histogram(
        ax, # Matplotlib axis objects.
        data_type, # One of "Best", "Minimum-maximum Differences" and "Standard Deviation". Used as a label.
        best_results_dict,
        plot_range_percentile=98, # cap the range of the x axis to the percentile of all the data points combined.
):
    '''
    Plot the distributions of given data sequences on the ax (a matplotlib axis object). Must give at least one data sequence and its name for labeling
    purpose. Please note that data4 is reserved to be used for GenImage data points as we expect it will have more datapoints. We plot it on a separate
    y axis so the scale will not be distorted.
    '''
    all_data = []
    for data in best_results_dict.values():
        if (data.ndim != 1):
             raise ValueError("Input data must be a 1D numpy array.")
        
        all_data.extend(data.tolist())

    bottom = min(all_data)
    top = np.percentile(np.asarray(all_data), plot_range_percentile)
    num_bins = 40
    bin_width = (top - bottom) / num_bins
    bin_edges = np.arange(bottom, top + bin_width, bin_width)
    bins = bin_edges
    alpha = 0.5

    colors = ['blue', 'red', 'green', 'yellow', 'orange']

    # plot the same data on both axes
    for (item, color) in zip(best_results_dict.items(), colors):
        ax.hist(item[1], bins=bins, color=color, alpha=alpha,
                label=textwrap.fill(f"{item[0]}, mean={np.mean(item[1]):.4f}, min={np.min(item[1]):.4f}, max={np.max(item[1]):.4f}", width=75))

    ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel(f'Variance')
    ax.set_ylabel("Number of Images")

    ax.legend(loc='lower center', shadow=True, bbox_to_anchor=(0.5, 1.05), ncol=1)

def analyze_image_comparison_distribution_and_plot(
        results_dict,
        ax,
        read_func,
        metric=None,
        plot_range_percentile=98,
):
    '''
    :param dataset_names: a dictionary with dataset nicknames as keys and dataset result canonical names as values. Dataset nicknames should be one of ['real', 'fake', 'fake2', 'fake3', 'fake4']. Dataset result canonical names should be the string between results JSONs' prefix and "_results.json". For example, dataset_names = {"real": "raise1000_ea","fake": "diffusionDB1000"}. "real" should always be a key of "dataset_names". "fake3" should only be used for specifying the result JSON file of GenImage dataset.
    :param metric_file_prefix: a dictionary with image comparison metric canonical names as keys and results JSONs' prefixes as values. Will look for the image comparison metric rsults from the corresponding result JSON files. The default value of this dictionary is given above. The metric canonical names acceptable are ["PSNR","SSIM","vmaf_v0.6.1","vmaf_4k_v0.6.1", "DeltaE2000"].
    :param display_ds_names: a dictionary that has the same keys as dataset_names and the proper name of the corresponding dataset for display on the plots. For example, display_ds_names = {'real': "RAISE 1000; EA Demosaicing",'fake': "RAISE 1000; Malvar et al. Demosaicing"}.
    :param axs: Matplotlib axis objects for drawing the distributions. The number os axis object in the list should equal to the number of keys in the metric_file_prefix dictionary so we can plot each metric on one axis.
    :param read_func: should be one of read_comparison_results, read_comparison_minmaxdiffs, read_comparison_std
    :param plot_range_percentile: specify the maximum value on the x-axis. For example, if this parameter is set to 98, then the maximum value on the x-axis is the 98 percentile of all the results read. We do not want to always set it to 100 as sometimes there are really large outlier values that stretches the x-axis so the majority of the bars only lie on the very left of the plot.
    This function does not return anything. It draws image comparison results as distributions using bar graphs.
    '''
    data_type = "Standard Deviation"

    best_results_dict = {}

    for k in results_dict.keys():
        best_results_dict[k] = read_func(k, json=results_dict[k], metric_name=metric)

    plot_histogram(
        ax,
        data_type,
        best_results_dict,
        plot_range_percentile=plot_range_percentile
    )

def do_plot(results_dict: dict, data_type: str, plot_range_percentile = 98):
    '''
    :param dataset_names: a dictionary with dataset nicknames as keys and dataset result canonical names as values. Dataset nicknames should be one of ['real', 'fake', 'fake2', 'fake3', 'fake4']. Dataset result canonical names should be the string between results JSONs' prefix and "_results.json". For example, dataset_names = {"real": "raise1000_ea","fake": "diffusionDB1000"}. "real" should always be a key of "dataset_names". "fake3" should only be used for specifying the result JSON file of GenImage dataset.
    :param metric_file_prefix: a dictionary with image comparison metric canonical names as keys and results JSONs' prefixes as values. Will look for the image comparison metric rsults from the corresponding result JSON files. The default value of this dictionary is given above. The metric canonical names acceptable are ["PSNR","SSIM","vmaf_v0.6.1","vmaf_4k_v0.6.1", "DeltaE2000"].
    :param display_ds_names: a dictionary that has the same keys as dataset_names and the proper name of the corresponding dataset for display on the plots. For example, display_ds_names = {'real': "RAISE 1000; EA Demosaicing",'fake': "RAISE 1000; Malvar et al. Demosaicing"}.
    :param data_type: one of ["best", "min_max_diff", "std"]. See the next cell for how to set this parameter.
    :param plot_range_percentile: specify the maximum value on the x-axis. For example, if this parameter is set to 98, then the maximum value on the x-axis is the 98 percentile of all the results read. We do not want to always set it to 100 as sometimes there are really large outlier values that stretches the x-axis so the majority of the bars only lie on the very left of the plot.
    Plot the distribution histograms.
    '''
    assert data_type in ["best", "min_max_diff", "std"]
    read_func =  read_comparison_std

    for metric in ["PSNR", "SSIM", "DeltaE2000"]:
        fig, ax = plt.subplots(figsize=(14, 12))

        analyze_image_comparison_distribution_and_plot(results_dict, ax, metric=metric, read_func=read_func,plot_range_percentile=plot_range_percentile)

        plt.tight_layout()
        plt.savefig('plot_train_' + metric + '.png')
        
        plt.close(fig)

## main
path_files_dict = {
    'RAISE_train': './data/RD_train_real.txt',
    'DiffusionDB_train': './data/RD_train_fake.txt'
}

results_dict = {}
for item in path_files_dict.items():
    results_dict[item[0]] = redemosaic_directory(item[1], batch_redemosaic_malvar_numpy_adapter)

data_type = "std"

do_plot(results_dict, data_type=data_type, plot_range_percentile=99)
