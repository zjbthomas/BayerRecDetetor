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

from image_similarity_metrics.image_similarity_metrics import PSNR
from image_similarity_metrics.psnr_ssim_all_img_directory import batch_psnr_numpy_adapter, compile_single_img_patterns_results

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

def read_comparison_std(prefix, metric) -> np.ndarray:
    filename = prefix + '_' + metric + '.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        variance_values = []
        for l in lines:
            parts = l.rstrip().split(',')
            variance_values.append(float(parts[1]))

    return np.asarray(variance_values)

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

    ax2 = ax.twinx()
    ax2.set_ylabel("Number of Images (GenImage)")

    # plot the same data on both axes
    for (item, color) in zip(best_results_dict.items(), colors):
        if item[0] == 'GenImage':
            ax2.hist(item[1], bins=bins, color=color, alpha=alpha,
                label=textwrap.fill(f"{item[0]}, mean={np.mean(item[1]):.4f}, min={np.min(item[1]):.4f}, max={np.max(item[1]):.4f}", width=75))
        else:
            ax.hist(item[1], bins=bins, color=color, alpha=alpha,
                    label=textwrap.fill(f"{item[0]}, mean={np.mean(item[1]):.4f}, min={np.min(item[1]):.4f}, max={np.max(item[1]):.4f}", width=75))

    ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel(f'Variance')
    ax.set_ylabel("Number of Images (other datasets besides GenImage)")

    handles = []
    labels = []
    for axi in [ax, ax2]:
        for handle, label in zip(*axi.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    ax.legend(handles, labels, loc='lower center', shadow=True, bbox_to_anchor=(0.5, 1.05), ncol=1)

def analyze_image_comparison_distribution_and_plot(
        results_dict,
        ax,
        plot_range_percentile=98
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

    plot_histogram(
        ax,
        data_type,
        results_dict,
        plot_range_percentile=plot_range_percentile
    )

def do_plot(path_files_dict: dict, plot_range_percentile = 98,
            detection_bounds = 0.0):
    '''
    :param dataset_names: a dictionary with dataset nicknames as keys and dataset result canonical names as values. Dataset nicknames should be one of ['real', 'fake', 'fake2', 'fake3', 'fake4']. Dataset result canonical names should be the string between results JSONs' prefix and "_results.json". For example, dataset_names = {"real": "raise1000_ea","fake": "diffusionDB1000"}. "real" should always be a key of "dataset_names". "fake3" should only be used for specifying the result JSON file of GenImage dataset.
    :param metric_file_prefix: a dictionary with image comparison metric canonical names as keys and results JSONs' prefixes as values. Will look for the image comparison metric rsults from the corresponding result JSON files. The default value of this dictionary is given above. The metric canonical names acceptable are ["PSNR","SSIM","vmaf_v0.6.1","vmaf_4k_v0.6.1", "DeltaE2000"].
    :param display_ds_names: a dictionary that has the same keys as dataset_names and the proper name of the corresponding dataset for display on the plots. For example, display_ds_names = {'real': "RAISE 1000; EA Demosaicing",'fake': "RAISE 1000; Malvar et al. Demosaicing"}.
    :param data_type: one of ["best", "min_max_diff", "std"]. See the next cell for how to set this parameter.
    :param plot_range_percentile: specify the maximum value on the x-axis. For example, if this parameter is set to 98, then the maximum value on the x-axis is the 98 percentile of all the results read. We do not want to always set it to 100 as sometimes there are really large outlier values that stretches the x-axis so the majority of the bars only lie on the very left of the plot.
    Plot the distribution histograms.
    '''
    for metric in ["PSNR", "SSIM", "DeltaE2000"]:
        results_dict = {}
        for item in path_files_dict:
            results_dict[item] = read_comparison_std(item, metric)

        fig, ax = plt.subplots(figsize=(14, 12))

        analyze_image_comparison_distribution_and_plot(results_dict, ax, plot_range_percentile=plot_range_percentile)

        plt.axvline(x=detection_bounds[metric], color='r')

        plt.tight_layout()
        plt.savefig('plot_test_' + metric + '.png')
        
        plt.close(fig)

## main
path_files_dict = ['RAISE_test','DiffusionDB_test','GenImage','VISION','PCS']

detection_bounds = {
    'PSNR': 0.14453367719257365,
    'SSIM': 0.0,
    'DeltaE2000': 0.0
}

do_plot(path_files_dict, plot_range_percentile=99, detection_bounds=detection_bounds)
