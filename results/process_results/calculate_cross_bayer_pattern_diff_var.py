import json
import os
from typing import Union

import numpy as np
from tqdm import tqdm

from redemosaic.redemosaic_directory import get_bayer_patterns


def diff_variance_analysis_single_image(img_redemosaic_result: dict) -> dict:
    '''
    For a dictionary like {"all_results": {"rggb": {"DeltaE2000": 5.939765930175781}, "bggr": {"DeltaE2000": 5.977053642272949}, "grbg": {"DeltaE2000": 5.992058277130127}, "gbrg": {"DeltaE2000": 5.988003730773926}}, "best_results": {"DeltaE2000": {"best_result": 5.939765930175781, "best_pattern": "rggb"}}}
    passed in as img_redemosaic_result, it calculates the min-max difference and the variance for each metric's four original-remodaic comparison results.
    It returns a dictionary like {"all_results": {"rggb": {"DeltaE2000": 5.939765930175781}, "bggr": {"DeltaE2000": 5.977053642272949}, "grbg": {"DeltaE2000": 5.992058277130127}, "gbrg": {"DeltaE2000": 5.988003730773926}}, "best_results": {"DeltaE2000": {"best_result": 5.939765930175781, "best_pattern": "rggb"}}, "analysis_results": {"DeltaE2000": {"min_max_diff": 0.0522923469543457, "variance": 0.0004258269550945215}}}.
    This dictionary is the same dictionary passed in with a new field called "analysis_results" keeping track of each metric's min-max difference and variance.
    '''
    patterns = get_bayer_patterns()
    all_results = img_redemosaic_result["all_results"]
    metrics = list(all_results[patterns[0]].keys())
    assert metrics == list(img_redemosaic_result["best_results"].keys()), \
        "The metrics in the all_results dict is different from the metrics in the best_results dict."
    analysis_results = {
        metric: {
            "min_max_diff": float("nan"),
            "variance": float("nan")
        } for metric in metrics
    }
    for metric in metrics:
        metric_values = np.asarray([all_results[pattern][metric] for pattern in patterns])
        analysis_results[metric]["min_max_diff"] = metric_values.max() - metric_values.min()
        analysis_results[metric]["variance"] = metric_values.var()
    img_redemosaic_result["analysis_results"] = analysis_results
    return img_redemosaic_result

def get_all_result_jsons_in_dir(root_dir: Union[str, bytes, os.PathLike]) -> list:
    'Return a list of full paths of result JSONs in a folder.'
    json_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith("_results.json"):
                json_paths.append(os.path.join(root, file))
    return json_paths