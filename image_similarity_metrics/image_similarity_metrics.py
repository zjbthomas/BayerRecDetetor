import time
from abc import abstractmethod, ABC
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import torch
from torchvision.transforms.functional import pad

'''
These image comparison metric implementations are based on the implementations 
(https://github.com/scikit-image/scikit-image/blob/v0.23.1/skimage/metrics/_structural_similarity.py#L15-L292,
https://github.com/scikit-image/scikit-image/blob/v0.23.1/skimage/metrics/simple_metrics.py#L112-L167) in scikit-image[3] library. Again the modifications are made, so 
we can use PyTorch for batch processing. The PyTorch adaptation works well in batch processing.'''
doTimer = False


class Metric(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(image1: np.ndarray, image2: np.ndarray) -> float:
        '''

        :param image1: One image to be compared.
        :param image2: The image to be compared against.
        :return: The comparison result.
        '''
        pass

    @staticmethod
    @abstractmethod
    def metric_name() -> str:
        '''
        :return: The canonical name of the image comparison metric. This name is used as the key to store metric comparison result in the result JSON files.
        '''
        pass

    @staticmethod
    @abstractmethod
    def higher_is_better() -> bool:
        '''

        :return: If a higher score of this metric means the images being compared are more similar to each other.
        '''
        pass


class PSNR(Metric):
    @staticmethod
    def evaluate(image1: np.ndarray, image2: np.ndarray) -> float:
        start_time = time.process_time()
        result = peak_signal_noise_ratio(image1, image2)
        end_time = time.process_time()
        if doTimer:
            print(f"The PSNR calculation took {end_time - start_time} seconds.")
        return result

    @staticmethod
    def metric_name() -> str:
        return "PSNR"

    @staticmethod
    def higher_is_better() -> bool:
        return True


class SSIM(Metric):

    @staticmethod
    def evaluate(image1: np.ndarray, image2: np.ndarray) -> float:
        start_time = time.process_time()
        result = structural_similarity(image1, image2, channel_axis=2)
        end_time = time.process_time()
        if doTimer:
            print(f"The SSIM calculation took {end_time - start_time} seconds.")
        return result

    @staticmethod
    def metric_name() -> str:
        return "SSIM"

    @staticmethod
    def higher_is_better() -> bool:
        return True


def batch_psnr(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: int = 255
) -> torch.Tensor:
    """
    The function calculates mean PSNR of rgb channels between B predictions and the target.
    This function is based on the implementation of scikit-image:
     https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio
     https://github.com/scikit-image/scikit-image/blob/v0.23.1/skimage/metrics/simple_metrics.py#L112-L167
    data_range: 255 for uint8, 1 for float.
    preds should be the four redemosaiced image and target should be the original image.
    Input: preds(B, H, W, 3) and target(H, W, 3).
    Return: psnr(B).
    """
    assert preds.ndim == 4
    B = preds.size(0)

    MSE = torch.mean(torch.pow(preds.double() - target.double().expand(B, -1, -1, -1), 2),
                     dim=(1, 2, 3), dtype=torch.float64)
    return 10 * (2 * torch.log10(torch.full((B,), data_range, dtype=torch.float64, device=preds.device)) - torch.log10(MSE))


def batch_psnr_numpy_adapter(
        preds: np.ndarray,
        target: np.ndarray,
        data_range: int = 255
) -> np.ndarray:
    'This is an adapter to call batch_psnr that accepts numpy arrays and returns numpy arrays. See the comments of batch_psnr for input and output array shapes.'
    #device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    pt_preds = torch.from_numpy(preds).to(device)
    pt_target = torch.from_numpy(target).to(device)
    results = batch_psnr(preds=pt_preds, target=pt_target, data_range=data_range)
    return results.cpu().detach().numpy()


def batch_ssim(
        preds: torch.Tensor,
        target: torch.Tensor,
        data_range: int = 255,
        window_size: int = 7,
        K1: float = .01,
        K2: float = .03
) -> torch.Tensor:
    """
    The function calculates mean SSIM on rgb channels between B predictions and the target.
    This function is based on the implementation of scikit-image:
     https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
     https://github.com/scikit-image/scikit-image/blob/v0.23.1/skimage/metrics/_structural_similarity.py#L15-L292
    data_range: 255 for uint8, 1 for float.

    window_size: length of convolution kernel, default 7 gives padding 3, must be odd number.
    preds should be the four redemosaiced image and target should be the original image.
    Input: preds(B, H, W, 3) and target(H, W, 3).
    Return: ssim(B).
    """
    device = preds.device

    target = target.cpu().numpy()

    results = []
    for p in preds:
        results.append(structural_similarity(target, p.cpu().numpy(), win_size = window_size, data_range = data_range, multichannel = True))

    return torch.tensor(results, device = device)


def batch_ssim_numpy_adapter(preds: np.ndarray,
                             target: np.ndarray,
                             data_range: int = 255,
                             window_size: int = 7,
                             K1: float = .01,
                             K2: float = .03) -> np.ndarray:
    'This is an adapter to call batch_psnr that accepts numpy arrays and returns numpy arrays. See the comments of batch_ssim for input and output shapes.'
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pt_preds = torch.from_numpy(preds).to(device)
    pt_target = torch.from_numpy(target).to(device)
    results = batch_ssim(preds=pt_preds, target=pt_target, data_range=data_range, window_size=window_size, K1=K1, K2=K2)
    return results.cpu().detach().numpy()
