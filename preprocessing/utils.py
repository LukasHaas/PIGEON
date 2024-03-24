import torch
from torch import Tensor
import numpy as np
from PIL import Image
from config import LABEL_SMOOTHING_CONSTANT, IMAGE_PATH, INPUT_PATH, LABEL_SMOOTHING_MONTHS

def smooth_labels(distances: Tensor) -> Tensor:
    """Haversine smooths labels for shared representation learning across geocells.

    Args:
        distances (Tensor): distance (km) matrix of size (batch_size, num_geocells)

    Returns:
        Tensor: smoothed labels
    """
    adj_distances = distances - distances.min(dim=-1, keepdim=True)[0]
    smoothed_labels = torch.exp(-adj_distances / LABEL_SMOOTHING_CONSTANT)
    smoothed_labels = torch.nan_to_num(smoothed_labels, nan=0.0, posinf=0.0, neginf=0.0)
    return smoothed_labels

def __scale_factor(original_fov: int) -> float:
    """Calculates the scaling factor to scale to 90 degree FOV

    Args:
        original_fov: the FOV of the image which should be rescaled.

    Returns:
        float: scaling factor
    """
    fov_old = np.radians(original_fov / 2)
    fov_90 = np.radians(45)
    factor = np.arcsin(fov_90) / np.arcsin(fov_old)
    return factor

def center_crop(filename: str, original_fov: int=96):
    """Center crops the given image to 90 degree FOV

    Args:
        filename (str): image location
        original_fov (int, optional): original FOV. Defaults to 96.
    """
    image = Image.open(filename)
    img = np.asarray(image)

    width = img.shape[1]
    height = img.shape[0]

    factor = __scale_factor(original_fov)
    new_width = factor * width
    new_height = factor * height

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    output = Image.fromarray(center_cropped_img)
    output.save(filename)


def alternative_crop(filename: str):
    """Crops the given image from a given degree FOV to 90 degree FOV

    Args:
        filename (str): image location
    """
    image = Image.open(f'{INPUT_PATH}/{filename}')
    img = np.asarray(image)

    left = 11
    right = 629

    top = 0
    bottom = 618

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    output = Image.fromarray(center_cropped_img)
    output_filename = filename.split('.')[0] + '_cropped.jpg'
    output = output.save(f'{IMAGE_PATH}/{output_filename}')