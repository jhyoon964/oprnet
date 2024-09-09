import torch
import torch.nn.functional as F
import numpy as np

def range_constrained_noise_filter(images, filter_size=5):
    """
    Apply a median filter to each channel of the images using PyTorch.

    Parameters:
    images (4D torch tensor): The input images with shape (batch_size, channels, height, width).
    filter_size (int): The size of the filter window. Default is 3.

    Returns:
    filtered_images (4D torch tensor): The filtered images with the same shape as the input.
    """
    # Ensure the images are a torch tensor
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images, dtype=torch.float32)
    
    # Calculate the padding size
    pad_size = filter_size // 2
    
    # Pad the images to handle the borders
    padded_images = F.pad(images, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # Unfold the padded image to get all sliding windows
    unfolded = padded_images.unfold(2, filter_size, 1).unfold(3, filter_size, 1)
    
    # Reshape the unfolded tensor to prepare for median computation
    unfolded = unfolded.contiguous().view(*unfolded.size()[:4], -1)
    
    # Compute the median of each sliding window
    median_values = unfolded.median(dim=-1)[0]
    
    # Initialize the filtered image and copy the filtered values back
    filtered_images = images.clone()
    for c in range(images.size(1)):
        filtered_images[:, c, :, :] = median_values[:, c, :, :]
    filtered_images = torch.where((filtered_images == 0) & (filtered_images < images), images, filtered_images)
    return filtered_images