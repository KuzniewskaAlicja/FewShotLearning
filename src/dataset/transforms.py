import torch
from torchvision import transforms as T
import numpy as np


def get_transforms(subset_name: str, img_size: int) -> T.Compose:
    """
    Gets image transformation pipeline based on dataset subset

    Args:
        subset_name:
            Name of the dataset subset ('train' or 'val')
        img_size:
            Target image size
    """
    if subset_name == "train":
        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomHorizontalFlip(p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif subset_name == "val":
        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transforms


def mixup_data(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    alpha: float = 0.2,
):
    """
    Applies mixup augmentation to a batch of data

    Args:
        inputs:
            Batch of input images with shape
            [batch_size, channels, height, width]
        labels:
            Corresponding labels
        device:
            Device (CPU/GPU) where the computation should be performed
        alpha:
            Parameter controlling the strength of interpolation
            in the Beta distribution
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    label_a, label_b = labels, labels[index]

    return mixed_inputs, label_a, label_b, lam
