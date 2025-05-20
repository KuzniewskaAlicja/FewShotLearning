from pathlib import Path
from torch.utils.data import DataLoader

from .dataset import ProductDataset
from .transforms import get_transforms


def get_dataset_info(data_dir: Path) -> tuple:
    """
    Gets information about the dataset from directory structure

    This function scans a directory where each subdirectory represents
    a class, collecting image paths, class indices, and class names

    Args:
        data_dir:
            Path to dataset directory where each subdirectory is a class
    """
    image_paths, labels, classes = [], [], []
    for class_idx, class_dir in enumerate(sorted(data_dir.iterdir())):
        if not class_dir.is_dir():
            continue

        files = list(class_dir.iterdir())
        classes.append(class_dir.name)
        image_paths.extend(files)
        labels.extend([class_idx] * len(files))

    return image_paths, labels, classes


def load_data(
    data_paths: list[Path],
    labels: list[int],
    classes: list[str],
    img_size: int,
    dataset_type: str,
    batch_size: int
):
    """
    Creates a DataLoader for the provided image data

    Args:
        data_paths:
            List of paths to dataset images
        labels:
            Ground truth labels (class indices) for provided images
        classes:
            List of all possible class names
        img_size:
            Target size for image resizing
        dataset_type:
            Type of loaded dataset ('train' or 'val') to determine 
            appropriate transformations
        batch_size:
            Number of samples per batch
    """
    dataset = ProductDataset(
        data_paths,
        labels,
        classes,
        get_transforms(dataset_type, img_size)
    )
    dataset_loader = DataLoader(dataset, batch_size, shuffle=True)

    return dataset_loader
