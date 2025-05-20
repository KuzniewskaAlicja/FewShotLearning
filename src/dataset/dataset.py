from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms as T


class ProductDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading product images
    with their corresponding labels
    """
    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        classes: list[str],
        transforms: T.Compose | None = None
    ):
        super(ProductDataset, self).__init__()
        self.image_paths = paths
        self.classes = classes
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        """
        Gets the total number of samples in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """
        Get a sample from the dataset at the specified index
        """
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)

        return image, label
