"""Functionality to receive data from the ImageNet dataset"""

from os import path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataSet(Dataset):
    """Pytorch Dataset for ImageNet"""

    def __init__(self, image_root, label_file, transform=None):
        self.image_root = image_root
        self.labels = np.fromfile(label_file, sep="\n")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = Image.open(path.join(self.image_root, f"ILSVRC2012_val_{idx+1:08d}.JPEG"))
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]
