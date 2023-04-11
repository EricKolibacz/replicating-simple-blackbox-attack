"""Functionality to receive data from the ImageNet dataset"""

import json
import os
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from mat4py import loadmat
from PIL import Image
from torch.utils.data import Dataset

FILE_LOCATION = Path(__file__).parent.absolute()


class ImageNetDataSet(Dataset):
    """Pytorch Dataset for ImageNet"""

    def __init__(self, image_root, label_file, meta_file: str, transform=None):
        self.image_root = image_root
        self.labels = np.fromfile(label_file, sep="\n").astype(np.int64)
        self.transform = transform

        self.meta = pd.DataFrame.from_dict(loadmat(meta_file)["synsets"])
        self.meta.set_index("WNID", inplace=True)
        with open(os.path.join(str(FILE_LOCATION) + "/imagenet_class_index.json"), "r", encoding="utf-8") as file:
            pytorch_label_indices = json.load(file)

        for label, (wnid, label_str) in pytorch_label_indices.items():
            self.meta.loc[wnid, "label"] = label
            self.meta.loc[wnid, "label_str"] = label_str.replace("_", " ")

        self.meta.set_index("ILSVRC2012_ID", inplace=True)
        self.labels = [self.meta.loc[label, "label"] for label in self.labels]
        self.meta.set_index("label", inplace=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(path.join(self.image_root, f"ILSVRC2012_val_{idx+1:08d}.JPEG")).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
