"""Descrete cosine transform search vector set"""
import random

import numpy as np
import torch
from scipy.fftpack import idct

from simple_blackbox_attack.set_interface import SearchVectors


class DCTSearchVectors(SearchVectors):
    """Search Vectors with base derived from the discrete cosine transform.
    Defined by Guo et al. 2018."""

    def __init__(self, size: torch.Size, ratio: float) -> None:
        if size[0] != 3 and size[0] != 1:
            raise ValueError(f"size = (3, w, h) or size = (1, w, h). Passed image has dimensions {size}")
        self.size = size
        self.frequency_dimensions = [
            (i, j, k) for i in range(3) for j in range(int(size[1] * ratio)) for k in range(int(size[2] * ratio))
        ]

    def get_random_vector(self) -> torch.Tensor:
        if len(self.frequency_dimensions) == 0:
            raise IndexError("No Vectors left")
        dimension = self.frequency_dimensions.pop(random.randrange(len(self.frequency_dimensions)))
        frequency_coefficients = np.zeros(self.size)
        frequency_coefficients[dimension] = 1.0
        return torch.from_numpy(self.idct_2d(frequency_coefficients)).float()

    def idct_2d(self, frequency_coefficients: np.array) -> np.array:
        """2 dimension discrete cosine transform (DCT)

        Args:
            frequency_coefficients (np.array): frequency coefficients with shape (h, w, 3)

        Returns:
            np.array: signal in 2D image space
        """
        return idct(
            idct(frequency_coefficients.T, axis=1, norm="ortho").T,
            axis=2,
            norm="ortho",
        )
