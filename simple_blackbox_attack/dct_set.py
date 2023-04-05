"""Descrete cosine transform search vector set"""
import random

import numpy as np
import torch
from scipy.fftpack import idct

from simple_blackbox_attack.set_interface import SearchVectors


class DCTSearchVectors(SearchVectors):
    """Search Vectors with base derived from the discrete cosine transform.
    Defined by Guo et al. 2018."""

    def __init__(self, size: torch.Size, ratio: float, color_dimension: int = 2) -> None:
        self.size = size
        self.pixel_dimensions = [i for i in range(3) if i != color_dimension]
        self.frequency_dimensions = [
            (k, i, j)
            for k in range(3)
            for i in range(int(size[self.pixel_dimensions[0]] * ratio))
            for j in range(int(size[self.pixel_dimensions[1]] * ratio))
        ]

    def get_random_vector(self) -> torch.Tensor:
        if len(self.frequency_dimensions) == 0:
            raise IndexError("No Vectors left")
        dimension = self.frequency_dimensions.pop(random.randrange(len(self.frequency_dimensions)))
        frequency_coefficients = np.zeros(self.size)
        frequency_coefficients[dimension] = 1.0
        return torch.from_numpy(self.idct_2d(frequency_coefficients)).type(torch.LongTensor)

    def idct_2d(self, frequency_coefficients: np.array) -> np.array:
        """2 dimension discrete cosine transform (DCT)

        Args:
            frequency_coefficients (np.array): frequency coefficients with shape (h, w, 3)

        Returns:
            np.array: signal in 2D image space
        """
        return idct(
            idct(frequency_coefficients.T, axis=self.pixel_dimensions[0], norm="ortho").T,
            axis=self.pixel_dimensions[1],
            norm="ortho",
        )
