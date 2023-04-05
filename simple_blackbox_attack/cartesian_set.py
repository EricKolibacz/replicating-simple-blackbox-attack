"""Module containing the class for the cartesian search vector set"""

import random

import numpy as np
import torch

from simple_blackbox_attack.set_interface import SearchVectors


# pylint: disable-msg=R0903
class CartesianSearchVectors(SearchVectors):
    """Search Vectors with cartesian base."""

    def __init__(self, size: torch.Size) -> None:
        self.size = size
        self.vector_dimensions = list(np.arange(0, size.numel()))

    def get_random_vector(self) -> torch.Tensor:
        if len(self.vector_dimensions) == 0:
            raise IndexError("No Vectors left")
        dimension = self.vector_dimensions.pop(random.randrange(len(self.vector_dimensions)))
        search_vector = torch.zeros((self.size.numel()))
        search_vector[dimension] = 1
        return search_vector.reshape(self.size)
