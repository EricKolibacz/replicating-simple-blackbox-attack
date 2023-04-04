"""Method implementation from the paper Simple Black-box Adversarial Attacks (Guo et al. 2019)"""
import random
from abc import ABC, abstractmethod

import numpy as np
import torch


class SearchVectors(ABC):
    """Abstract class for the set of search vectors"""

    @abstractmethod
    def get_random_vector(self) -> torch.Tensor:
        """Get a random new search vector (without replacement)

        Returns:
            torch.Tensor: search vectors
        """


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


def simba(model, image: torch.Tensor, label: int, basis: SearchVectors, step_size: float) -> torch.Tensor:
    """Adversarial method from the paper Simple Black-box Adversarial Attacks (Guo et al. 2019)
    Named how it is named in the paper

    Args:
        model: the to-be-attacked model (no access on the parameters)
        image (torch.Tensor): input image
        label (int): correct label of the image (ground truth)
        basis (SearchVectors): set of orthogonal search vectors
        step_size (float): the magnitude of the image pertubation in search vector direction

    Returns:
        torch.Tensor: adversarial image
    """
    pertubation: torch.Tensor = torch.zeros(image.shape)

    logits = model(image.unsqueeze(0))
    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    probability, prediction = torch.topk(probabilities, 1)

    while prediction.item() == label:
        search_vector = basis.get_random_vector()
        for alpha in [-step_size, step_size]:
            pertubed_image: torch.Tensor = image + pertubation + alpha * search_vector

            logits = model(pertubed_image.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            probability_perturbed, prediction_perturbed = torch.topk(probabilities, 1)

            if probability_perturbed < probability:
                pertubation += alpha * search_vector
                probability = probability_perturbed
                prediction = prediction_perturbed

    return pertubation
