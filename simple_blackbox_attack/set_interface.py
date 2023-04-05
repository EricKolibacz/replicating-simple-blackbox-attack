"""Interface defining the structure of the search vector sets"""

from abc import ABC, abstractmethod

import torch


# pylint: disable-msg=R0903
class SearchVectors(ABC):
    """Abstract class for the set of search vectors"""

    @abstractmethod
    def get_random_vector(self) -> torch.Tensor:
        """Get a random new search vector (without replacement)

        Returns:
            torch.Tensor: search vectors
        """
