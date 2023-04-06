"""Code required to reproduce figure 1 of the paper (Guo et al.)"""


import torch

from simple_blackbox_attack.simba.simba import predict


def simba_best_epsilon(model, image: torch.Tensor, search_vector: torch.Tensor, step_size: float) -> torch.Tensor:
    """Same idea as simba but finding best epsilon for a given search vector

    Args:
        model: the to-be-attacked model (no access on the parameters)
        image (torch.Tensor): input image
        search_vector (torch.Tensor): specific search vector
        step_size (float): the magnitude of the image pertubation in search vector direction

    Returns:
        torch.Tensor: the best probability of either -epsilon or +epsilon
    """
    device = image.get_device()

    pertubed_image: torch.Tensor = image + step_size * search_vector
    probability_plus, _ = predict(model, pertubed_image.to(device))

    pertubed_image: torch.Tensor = image - step_size * search_vector
    probability_minus, _ = predict(model, pertubed_image.to(device))

    return min(probability_plus, probability_minus)
