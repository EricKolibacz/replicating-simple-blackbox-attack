"""Method proposed by Guo et al."""

from typing import Union

import torch
from torch.linalg import vector_norm

from simple_blackbox_attack.set_interface import SearchVectors


# pylint: disable-msg=R0913,R0914
def simba(
    model, image: torch.Tensor, label: int, basis: SearchVectors, step_size: float, budget: float
) -> torch.Tensor:
    """Adversarial method from the paper Simple Black-box Adversarial Attacks (Guo et al. 2019)
    Named how it is named in the paper

    Args:
        model: the to-be-attacked model (no access on the parameters)
        image (torch.Tensor): input image
        label (int): correct label of the image (ground truth)
        basis (SearchVectors): set of orthogonal search vectors
        step_size (float): the magnitude of the image pertubation in search vector direction
        budget (float): the budget of pertubation allowed

    Returns:
        bool: was the attack successful
        torch.Tensor: pertubation
        int: number of base vectors included in the pertubation
        int: number of queries sent to the model
    """
    device = image.get_device()
    pertubation: torch.Tensor = torch.zeros(image.shape).to(device)

    probability, prediction = predict(model, image.to(device))

    steps, queries, l2_norm = 0, 0, []
    while prediction.item() == label and steps + 1 < (budget / step_size) ** 2:
        try:
            search_vector = basis.get_random_vector().to(device)
        except IndexError:
            print("Evaluated all vectors")
            break
        for alpha in [-step_size, step_size]:
            pertubed_image: torch.Tensor = image + pertubation + alpha * search_vector

            queries += 1
            probability_perturbed, prediction_perturbed = predict(model, pertubed_image.to(device))

            if probability_perturbed < probability:
                if steps % 512 == 0:
                    print(f"{steps}: Label probability {probability.item():.4f} - queries {queries}")
                steps += 1

                pertubation += alpha * search_vector
                if queries % 64 == 0:
                    l2_norm.append(vector_norm(pertubation.cpu()))
                probability = probability_perturbed
                prediction = prediction_perturbed

    l2_norm.append(vector_norm(pertubation.cpu()))

    return prediction.item() != label, pertubation, steps, queries, l2_norm


def predict(model, image: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
    """Simple helper function to predict class (with probability)"""
    logits = model(image.unsqueeze(0))
    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    probability, prediction = torch.topk(probabilities, 1)
    return probability, prediction
