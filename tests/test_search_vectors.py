"""Testing suites for the search vector classes for the simple black box attack"""

import numpy as np
import torch
from torch.linalg import vector_norm

from simple_blackbox_attack.cartesian_set import CartesianSearchVectors
from simple_blackbox_attack.dct_set import DCTSearchVectors

IMAGE = torch.zeros((3, 8, 8))
RATIO = 0.8


def test_cartesian_correctly_initialized():
    """Testing if cartesian search vector class has implemented required abstract methods"""
    CartesianSearchVectors(IMAGE.size())


def test_cartesian_get_correct_search_vector():
    """Testing if cartesian search vector class returns a valid vector"""
    base = CartesianSearchVectors(IMAGE.size())
    vector1 = base.get_random_vector()
    vector2 = base.get_random_vector()
    vector3 = base.get_random_vector()

    assert np.isclose(vector_norm(vector1).numpy(), 1)
    assert np.isclose(vector_norm(vector2).numpy(), 1)
    assert np.isclose(vector_norm(vector3).numpy(), 1)

    assert not vector1.equal(vector2)
    assert not vector1.equal(vector3)
    assert not vector2.equal(vector3)


def test_cartesian_vector_size():
    """Testing if cartesian search vector class returns a vector with correct size"""
    base = CartesianSearchVectors(IMAGE.size())
    vector = base.get_random_vector()

    assert vector.size() == IMAGE.size()


def test_dct_correctly_initialized():
    """Testing if dct search vector class has implemented required abstract methods"""
    DCTSearchVectors(IMAGE.size(), RATIO)


def test_dct_vector_size():
    """Testing if dct search vector class returns a vector with correct size"""
    base = DCTSearchVectors(IMAGE.size(), RATIO)
    vector = base.get_random_vector()

    assert vector.size() == IMAGE.size()
