"""Testing suites for the search vector classes for the simple black box attack"""

import numpy as np
import pytest
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


def test_cartesian_vector_type():
    """Testing if cartesian search vector class returns a vector with correct type"""
    base = CartesianSearchVectors(IMAGE.size())
    vector = base.get_random_vector()

    assert vector.dtype == torch.float


def test_cartesian_vector_not_enough_vectors():
    """Testing if cartesian search vector class raises error as soon as no search vectors are left"""
    base = CartesianSearchVectors(IMAGE.size())
    for _ in range(IMAGE.numel()):
        base.get_random_vector()

    with pytest.raises(IndexError):
        base.get_random_vector()


def test_dct_correctly_initialized():
    """Testing if dct search vector class has implemented required abstract methods"""
    DCTSearchVectors(IMAGE.size(), RATIO)


def test_dct_get_correct_search_vector():
    """Testing if dct search vector class returns a valid vector"""
    base = DCTSearchVectors(IMAGE.size(), RATIO)
    vector1 = base.get_random_vector()
    vector2 = base.get_random_vector()
    vector3 = base.get_random_vector()

    assert np.isclose(vector_norm(vector1).numpy(), 1)
    assert np.isclose(vector_norm(vector2).numpy(), 1)
    assert np.isclose(vector_norm(vector3).numpy(), 1)

    assert not vector1.equal(vector2)
    assert not vector1.equal(vector3)
    assert not vector2.equal(vector3)


def test_dct_vector_size():
    """Testing if dct search vector class returns a vector with correct size"""
    base = DCTSearchVectors(IMAGE.size(), RATIO)
    vector = base.get_random_vector()

    assert vector.size() == IMAGE.size()


def test_dct_vector_type():
    """Testing if dct search vector class returns a vector with correct type"""
    base = DCTSearchVectors(IMAGE.size(), RATIO)
    vector = base.get_random_vector()

    assert vector.dtype == torch.float


def test_inverse_dct_2d():
    """Testing if dct class has a valid inverse dct in 2D"""
    base = DCTSearchVectors(IMAGE.size(), RATIO)

    frequency_coefficient = np.zeros((3, 2, 2))
    frequency_coefficient[0, 0, 0] = 1.0
    image = base.idct_2d(frequency_coefficient)

    assert np.allclose(image[0, :, :], 0.5)
    assert np.allclose(image[1:2, :, :], 0.0)

    frequency_coefficient = np.zeros((3, 2, 2))
    frequency_coefficient[2, 0, 0] = 1.0
    image = base.idct_2d(frequency_coefficient)

    assert np.allclose(image[2, :, :], 0.5)
    assert np.allclose(image[0:1, :, :], 0.0)

    frequency_coefficient = np.zeros((3, 2, 2))
    frequency_coefficient[0, 1, 0] = 1.0
    image = base.idct_2d(frequency_coefficient)

    assert np.isclose(image[0, :, :], np.array([[0.5, 0.5], [-0.5, -0.5]])).all()
    assert np.allclose(image[1:2, :, :], 0.0)


def test_inverse_dct_2d_gray_image():
    """Testing if dct class has a valid inverse dct in 2D for gray images (dimensions (1, w, h))"""
    gray_image = torch.zeros((1, 8, 8))
    base = DCTSearchVectors(gray_image.size(), RATIO)

    frequency_coefficient = np.zeros((1, 2, 2))
    frequency_coefficient[0, 0, 0] = 1.0
    image = base.idct_2d(frequency_coefficient)

    assert np.allclose(image[0, :, :], 0.5)

    frequency_coefficient = np.zeros((1, 2, 2))
    frequency_coefficient[0, 1, 0] = 1.0
    image = base.idct_2d(frequency_coefficient)

    assert np.isclose(image[0, :, :], np.array([[0.5, 0.5], [-0.5, -0.5]])).all()


def test_dct_vector_not_enough_vectors():
    """Testing if dct search vector class raises error as soon as no search vectors are left"""
    base = DCTSearchVectors(IMAGE.size(), RATIO)

    with pytest.raises(IndexError):
        for _ in range(IMAGE.numel() + 1):
            base.get_random_vector()


def test_dct_error_for_wrong_input_dims():
    """Testing if dct search vector class raises error as soon as no search vectors are left"""
    image_with_wrong_dimension = torch.zeros((8, 8, 3))
    with pytest.raises(ValueError):
        DCTSearchVectors(image_with_wrong_dimension.size(), RATIO)
