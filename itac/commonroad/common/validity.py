import numpy as npy
import warnings
from typing import Union

__author__ = "Christian Pek"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["BMW Group CAR@TUM"]
__version__ = "2022.1"
__maintainer__ = "Christian Pek"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"

from commonroad import TWO_PI


class ValidTypes:
    """
    Default Type Lists
    """
    NUMBERS = (float, int, npy.number)  # real numbers
    INT_NUMBERS = (int, npy.integer)  # integer numbers

    LISTS = list  # vectors
    ARRAY = (npy.ndarray,)


def is_real_number(n: float) -> bool:
    """
    Checks if a provided variable is a scalar number
    :param n: The number to check
    :return: True if the provided variable is a scalar number, False otherwise
    """
    return isinstance(n, ValidTypes.NUMBERS)


def is_integer_number(n: int) -> bool:
    """
    Checks if a provided variable is an integer number
    :param n: The number to check
    :return: True if the provided variable is a integer number, False otherwise
    """
    return isinstance(n, ValidTypes.INT_NUMBERS)


def is_natural_number(n: int) -> bool:
    """
    Checks if a provided variable is a natural number
    :param n: The number to check
    :return: True if the provided variable is a natural number, False otherwise
    """
    return is_integer_number(n) and n >= 0


def is_positive(n: float) -> bool:
    """
    Checks if a provided variable is a positive number
    :param n: The number to check
    :return: True if the provided number is a positive scalar value (0 is NOT positive)
    """
    return is_real_number(n) and (True if (npy.sign(n) > 0) else False)


def is_negative(n: float) -> bool:
    """
    Checks if a provided variable is a negative number
    :param n: The number to check
    :return: True if the provided number is a positive scalar value (0 is NOT positive)
    """
    return is_real_number(n) and not (True if (npy.sign(n) > 0) else False)


def is_valid_length(length: int) -> bool:
    """
    Checks if a provided length/width is non-zero and positive
    :param length: The length/width to check
    :return: True if the provided length/width is valid
    """
    return is_natural_number(length) and length > 0


def is_real_number_vector(x: Union[npy.ndarray, float], length=None):
    """
    Checks if a provided variable is a vector of real numbers
    :param x: The variable to check
    :param length: optional parameter which tests if the vector is a real vector of specified length
    :return: True if the specified variable is a vector of real numbers
    """
    return isinstance(x, ValidTypes.ARRAY) and all(is_real_number(elem) for elem in x) and (
        len(x) >= 0 if length is None else len(x) == length)


def is_integer_number_vector(x: Union[npy.ndarray, float], length=None):
    """
    Checks if a provided variable is a vector of integer numbers
    :param x: The variable to check
    :param length: optional parameter which tests if the vector is a real vector of specified length
    :return: True if the specified variable is a vector of integer numbers
    """
    return isinstance(x, ValidTypes.ARRAY) and all(is_integer_number(elem) for elem in x) and (
        len(x) >= 0 if length is None else len(x) == length)


def is_natural_number_vector(x: npy.ndarray, length=None):
    """
    Checks if a provided variable is a vector of natural numbers
    :param x: The variable to check
    :param length: optional parameter which tests if the vector is a natural number vector of specified length
    :return: True if the specified variable is a vector of real numbers
    """
    return isinstance(x, ValidTypes.ARRAY) and all(is_natural_number(elem) for elem in x) and (
        len(x) >= 0 if length is None else len(x) == length)


def is_list_of_numbers(x: list, length=None):
    """
    Checks if a provided variable is a list of numbers
    :param x: The variable to check
    :param length: optional parameter which tests if the list is a list of specified length
    :return: True if the specified variable is a list of numbers
    """
    return isinstance(x, list) and is_real_number_vector(npy.array(x), length)


def is_list_of_integer_numbers(x: list, length=None):
    """
    Checks if a provided variable is a list of numbers
    :param x: The variable to check
    :param length: optional parameter which tests if the list is a list of specified length
    :return: True if the specified variable is a list of numbers
    """
    return isinstance(x, list) and is_integer_number_vector(npy.array(x), length)


def is_list_of_natural_numbers(x: list, length=None):
    """
    Checks if a provided variable is a list of natural numbers
    :param x: The variable to check
    :param length: optional parameter which tests if the list is a list of specified length
    :return: True if the specified variable is a list of natural numbers
    """
    return isinstance(x, list) and is_natural_number_vector(npy.array(x), length)


def is_set_of_natural_numbers(x: set, length=None):
    """
    Checks if a provided variable is a set of natural numbers
    :param x: The variable to check
    :param length: optional parameter which tests if the list is a list of specified length
    :return: True if the specified variable is a set of natural numbers
    """
    return isinstance(x, set) and is_natural_number_vector(npy.array(x), length)


def is_in_interval(x: float, x_min: float, x_max: float) -> bool:
    """
    Checks if a provided number (or vector) is within a specified interval
    :param x: The number (or vector) to check
    :param x_min: The lower bound of the interval
    :param x_max: The upper bound of the interval
    :return: True if the specified number (or vector) is within [x_min,x_max], False otherwise
    """
    if x_min is not None:
        assert is_real_number(x_min), '<Validity>: provided lower bound is not valid'

    if x_max is not None:
        assert is_real_number(x_max), '<Validity>: provided upper bound is not valid'

    if x_min is not None and x_max is not None and npy.greater(x_min, x_max):
        warnings.warn("<Validity>: x_min is greater than x_max", UserWarning)

    if is_real_number(x):
        return (npy.greater_equal(x, x_min) if x_min is not None else True) and (
            npy.greater_equal(x_max, x) if x_max is not None else True)
    if is_real_number_vector(x):
        return (npy.greater_equal(x, x_min).all() if x_min is not None else True) and (
            npy.greater_equal(x_max, x).all() if x_max is not None else True)

    return False


def is_valid_velocity(v: float, v_min=None, v_max=None):
    """
    Checks if a provided velocity is a valid velocity
    :param v: The velocity to check (either scalar or vector)
    :param v_min: Default parameter: if provided this is the minimum velocity
    :param v_max: Default parameter: if provided this is the maximum velocity
    :return: True if the provided velocity is a valid velocity (with respect to specified velocity range)
    """
    if v_min is None and v_max is None:
        return is_real_number(v) or is_real_number_vector(v)
    else:
        return is_in_interval(v, v_min, v_max)


def is_valid_acceleration(a: float, a_min=None, a_max=None):
    """
    Checks if a provided acceleration is a valid acceleration
    :param a: The acceleration to check (either scalar or vector)
    :param a_min: Default parameter: if provided this is the minimum acceleration
    :param a_max: Default parameter: if provided this is the maximum acceleration
    :return: True if the provided acceleration is a valid acceleration (with respect to specified acceleration range)
    """
    if a_min is None and a_max is None:
        return is_real_number(a) or is_real_number_vector(a)
    else:
        return is_in_interval(a, a_min, a_max)


def is_valid_orientation(theta: float) -> bool:
    """
    Checks if a provided orientation (scalar or vector) is a valid orientation in the interval [-2pi,2pi]
    :param theta: The orientation to check
    :return: True if the orientation is a valid orientation (radian), False otherwise
    """
    return is_in_interval(theta, -TWO_PI, TWO_PI)


def is_valid_polyline(polyline: npy.ndarray, length=None):
    """
    Checks if a provided polyline is a valid polyline, i.e. it is a list of points (xi,yi)^T.
    The list must have a shape of (2,n), resulting in [[x0,x1,...,xn],[y0,y1,...,yn]].
    By providing the optional parameter length, it is checked whether the provided has the desired length.
    :param polyline: The polyline to check
    :param length: The assumed length of the polyline
    :return: True if the polyline is a valid polyline, False otherwise
    """
    if length is not None:
        assert is_valid_length(length)

    return isinstance(polyline, ValidTypes.ARRAY) and len(polyline) >= 2 and all(
            is_real_number_vector(elem, 2) for elem in polyline) and (
               len(polyline) == length if length is not None else True)


def is_valid_list_of_vertices(vertices: npy.ndarray, number_of_vertices=None):
    """
    Checks if a provided list of vertices is a valid, i.e., it is a list of points (xi,yi). The list must
    have a shape of (n, 2), resulting in [[x0, y0], [x1, y1], ...]. The number of vertices can be checked with the
    provided parameter number_of_vertices.
    :param vertices: list of vertices [[x0, y0], [x1, y1], ...]
    :param number_of_vertices: The assumed number of vertices
    :return: True if vertices is a valid list of vertices, False otherwise
    """
    if number_of_vertices is not None:
        assert is_valid_length(number_of_vertices)

    return isinstance(vertices, ValidTypes.LISTS) and len(vertices) >= 1 and all(
            is_real_number_vector(elem, 2) for elem in vertices) and (
               len(vertices) == number_of_vertices if number_of_vertices is not None else True)


def is_valid_array_of_vertices(vertices: npy.ndarray, number_of_vertices=None):
    """
    Checks if a provided array of vertices is a valid, i.e., it is a array of points (xi,yi). The array must
    have a shape of (n, 2), resulting in [[x0, y0], [x1, y1], ...]. The number of vertices can be checked with the
    provided parameter number_of_vertices.
    :param vertices: array of vertices [[x0, y0], [x1, y1], ...]
    :param number_of_vertices: The assumed number of vertices
    :return: True if vertices is a valid array of vertices, False otherwise
    """
    if number_of_vertices is not None:
        assert is_valid_length(number_of_vertices)

    return isinstance(vertices, ValidTypes.ARRAY) and len(vertices) >= 1 and all(
            is_real_number_vector(elem, 2) for elem in vertices) and (
               len(vertices) == number_of_vertices if number_of_vertices is not None else True)
