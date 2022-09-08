import numpy as np
import math
from typing import List, Union

__author__ = "Christina Miller"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["BMW CAR@TUM"]
__version__ = "2022.1"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"


def translate_rotate(vertices: np.ndarray, translation: Union[np.array, List[float]], angle: Union[float, int])\
        -> np.ndarray:
    """
    First translates the list of vertices, then rotates the list of vertices around the origin.

    :param vertices: array of 2D vertices [[x_0, y_0], [x_1, y_1], ...]
    :param translation: translation vector [x_off, y_off] in x- and y-direction
    :param angle: rotation angle in radian (counter-clockwise)
    :return: array of transformed vertices [[x'_0, y'_0], [x'_1, y'_1], ...]
    """

    h_vertices = to_homogeneous_coordinates(vertices)
    return from_homogeneous_coordinates(translation_rotation_matrix(translation, angle).
                                        dot(h_vertices.transpose()).transpose())


def rotate_translate(vertices: np.ndarray, translation: Union[np.array, List[float]], angle: Union[float, int])\
        -> np.ndarray:
    """
    First rotates the list of vertices around the origin and then translates the list of vertices.

    :param vertices: array of 2D vertices [[x_0, y_0], [x_1, y_1], ...]
    :param translation: translation vector [x_off, y_off] in x- and y-direction
    :param angle: rotation angle in radian (counter-clockwise)
    :return: array of transformed vertices [[x'_0, y'_0], [x'_1, y'_1], ...]
    """

    h_vertices = to_homogeneous_coordinates(vertices)
    return from_homogeneous_coordinates(rotation_translation_matrix(translation, angle).
                                        dot(h_vertices.transpose()).transpose())


def rotation_translation_matrix(translation: Union[np.array, List[float]], angle: Union[float, int]) -> np.ndarray:
    """
    Creates a matrix that first rotates a vector around the origin and then translates it.

    :param translation: offset in (x, y) for translating the vector
    :param angle: angle in rad [-2pi, +2pi]
    :return: matrix
    """
    if angle == 0:
        cos_angle = 1.0
        sin_angle = 0.0
    else:
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

    return np.array([[cos_angle, -sin_angle, translation[0]],
                     [sin_angle,  cos_angle, translation[1]],
                     [0, 0, 1]])


def translation_rotation_matrix(translation: Union[np.array, List[float]], angle: Union[float, int]) -> np.ndarray:
    """
    Creates a matrix that first translates a homogeneous point, and then rotates it around the origin.

    :param translation: offset in (x, y) for translating the vector
    :param angle: angle in rad [-2pi, +2pi]
    :return: matrix
    """
    translation_matrix = np.array([[1.0, 0.0, translation[0]],
                                   [0.0, 1.0, translation[1]],
                                   [0.0, 0.0, 1.0]], dtype=np.float64)
    if np.abs(angle) <= 0.05:
        cos_angle = 1.0
        sin_angle = angle
    else:
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

    rotation_matrix = np.array([[cos_angle, -sin_angle, 0.0],
                                [sin_angle, cos_angle, 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float64)
    return rotation_matrix.dot(translation_matrix)


def to_homogeneous_coordinates(points: np.array) -> np.ndarray:
    """
    Converts an array of vertices to homogeneous coordinates.

    :param points: array of points
    :return: homogeneous points
    """
    assert(len(points.shape) == 2)
    assert(points.shape[1] == 2)
    return np.hstack((points, np.ones((len(points), 1))))


def from_homogeneous_coordinates(points) -> np.ndarray:
    """
    Converts an array of homogeneous vertices back to 2D.

    :param points: array of points
    :return: array of 2D points
    """
    assert(len(points.shape) == 2)
    assert(points.shape[1] == 3)
    return points[:, 0:2]
