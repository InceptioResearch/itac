from typing import Union, List
import numpy as np
import enum
import commonroad_dc.pycrccosy as pycrccosy


def chaikins_corner_cutting(polyline: np.ndarray, refinements: int = 1) -> np.ndarray:
    """
    Chaikin's corner cutting algorithm to smooth a polyline by replacing each original point with two new points.
    The new points are at 1/4 and 3/4 along the way of an edge.

    :param polyline: polyline with 2D points
    :param refinements: how many times apply the chaikins corner cutting algorithm
    :return: smoothed polyline
    """
    new_polyline = pycrccosy.Util.chaikins_corner_cutting(polyline, refinements)
    return np.array(new_polyline)


def resample_polyline(polyline: np.ndarray, step: float = 2.0) -> np.ndarray:
    """
    Resamples point with equidistant spacing.

    :param polyline: polyline with 2D points
    :param step: sampling interval
    :return: resampled polyline
    """
    new_polyline = pycrccosy.Util.resample_polyline(polyline, step)
    return np.array(new_polyline)


def resample_polyline_with_length_check(polyline, length_to_check: float = 2.0):
    """
    Resamples point with length check.
    TODO: This is a helper functions to avoid exceptions during creating CurvilinearCoordinateSystem

    :param polyline: polyline with 2D points
    :return: resampled polyline
    """
    length = np.linalg.norm(polyline[-1] - polyline[0])
    if length > length_to_check:
        polyline = resample_polyline(polyline, 1.0)
    else:
        polyline = resample_polyline(polyline, length / 10.0)

    return polyline


def compute_pathlength_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the path length of a given polyline

    :param polyline: polyline with 2D points
    :return: path length of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for pathlength computation p={}'.format(polyline)
    distance = [0]
    for i in range(1, len(polyline)):
        distance.append(distance[i - 1] + np.linalg.norm(polyline[i] - polyline[i - 1]))
    return np.array(distance)


def compute_polyline_length(polyline: np.ndarray) -> float:
    """
    Computes the length of a given polyline
    :param polyline: The polyline
    :return: The path length of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:,
        0]) > 2, 'Polyline malformed for path length computation p={}'.format(polyline)

    distance_between_points = np.diff(polyline, axis=0)
    # noinspection PyTypeChecker
    return np.sum(np.sqrt(np.sum(distance_between_points ** 2, axis=1)))


def compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the curvature of a given polyline

    :param polyline: The polyline for the curvature computation
    :return: The curvature of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for curvature computation p={}'.format(polyline)

    curvature=pycrccosy.Util.compute_curvature(polyline)
    return curvature


def compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the orientation of a given polyline

    :param polyline: polyline with 2D points
    :return: orientation of polyline
    """
    assert isinstance(polyline, np.ndarray) and len(polyline) > 1 and polyline.ndim == 2 and len(polyline[0, :]) == 2, \
        'not a valid polyline. polyline = {}'.format(polyline)

    if len(polyline) < 2:
        raise NameError('Cannot create orientation from polyline of length < 2')

    orientation = []
    for i in range(0, len(polyline) - 1):
        pt1 = polyline[i]
        pt2 = polyline[i + 1]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    for i in range(len(polyline) - 1, len(polyline)):
        pt1 = polyline[i - 1]
        pt2 = polyline[i]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    return np.array(orientation)
