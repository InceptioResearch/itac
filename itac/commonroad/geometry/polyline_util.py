import math
from typing import List

import numpy as np
from shapely.geometry import LineString

from commonroad.common.validity import is_valid_polyline


def compute_polyline_lengths(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the path lengths of a given polyline in steps travelled
    from initial to final coordinate.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Path lengths of the polyline for each coordinate in m
    """
    assert is_valid_polyline(polyline), "Polyline p={} is malformed!".format(polyline)

    distance = [0]
    for i in range(1, len(polyline)):
        distance.append(distance[i - 1] + np.linalg.norm(polyline[i] - polyline[i - 1]))

    return np.array(distance)


def compute_total_polyline_length(polyline: np.ndarray) -> float:
    """
    Computes the complete path length of a given polyline.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Path length of the polyline [m]
    """
    lengths = compute_polyline_lengths(polyline)

    return float(lengths[-1])


def compute_polyline_curvatures(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the curvatures along a given polyline travelled from initial
    to final coordinate.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Curvatures of the polyline for each coordinate [1/rad]
    """
    assert is_valid_polyline(polyline) and len(polyline) >= 3, "Polyline p={} is malformed!".format(polyline)

    x_d = np.gradient(polyline[:, 0])
    x_dd = np.gradient(x_d)
    y_d = np.gradient(polyline[:, 1])
    y_dd = np.gradient(y_d)

    return (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))


def compute_polyline_orientations(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the orientation of a given polyline travelled from initial
    to final coordinate. The orientation of the last coordinate is always
    assigned with the computed orientation of the penultimate one.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Orientations of the polyline for each coordinate [rad]
    """
    assert is_valid_polyline(polyline), "Polyline p={} is malformed!".format(polyline)

    orientation = []
    for i in range(0, len(polyline) - 1):
        pt_1 = polyline[i]
        pt_2 = polyline[i + 1]
        tmp = pt_2 - pt_1
        orient = np.arctan2(tmp[1], tmp[0])
        orientation.append(orient)
        if i == len(polyline) - 2:
            orientation.append(orient)

    return np.array(orientation)


def compute_polyline_initial_orientation(polyline: np.ndarray) -> float:
    """
    Computes the orientation of the initial coordinate with respect to the succeeding
    coordinate.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Orientation of the initial coordinate [rad]
    """
    orientations = compute_polyline_orientations(polyline)

    return orientations[0]


def is_point_on_polyline(polyline: np.ndarray, point: np.ndarray) -> bool:
    """
    Computes whether a given point lies on a polyline. That means, the point is between the starting
    and ending point of the polyline.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :param point: 2D point [x, y]
    :return: Boolean indicating whether point lies on polyline or not
    """
    assert is_valid_polyline(polyline), "Polyline p={} is malformed!".format(polyline)

    for i in range(0, len(polyline) - 1):
        l_x_1 = polyline[i][0]
        l_y_1 = polyline[i][1]
        l_x_2 = polyline[i + 1][0]
        l_y_2 = polyline[i + 1][1]

        cross_product = (point[1] - l_y_1) * (l_x_2 - l_x_1) - (point[0] - l_x_1) * (l_y_2 - l_y_1)

        epsilon = 1e-12
        if abs(cross_product) > epsilon:
            continue

        dot_product = (point[0] - l_x_1) * (l_x_2 - l_x_1) + (point[1] - l_y_1) * (l_y_2 - l_y_1)
        if dot_product < 0:
            continue

        squared_length = (l_x_2 - l_x_1) * (l_x_2 - l_x_1) + (l_y_2 - l_y_1) * (l_y_2 - l_y_1)
        if dot_product > squared_length:
            continue

        return True

    return False


def compute_polyline_intersections(polyline_1: np.ndarray, polyline_2: np.ndarray) -> np.ndarray:
    """
    Computes the intersection points of two polylines.

    :param polyline_1: First polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :param polyline_2: Second polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Intersection points
    """
    assert is_valid_polyline(polyline_1), "First polyline p={} is malformed!".format(polyline_1)
    assert is_valid_polyline(polyline_2), "Second polyline p={} is malformed!".format(polyline_2)

    intersection_points = []

    for i in range(0, len(polyline_1) - 1):
        for j in range(0, len(polyline_2) - 1):

            x_diff = [polyline_1[i][0] - polyline_1[i + 1][0], polyline_2[j][0] - polyline_2[j + 1][0]]
            y_diff = [polyline_1[i][1] - polyline_1[i + 1][1], polyline_2[j][1] - polyline_2[j + 1][1]]

            div = np.linalg.det(np.array([x_diff, y_diff]))
            if math.isclose(div, 0.0):
                continue

            d_1 = np.linalg.det(
                np.array([[polyline_1[i][0], polyline_1[i][1]], [polyline_1[i + 1][0], polyline_1[i + 1][1]]]))
            d_2 = np.linalg.det(
                np.array([[polyline_2[j][0], polyline_2[j][1]], [polyline_2[j + 1][0], polyline_2[j + 1][1]]]))
            d = [d_1, d_2]
            x = np.linalg.det(np.array([d, x_diff])) / div
            y = np.linalg.det(np.array([d, y_diff])) / div

            point = np.array([x, y])
            between_polyline_1 = is_point_on_polyline(np.array([polyline_1[i], polyline_1[i + 1]]), point)
            between_polyline_2 = is_point_on_polyline(np.array([polyline_2[j], polyline_2[j + 1]]), point)
            if [x, y] not in intersection_points and between_polyline_1 and between_polyline_2:
                intersection_points.append([x, y])

    return np.array(intersection_points)


def is_polyline_self_intersection(polyline: np.ndarray) -> bool:
    """
    Computes whether the given polyline contains self-intersection. Intersection
    at boundary points are considered as self-intersection.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Self-intersection or not
    """
    assert is_valid_polyline(polyline), "Polyline p={} is malformed!".format(polyline)

    line = [(x, y) for x, y in polyline]
    line_string = LineString(line)

    return not line_string.is_simple


def compare_polylines_equality(polyline_1: np.ndarray, polyline_2: np.ndarray, threshold=1e-10) -> bool:
    """
    Compares two polylines for equality. For equality of the values a threshold can be given.

    :param polyline_1: First polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :param polyline_2: Second polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :param threshold: Threshold for equality of values
    :return: Equality of both polylines or not
    """
    assert is_valid_polyline(polyline_1), "First polyline p={} is malformed!".format(polyline_1)
    assert is_valid_polyline(polyline_2), "Second polyline p={} is malformed!".format(polyline_2)

    return np.allclose(polyline_1, polyline_2, rtol=threshold, atol=threshold)


def resample_polyline_with_number(polyline: np.ndarray, number: int) -> np.ndarray:
    """
    Resamples the given polyline with a fixed number of points. The number
    of coordinates can be resampled down or up.
    There exists also an efficient C++ implementation with Python interface in the CommonRoad Drivability Checker.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :param number: Fixed number of 2D points
    :return: Resampled polyline
    """
    assert is_valid_polyline(polyline), "Polyline p={} is malformed!".format(polyline)
    assert number > 1, 'Number n={} has to be at least two'.format(number)

    line = LineString(polyline)
    dists = np.linspace(0, line.length, number)
    polyline_resampled = []
    for d in dists:
        point = line.interpolate(d)
        polyline_resampled.append([point.x, point.y])

    return np.array(polyline_resampled)


def resample_polyline_with_distance(polyline: np.ndarray, distance: float) -> np.ndarray:
    """
    Resamples the given polyline with a specific distance. For a higher distance than the length
    of the given polyline the polyline is not resampled.
    There exists also an efficient C++ implementation with Python interface in the CommonRoad Drivability Checker.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :param distance: Specific distance [m]
    :return: Resampled polyline
    """
    assert is_valid_polyline(polyline), "Polyline p={} is malformed!".format(polyline)
    assert distance > 0, 'Distance d={} has to be greater than 0'.format(distance)

    line = LineString(polyline)
    dists = np.arange(0, line.length, distance)
    polyline_resampled = []
    for d in dists:
        point = line.interpolate(d)
        polyline_resampled.append([point.x, point.y])

    last_point = polyline[-1]
    polyline_resampled.append(last_point)

    return np.array(polyline_resampled)


def equalize_polyline_length(long_polyline: np.ndarray, short_polyline: np.ndarray) -> np.ndarray:
    """
    Inserts vertices into a polyline to be of the same length than other polyline.

    :param long_polyline: Polyline with higher number of vertices
    :param short_polyline: Polyline with lower number of vertices
    :return: Short polyline with equal length as long polyline
    """
    assert is_valid_polyline(long_polyline), "Long polyline p={} is malformed!".format(long_polyline)
    assert is_valid_polyline(short_polyline), "Short polyline p={} is malformed!".format(short_polyline)
    assert len(long_polyline) > len(short_polyline), "The number of vertices of long polyline p={} must be greater " \
                                                     "compared to short polyline p={}!".format(long_polyline,
                                                                                               short_polyline)

    path_length_long = compute_polyline_lengths(long_polyline)
    path_length_percentage_long = path_length_long / path_length_long[-1]
    if len(short_polyline) > 2:
        path_length_short = compute_polyline_lengths(short_polyline)
        path_length_percentage_short = path_length_short / path_length_short[-1]
    else:
        path_length_percentage_short = [0, 1]

    index_mapping = create_indices_mapping(path_length_percentage_long, path_length_percentage_short)

    org_polyline = short_polyline
    last_key = 0
    counter = 0
    for key, value in enumerate(index_mapping):
        if value == -1:
            counter += 1
        elif counter > 0 and value > -1:
            ub = org_polyline[value]
            lb = short_polyline[last_key]
            for idx in range(1, counter + 1):
                insertion_factor = \
                    (path_length_percentage_long[last_key + idx] - path_length_percentage_long[last_key]) / \
                    (path_length_percentage_long[key] - path_length_percentage_long[last_key])
                new_vertex = insertion_factor * (ub - lb) + lb
                short_polyline_updated = np.insert(short_polyline, last_key + idx, new_vertex, 0)
                short_polyline = short_polyline_updated
            last_key = key
            counter = 0
        else:
            last_key = key
    return short_polyline


def create_indices_mapping(path_length_percentage_long: np.ndarray,
                           path_length_percentage_short: np.ndarray) -> List[int]:
    """
    Extracts places (indices) where new vertices have to be added in shorter polyline.
    Helper function for insert_vertices

    :param path_length_percentage_long: Proportional path length along longer polyline
    :param path_length_percentage_short: Proportional path length along shorter polyline
    :return: Mapping of existing indices of shorter polyline to longer polyline
    """
    path_length_percentages = [path_length_percentage_long, path_length_percentage_short]
    path_length_percentage_names = ['Long path length percentage', 'Short path length percentage']
    for i in range(0, len(path_length_percentages)):
        path_length_percentage = path_length_percentages[i]
        path_length_percentage_name = path_length_percentage_names[i]
        valid_path_percentage = True
        p = 0
        for percentage in path_length_percentage:
            if p > percentage:
                valid_path_percentage = False
                break
            p = percentage
        assert valid_path_percentage and len(path_length_percentage) > 1, \
            str(path_length_percentage_name) + " p={} is malformed!".format(path_length_percentage)
    assert len(path_length_percentage_long) > len(path_length_percentage_short), \
        "The number of long path length percentage p={} must be greater compared to the number of short path length " \
        "percentage p={}!".format(path_length_percentage_long, path_length_percentage_short)

    index_mapping = [-1] * len(path_length_percentage_long)
    index_mapping[0] = 0
    index_mapping[-1] = len(path_length_percentage_short) - 1

    finished = False

    last_idx_long = 1
    for key in range(1, len(path_length_percentage_short) - 1):
        value = path_length_percentage_short[key]
        threshold = 0.01
        while key not in index_mapping and not finished:
            for idx_long in range(last_idx_long,
                                  len(path_length_percentage_long) - (len(path_length_percentage_short) - key) + 1):
                if abs(path_length_percentage_long[idx_long] - value) < threshold and index_mapping[idx_long] == -1:
                    index_mapping[idx_long] = key
                    last_idx_long = idx_long
                    if len(path_length_percentage_short) - key + 1 == len(index_mapping) - idx_long + 1:
                        for idx in range(idx_long + 1, len(index_mapping)):
                            index_mapping[idx] = key + 1
                            key += 1
                        finished = True
                    break
            threshold *= 2
        if finished:
            break

    return index_mapping


def concatenate_polylines(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """
    Concatenates two polylines. The head represents the first part of the new polyline.

    :param head: First part of new polyline
    :param tail: Second part of new polyline
    """
    assert is_valid_polyline(head), "Left polyline p={} is malformed!".format(head)
    assert is_valid_polyline(tail), "Right polyline p={} is malformed!".format(tail)

    return np.concatenate((head, tail), axis=0)
