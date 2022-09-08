""" Provides a function to classify a trajectory based on its curvature profile
"""

import math
import os
from enum import Enum

import numpy as np
from commonroad.scenario.trajectory import Trajectory


class TrajectoryType(Enum):
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    BOTH = 4


def classify_trajectory(
    t: Trajectory, min_velocity=1.0, turn_threshold=0.03
):
    """
    Get TrajectoryType of the given trajectory
    :param t: trajectory to classify
    :param min_velocity: curvatures for states with velocity < min_velocity are clamped to 0
    :param turn_threshold: minimum curvature at any point for trajectory to be classified as a turn
    :return: a goal shape of area larger than the ego vehicle,
    """

    v = [s.velocity for s in t.state_list]
    c = np.array(_calc_curvature(t))
    c[np.abs(v) < min_velocity] = 0.0
    c = _smooth(c, 2, 13)

    traj_class = _classify_curvature(c, turn_threshold)

    return traj_class, c


def _classify_curvature(c, turn_threshold):
    """Get TrajectoryType based off curvatures"""

    min_c = min(c)
    max_c = max(c)

    is_right_turn = min_c <= -turn_threshold
    is_left_turn = max_c >= turn_threshold

    if is_left_turn and not is_right_turn:
        return TrajectoryType.LEFT
    elif is_right_turn and not is_left_turn:
        return TrajectoryType.RIGHT
    elif not is_left_turn and not is_right_turn:
        return TrajectoryType.STRAIGHT
    else:
        return TrajectoryType.BOTH


def _smooth(x, iterations=1, window=2):

    if iterations <= 0:
        return x

    x_new = x[:]
    for i in range(len(x)):
        lb = max(0, i - window)
        ub = min(len(x) - 1, i + window)
        x_new[i] = sum(x[lb:ub]) / (ub - lb)

    return _smooth(x_new, iterations - 1, window)


def _calc_derivative(x, t):
    if len(x) <= 2:
        return len(x) * [x[0]]
    dx = []
    for i in range(1, len(x) - 1):
        dx.append((x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1]))
    # pad start and end using linear extrapolation to get same length as before
    if len(dx) > 1:
        dx.append(dx[-1] + (dx[-1] - dx[-2]))
        dx.insert(0, dx[0] + (dx[0] - dx[1]))
    elif len(dx) == 1:
        dx.append(dx[-1])
        dx.append(dx[-1])
    return dx


def _calc_curvature(traj: Trajectory):

    t = list(map(lambda s: s.time_step, traj.state_list))
    x = list(map(lambda s: s.position[0], traj.state_list))
    y = list(map(lambda s: s.position[1], traj.state_list))

    x1 = _calc_derivative(x, t)
    y1 = _calc_derivative(y, t)
    x2 = _calc_derivative(x1, t)
    y2 = _calc_derivative(y1, t)

    c = map(
        lambda a: (a[0] * a[3] - a[2] * a[1]) / ((a[0] ** 2 + a[2] ** 2) ** 1.5) if (a[0]**2 + a[2] ** 2) ** 1.5 != 0
        else (a[0] * a[3] - a[2] * a[1]) / 1E-7,
        zip(x1, x2, y1, y2),
    )
    c = list(c)

    c = _smooth(c, 2, 5)

    return c