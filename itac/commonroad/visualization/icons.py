"""Module for drawing obstacle icons."""
from typing import Union

import matplotlib as mpl
import numpy as np
import math

from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.obstacle import ObstacleType

__author__ = "Simon Sagmeister"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2022.1"
__maintainer__ = "Luis Gressenbuch"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"


def _obstacle_icon_assignment():
    """Assign obstacle type to icon."""
    assign_dict = {ObstacleType.CAR: draw_car_icon, ObstacleType.PARKED_VEHICLE: draw_car_icon,
                   ObstacleType.TAXI: draw_car_icon, ObstacleType.TRUCK: draw_truck_icon,
                   ObstacleType.BUS: draw_bus_icon, ObstacleType.BICYCLE: draw_bicycle_icon, }

    return assign_dict


def supported_icons():
    """Return a list of obstacle types, that have a icon."""
    return list(_obstacle_icon_assignment().keys())


def get_obstacle_icon_patch(obstacle_type: ObstacleType, pos_x: Union[int, float], pos_y: Union[int, float],
                            orientation: Union[int, float], vehicle_length: Union[int, float],
                            vehicle_width: Union[int, float], zorder: float = 5, vehicle_color: str = "#ffffff",
                            edgecolor="black", lw=0.5, ):
    """Get a list of mpl.patches to draw a obstacle specific icon."""
    if obstacle_type not in supported_icons():
        error_string = (f"There is no icon available for vehicle type: {str(obstacle_type)}\n\nEnsure to call the "
                        f"get_obstacle_icon_patch(...) function\nonly for vehicle types supported.\nThese can be "
                        f"retrieved by "
                        f"calling commonroad.visualization.icons.supported_icons()")
        raise TypeError(error_string)
    draw_func = _obstacle_icon_assignment()[obstacle_type]
    patch = draw_func(pos_x=pos_x, pos_y=pos_y, orientation=orientation, vehicle_length=vehicle_length,
                      vehicle_width=vehicle_width, zorder=zorder, vehicle_color=vehicle_color, edgecolor=edgecolor,
                      lw=lw, )
    return patch


def _transform_to_global(vertices: list, pos_x: Union[int, float], pos_y: Union[int, float],
                         orientation: Union[int, float], vehicle_length: Union[int, float],
                         vehicle_width: Union[int, float], ):
    """Transform absolute coordinate to car-relative coordinate.

    Args:
        vertices: Shape: (N,2)
        pos_x: -
        pos_y: -
        orientation: -
        vehicle_length: -
        vehicle_width: -

    Returns:
        np_array: transformed absolute coordinate in the form (x,y) (shape: (N,2))
    """
    # Norm the array
    vertices = np.array(vertices)
    vertices = vertices * 0.01
    # Scale it to vehicle dim
    vertices[:, 0] = vertices[:, 0] * vehicle_length
    vertices[:, 1] = vertices[:, 1] * vehicle_width
    # Preprocess current pos
    curr_pos = np.array([pos_x, pos_y])
    vertices = rotate_translate(vertices, curr_pos, orientation)
    return vertices


def draw_bus_icon(pos_x: Union[int, float], pos_y: Union[int, float], orientation: Union[int, float],
                  vehicle_length: Union[int, float] = 12, vehicle_width: Union[int, float] = 2.5, zorder: float = 5,
                  vehicle_color: str = "#ffffff", edgecolor="black", lw=0.5, ):
    """Return the patches of the truck icon.

    Define vertices in a normed rectangle.
    -50 <= x <= 50 and -50 <= y <= 50
    """
    window_color = edgecolor

    outline = np.array([[-50, -50], [50, -50], [50, 50], [-50, 50]])
    front_window = np.array([[47, -42], [50, -46], [50, 46], [47, 42]])
    right_window = np.array([[-20, -50], [-15, -42], [40, -42], [45, -50]])
    left_window = np.array([[-20, 50], [-15, 42], [40, 42], [45, 50]])
    roof_hatch = np.array([[-40, -27], [-15, -27], [-15, 27], [-40, 27]])
    hatch_circles = [[-35, 0], [-27.5, 0], [-20, 0]]
    roof_line = np.array([[-7, -27], [-7, 27]])
    bus_list = [outline, roof_hatch, roof_line]
    window_list = [front_window, right_window, left_window]

    bus_list = [_transform_to_global(vertices=part, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
                                     vehicle_length=vehicle_length, vehicle_width=vehicle_width, ) for part in bus_list]
    window_list = [_transform_to_global(vertices=window, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
                                        vehicle_length=vehicle_length, vehicle_width=vehicle_width, ) for window in
                   window_list]
    hatch_circles = _transform_to_global(vertices=hatch_circles, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
                                         vehicle_length=vehicle_length, vehicle_width=vehicle_width, )

    bus_list_patches = [mpl.patches.Polygon(part, fc=vehicle_color, ec=edgecolor, lw=lw, zorder=zorder, closed=True, )
                        for part in bus_list]
    window_list_patches = [
        mpl.patches.Polygon(window, fc=window_color, ec=edgecolor, lw=lw, zorder=zorder + 1, closed=True, ) for window
        in window_list]
    hatch_circle_patches = [
        mpl.patches.Circle(point, radius=vehicle_length * 2.5 / 100, facecolor=vehicle_color, zorder=zorder + 1,
                           linewidth=lw, edgecolor=edgecolor, ) for point in hatch_circles]

    return bus_list_patches + window_list_patches + hatch_circle_patches


def draw_truck_icon(pos_x: Union[int, float], pos_y: Union[int, float], orientation: Union[int, float],
                    vehicle_length: Union[int, float] = 10, vehicle_width: Union[int, float] = 2.5, zorder: float = 5,
                    vehicle_color: str = "#ffffff", edgecolor="black", lw=0.5, ):
    """Return the patches of the truck icon.

    Define vertices in a normed rectangle.
    -50 <= x <= 50 and -50 <= y <= 50

    Credits to Tobias Geißenberger for defining the vertices.
    """
    # region Define your points in the norm square (-50<=x<=50, -50<=y<=50)
    # x -> length |  y -> width
    v_trailer = np.array([[-50, -46], [20, -46], [20, 46], [-50, 46]])
    v_driver_cabin = np.array([[25, -42], [50, -42], [50, 42], [25, 42]])
    v_roof = np.array([[25, -34], [44, -34], [44, 34], [25, 34]])
    v_a_col_l = np.array([v_roof[2], v_driver_cabin[2]])
    v_a_col_r = np.array([v_roof[1], v_driver_cabin[1]])
    v_connection = np.array([v_trailer[2], [v_driver_cabin[3][0], v_driver_cabin[3][1] - 3],
                             [v_driver_cabin[0][0], v_driver_cabin[0][1] + 3], v_trailer[1], ])
    v_mirror_l = np.array([[43, 42], [41, 42], [41, 50], [43, 50]])
    v_mirror_r = np.array([[43, -42], [41, -42], [41, -50], [43, -50]])
    # endregion

    # Transform your coords
    truck = [v_trailer, v_driver_cabin, v_roof, v_a_col_l, v_a_col_r, v_connection, v_mirror_l, v_mirror_r, ]
    truck = [_transform_to_global(vertices=part, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
                                  vehicle_length=vehicle_length, vehicle_width=vehicle_width, ) for part in truck]
    patch_list = [mpl.patches.Polygon(part, fc=vehicle_color, ec=edgecolor, lw=lw, zorder=zorder, closed=True) for part
                  in truck]

    return patch_list


def draw_bicycle_icon(pos_x: Union[int, float], pos_y: Union[int, float], orientation: Union[int, float],
                      vehicle_length: Union[int, float] = 2.5, vehicle_width: Union[int, float] = 0.8,
                      zorder: float = 5, vehicle_color: str = "#ffffff", edgecolor="black", lw=0.5, ):
    """Return the patches of the truck icon.

    Define vertices in a normed rectangle.
    -50 <= x <= 50 and -50 <= y <= 50

    Credits to Tobias Geißenberger for defining the vertices.
    """

    def elliptic_arc(center, major, minor, start_angle, end_angle):
        """Create the vertices of an elliptic arc."""
        arc = []
        angle_list = np.linspace(start_angle, end_angle, 50)
        for angle in angle_list:
            arc.append([center[0] + major * math.cos(angle), center[1] + minor * math.sin(angle)])

        return np.array(arc)

    # region Define your points in the norm square (-50<=x<=50, -50<=y<=50)
    # x -> length |  y -> width
    v_front_wheel = elliptic_arc((30, 0), 20, 6, 0, 2 * np.pi)
    v_rear_wheel = elliptic_arc((-30, 0), 20, 6, 0, 2 * np.pi)
    v_handlebar = np.array([[18, 50], [16, 50], [16, -50], [18, -50]])
    v_frame = np.array([[18, 3], [18, -3], [-30, -3], [-30, 3]])
    v_body = elliptic_arc((5, 0), 20, 40, np.pi / 2 + 0.2, np.pi * 3 / 2 - 0.2)
    v_arm_r = np.array([v_body[-1], v_handlebar[3], [v_handlebar[3][0], v_handlebar[3][1] + 7.5],
                        [v_body[-1][0], v_body[-1][1] + 15], ])
    v_arm_l = np.array([[v_body[0][0], v_body[0][1] - 15], [v_handlebar[0][0], v_handlebar[0][1] - 7.5], v_handlebar[0],
                        v_body[0], ])
    v_body = np.concatenate([v_body, v_arm_r, v_arm_l])
    v_head = elliptic_arc((3, 0), 6, 15, 0, 2 * np.pi)
    # endregion

    # Transform your coords
    list_bicycle = [v_front_wheel, v_frame, v_rear_wheel, v_handlebar, v_body, v_head]

    list_bicycle = [_transform_to_global(vertices=part, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
                                         vehicle_length=vehicle_length, vehicle_width=vehicle_width, ) for part in
                    list_bicycle]
    patch_list = [mpl.patches.Polygon(part, fc=vehicle_color, ec=edgecolor, lw=lw, zorder=zorder, closed=True) for part
                  in list_bicycle]

    # Return this patch collection
    return patch_list


def draw_car_icon(pos_x: Union[int, float], pos_y: Union[int, float], orientation: Union[int, float],
                  vehicle_length: Union[int, float] = 5, vehicle_width: Union[int, float] = 2, zorder: float = 5,
                  vehicle_color: str = "#ffffff", edgecolor="black", lw=0.5, ):
    """Return the patches of the car icon.

    Define vertices in a normed rectangle.
    -50 <= x <= 50 and -50 <= y <= 50
    """
    window_color = edgecolor

    front_window = np.array(
            [[-21.36, -38.33], [-23.93, -27.66], [-24.98, -12.88], [-25.28, -0.3], [-25.29, -0.3], [-25.28, -0.04],
             [-25.29, 0.22], [-25.28, 0.22], [-24.98, 12.8], [-23.93, 27.58], [-21.36, 38.24], [-14.65, 36.18],
             [-7.64, 33.19], [-8.32, 19.16], [-8.62, -0.04], [-8.32, -19.24], [-7.64, -33.27], [-14.65, -36.27], ])

    rear_window = np.array(
            [[37.68, -34.02], [26.22, -32.15], [27.43, -14.56], [27.8, -0.41], [27.43, 13.74], [26.22, 31.32],
             [37.68, 33.19], [40.17, 21.22], [41.3, -0.34], [40.17, -21.91], [40.17, -21.91], ])

    left_window = np.array(
            [[4.32, -38.7], [25.84, -37.76], [27.35, -36.27], [15.06, -32.71], [-0.1, -32.71], [-13.6, -37.95],
             [0.84, -38.78], ])

    left_mirror = np.array(
            [[-12.62, -49.78], [-13.3, -50.0], [-15.11, -46.63], [-16.78, -41.24], [-17.23, -39.56], [-14.92, -39.45],
             [-14.52, -40.68], [-13.97, -41.47], ])

    engine_hood = np.array(
            [[-21.67, -38.04], [-32.98, -34.96], [-40.1, -29.77], [-46.78, -18.96], [-49.04, 2.65], [-46.78, 19.35],
             [-40.33, 29.6], [-32.98, 35.35], [-21.67, 38.44], ])

    right_window = np.array(
            [[4.32, 38.7], [25.84, 37.76], [27.35, 36.27], [15.06, 32.71], [-0.1, 32.71], [-13.6, 37.95],
             [0.84, 38.78], ])

    right_mirror = np.array(
            [[-12.62, 49.78], [-13.3, 50.0], [-15.11, 46.63], [-16.78, 41.24], [-17.23, 39.56], [-14.92, 39.45],
             [-14.52, 40.68], [-13.97, 41.47], ])

    outline = np.array(
            [[0.78, -45.23], [-38.09, -42.38], [-45.85, -36.08], [-49.16, -15.15], [-49.99, 1.79], [-50.0, 1.79],
             [-50.0, 2.0], [-50.0, 2.22], [-49.99, 2.22], [-49.16, 14.1], [-45.85, 35.03], [-38.09, 41.33],
             [0.78, 44.18], [30.15, 42.88], [44.88, 37.96], [47.6, 32.77], [49.58, 14.36], [50.0, 3.86], [50.0, 0.14],
             [49.58, -15.41], [47.6, -33.82], [44.88, -39.01], [30.15, -43.93], ])

    windows = [-front_window, -rear_window, -left_window, -right_window]
    car = [-outline, -left_mirror, -right_mirror, -engine_hood]

    windows = [_transform_to_global(vertices=window, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
                                    vehicle_length=vehicle_length, vehicle_width=vehicle_width, ) for window in windows]
    car = [_transform_to_global(vertices=part, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
                                vehicle_length=vehicle_length, vehicle_width=vehicle_width, ) for part in car]

    window_patches = [
        mpl.patches.Polygon(window, fc=window_color, ec=edgecolor, lw=lw, zorder=zorder + 1, closed=True, ) for window
        in windows]
    car_patches = [mpl.patches.Polygon(part, fc=vehicle_color, ec=edgecolor, lw=lw, zorder=zorder, closed=True) for part
                   in car]

    return car_patches + window_patches
