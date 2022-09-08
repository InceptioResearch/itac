import sys

import commonroad.geometry.shape
import commonroad_dc.pycrcc as pycrcc

import commonroad_dc.collision.collision_detection.scenario
from commonroad_dc.collision.collision_detection.minkowski_sum import minkowski_sum_circle


def _create_collision_object_dict():
    collision_object_func = {}
    if 'commonroad_dc.collision.collision_detection.scenario' in sys.modules.keys():
        collision_object_func.update(
            commonroad_dc.collision.collision_detection.scenario.collision_object_func_dict)
    return collision_object_func


def _create_collision_checker_dict():
    collision_checker_func = {}
    if 'commonroad_dc.collision.collision_detection.scenario' in sys.modules.keys():
        collision_checker_func.update(
            commonroad_dc.collision.collision_detection.scenario.collision_checker_func_dict)
    return collision_checker_func


def _create_default_params():
    params = {'minkowski_sum_circle': False,
              'minkowski_sum_circle_radius': 1.0,
              'resolution': 16}
    return params


def create_collision_checker(obj, params=None, collision_checker_func=None) -> pycrcc.CollisionChecker:
    """
    Function to convert a commonroad-io scenario to a C++ collision checker.

    :param obj: the commonroad-io scenario
    :param params: optional parameters, e.g. adding the minkowski sum with a circle
    :param collision_object_func: specify the converter function (it is usually not necessary to change the default)
    :return: Returns the C++ collision checker
    """
    if params is None:
        params = _create_default_params()

    if collision_checker_func is None:
        collision_checker_func = _create_collision_checker_dict()

    return collision_checker_func[type(obj)](obj, params)


def create_collision_object(obj, params=None, collision_object_func=None) -> pycrcc.CollisionObject:
    """
    Main function to convert a commonroad-io shape to a C++ collision object.

    :param obj: the object or list of objects (with all the same type) to be converted to C++ collision objects
    :param params: optional parameters, e.g. adding the minkowski sum with a circle
    :param collision_object_func: specify the converter function (it is usually not necessary to change the default)
    :return: Returns the C++ collision object
    """

    if params is None:
        params = _create_default_params()

    if collision_object_func is None:
        collision_object_func = _create_collision_object_dict()

    if type(obj) is list:
        collision_objects = list()
        for o in obj:
            collision_objects.append(create_collision_object(o, params))
        return collision_objects
    elif isinstance(obj, commonroad.geometry.shape.Shape) \
            and params['minkowski_sum_circle'] and not isinstance(obj, commonroad.geometry.shape.ShapeGroup):
        shape = minkowski_sum_circle(obj, params['minkowski_sum_circle_radius'], params['resolution'])
        return collision_object_func[type(shape)](shape, params)
    else:
        return collision_object_func[type(obj)](obj, params)
