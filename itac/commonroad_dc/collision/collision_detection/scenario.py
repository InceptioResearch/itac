import math
import warnings

import commonroad.geometry.shape
import commonroad.prediction
import commonroad.scenario.obstacle
import matplotlib.pyplot as plt
import numpy as np
import commonroad_dc.pycrcc as pycrcc
from commonroad.scenario.scenario import Scenario
import sys

try:
    import triangle
except:
    pass

import commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch


def create_collision_checker_scenario(scenario: Scenario, params=None, collision_object_func=None):
    cc = pycrcc.CollisionChecker()
    for co in scenario.dynamic_obstacles:
        cc.add_collision_object(commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch.
                                create_collision_object(co, params, collision_object_func))
    shape_group = pycrcc.ShapeGroup()
    for co in scenario.static_obstacles:
        collision_object = commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch. \
            create_collision_object(co, params, collision_object_func)
        if isinstance(collision_object, pycrcc.ShapeGroup):
            for shape in collision_object.unpack():
                shape_group.add_shape(shape)
        else:
            shape_group.add_shape(collision_object)
    cc.add_collision_object(shape_group)
    return cc


def create_collision_object_rectangle(rect, params=None, collision_object_func=None):
    if math.isclose(rect.orientation, 0.0):
        return pycrcc.RectAABB(
            0.5 * rect.length, 0.5 * rect.width, rect.center[0], rect.center[1])
    else:
        return pycrcc.RectOBB(0.5 * rect.length, 0.5 * rect.width,
                              rect.orientation, rect.center[0],
                              rect.center[1])


def create_collision_object_circle(circle, params=None, collision_object_func=None):
    return pycrcc.Circle(circle.radius, circle.center[0], circle.center[1])


def create_collision_object_polygon(polygon, params=None, collision_object_func=None):
    if len(polygon.vertices) == 3:
        return pycrcc.Triangle(polygon.vertices[0][0], polygon.vertices[0][1],
                               polygon.vertices[1][0], polygon.vertices[1][1],
                               polygon.vertices[2][0], polygon.vertices[2][1])
    else:

        if (type(params) is dict and params.get('triangulation_method', 'gpc') == 'triangle'):

            if 'triangle' not in sys.modules:
                raise Exception(
                    'This operation requires a non-free third-party python package triangle to be installed. It can be installed using pip (pip install triangle). Please refer to its license agreement for more details.')

            if all(np.equal(polygon.vertices[0], polygon.vertices[-1])):
                vertices = polygon.vertices[:-1]
            else:
                vertices = polygon.vertices

            # Randomly appearing segfault in triangle library if duplicate vertices
            # https://github.com/drufat/triangle/issues/2#issuecomment-583812662

            _, ind = np.unique(vertices, axis=0, return_index=True)
            ind.sort()
            vertices = vertices[ind]

            number_of_vertices = len(vertices)
            segments = list(zip(range(0, number_of_vertices - 1), range(1, number_of_vertices)))
            segments.append((0, number_of_vertices - 1))
            triangles = triangle.triangulate({'vertices': vertices, 'segments': segments}, opts='pqS2.4')
            mesh = list()
            if not 'triangles' in triangles:
                warnings.warn(f"Triangulation of polygon with vertices\n {polygon.vertices} \n not successful.",
                              stacklevel=1)
                return None
            else:
                for t in triangles['triangles']:
                    v0 = triangles['vertices'][t[0]]
                    v1 = triangles['vertices'][t[1]]
                    v2 = triangles['vertices'][t[2]]
                    mesh.append(pycrcc.Triangle(v0[0], v0[1],
                                                v1[0], v1[1],
                                                v2[0], v2[1]))

            return pycrcc.Polygon(polygon.vertices.tolist(), list(), mesh)
        else:
            return pycrcc.Polygon(polygon.vertices.tolist(), list())


def create_collision_object_shape_group(shape_group, params=None, collision_object_func=None):
    sg = pycrcc.ShapeGroup()
    for shape in shape_group.shapes:
        co = commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch.create_collision_object(
            shape, params, collision_object_func)
        if co is not None:
            sg.add_shape(co)
    return sg


def create_collision_object_static_obstacle(static_obstacle, params=None, collision_object_func=None):
    initial_time_step = static_obstacle.initial_state.time_step
    occupancy = static_obstacle.occupancy_at_time(initial_time_step)
    return commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch.create_collision_object(
        occupancy.shape, params, collision_object_func)


def create_collision_object_dynamic_obstacle(dynamic_obstacle, params=None, collision_object_func=None):
    initial_time_step = dynamic_obstacle.initial_state.time_step
    tvo = pycrcc.TimeVariantCollisionObject(initial_time_step)
    # add occupancy of initial state
    tvo.append_obstacle(commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch.create_collision_object(
        dynamic_obstacle.occupancy_at_time(initial_time_step).shape, params, collision_object_func))
    # add occupancies of prediction
    if dynamic_obstacle.prediction is not None:
        for occupancy in dynamic_obstacle.prediction.occupancy_set:
            tvo.append_obstacle(
                commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch.create_collision_object(
                    occupancy.shape, params, collision_object_func))
    return tvo


def create_collision_object_prediction(prediction, params=None, collision_object_func=None):
    tvo = pycrcc.TimeVariantCollisionObject(prediction.initial_time_step)
    for occupancy in prediction.occupancy_set:
        tvo.append_obstacle(
            commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch.create_collision_object(
                occupancy.shape, params, collision_object_func))
    return tvo


collision_object_func_dict = {
    commonroad.geometry.shape.ShapeGroup: create_collision_object_shape_group,
    commonroad.geometry.shape.Polygon: create_collision_object_polygon,
    commonroad.geometry.shape.Circle: create_collision_object_circle,
    commonroad.geometry.shape.Rectangle: create_collision_object_rectangle,
    commonroad.scenario.obstacle.StaticObstacle: create_collision_object_static_obstacle,
    commonroad.scenario.obstacle.DynamicObstacle: create_collision_object_dynamic_obstacle,
    commonroad.prediction.prediction.SetBasedPrediction: create_collision_object_prediction,
    commonroad.prediction.prediction.TrajectoryPrediction: create_collision_object_prediction
}

collision_checker_func_dict = {
    commonroad.scenario.scenario.Scenario: create_collision_checker_scenario,
}
