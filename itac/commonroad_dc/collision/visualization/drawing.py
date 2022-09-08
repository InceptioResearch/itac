from typing import *

import commonroad_dc.collision
import matplotlib.colors
import numpy as np
from commonroad.visualization.param_server import ParamServer
from commonroad.visualization.renderer import IRenderer


def draw_collision_point(obj, renderer: IRenderer,
                            draw_params: Union[ParamServer, dict, None] = None,
                            call_stack: Optional[
                                Tuple[str, ...]] = tuple()) -> None:
    call_stack += ('collision', 'point')
    renderer.draw_ellipse(obj.center, 0.125, 0.125, draw_params, call_stack)


def draw_collision_rectaabb(obj, renderer: IRenderer,
                            draw_params: Union[ParamServer, dict, None] = None,
                            call_stack: Optional[
                                Tuple[str, ...]] = tuple()) -> None:
    call_stack += ('collision', 'rectaabb')
    vertices = np.array([(obj.min_x(), obj.min_y()),
                         (obj.min_x(), obj.max_y()),
                         (obj.max_x(), obj.max_y()),
                         (obj.max_x(), obj.min_y())])
    renderer.draw_rectangle(vertices, draw_params, call_stack)


def draw_collision_rectobb(obj, renderer: IRenderer,
                           draw_params: Union[ParamServer, dict, None] = None,
                           call_stack: Optional[
                               Tuple[str, ...]] = tuple()) -> None:
    call_stack += ('collision', 'rectobb')
    center = obj.center()
    r_x = obj.r_x()
    r_y = obj.r_y()
    a_x = obj.local_x_axis()
    a_y = obj.local_y_axis()
    pt1 = center + r_x * a_x + r_y * a_y
    pt2 = center - r_x * a_x + r_y * a_y
    pt3 = center - r_x * a_x - r_y * a_y
    pt4 = center + r_x * a_x - r_y * a_y
    vertices = [(pt1[0], pt1[1]),
                (pt2[0], pt2[1]),
                (pt3[0], pt3[1]),
                (pt4[0], pt4[1])]

    renderer.draw_rectangle(vertices, draw_params, call_stack)


def draw_collision_triangle(obj, renderer: IRenderer,
                            draw_params: Union[ParamServer, dict, None] = None,
                            call_stack: Optional[
                                Tuple[str, ...]] = tuple()) -> None:
    call_stack += ('collision', 'triangle')
    v = obj.vertices()
    vertices = [(v[0][0], v[0][1]),
                (v[1][0], v[1][1]),
                (v[2][0], v[2][1])]

    renderer.draw_polygon(vertices, draw_params, call_stack)


def draw_collision_circle(obj, renderer: IRenderer,
                          draw_params: Union[ParamServer, dict, None] = None,
                          call_stack: Optional[
                              Tuple[str, ...]] = tuple()) -> None:
    call_stack += ('collision',)
    renderer.draw_ellipse([obj.x(), obj.y()], obj.r(),
                          obj.r(), draw_params, call_stack)


def draw_collision_timevariantcollisionobject(obj, renderer: IRenderer,
                                              draw_params: Union[
                                                  ParamServer, dict, None] = None,
                                              call_stack: Optional[Tuple[
                                                  str, ...]] = tuple()) -> None:
    call_stack = call_stack + ('time_variant_obstacle',)
    for i in range(obj.time_start_idx(), obj.time_end_idx() + 1):
        tmp = obj.obstacle_at_time(i)
        if draw_params is not None and "time_to_color" in draw_params.keys():
            draw_params = draw_params.copy()
            draw_params['color'] = draw_params["time_to_color"].to_rgba(i)
        tmp.draw(renderer, draw_params, call_stack)


def draw_collision_shapegroup(obj, renderer: IRenderer,
                              draw_params: Union[
                                  ParamServer, dict, None] = None,
                              call_stack: Optional[
                                  Tuple[str, ...]] = tuple()) -> None:
    call_stack = call_stack + ('shape_group',)
    for o in obj.unpack():
        o.draw(renderer, draw_params, call_stack)


def draw_collision_polygon(obj, renderer: IRenderer,
                           draw_params: Union[ParamServer, dict, None] = None,
                           call_stack: Optional[
                               Tuple[str, ...]] = tuple()) -> None:
    if draw_params is None:
        draw_params = ParamServer()
    elif isinstance(draw_params, dict):
        draw_params = ParamServer(params=draw_params)
    call_stack = call_stack + ('collision',)
    draw_mesh = draw_params.by_callstack(call_stack,
                                             ('polygon', 'draw_mesh'))
    if draw_mesh:
        for o in obj.triangle_mesh():
            o.draw(renderer, draw_params, call_stack)
    else:
        renderer.draw_polygon(obj.vertices(), draw_params, call_stack)


def draw_collision_collisionchecker(obj, renderer: IRenderer,
                                    draw_params: Union[
                                        ParamServer, dict, None] = None,
                                    call_stack: Optional[
                                        Tuple[str, ...]] = tuple()) -> None:
    call_stack = call_stack + ('collision_checker',)
    for o in obj.obstacles():
        o.draw(renderer, draw_params, call_stack)
