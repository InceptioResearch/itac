import warnings
from typing import List, Dict, Tuple, Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import math

from commonroad.geometry.shape import Rectangle
from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.lanelet import LaneletNetwork, LineMarking
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficLightState, \
    TrafficLight, \
    TrafficLightDirection
from matplotlib.lines import Line2D
from matplotlib.path import Path

__author__ = "Moritz Klischat"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = [""]
__version__ = "2022.1"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"


class LineDataUnits(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        # _dashes_data = kwargs.pop("dashes", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data  # self._dashes_data = _dashes_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72. / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def draw_polygon_as_patch(vertices, ax, zorder=5, facecolor='#ffffff',
                          edgecolor='#000000', lw=0.5,
                          alpha=1.0) -> mpl.patches.Patch:
    """
    vertices are no closed polygon (first element != last element)
    """
    verts = []
    codes = [Path.MOVETO]
    for p in vertices:
        verts.append(p)
        codes.append(Path.LINETO)
    del codes[-1]
    codes.append(Path.CLOSEPOLY)
    verts.append((0, 0))

    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                              lw=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(patch)

    return patch


def draw_polygon_collection_as_patch(vertices: List[list], ax, zorder=5,
                                     facecolor='#ffffff', edgecolor='#000000',
                                     lw=0.5, alpha=1.0,
                                     antialiased=True) -> \
        mpl.collections.Collection:
    """
    vertices are no closed polygon (first element != last element)
    """
    path_list = list()
    for v in vertices:
        verts = []
        codes = [Path.MOVETO]
        for p in v:
            verts.append(p)
            codes.append(Path.LINETO)
        del codes[-1]
        codes.append(Path.CLOSEPOLY)
        verts.append((0, 0))

        path_list.append(Path(verts, codes))
        collection_tmp = collections.PathCollection(path_list,
                                                    facecolor=facecolor,
                                                    edgecolor=edgecolor, lw=lw,
                                                    zorder=zorder, alpha=alpha,
                                                    antialiaseds=antialiased)
        collection = ax.add_collection(collection_tmp)

    return collection


def collect_center_line_colors(lanelet_network: LaneletNetwork,
                               traffic_lights: List[TrafficLight], time_step)\
        -> \
Dict[int, TrafficLightState]:
    """Collects traffic light states that each lanelet is affected by."""

    def update_state_dict(new_dict: Dict[int, TrafficLightState]):
        """ Updates state in dict. If new_state is inactive, an existing state is not overwritten."""
        for lanelet_id, new_state in new_dict.items():
            if lanelet_id in l2state:
                if new_state == TrafficLightState.INACTIVE or (
                        new_state == TrafficLightState.RED and l2state[
                    lanelet_id] == TrafficLightState.GREEN):
                    continue

            l2state[lanelet_id] = new_state

    l2int = lanelet_network.map_inc_lanelets_to_intersections
    l2state = {}
    for lanelet in lanelet_network.lanelets:
        intersection = l2int[
            lanelet.lanelet_id] if lanelet.lanelet_id in l2int else None
        for tl_id in lanelet.traffic_lights:
            tl = lanelet_network.find_traffic_light_by_id(tl_id)
            direction = tl.direction
            state = tl.get_state_at_time_step(time_step)
            if direction == TrafficLightDirection.ALL:
                update_state_dict(
                        {succ_id: state for succ_id in lanelet.successor})
            elif intersection is not None:
                inc_ele = intersection.map_incoming_lanelets[lanelet.lanelet_id]
                if direction in (
                TrafficLightDirection.RIGHT, TrafficLightDirection.LEFT_RIGHT,
                TrafficLightDirection.STRAIGHT_RIGHT):
                    update_state_dict(
                            {l: state for l in inc_ele.successors_right})
                if direction in (
                TrafficLightDirection.LEFT, TrafficLightDirection.LEFT_RIGHT,
                TrafficLightDirection.LEFT_STRAIGHT):
                    update_state_dict(
                            {l: state for l in inc_ele.successors_left})
                if direction in (TrafficLightDirection.STRAIGHT,
                                 TrafficLightDirection.STRAIGHT_RIGHT,
                                 TrafficLightDirection.LEFT_STRAIGHT):
                    update_state_dict(
                            {l: state for l in inc_ele.successors_straight})
            elif len(lanelet.successor) == 1:
                update_state_dict({lanelet.successor[0]: state})
            else:
                warnings.warn(
                    'Direction of traffic light cannot be visualized.')

    return l2state


def approximate_bounding_box_dyn_obstacles(obj: list, time_step=0) -> Union[
    Tuple[list], None]:
    """
    Compute bounding box of dynamic obstacles at time step
    :param obj: All possible objects. DynamicObstacles are filtered.
    :return:
    """

    def update_bounds(new_point: np.ndarray, bounds: List[list]):
        """Update bounds with new point"""
        if new_point[0] < bounds[0][0]:
            bounds[0][0] = new_point[0]
        if new_point[1] < bounds[1][0]:
            bounds[1][0] = new_point[1]
        if new_point[0] > bounds[0][1]:
            bounds[0][1] = new_point[0]
        if new_point[1] > bounds[1][1]:
            bounds[1][1] = new_point[1]

        return bounds

    dynamic_obstacles_filtered = []
    for o in obj:
        if type(o) == DynamicObstacle:
            dynamic_obstacles_filtered.append(o)
        elif type(o) == Scenario:
            dynamic_obstacles_filtered.extend(o.dynamic_obstacles)

    x_int = [np.inf, -np.inf]
    y_int = [np.inf, -np.inf]
    bounds = [x_int, y_int]
    shapely_set = None
    for obs in dynamic_obstacles_filtered:
        occ = obs.occupancy_at_time(time_step)
        if occ is None:
            continue
        shape = occ.shape
        if hasattr(shape, "_shapely_polygon"):
            if shapely_set is None:
                shapely_set = shape._shapely_polygon
            else:
                shapely_set = shapely_set.union(shape._shapely_polygon)
        elif hasattr(shape, 'center'):  # Rectangle, Circle
            bounds = update_bounds(shape.center, bounds=bounds)
        elif hasattr(shape, 'vertices'):  # Polygon, Triangle
            v = shape.vertices
            bounds = update_bounds(np.min(v, axis=0), bounds=bounds)
            bounds = update_bounds(np.max(v, axis=0), bounds=bounds)
    envelope_bounds = shapely_set.envelope.bounds
    envelope_bounds = np.array(envelope_bounds).reshape((2, 2))
    bounds = update_bounds(envelope_bounds[0], bounds)
    bounds = update_bounds(envelope_bounds[1], bounds)
    if np.inf in bounds[0] or -np.inf in bounds[0] or np.inf in bounds[
        1] or -np.inf in bounds[1]:
        return None
    else:
        return tuple(bounds)


def get_arrow_path_at(x, y, angle):
    """Returns path of arrow shape"""
    # direction arrow
    codes_direction = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

    scale_direction = 1.5
    pts = np.array([[0.0, -0.5, 1.0], [1.0, 0.0, 1.0], [0.0, 0.5, 1.0],
                    [0.0, -0.5, 1.0]])
    scale_m = np.array(
            [[scale_direction, 0, 0], [0, scale_direction, 0], [0, 0, 1]])
    transform = np.array([[math.cos(angle), -math.sin(angle), x],
                          [math.sin(angle), math.cos(angle), y], [0, 0, 1]])
    ptr_trans = transform.dot(scale_m.dot(pts.transpose()))
    ptr_trans = ptr_trans[0:2, :]
    ptr_trans = ptr_trans.transpose()

    path = Path(ptr_trans, codes_direction)
    return path


def colormap_idx(max_x):
    norm = mpl.colors.Normalize(vmin=0, vmax=max_x)
    colormap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    # Closure
    return lambda x: colormap.to_rgba(x)


def get_vehicle_direction_triangle(rect: Rectangle) -> np.ndarray:
    """
    :returns vertices of triangle pointing in the driving direction
    """
    l = rect.length * 0.49
    w = rect.width * 0.49
    dist = min(l+1.0, 0.65*rect.width)
    vertices = np.array([[l - dist,  w],
                         [l - dist, -w],
                         [l, 0.0]])
    return rotate_translate(vertices, rect.center, rect.orientation)

def set_non_blocking() -> None:
    """
    Ensures that interactive plotting is enabled for non-blocking plotting.

    :return: None
    """

    plt.ion()
    if not mpl.is_interactive():
        warnings.warn(
                'The current backend of matplotlib does not support '
                'interactive '
                'mode: ' + str(
                        mpl.get_backend()) + '. Select another backend with: '
                                             '\"matplotlib.use(\'TkAgg\')\"',
                UserWarning, stacklevel=3)


def line_marking_to_linestyle(line_marking: LineMarking) -> Tuple:
    """:returns: Tuple[line_style, dashes, line_width] for matplotlib
    plotting options."""
    return {
            LineMarking.DASHED:       ('--', (10, 10), 0.25,),
            LineMarking.SOLID:        ('-', (None, None), 0.25),
            LineMarking.BROAD_DASHED: ('--', (10, 10), 0.5),
            LineMarking.BROAD_SOLID:  ('-', (None, None), 0.5)
    }[line_marking]


def traffic_light_color_dict(traffic_light_state: TrafficLightState,
                             params: dict):
    """Retrieve color code for traffic light state."""
    return {
            TrafficLightState.RED:        params['red_color'],
            TrafficLightState.YELLOW:     params['yellow_color'],
            TrafficLightState.GREEN:      params['green_color'],
            TrafficLightState.RED_YELLOW: params['red_yellow_color']
    }[traffic_light_state]


def get_tangent_angle(points, rel_angle=0.0):
    vec = points[1:] - points[0:-1]
    angle = np.arctan2(vec[:, 1], vec[:, 0])
    angle -= rel_angle
    angle = np.where(angle >= 0.0, angle, angle + 2 * np.pi) * 180.0 / np.pi
    return angle
