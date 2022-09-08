import math
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import lru_cache
from typing import Union, Any, Optional

import shapely
import shapely.geometry
from commonroad.common.solution import VehicleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.scenario.trajectory import Trajectory, State
from commonroad_dc.collision.collision_detection.scenario import create_collision_checker_scenario
from scipy.signal import savgol_filter

from commonroad_dc import pycrcc
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping
from commonroad_dc.geometry.util import chaikins_corner_cutting, resample_polyline
from commonroad_dc.pycrcc import CollisionObject, CollisionChecker, Circle, RectAABB

from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from commonroad.common.util import Interval, subtract_orientations

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from shapely.geometry import Polygon

draw_lanelet_path = True
use_shapely = True


def merge_trajectories(traj_1: Trajectory, traj_2: Trajectory):
    traj = deepcopy(traj_1)
    for s, s2 in zip(traj.state_list, traj_2.state_list):
        for attr in s.__slots__:
            if not hasattr(s, attr) and hasattr(s2, attr):
                setattr(s, attr, getattr(s2, attr))

    return traj


def smoothen_polyline(polyline, resampling_distance: float = 2.0, n_lengthen=3):
    for _ in range(3):
        polyline = np.array(chaikins_corner_cutting(polyline))

    resampled_polyline = resample_polyline(polyline, resampling_distance)

    # lengthen by n_lengthen points
    for _ in range(n_lengthen):
        resampled_polyline = np.insert(resampled_polyline, 0,
                                       2 * resampled_polyline[0] - resampled_polyline[1], axis=0)
        resampled_polyline = np.insert(resampled_polyline, len(resampled_polyline),
                                       2 * resampled_polyline[-1] - resampled_polyline[-2], axis=0)

    return resampled_polyline


def extrapolate_polyline(polyline: np.ndarray, offset: float = 10) -> np.ndarray:
    """
    Current ccosy (https://gitlab.lrz.de/cps/commonroad-curvilinear-coordinate-system/-/tree/development) creates
    wrong projection domain if polyline has large distance between waypoints --> resampling;
    initial and final points are not within projection domain -> extrapolation
    :param polyline: polyline to be used to create ccosy
    :param offset: offset of newly created vertices
    :return: extrapolated polyline
    """
    d1 = (polyline[0] - polyline[1]) / np.linalg.norm(polyline[0] - polyline[1])
    d2 = (polyline[-1] - polyline[-2]) / np.linalg.norm(polyline[-1] - polyline[-2])
    first = polyline[0] + d1 * offset
    first = first[np.newaxis]
    last = polyline[-1] + d2 * offset
    last = last[np.newaxis]

    return np.concatenate((first, polyline, last), axis=0)


def create_cosy_from_lanelet(lanelet):
    v0=lanelet.center_vertices
    v = smoothen_polyline(extrapolate_polyline(v0), resampling_distance=2.5, n_lengthen=0)
    tt = CurvilinearCoordinateSystem(v)
    return tt
    # raise ValueError

    dom = np.array(tt.projection_domain())
    plt.fill(dom[:,0], dom[:,1], fill=False)
    domc = np.array(tt.curvilinear_projection_domain())
    for ii, d in enumerate(dom):
        try:
            tt.convert_to_curvilinear_coords(d[0], d[1])
            plt.scatter(d[0], d[1], marker='x', c='r')
        except ValueError:
            plt.scatter(d[0], d[1], marker='x', c='g')

    poly = Polygon(dom)
    poly = poly.buffer(-.5)
    plt.plot(np.array(poly.exterior.coords)[:,0], np.array(poly.exterior.coords)[:,1])
    for ii, d in enumerate(np.array(poly.exterior.coords)):
        try:
            tt.convert_to_curvilinear_coords(d[0], d[1])
            plt.scatter(d[0], d[1], marker='x', c='r')
        except ValueError:
            plt.scatter(d[0], d[1], marker='x', c='g')

    plt.show(block=True)
    # print(tt)
    return tt
        # return CurvilinearCoordinateSystem(lanelet.center_vertices)

    # except RuntimeError:
    #     return None


def create_cosy_from_vertices(center_vertices):
    try:
        return CurvilinearCoordinateSystem(smoothen_polyline(extrapolate_polyline(center_vertices, 2.0)))
    except RuntimeError:
        return None


def get_orientation_at_position(cosy, position):
    try:
        s, d = cosy.convert_to_curvilinear_coords(position[0], position[1])
        tangent = cosy.tangent(s)
    except ValueError as e:
        # print(str(e))
        return None

    return np.arctan2(tangent[1], tangent[0])


def cleanup_discontinuities(positions: np.ndarray, ds_0: float, tol: float, dt: float):
    """
    remove discontinuities from a signal.
    :param signal: signal to be checked
    :param ds0: initial first order derivative of signal
    :param tol: max. absolute deviation between signal states at subsequent time steps
    :param dt: time step
    :return:
    """
    ds = ds_0
    ds2_0 = 0.0
    dt2 = 0.5 * dt ** 2
    positions_new = deepcopy(positions)
    delta_s = 0
    for i, (s_prev, s) in enumerate(zip(positions[:-1], positions[1:])):
        s_pred_new = positions_new[i] + ds * dt + ds2_0 * dt2
        s_pred     = s_prev           + ds * dt + ds2_0 * dt2
        # replace with prediction if unreasonable discontinuity
        if abs(s - s_pred) > tol:
            positions_new[i+1] = s_pred_new
            delta_s += s_pred - s
        else:
            positions_new[i + 1] = s + delta_s
            ds_prev = ds
            ds = (s - s_prev) / dt
            if i > 0:
                dt2 = (ds - ds_prev) / dt

    return positions_new


class SolutionProperties(Enum):
    AllowedVelocityInterval = "ALLOWED_VELOCITY_INTERVAL"
    LongPosition = "LONG_POSITION"
    LatPosition = "LAT_POSITION"
    LonJerk = "LON_JERK"
    LonVelocity = "LAT_VELOCITY"
    LatJerk = "LAT_JERK"
    LatVelocity = "LAT_VELOCITY"
    LonDistanceObstacles = "LON_DISTANCE_OBSTACLES"
    DeltaOrientation = "DELTA_ORIENTATION"


class CurvilinearState(State):
    __slots__ = State.__slots__ + [
        'lon_position',
        'lon_velocity',
        'lon_acceleration',
        'lon_jerk',
        'lat_position',
        'lat_velocity',
        'lat_acceleration',
        'lat_jerk',
        'delta_orientation'
    ]


class LaneletRouteMatcher:
    """
    Finds lanelet paths of vehicles' trajectories and transforms to lane-based coordinate systems.
    """
    def __init__(self, scenario: Scenario, vehicle_type: VehicleType):
        param = VehicleParameterMapping.from_vehicle_type(vehicle_type)
        self.vehicle_type = vehicle_type
        self.vehicle_param = param
        self.ego_radius = param.w / 2.5
        self.scenario: Scenario = scenario
        self.lanelet_network: LaneletNetwork = scenario.lanelet_network
        self.lanelet_cc: Union[CollisionChecker, Dict[int, shapely.geometry.Polygon]] = CollisionChecker()
        self.co2lanelet: Dict[CollisionObject, int] = {}
        self.lanelet_cc, self.co2lanelet = self._create_cc_from_lanelet_network(self.lanelet_network)
        self._lanelet_cosys = {}

    @lru_cache(1)
    def scenario_cc(self):
        return create_collision_checker_scenario(self.scenario)

    @staticmethod
    def _create_cc_from_lanelet_network(ln: LaneletNetwork) -> Tuple[CollisionChecker, Dict[CollisionObject, int]]:
        """Creates Collision Checker"""
        if use_shapely is True:
            cc = {}
        else:
            cc = CollisionChecker()
        co2lanelet: Dict[CollisionObject, int] = {}
        for l in ln.lanelets:
            poly = l.convert_to_polygon()
            # assert poly.shapely_object.is_valid
            if use_shapely is True:
                cc[l.lanelet_id] = poly.shapely_object
            else:
                co: pycrcc.Polygon = create_collision_object(poly)
                co2lanelet[co] = l.lanelet_id
                cc.add_collision_object(co)

        return cc, co2lanelet

    def find_lanelet_by_position(self, position: np.ndarray) -> List[int]:
        if use_shapely is True:
            point = shapely.geometry.Point(position).buffer(self.ego_radius, resolution=8)
            return [id_l for id_l, shape in self.lanelet_cc.items() if shape.intersects(point)]
        else:
            cc_obs = self.lanelet_cc.find_all_colliding_objects(Circle(self.ego_radius, position[0], position[1]))
            return [self.co2lanelet[o] for o in cc_obs]

    def get_lanelet_cosy(self, lanelet_id: int) -> CurvilinearCoordinateSystem:
        if lanelet_id not in self._lanelet_cosys:
            self._lanelet_cosys[lanelet_id] = create_cosy_from_lanelet(
                self.lanelet_network.find_lanelet_by_id(lanelet_id))

        return self._lanelet_cosys[lanelet_id]

    def _select_by_best_alignment(self, lanelets2states: Dict[int, List[State]],
                                  successor_candidates: List[List[int]]) -> Optional[List[int]]:
        """
        Computes mean square error of deviation of orientations compared to lanelets in successor_candidates
        :param obstacle
        :param successor_candidates_tmp:
        :return: list with ids (only those which have a feasible projection/tangent) ranked by deviation (best first)
        """
        if len(successor_candidates)==1:
            return successor_candidates[0]

        # compute tangential vectors of trajectory
        errors = {}
        # print(successor_candidates)
        for i, lanelets in enumerate(successor_candidates):
            errors_tmp = []

            v = np.concatenate(
                [self.lanelet_network.find_lanelet_by_id(l).center_vertices for l in lanelets if l is not None])
            cosy = create_cosy_from_vertices(v)
            if cosy is None:
                continue
            for l in lanelets:
                for s in lanelets2states[l]:
                    ori = get_orientation_at_position(cosy, s.position)
                    if ori is not None:
                        errors_tmp.append(subtract_orientations(s.orientation, ori))

            # compute mean square error for deviation of tangent (only if tangent was feasible)
            if len(errors_tmp) > 0:
                errors[i] = np.square(errors_tmp).mean(axis=0)
        
        try:
            best_index = sorted(errors.keys(), key=errors.get)[0]
        except IndexError:
            return None
        return successor_candidates[best_index]

    def find_lanelets_by_trajectory(self, trajectory: Trajectory, required_properties: List[SolutionProperties],
                                    exclude_oncoming_lanes=True) \
            -> Tuple[List[int], Dict[SolutionProperties, Dict[int, Any]]]:
        properties = {}

        if len(trajectory.state_list) < 1:
            return [], None

        max_dist = max(s.velocity for s in trajectory.state_list) * self.scenario.dt  # per time_step

        assert hasattr(trajectory.state_list[0], "position"), "Trajectory must have slot 'position'!"
        # find all lanelets at each time step
        lanelets = []
        lanelets2states = defaultdict(list)
        for state in trajectory.state_list:
            lanelets.append(self.find_lanelet_by_position(state.position))
            for l in lanelets[-1]:
                lanelets2states[l].append(state)

        # find sequence of lanelets considering adjacencies
        l_seq = []
        candidate_paths_next = []
        for i, l_tmp in enumerate(lanelets):
            candidate_paths = candidate_paths_next
            candidate_paths_next = []
            if not candidate_paths and l_seq:
                candidate_paths = [[l_seq[-1]]]

            if len(l_tmp) > 0:
                if candidate_paths:
                    # check for longitudinal adjacency
                    for c_path in candidate_paths:
                        if set(c_path) & set(l_tmp):
                            candidate_paths_next.append(c_path)
                            continue

                        # find successor paths that lead to one current lanelet of t_tmp
                        # (considers that short lanelets can be skipped)
                        lanelet_prev = self.lanelet_network.find_lanelet_by_id(c_path[-1])
                        successor_paths = lanelet_prev.find_lanelet_successors_in_range(self.lanelet_network,
                                                                                        max_length=max_dist)
                        in_successor_list = []
                        for succ_p in successor_paths:
                            if set(l_tmp) & set(succ_p):
                                succ_path = []
                                for s_tmp in succ_p:
                                    succ_path.append(s_tmp)
                                    if s_tmp in l_tmp:
                                        break
                                in_successor_list.append(succ_path)

                        if in_successor_list:
                            # create new candidates
                            candidate_paths_next.extend([c_path + l for l in in_successor_list])
                        else:
                            # check for lateral adjacency
                            adj = set()
                            if lanelet_prev.adj_left and lanelet_prev.adj_left_same_direction:
                                adj.add(lanelet_prev.adj_left)
                            if lanelet_prev.adj_right and lanelet_prev.adj_right_same_direction:
                                adj.add(lanelet_prev.adj_right)

                            adj_list = adj & set(l_tmp)
                            if adj_list:
                                candidate_paths_next.extend([c_path + [l] for l in adj_list])
                                continue

                else:
                    # first state -> no adjacency checks possible
                    candidate_paths_next = [[None, l] for l in l_tmp]

                if len(candidate_paths_next) == 0:
                    # leaves route (e.g. drives to diagonal successor or conducts U-turn to oncoming lane)
                    # 1. check if still in projection domain of previous cosy or successor
                    for c_path in candidate_paths:
                        if set(c_path) & set(l_tmp):
                            continue
                        if self.get_lanelet_cosy(c_path[-1]). \
                                cartesian_point_inside_projection_domain(trajectory.state_list[i].position[0],
                                                                         trajectory.state_list[i].position[1]):
                            candidate_paths_next.append(c_path)
                        else:
                            # select successor path with best alignment
                            lanelet_tmp = self.lanelet_network.find_lanelet_by_id(c_path[-1])
                            if i > 0:
                                dist = 10
                            else:
                                dist = 10 + np.linalg.norm([trajectory.state_list[i].position,
                                                            trajectory.state_list[i-1].position],
                                                           ord=np.inf)

                            successors = lanelet_tmp.find_lanelet_successors_in_range(self.lanelet_network, dist)
                            succ_candidates = []
                            for path in successors:
                                select_path = False
                                for i_l, l_id_tmp in enumerate(path):
                                    if self.get_lanelet_cosy(l_id_tmp). \
                                            cartesian_point_inside_projection_domain(
                                            trajectory.state_list[i].position[0],
                                            trajectory.state_list[i].position[1]):
                                        select_path = True
                                    else:
                                        break

                                if select_path:
                                    succ_candidates.append(c_path + path[:i_l+1])

                            if len(succ_candidates) > 0:
                                best_path = self._select_by_best_alignment(lanelets2states, succ_candidates)
                                if best_path:
                                    candidate_paths_next.append(best_path[1:])

                    if len(candidate_paths_next) == 0:
                        # still no candidate -> add by best alignement
                        if candidate_paths:
                            best_path = self._select_by_best_alignment(lanelets2states, candidate_paths)
                            if best_path:
                                l_seq.extend(best_path[1:])
                                candidate_paths_next = [[None, l] for l in l_tmp]

                if len(candidate_paths_next) == 1:
                    # only one candidate path left -> add to sequence and reset
                    l_seq.extend(candidate_paths_next[0][1:])
                    candidate_paths_next = []

            else:
                continue

            if SolutionProperties.AllowedVelocityInterval in required_properties:

                speed_intervals = {}
                # try:
                country = SupportedTrafficSignCountry(self.scenario.scenario_id.country_id)
                # except:
                tsi = TrafficSigInterpreter(country, self.lanelet_network)
                for i, l_list_tmp in enumerate(lanelets):
                    l_tmp = frozenset(l_list_tmp) & frozenset(l_seq)
                    if not l_list_tmp:
                        l_tmp = frozenset(l_list_tmp)

                    min_speed = tsi.required_speed(l_tmp)
                    min_speed = 0.0 if min_speed is None else min_speed
                    max_speed = tsi.speed_limit(l_tmp)
                    max_speed = np.inf if max_speed is None else max_speed
                    speed_intervals[i] = Interval(min_speed, max_speed)

                properties[SolutionProperties.AllowedVelocityInterval] = speed_intervals

        # check if there are candidates left and use best aligned candidate
        if candidate_paths_next:
            best_path = self._select_by_best_alignment(lanelets2states, candidate_paths_next)
            if best_path:
                l_seq.extend(best_path[1:])

        if exclude_oncoming_lanes and len(l_seq) > 1:
            # exclude oncoming lanes when switching back to previous lane
            onc_tmp = None
            delete_i = []
            lanelet_prev = self.lanelet_network.find_lanelet_by_id(l_seq[0])
            for i, l in enumerate(l_seq[1:]):
                if onc_tmp is not None:
                    if l in l_seq[:i]:
                        delete_i.append(i)
                    else:
                        onc_tmp = None

                lanlet = self.lanelet_network.find_lanelet_by_id(l)
                oncomings = []
                if lanelet_prev.adj_left and not lanelet_prev.adj_left_same_direction:
                    oncomings.append(lanelet_prev.adj_left)
                if lanelet_prev.adj_right and not lanelet_prev.adj_right_same_direction:
                    oncomings.append(lanelet_prev.adj_right)

                if l in oncomings:
                    onc_tmp = l
                else:
                    onc_tmp = None

                lanelet_prev = lanlet

            # for i_del in reversed(delete_i):
            #     del l_seq[i_del]

        return l_seq, properties

    def compute_curvilinear_coordinates(self, trajectory: Trajectory, required_properties: List[SolutionProperties],
                                        draw_lanelet_path=False, debug_plot=False,
                                        trajectory_smoothing_window = 13) \
            -> Tuple[Trajectory, List[int], Dict[SolutionProperties, Dict[int, Any]]]:
        """
        Converts trajectory to curvilinear coordinates
        :param trajectory: trajectory in cartesian coordinates
        :param required_properties: additional properties that should be retrieved
        :param draw_lanelet_path: draw the lanelet path of the reference path on which the trajectory is mapped
        :param debug_plot: create plots for debugging in case of projection errors
        :param trajectory_smoothing_window: number of time steps for smoothing filter of longitudinal positions
        :return:
        """
        lanelets, properties = self.find_lanelets_by_trajectory(trajectory, required_properties)
        cosys = []
        border_vertices = []
        lanelet = None
        lanelets_tmp = []
        next_lanelets = []

        def merge_lanelets(lanelet_list):
            lanelet_merged = lanelet_list[0]
            for l_tmp in lanelet_list[1:]:
                lanelet_merged = Lanelet.merge_lanelets(lanelet_merged, l_tmp)
            tmp = create_cosy_from_lanelet(lanelet_merged)
            cosys.append(tmp)
            border_vertices.append((lanelet_merged.center_vertices[0], lanelet_merged.center_vertices[-1]))

        for i, l in enumerate(lanelets):
            next_lanelets = []
            if lanelet is None or l in lanelet.successor:
                lanelet = self.lanelet_network.find_lanelet_by_id(l)
                lanelets_tmp.append(lanelet)
                merge = False
            else:
                lanelet = self.lanelet_network.find_lanelet_by_id(l)
                next_lanelets = [lanelet]
                merge = True

            if merge is True or i + 1 == len(lanelets):
                merge_lanelets(lanelets_tmp)
                lanelets_tmp = next_lanelets

        if next_lanelets:
            merge_lanelets(next_lanelets)

        cosy_length = {}
        for i_c in range(len(cosys)):
            cosy_length[i_c] = cosys[i_c].convert_to_curvilinear_coords(
                border_vertices[i_c][-1][0], border_vertices[i_c][-1][1])[0]

        i_c = 0
        c_tmp = cosys[i_c]
        ds = 0  # consider long. offset of coordinate system origins
        curvilinear_states = []
        s_return = None
        ghost_driving = False

        if SolutionProperties.LonDistanceObstacles in required_properties:
            properties[SolutionProperties.LonDistanceObstacles] = {}
        for i, state in enumerate(trajectory.state_list):
            s = None
            s2 = None
            d2 = None
            delta_orientation = None
            try:
                s, d = cosys[i_c].convert_to_curvilinear_coords(state.position[0], state.position[1])

            except ValueError:
                if debug_plot is True:
                    rnd = MPRenderer()
                    self.lanelet_network.draw(rnd, draw_params={'lanelet': {'show_label':True}})
                    l_tmp = LaneletNetwork.create_from_lanelet_list([self.lanelet_network._lanelets[l] for l in lanelets])
                    l_tmp.draw(draw_params={'lanelet': {'facecolor': 'red', 'show_label':True}, "traffic_sign": {
                        "draw_traffic_signs": False}, "traffic_light": {
                        "draw_traffic_lights": False}}, renderer=rnd)
                    trajectory.draw(draw_params={'time_begin': 0, 'time_end': trajectory.final_state.time_step + 1},
                                    renderer=rnd)
                    rnd.render(show=False)

                    dom = np.array(cosys[i_c].projection_domain())
                    plt.fill(dom[:, 0], dom[:, 1], fill=False, zorder=1000)
                    plt.scatter(state.position[0], state.position[1], marker='x', zorder=1000)

                    for state in trajectory.state_list:
                        plt.scatter(state.position[0], state.position[1], )
                        plt.text(state.position[0], state.position[1], s=str(state.time_step), zorder=1e6)

                    plt.axis('equal')
                    plt.title('s2 failed')
                    plt.show(block=True)

                    plt.pause(0.01)

            if i_c + 1 < len(cosys):
                try:
                    s2, d2 = cosys[i_c + 1].convert_to_curvilinear_coords(state.position[0], state.position[1])
                except ValueError:
                    pass

                if not (s is None and s2 is None):
                    if s is not None and s2 is not None:
                        tangent = cosys[i_c].tangent(s)
                        tangent2 = cosys[i_c + 1].tangent(s2)
                        delta_orientation = subtract_orientations(state.orientation,
                                                                  np.arctan2(tangent[1], tangent[0]))
                        delta_orientation2 = subtract_orientations(state.orientation,
                                                                   np.arctan2(tangent2[1], tangent2[0]))

                        ori_diff = subtract_orientations(abs(delta_orientation2),
                                                         abs(delta_orientation))
                    else:
                        ori_diff = 0

                    if s is None or (s2 is not None and (abs(d2) < abs(d) and ori_diff < 0.1)):
                        # switch to next coordinate system, once the center line is closer
                        if delta_orientation2:
                            delta_orientation = delta_orientation2
                        d = d2
                        if s is None:
                            if not ghost_driving:
                                ds += cosy_length[i_c]
                            else:
                                ds += s_return - s2
                        else:
                            if not ghost_driving:
                                ds += s - s2
                            else:
                                ds += s_return - s
                                s_return = cosy_length[i_c + 1]

                        s = s2
                        i_c += 1
                else:
                    raise RuntimeError("Vehicle out of lane!")

            if s is None and s2 is None:
                raise RuntimeError('No curvilinear system found for projection.'
                                   'Check whether the trajectory leaves the lanelet_network.')

            if not delta_orientation:
                tangent = cosys[i_c].tangent(s)
                delta_orientation = subtract_orientations(state.orientation, np.arctan2(tangent[1], tangent[0]))

            # negate s when driving against driving direction
            ghost_driving = subtract_orientations(delta_orientation, math.pi / 2) > 0.0 or subtract_orientations(
                -math.pi / 2, delta_orientation) > 0.0
            if s_return is None and ghost_driving:
                s_return = s
                ds += s_return
            elif s_return is not None and not ghost_driving:
                s_return = None

            state_tmp = CurvilinearState(**{s: getattr(state, s) for s in state.__slots__ if hasattr(state, s)})
            if s_return is None:
                state_tmp.lon_position = s + ds
            else:
                state_tmp.lon_position = ds + s_return - s
            state_tmp.lat_position = d
            state_tmp.delta_orientation = delta_orientation

            curvilinear_states.append(state_tmp)

            if SolutionProperties.LonDistanceObstacles in required_properties:
                # compute longitudinal distance to vehicles ahead in the same lane
                cc_tmp: CollisionChecker = self.scenario_cc().time_slice(state.time_step)
                # pre-filter vehicles in surrounding
                cc_tmp = cc_tmp.window_query(RectAABB(100, 100, state_tmp.position[0], state_tmp.position[1]))
                dist_lon = [np.inf]
                for obj in cc_tmp.obstacles():
                    if hasattr(obj, "center"):
                        try:
                            c = obj.center()
                            s_ego, d_ego = cosys[i_c].convert_to_curvilinear_coords(state_tmp.position[0],
                                                                                    state_tmp.position[1])
                            s_obs, d_obs = cosys[i_c].convert_to_curvilinear_coords(c[0], c[1])
                            dist = s_obs - s_ego
                            if abs(d_ego - d_obs) < 2.8 and dist > 0:
                                if self.vehicle_type.name == 'SEMI_TRAILER':
                                    t_radius = self.vehicle_param.trailer.l_total - self.vehicle_param.trailer.l_hitch
                                    dist_lon.append(dist - t_radius)
                                else:
                                    dist_lon.append(dist - self.ego_radius)
                        except:
                            continue

                properties[SolutionProperties.LonDistanceObstacles][state_tmp.time_step] = dist_lon

        # longitudinal states
        positions_long = np.array([s.lon_position for s in curvilinear_states])
        positions_long = cleanup_discontinuities(positions_long, ds_0=trajectory.state_list[0].velocity, tol=1,
                                                 dt=self.scenario.dt)

        # ensure window does not exceed trajectory length and is odd
        trajectory_smoothing_window = min(math.floor(len(positions_long) / 1.5), trajectory_smoothing_window)
        trajectory_smoothing_window -= trajectory_smoothing_window % 2 + 1
        if trajectory_smoothing_window > 4:
            positions_long = savgol_filter(positions_long, trajectory_smoothing_window, 3)

        velocities_long = np.gradient(positions_long, self.scenario.dt)
        accelerations_long = np.gradient(velocities_long, self.scenario.dt)
        jerks_long = np.gradient(accelerations_long, self.scenario.dt)

        # lateral states
        positions_lat = np.array([s.lat_position for s in curvilinear_states])
        # unwrapped lateral positions are used for derivatives only, because in the positions, we are actually
        # interested in the closest center line:
        positions_lat_unwrapped = cleanup_discontinuities(positions_lat, ds_0=0.0, tol=0.5,
                                                 dt=self.scenario.dt)
        if trajectory_smoothing_window > 4:
            positions_lat_unwrapped = savgol_filter(positions_lat_unwrapped, trajectory_smoothing_window, 3)
        velocities_lat = np.gradient(positions_lat_unwrapped, self.scenario.dt)
        accelerations_lat = np.gradient(positions_lat_unwrapped, self.scenario.dt)
        jerks_lat = np.gradient(positions_lat_unwrapped, self.scenario.dt)

        for i, c in enumerate(curvilinear_states):
            c.lon_position = positions_long[i]
            c.lon_velocity = velocities_long[i]
            c.lon_acceleration = accelerations_long[i]
            c.lon_jerk = jerks_long[i]
            c.lat_velocity = velocities_lat[i]
            c.lat_acceleration = accelerations_lat[i]
            c.lat_jerk = jerks_lat[i]

        trajectory.state_list = curvilinear_states

        if draw_lanelet_path is True:
            plt.figure(figsize=(30, 15))
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15), gridspec_kw={'width_ratios': [4, 1]})
            plt.suptitle(f"Debug <costs/compute_curvilinear_coordinates> {self.scenario.scenario_id}")

            positions_tmp = np.array([s.position for s in trajectory.state_list])
            plot_limits = [np.min(positions_tmp[:, 0], axis=0) - 50, np.max(positions_tmp[:, 0], axis=0) + 50,
                           np.min(positions_tmp[:, 1], axis=0) - 30, np.max(positions_tmp[:, 1], axis=0) + 30]

            plt.sca(ax1)
            rnd = MPRenderer(plot_limits=plot_limits)
            self.lanelet_network.draw(rnd, draw_params={'lanelet': {'show_label': True}})
            l_tmp = LaneletNetwork.create_from_lanelet_list([self.lanelet_network._lanelets[l] for l in lanelets])
            l_tmp.draw(draw_params={'lanelet': {'facecolor': 'red'}, "traffic_sign": {
                "draw_traffic_signs": False}, "traffic_light": {
                "draw_traffic_lights": False}}, renderer=rnd)
            trajectory.draw(draw_params={'time_begin': 0, 'time_end': trajectory.final_state.time_step + 1},
                            renderer=rnd)

            rnd.render(show=False)
            for s in trajectory.state_list:
                plt.scatter(s.position[0], s.position[1], )
                plt.text(s.position[0], s.position[1], s=str(s.time_step), zorder=1e6)

            ax1.title.set_text("matched lanelet route (red)")
            for l in lanelets:
                proj_dom = np.array(
                        create_cosy_from_lanelet(self.lanelet_network.find_lanelet_by_id(l)).projection_domain())
                plt.fill(proj_dom[:, 0], proj_dom[:, 1], fill=False)

            plt.sca(ax2)
            ax2.title.set_text("Lon/lat trajectories")
            maxs = max([s.lon_position for s in trajectory.state_list],)
            plt.plot([s.lon_position / maxs for s in trajectory.state_list], label="lon_position")
            plt.plot([s.lat_position for s in trajectory.state_list], label="lat_position")
            plt.plot([s.lat_jerk for s in trajectory.state_list], label="lat_jerk")
            plt.plot([s.lon_jerk for s in trajectory.state_list], label="lon_jerk")
            plt.plot([s.delta_orientation for s in trajectory.state_list], label="delta_orientation")
            plt.legend(loc="upper right")
            plt.autoscale()
            plt.close(1)
            plt.show(block=True)

        return trajectory, lanelets, properties


