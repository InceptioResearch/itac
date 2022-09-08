from collections import defaultdict, OrderedDict
from typing import Union, Dict, List, Tuple
import gym
import numpy as np
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import State
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer, ZOrders
from commonroad.visualization.util import LineDataUnits
from commonroad_rl.gym_commonroad.action.vehicle import Vehicle
from commonroad_rl.gym_commonroad.observation.observation import Observation
from commonroad_rl.gym_commonroad.observation.goal_observation import GoalObservation
from commonroad_rl.gym_commonroad.utils.navigator import Navigator
from numpy import ndarray
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad_dc.geometry.util import compute_curvature_from_polyline, compute_pathlength_from_polyline
from shapely.geometry import Point, LineString

from commonroad_rl.gym_commonroad.utils.scenario import approx_orientation_vector, get_lane_marker


class LaneletNetworkObservation(Observation):
    def __init__(self, configs: Dict, configs_name: str = "lanelet_configs"):
        # Read configs
        configs = configs[configs_name]
        self.strict_off_road_check: bool = configs.get("strict_off_road_check")
        self.non_strict_check_circle_radius: float = configs.get("non_strict_check_circle_radius")

        self.observe_lat_offset: bool = configs.get("observe_lat_offset")
        self.observe_left_marker_distance: bool = configs.get("observe_left_marker_distance")
        self.observe_right_marker_distance: bool = configs.get("observe_right_marker_distance")
        self.observe_left_road_edge_distance: bool = configs.get("observe_left_road_edge_distance")
        self.observe_right_road_edge_distance: bool = configs.get("observe_right_road_edge_distance")
        self.observe_is_off_road: bool = configs.get("observe_is_off_road")
        self.observe_lane_curvature: bool = configs.get("observe_lane_curvature")

        # extrapolated_positions
        self.observe_static_extrapolated_positions: bool = configs.get("observe_static_extrapolated_positions")
        self.static_extrapolation_samples: List[float] = configs.get("static_extrapolation_samples")
        self.observe_dynamic_extrapolated_positions: bool = configs.get("observe_dynamic_extrapolated_positions")
        self.dynamic_extrapolation_samples: List[float] = configs.get("dynamic_extrapolation_samples")

        # reference path waypoints configs
        self.observe_route_reference_path: bool = configs.get("observe_route_reference_path")
        self.distances_route_reference_path: List[float] = configs.get("distances_route_reference_path")
        # reference lanelets waypoints configs
        self.observe_route_multilanelet_waypoints: bool = configs.get("observe_route_multilanelet_waypoints")
        self.distances_and_ids_multilanelet_waypoints: Tuple[List[float], List[int]] = configs.get("distances_and_ids_multilanelet_waypoints")
        # distance to reference path
        self.observe_distance_togoal_via_referencepath: bool = configs.get("observe_distance_togoal_via_referencepath")

        self.observation_dict = OrderedDict()
        self.observation_history_dict = defaultdict(list)

    def build_observation_space(self) -> OrderedDict:
        """ builds observation space for LaneletNetworkObservation """
        observation_space_dict = OrderedDict()

        if self.observe_is_off_road:
            observation_space_dict["is_off_road"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        if self.observe_left_marker_distance:
            observation_space_dict["left_marker_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_right_marker_distance:
            observation_space_dict["right_marker_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_left_road_edge_distance:
            observation_space_dict["left_road_edge_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_right_road_edge_distance:
            observation_space_dict["right_road_edge_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_lat_offset:
            observation_space_dict["lat_offset"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_lane_curvature:
            observation_space_dict["lane_curvature"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        if self.observe_dynamic_extrapolated_positions:
            sampling_points = self.dynamic_extrapolation_samples
            observation_space_dict["extrapolation_dynamic_off"] = gym.spaces.Box(-np.inf, np.inf,
                                                                                 (len(sampling_points),),
                                                                                 dtype=np.float32)
        if self.observe_static_extrapolated_positions:
            sampling_points = self.static_extrapolation_samples
            observation_space_dict["extrapolation_static_off"] = gym.spaces.Box(-np.inf, np.inf,
                                                                                (len(sampling_points),),
                                                                                dtype=np.float32)
        if self.observe_route_reference_path:
            distances = self.distances_route_reference_path
            assert sorted(set(distances)) == sorted(distances) and all(isinstance(x, int) or isinstance(x, float) for x in distances), \
                "the config setting distances_and_ids_multilanelet_waypoints[0] / distances must be a set of floats"
            observation_space_dict["route_reference_path_positions"] = gym.spaces.Box(
                -np.inf, np.inf, (len(distances) * 2,), dtype=np.float32
            )
            observation_space_dict["route_reference_path_orientations"] = gym.spaces.Box(
                -np.inf, np.inf, (len(distances),), dtype=np.float32
            )
        if self.observe_route_multilanelet_waypoints:
            distances, ids = self.distances_and_ids_multilanelet_waypoints
            assert sorted(set(distances)) == sorted(distances) and all(isinstance(x, int) or isinstance(x, float) for x in distances), \
                "the config setting distances_and_ids_multilanelet_waypoints[0] / distances must be a set of float/int"
            assert sorted(set(ids)) == sorted(ids) and all(isinstance(x, int) for x in ids), \
                "the config setting distances_and_ids_multilanelet_waypoints[1] / relative ids must be a set of ints"

            observation_space_dict["route_multilanelet_waypoints_positions"] = gym.spaces.Box(-np.inf, np.inf,
                                                                                (len(ids) * len(distances) * 2,),
                                                                                dtype=np.float32)
            observation_space_dict["route_multilanelet_waypoints_orientations"] = gym.spaces.Box(-np.inf, np.inf,
                                                                                (len(ids) * len(distances), ),
                                                                                dtype=np.float32)
        if self.observe_distance_togoal_via_referencepath:
            observation_space_dict["distance_togoal_via_referencepath"] = gym.spaces.Box(-np.inf, np.inf,
                                                                                (3,),
                                                                                dtype=np.float32)

        return observation_space_dict

    def observe(self, scenario: Scenario, ego_vehicle: Vehicle, ego_lanelet: Lanelet, road_edge: dict,
                local_ccosy: Union[None, CurvilinearCoordinateSystem] = None, navigator: Union[Navigator, None] = None
                ) -> Union[ndarray, Dict]:
        self._scenario = scenario

        # Check if the ego vehicle is off road
        is_off_road = self._check_is_off_road(ego_vehicle, road_edge)

        if self.observe_is_off_road:
            self.observation_history_dict["is_off_road"].append(is_off_road)
            self.observation_dict["is_off_road"] = np.array([is_off_road])

        if any((self.observe_left_marker_distance, self.observe_right_marker_distance,
                self.observe_left_road_edge_distance, self.observe_right_road_edge_distance)):
            self._get_distance_to_marker_and_road_edge(ego_vehicle.state, ego_lanelet, road_edge)

        # TODO: merge two functions to reduce run time
        if self.observe_lat_offset:
            self._get_lat_offset(ego_vehicle.state, local_ccosy)

        if self.observe_lane_curvature:
            self._get_lane_curvature(ego_vehicle.state.position, local_ccosy)

        # Get the relative offset of current and future positions from center vertices TODO
        if self.observe_static_extrapolated_positions:
            sampling_points = self.static_extrapolation_samples
            static_lat_offset, static_pos = self._get_relative_future_goal_offsets(ego_vehicle.state, sampling_points,
                                                                                   static=True, navigator=navigator)

            self._extrapolation_static_off = (np.array(static_lat_offset))
            self._extrapolation_static_pos = (np.array(static_pos))
            self.observation_dict["extrapolation_static_off"] = np.array(static_lat_offset)

        if self.observe_dynamic_extrapolated_positions:
            sampling_points = self.dynamic_extrapolation_samples
            dynamic_lat_offset, dynamic_pos = self._get_relative_future_goal_offsets(ego_vehicle.state, sampling_points,
                                                                                     static=False, navigator=navigator)
            self._extrapolation_dynamic_off = (np.array(dynamic_lat_offset))
            self._extrapolation_dynamic_pos = (np.array(dynamic_pos))
            self.observation_dict["extrapolation_dynamic_off"] = np.array(dynamic_lat_offset)

        # get waypoints of the reference path
        if self.observe_route_reference_path:
            pos, orient = navigator.get_waypoints_of_reference_path(
                ego_vehicle.state, distances_ref_path=self.distances_route_reference_path,
                observation_cos=Navigator.CosyVehicleObservation.AUTOMATIC
            )
            assert pos.shape == (len(self.distances_route_reference_path), 2)
            assert orient.shape == (len(self.distances_route_reference_path),)
            # assert orient.shape == (len(self.distances_route_reference_path),1), "assert works"
            self.observation_dict["route_reference_path_positions"] = pos.flatten()
            self.observation_dict["route_reference_path_orientations"] = orient.flatten()

        if self.observe_route_multilanelet_waypoints:
            distances, ids = self.distances_and_ids_multilanelet_waypoints

            pos, orient = navigator.get_referencepath_multilanelets_waypoints(
                ego_vehicle.state, distances_per_lanelet=distances, lanelets_id_rel=ids,
                observation_cos=Navigator.CosyVehicleObservation.AUTOMATIC
            )
            self.observation_dict["route_multilanelet_waypoints_positions"] = pos.flatten()
            self.observation_dict["route_multilanelet_waypoints_orientations"] = orient.flatten()
        
        if self.observe_distance_togoal_via_referencepath:
            distance_long, distance_lat, indomain = navigator.get_longlat_togoal_on_reference_path(ego_vehicle.state)
            self.observation_dict["distance_togoal_via_referencepath"] = np.array((distance_long, distance_lat, indomain))

        return self.observation_dict

    def draw(self, render_configs: Dict, render: MPRenderer, ego_vehicle: Vehicle, road_edge: Union[None, Dict] = None,
             ego_lanelet: Union[Lanelet, None] = None, navigator: Union[Navigator, None] = None):
        """ Method to draw the observation """
        # Draw road boundaries
        if render_configs["render_road_boundaries"]:
            road_edge["boundary_collision_object"].draw(render)
        # Plot ego lanelet center vertices
        if render_configs["render_ego_lanelet_center_vertices"]:
            line = LineDataUnits(ego_lanelet.center_vertices[:, 0], ego_lanelet.center_vertices[:, 1],
                                 zorder=ZOrders.LANELET_LABEL, markersize=1., color="pink",
                                 marker="x", label="Ego center vertices")
            render.dynamic_artists.append(line)

        # Extrapolated future positions
        if self.observe_static_extrapolated_positions and render_configs["render_static_extrapolated_positions"]:
            for future_pos in self._extrapolation_static_pos:
                render.dynamic_artists.append(
                    LineDataUnits(future_pos[0], future_pos[1], color="r", marker="x",
                                  zorder=21, label="static_extrapolated_positions"))

        if self.observe_dynamic_extrapolated_positions and render_configs["render_dynamic_extrapolated_positions"]:
            for future_pos in self._extrapolation_dynamic_pos:
                render.dynamic_artists.append(
                    LineDataUnits(future_pos[0], future_pos[1], color="b", marker="x",
                                  zorder=21, label="dynamic_extrapolated_positions"))

        # Plot Navigator Observations in non-local (global) CoSy
        if self.observe_route_reference_path and render_configs["render_ccosy_nav_observations"]:
            pos, _ = navigator.get_waypoints_of_reference_path(
                ego_vehicle.state, distances_ref_path=self.distances_route_reference_path,
                observation_cos=Navigator.CosyVehicleObservation.LOCALCARTESIAN
                # coordinate axes for plot in direction of global CoSy
            )
            pos_global = ego_vehicle.state.position + pos  # back to non local CoSy Origin
            render.dynamic_artists.append(
                LineDataUnits(pos_global[:, 0], pos_global[:, 1], zorder=22, marker='v',
                              color='yellow', label="ccosy_nav_observations"))

        if self.observe_route_multilanelet_waypoints and render_configs["render_ccosy_nav_observations"]:
            distances, ids = self.distances_and_ids_multilanelet_waypoints

            pos, _ = navigator.get_referencepath_multilanelets_waypoints(
                ego_vehicle.state, distances_per_lanelet=distances, lanelets_id_rel=ids,
                observation_cos=Navigator.CosyVehicleObservation.LOCALCARTESIAN
                # coordinate axes for plot in direction of global CoSy
            )
            for po in pos:
                po_global = ego_vehicle.state.position + po  # back to non local CoSy Origin
                render.dynamic_artists.append(
                    LineDataUnits(po_global[:, 0], po_global[:, 1], zorder=22, marker='v',
                                  color='purple', label="ccosy_nav_observations"))

    def _get_relative_future_goal_offsets(self, ego_state: State, sampling_points: List[float], static: bool,
                                          navigator: Navigator) -> Tuple[List[float], List[ndarray]]:
        """
        Get the relative offset of current and future positions from center vertices. Positive if left.
        For a given static extrapolation, the future position at "static" m/s after sampling_points seconds is given.
        For static = True this means the future position in exactly sampling_points meters.
        Otherwise for static = False, the future position at the current velocity after sampling_points seconds is
        given.

        :param ego_state: State of ego vehicle
        :param sampling_points: Parameter of evaluating the future position, see description above
        :param static: Curvilinear coordinate system
        :param navigator: Navigator of current planning problem
        :return: Offset of step_parameter future positions as well as the positions themselves
        """

        ego_state_orientation = ego_state.orientation if hasattr(ego_state, "orientation") else np.arctan2(
            ego_state.velocity_y, ego_state.velocity)
        v = approx_orientation_vector(ego_state_orientation) * (ego_state.velocity if static is False else 1.0)

        # quadratic steps may make sense based on the fact that braking distance is
        # proportional to the velocity squared (https://en.wikipedia.org/wiki/Braking_distance)
        positions = [ego_state.position + (v * i) for i in sampling_points]
        lat_offset = [GoalObservation.get_long_lat_distance_to_goal(p, navigator)[1] for p in positions]

        # Fix nans for positions outside the ccosy
        for i in range(len(lat_offset)):
            if np.isnan(lat_offset[i]):
                extrapolation_off = self._extrapolation_static_off if static else self._extrapolation_dynamic_off
                if len(extrapolation_off) > 0:
                    # We can assume that the ego vehicle starts inside the ccosy,
                    # however not that any extrapolated position is inside the ccosy
                    lat_offset[i] = extrapolation_off[i]
                else:
                    lat_offset[i] = 0

        return lat_offset, positions

    def _check_is_off_road(self, ego_vehicle: Vehicle, road_edge: dict) -> bool:
        """
        Check if the ego vehicle is off road.
        """
        strict_off_road_check = self.strict_off_road_check
        non_strict_check_circle_radius = self.non_strict_check_circle_radius if not self.strict_off_road_check else None

        collision_ego_vehicle = ego_vehicle.collision_object if strict_off_road_check else pycrcc.Circle(
            non_strict_check_circle_radius, ego_vehicle.state.position[0], ego_vehicle.state.position[1])

        is_off_road = collision_ego_vehicle.collide(road_edge["boundary_collision_object"])

        return is_off_road

    def _get_distance_to_marker_and_road_edge(self, ego_state: State, ego_lanelet: Lanelet, road_edge: dict):

        left_marker_line, right_marker_line = get_lane_marker(ego_lanelet)

        current_left_road_edge, current_right_road_edge = self._get_road_edge(road_edge, ego_lanelet.lanelet_id)

        (left_marker_distance, right_marker_distance, left_road_edge_distance, right_road_edge_distance) \
            = LaneletNetworkObservation.get_distance_to_marker_and_road_edge(ego_state, left_marker_line,
                                                                             right_marker_line, current_left_road_edge,
                                                                             current_right_road_edge)

        self.observation_dict["left_marker_distance"] = np.array([left_marker_distance])
        self.observation_dict["right_marker_distance"] = np.array([right_marker_distance])
        self.observation_dict["left_road_edge_distance"] = np.array([left_road_edge_distance])
        self.observation_dict["right_road_edge_distance"] = np.array([right_road_edge_distance])

    def _get_lane_curvature(self, ego_position: np.array, local_ccosy: CurvilinearCoordinateSystem):
        lane_curvature = LaneletNetworkObservation.get_lane_curvature(ego_position, local_ccosy)
        if np.isnan(lane_curvature):
            assert len(self.observation_history_dict["lane_curvature"]) > 0, \
                "Ego vehicle started outside the local coordinate system"
            lane_curvature = self.observation_history_dict["lane_curvature"][-1]
        self.observation_history_dict["lane_curvature"].append(lane_curvature)
        self.observation_dict["lane_curvature"] = np.array([lane_curvature])

    def _get_lat_offset(self, ego_state: State, local_ccosy: CurvilinearCoordinateSystem):

        lat_offset = LaneletNetworkObservation.get_relative_offset(local_ccosy, ego_state.position)
        if np.isnan(lat_offset):
            # we assume that the ego vehicle starts inside the curvi_cosy
            assert (len(self.observation_history_dict["lat_offset"]) > 0), \
                "Ego vehicle started outside the local coordinate system"
            lat_offset = self.observation_history_dict["lat_offset"][-1]

        self.observation_history_dict["lat_offset"].append(lat_offset)
        self.observation_dict["lat_offset"] = np.array([lat_offset])

    def _get_road_edge(self, road_edge: dict, ego_vehicle_lanelet_id: int) -> Tuple[LineString, LineString]:
        """
        Get the left and right road edge of ego vehicle lanelet.

        :param ego_vehicle_lanelet_id: id of ego vehicle lanelet
        :return: left and right road edge
        """
        left_most_lanelet_id = road_edge["left_road_edge_lanelet_id_dict"][ego_vehicle_lanelet_id]
        right_most_lanelet_id = road_edge["right_road_edge_lanelet_id_dict"][ego_vehicle_lanelet_id]
        left_road_edge = road_edge["left_road_edge_dict"][left_most_lanelet_id]
        right_road_edge = road_edge["right_road_edge_dict"][right_most_lanelet_id]
        return left_road_edge, right_road_edge

    @staticmethod
    def get_relative_offset(curvi_cosy: CurvilinearCoordinateSystem, position: np.ndarray) -> float:
        """
        Get the relative offset of ego vehicle from center vertices. Positive if left.

        :param curvi_cosy: curvilinear coordinate system
        :param position: The position of the ego vehicle
        :return: offset
        """
        try:
            _, ego_vehicle_lat_position = curvi_cosy.convert_to_curvilinear_coords(position[0], position[1])
        except ValueError:
            ego_vehicle_lat_position = np.nan

        return ego_vehicle_lat_position

    @staticmethod
    def get_distance_to_marker_and_road_edge(ego_vehicle_state: State, left_marker_line: LineString,
                                             right_marker_line: LineString, left_road_edge: LineString,
                                             right_road_edge: LineString, ) -> Tuple[float, float, float, float]:
        """
        Get the distance to lane markers and the road edge

        :param ego_vehicle_state: The state of the ego vehicle
        :param left_marker_line: The left marker line
        :param right_marker_line: The right marker line
        :param left_road_edge: The left road edge
        :param right_road_edge: The right road edge
        :return: Tuple of the distances to the left marker, right marker, left road edge and right road edge
        """
        ego_vehicle_point = Point(ego_vehicle_state.position[0], ego_vehicle_state.position[1])

        distance_left_marker = LaneletNetworkObservation.get_distance_point_to_linestring(ego_vehicle_point,
                                                                                          left_marker_line)
        distance_right_marker = LaneletNetworkObservation.get_distance_point_to_linestring(ego_vehicle_point,
                                                                                           right_marker_line)
        distance_left_road_edge = LaneletNetworkObservation.get_distance_point_to_linestring(ego_vehicle_point,
                                                                                             left_road_edge)
        distance_right_road_edge = LaneletNetworkObservation.get_distance_point_to_linestring(ego_vehicle_point,
                                                                                              right_road_edge)

        return distance_left_marker, distance_right_marker, distance_left_road_edge, distance_right_road_edge

    @staticmethod
    def get_distance_point_to_linestring(p: Point, line: LineString) -> float:
        """
        Get the distance of a point to the given line

        :param p: The point
        :param line: The line
        :return: The distance between the point and the line
        """
        nearest_point = line.interpolate(line.project(p))
        return nearest_point.distance(p)

    @staticmethod
    def get_lane_curvature(ego_position: np.array, ccosy: CurvilinearCoordinateSystem):
        polyline = np.array(ccosy.reference_path())
        try:
            s, _ = ccosy.convert_to_curvilinear_coords(ego_position[0], ego_position[1])
        except ValueError:
            return np.nan

        curvature = compute_curvature_from_polyline(polyline)
        ref_pos = compute_pathlength_from_polyline(polyline)

        return np.interp(s, ref_pos, curvature)


if __name__ == "__main__":
    import yaml
    from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

    config_file = PATH_PARAMS["configs"]["commonroad-v1"]
    with open(config_file, "r") as config_file:
        config = yaml.safe_load(config_file)
    configs = config["env_configs"]
    lanelet_network_observation = LaneletNetworkObservation(configs)
    print(lanelet_network_observation)
