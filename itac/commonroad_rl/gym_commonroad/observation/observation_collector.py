# """
# Module containing the observation base class
# """
import logging
import warnings
from collections import OrderedDict
from typing import Union, Dict, List, Tuple, Set

import commonroad_dc.pycrcc as pycrcc
import gym
import numpy as np
from commonroad.geometry.shape import Polygon, Rectangle, Circle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.obstacle import State, DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer, ZOrders
from commonroad.visualization.util import LineDataUnits
import commonroad_dc
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object, \
    create_collision_checker
from commonroad_dc.collision.trajectory_queries import trajectory_queries
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from commonroad_rl.gym_commonroad.action.vehicle import Vehicle
from commonroad_rl.gym_commonroad.observation.ego_observation import EgoObservation
from commonroad_rl.gym_commonroad.observation.goal_observation import GoalObservation
from commonroad_rl.gym_commonroad.observation.lanelet_network_observation import LaneletNetworkObservation
from commonroad_rl.gym_commonroad.observation.surrounding_observation import SurroundingObservation
from commonroad_rl.gym_commonroad.observation.traffic_sign_observation import TrafficSignObservation
from commonroad_rl.gym_commonroad.utils.navigator import Navigator

LOGGER = logging.getLogger(__name__)


class ObservationCollector:
    """
    This class is a wrapper for individual observation classes. It serves as an access point for the observations.
    Currently, this class supports the GoalObservation, SurroundingObservation, LaneletNetworkObservation,
    EgoObservation.

    :param configs: dictionary with the parameters of the configuration
        should include a individual dictionary for the individual observation classes

    Examples::

        observation_collector = ObservationCollector(configs)
        # build gym observation space
        observation_space = observation_collector._build_observation_space()
        # reset and observe
        observation_collector.reset()
        observation = observation_collector.observe(ego_vehicle)
    """

    def __init__(self, configs: Dict):
        self._scenario: Scenario = None
        self._goal_region: GoalRegion = None
        self._planning_problem: PlanningProblem = None
        self.observation_space_size: int = None
        self._flatten_observation = configs.get("flatten_observation")
        self._max_lane_merge_range = configs.get("max_lane_merge_range")
        self.observation_dict = OrderedDict()

        self.ego_observation = EgoObservation(configs)
        self.goal_observation = GoalObservation(configs)
        self.surrounding_observation = SurroundingObservation(configs)
        self.lanelet_network_observation = LaneletNetworkObservation(configs)
        self.traffic_sign_observation = TrafficSignObservation(configs)

        self._road_edge = dict()
        self._cache_lanelet_polygons_accel_struct = dict()
        self._cache_lanelet_polygons_sg_accel_struct = dict()
        self._cache_scenario_ref_path_dict = dict()
        self._connected_lanelet_dict: Dict[int: Set[int]] = dict()
        self._cache_collision_checker_templates: Dict[str: pycrcc.CollisionChecker] = dict()

        self.observation_space = self._build_observation_space()

        self.navigator: Navigator = None
        self._use_cache_navigator = configs.get("cache_navigators", False)
        self._continous_collision_check = configs["action_configs"].get("continuous_collision_checking", True)
        self._cache_navigator = dict()
        self.episode_length = None

    def _build_observation_space(self) -> Union[gym.spaces.Box, gym.spaces.Dict]:
        """
        builds the observation space dictionary

        :return: the function returns an OrderedDict with the observation spaces of each observation as an entry
        """
        observation_space_dict = OrderedDict()
        observation_space_dict.update(self.goal_observation.build_observation_space())
        observation_space_dict.update(self.ego_observation.build_observation_space())
        observation_space_dict.update(self.surrounding_observation.build_observation_space())
        observation_space_dict.update(self.lanelet_network_observation.build_observation_space())
        observation_space_dict.update(self.traffic_sign_observation.build_observation_space())

        self.observation_space_dict = observation_space_dict
        if self._flatten_observation:
            lower_bounds, upper_bounds = np.array([]), np.array([])
            for space in observation_space_dict.values():
                lower_bounds = np.concatenate((lower_bounds, space.low))
                upper_bounds = np.concatenate((upper_bounds, space.high))
            self.observation_space_size = lower_bounds.shape[0]
            observation_space = gym.spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.float64)
            LOGGER.debug(f"Size of flattened observation space: {self.observation_space_size}")
        else:
            observation_space = gym.spaces.Dict(self.observation_space_dict)
            LOGGER.debug(f"Length of dictionary observation space: {len(self.observation_space_dict)}")

        return observation_space

    def reset(self, scenario: Scenario, planning_problem: PlanningProblem, reset_config: dict, benchmark_id: str,
              clone_collision_checker: bool = True):

        self.time_step = 0
        self._scenario = scenario
        self._planning_problem = planning_problem
        if self._planning_problem is not None:
            self._goal_region: GoalRegion = planning_problem.goal
            self.episode_length = max(s.time_step.end for s in self._goal_region.state_list)
        else:
            self._goal_region = None
            self.episode_length = None
        self._benchmark_id = benchmark_id
        self._update_collision_checker(clone_collision_checker=clone_collision_checker)

        self.ego_lanelet_id = None
        self.ego_lanelet = None

        self.navigator = None

        # dictionary is not set anywhere
        if self._cache_scenario_ref_path_dict.get(str(self._scenario.scenario_id), None) is None:
            self._cache_scenario_ref_path_dict[str(self._scenario.scenario_id)] = dict()

        self._connected_lanelet_dict: Dict[int: Set[int]] = reset_config["connected_lanelet_dict"]

        self._road_edge = {
            "left_road_edge_lanelet_id_dict": reset_config["left_road_edge_lanelet_id_dict"],
            "right_road_edge_lanelet_id_dict": reset_config["right_road_edge_lanelet_id_dict"],
            "left_road_edge_dict": reset_config["left_road_edge_dict"],
            "right_road_edge_dict": reset_config["right_road_edge_dict"],
            "boundary_collision_object": reset_config["boundary_collision_object"],
        }

    @staticmethod
    def compute_convex_hull_circle(radius, previous_position, current_position) -> pycrcc.RectOBB:
        """ Compute obb based on last and current position to
            approximate the area covered by the collision circle between
            the last and current timestep.
        """
        position = (current_position + previous_position) / 2.0
        direction = current_position - previous_position
        direction_length = np.linalg.norm(direction)
        d_normed = direction / direction_length
        orientation = np.arctan2(d_normed[1], d_normed[0])

        return pycrcc.RectOBB(direction_length / 2 + radius, radius, orientation, position[0], position[1])

    @staticmethod
    def create_convex_hull_collision_circle(dynamic_obstacle: DynamicObstacle):
        assert isinstance(dynamic_obstacle.obstacle_shape, Circle)

        initial_time_step = dynamic_obstacle.initial_state.time_step
        tvo = pycrcc.TimeVariantCollisionObject(initial_time_step)
        if dynamic_obstacle.prediction is not None:
            for time_step in range(initial_time_step, dynamic_obstacle.prediction.final_time_step):
                previous_state = dynamic_obstacle.state_at_time(time_step)
                state = dynamic_obstacle.state_at_time(time_step + 1)
                convex_obb = ObservationCollector.compute_convex_hull_circle(
                    dynamic_obstacle.obstacle_shape.radius, previous_state.position, state.position)
                tvo.append_obstacle(convex_obb)
        else: # SUMO scenario
            tvo.append_obstacle(create_collision_object(dynamic_obstacle))

        return tvo

    @staticmethod
    def create_collision_checker_scenario(scenario: Scenario, params=None, collision_object_func=None, continous_collision_check=True):
        if not continous_collision_check:
            return create_collision_checker(scenario)
        cc = pycrcc.CollisionChecker()
        for co in scenario.dynamic_obstacles:
            if isinstance(co.obstacle_shape, Rectangle):
                collision_object = create_collision_object(co, params, collision_object_func)
                if co.prediction is not None:
                # TODO: remove if when https://gitlab.lrz.de/cps/commonroad-drivability-checker/-/issues/16 is fixed
                    collision_object, err = trajectory_queries.trajectory_preprocess_obb_sum(collision_object)
                    if err:
                        raise Exception("<ObservationCollector.create_collision_checker_scenario> Error create convex hull")
            elif isinstance(co.obstacle_shape, Circle):
                collision_object = ObservationCollector.create_convex_hull_collision_circle(co)
            else:
                raise NotImplementedError(f"Unsupported shape for convex hull collision object: {co.obstacle_shape}")
            cc.add_collision_object(collision_object)

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

    def _update_collision_checker(self, collision_checker=None, clone_collision_checker: bool = True):

        if collision_checker is None:
            if clone_collision_checker:
                full_reset = False
                cc_template = self._cache_collision_checker_templates.get(self._benchmark_id, None)
                if cc_template is None:
                    cc_template = self.create_collision_checker_scenario(
                        self._scenario, continous_collision_check=self._continous_collision_check)
                    if not full_reset:
                        self._cache_collision_checker_templates[self._benchmark_id] = cc_template
                self._collision_checker = cc_template.clone()
            else:
                self._collision_checker = self.create_collision_checker_scenario(
                    self._scenario, continous_collision_check=self._continous_collision_check)
        else:
            self._collision_checker = collision_checker

    def get_collision_checker(self):
        return self._collision_checker

    def observe(self, ego_vehicle: Vehicle) -> Union[np.array, Dict]:
        # initialize observation_dict
        observation_dict = OrderedDict()
        self._ego_vehicle = ego_vehicle
        self._ego_state = ego_vehicle.state
        self.time_step = self._ego_state.time_step
        # TODO: only valid for car env
        self._update_ego_lanelet_and_local_ccosy()
        self._create_navigator()

        # calculate all observations
        observation_dict_ego = self.ego_observation.observe(self.ego_lanelet, ego_vehicle, self.episode_length)
        observation_dict_goal = self.goal_observation.observe(self._ego_state, goal=self._goal_region,
                                                              scenario=self._scenario,
                                                              planning_problem=self._planning_problem,
                                                              ego_lanelet_ids=self._ego_lanelet_ids,
                                                              navigator=self.navigator,
                                                              episode_length=self.episode_length,
                                                              local_ccosy=self.local_ccosy)

        observation_dict_surrounding, ego_vehicle_lat_position = self.surrounding_observation.observe(
            self._scenario,
            ego_vehicle,
            self.time_step,
            self._connected_lanelet_dict,
            self.ego_lanelet,
            self._collision_checker,
            self.local_ccosy,
            self._ego_lanelet_ids)
        observation_dict_lanelet = self.lanelet_network_observation.observe(self._scenario, ego_vehicle,
                                                                            self.ego_lanelet, self._road_edge,
                                                                            self.local_ccosy, self.navigator)
        observation_dict_traffic_sign = self.traffic_sign_observation.observe(self._scenario, ego_vehicle,
                                                                              self.ego_lanelet, self.local_ccosy)

        observation_dict.update(observation_dict_ego)
        observation_dict.update(observation_dict_surrounding)
        observation_dict.update(observation_dict_lanelet)
        observation_dict.update(observation_dict_goal)
        observation_dict.update(observation_dict_traffic_sign)

        assert len(list(observation_dict.keys())) == len(list(self.observation_space_dict.keys()))
        self.observation_dict = OrderedDict((k, observation_dict[k]) for k in self.observation_space_dict.keys())
        assert list(self.observation_dict.keys()) == list(self.observation_space_dict.keys())

        if self._flatten_observation:
            observation_vector = np.zeros(self.observation_space.shape)
            index = 0
            for k in self.observation_dict.keys():
                size = np.prod(self.observation_dict[k].shape)
                observation_vector[index: index + size] = self.observation_dict[k].flat
                index += size
            return observation_vector
        else:
            return self.observation_dict

    def _update_ego_lanelet_and_local_ccosy(self):
        # update ego lanelet
        lanelet_polygons, lanelet_polygons_sg = self._get_lanelet_polygons(str(self._scenario.scenario_id))
        self._ego_lanelet_ids = self.sorted_lanelets_by_state(self._scenario, self._ego_state, lanelet_polygons,
                                                              lanelet_polygons_sg)
        if len(self._ego_lanelet_ids) == 0:
            ego_lanelet_id = self.ego_lanelet_id
        else:
            self.ego_lanelet_id = ego_lanelet_id = self._ego_lanelet_ids[0]

        self.ego_lanelet = self._scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        ref_path_dict = self._cache_scenario_ref_path_dict[str(self._scenario.scenario_id)]

        # update local ccosy
        self.local_ccosy, self._local_merged_lanelet = self.get_local_curvi_cosy(self._scenario.lanelet_network,
                                                                                 ego_lanelet_id, ref_path_dict,
                                                                                 self._max_lane_merge_range)

    def render(self, render_configs: Dict, render: MPRenderer):
        self.lanelet_network_observation.draw(render_configs, render,
                                              ego_vehicle=self._ego_vehicle, road_edge=self._road_edge,
                                              ego_lanelet=self.ego_lanelet, navigator=self.navigator)
        self.goal_observation.draw(render_configs, render, self.navigator)
        self.surrounding_observation.draw(render_configs, render)
        # Plot ccosys
        if render_configs["render_local_ccosy"]:
            # draw_params = ParamServer(
            #     {"lanelet":
            #         {"unique_colors": False,
            #         "draw_stop_line": False, "stop_line_color": "#ffffff", "draw_line_markings": False,
            #         "draw_left_bound": False, "draw_right_bound": False, "draw_center_bound": False,
            #         "draw_border_vertices": False, "draw_start_and_direction": False, "show_label": False,
            #         "draw_linewidth": 0.5, "fill_lanelet": True, "facecolor": "wheat"},
            #      "traffic_sign": {"draw_traffic_signs": False, "show_traffic_signs": "all", "show_label": True}
            #      }
            # )
            # render.draw_list(self._local_merged_lanelet, draw_params=draw_params)
            # # TODO: temporary fix for missing draw method in Lanelet, remove after fixed in cr-io
            # from commonroad.scenario.lanelet import LaneletNetwork
            # LaneletNetwork.create_from_lanelet_list([self._local_merged_lanelet]).draw(render, draw_params=draw_params)
            # self._local_merged_lanelet.draw(render, draw_params=draw_params)
            reference_path_local_ccosy = np.array(self.local_ccosy.reference_path())
            line = LineDataUnits(reference_path_local_ccosy[:, 0], reference_path_local_ccosy[:, 1],
                                 zorder=ZOrders.LANELET_LABEL, markersize=1., color="green",
                                 marker="x", label="Local CCOSy")
            render.dynamic_artists.append(line)

    def _get_lanelet_polygons(self, meta_scenario_id) -> Tuple[List[Tuple[int, Polygon]], pycrcc.ShapeGroup]:
        """
        returns lanelet_polygons and the shape group of the polygons
        # TODO: fix meta_scenario_id
        :param meta_scenario_id: id of the current meta scenario
        :return:
            lanelet_polygons: List of tuples for each lanelet the tuple contains the lanelet id and the corresponding
                polygon
            lanelet+polygons_sg: ShapeGroup of all the polygons of all the lanelets
        """
        # TODO: pre store polygon for all maps
        lanelet_polygons = self._cache_lanelet_polygons_accel_struct.get(meta_scenario_id, None)
        lanelet_polygons_sg = self._cache_lanelet_polygons_sg_accel_struct.get(meta_scenario_id, None)

        if lanelet_polygons is None:
            lanelet_polygons = [(lanelet.lanelet_id, lanelet.convert_to_polygon()) for lanelet in
                                self._scenario.lanelet_network.lanelets]
            lanelet_polygons_sg = pycrcc.ShapeGroup()
            for l_id, poly in lanelet_polygons:
                lanelet_polygons_sg.add_shape(create_collision_object(poly))
            self._cache_lanelet_polygons_sg_accel_struct[meta_scenario_id] = lanelet_polygons_sg
            self._cache_lanelet_polygons_accel_struct[meta_scenario_id] = lanelet_polygons
        return lanelet_polygons, lanelet_polygons_sg

    @staticmethod
    def sorted_lanelets_by_state(scenario: Scenario, state: State, lanelet_polygons: list,
                                 lanelet_polygons_sg: pycrcc.ShapeGroup) -> List[int]:
        """
        Returns the sorted list of lanelet ids which correspond to a given state

        :param scenario: The scenario to be used
        :param state: The state which lanelets ids are searched
        :param lanelet_polygons: The polygons of the lanelets
        :param lanelet_polygons_sg: Thy pycrcc polygons of the lanelets
        :return: The list of lanelet ids sorted by relative orientations, the nearest lanelet is the first elements
        """
        return Navigator.sorted_lanelet_ids(
            ObservationCollector._related_lanelets_by_state(state, lanelet_polygons, lanelet_polygons_sg),
            state.orientation if hasattr(state, "orientation") else np.arctan2(state.velocity_y, state.velocity),
            state.position, scenario, )

    @staticmethod
    def _related_lanelets_by_state(state: State, lanelet_polygons: List[Tuple[int, Polygon]],
                                   lanelet_polygons_sg: pycrcc.ShapeGroup) -> List[int]:
        """
        Get the lanelet of a state.

        :param state: The state to which the related lanelets should be found
        :param lanelet_polygons: The polygons of the lanelets
        :param lanelet_polygons_sg: The pycrcc polygons of the lanelets
        :return: The list of lanelet ids
        """
        # output list
        res = list()

        # look at each lanelet
        point_list = [state.position]

        point_sg = pycrcc.ShapeGroup()
        for el in point_list:
            point_sg.add_shape(pycrcc.Point(el[0], el[1]))

        lanelet_polygon_ids = point_sg.overlap_map(lanelet_polygons_sg)

        for lanelet_id_list in lanelet_polygon_ids.values():
            for lanelet_id in lanelet_id_list:
                res.append(lanelet_polygons[lanelet_id][0])

        return res

    @staticmethod
    def get_local_curvi_cosy(lanelet_network: LaneletNetwork, ego_vehicle_lanelet_id: int,
                             ref_path_dict: Dict[str, Tuple[np.ndarray or None, Lanelet or None]],
                             max_lane_merge_range: float) -> Tuple[CurvilinearCoordinateSystem, Lanelet]:
        """
        At every time step, update the local curvilinear coordinate system from the dict.

        :param lanelet_network: The lanelet network
        :param ego_vehicle_lanelet_id: The lanelet id where the ego vehicle is on
        :param ref_path_dict: The dictionary of the reference path, contains the paths by the starting lanelet ids
        :param max_lane_merge_range: Maximum range of lanes to be merged
        :return: Curvilinear coordinate system of the merged lanelets
        """
        if ref_path_dict is None:
            ref_path_dict = dict()

        ref_path, ref_merged_lanelet, curvi_cosy = ref_path_dict.get(ego_vehicle_lanelet_id, (None, None, None))

        if curvi_cosy is None:
            for lanelet in lanelet_network.lanelets:  # iterate in all lanelet in this scenario
                if lanelet.lanelet_id == ego_vehicle_lanelet_id and (
                        not lanelet.predecessor and not lanelet.successor):  # the lanelet is a lane itself
                    ref_path = lanelet.center_vertices
                    ref_merged_lanelet = lanelet
                elif not lanelet.predecessor:  # the lanelet is the start of a lane, the lane can be created from here
                    # TODO: cache merged lanelets in pickle or dict
                    merged_lanelet_list, sub_lanelet_ids_list = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                        lanelet, lanelet_network, max_lane_merge_range)
                    for merged_lanelet, sub_lanelet_ids in zip(merged_lanelet_list, sub_lanelet_ids_list):
                        if ego_vehicle_lanelet_id in sub_lanelet_ids:
                            ref_path = merged_lanelet.center_vertices
                            ref_merged_lanelet = merged_lanelet
                            break
            # TODO: Idea, the reference path dict could be updated
            #  on all successor of the current lanelet for optimization
            curvi_cosy = Navigator.create_coordinate_system_from_polyline(ref_path)
            ref_path_dict[ego_vehicle_lanelet_id] = (ref_path, ref_merged_lanelet, curvi_cosy)

        return curvi_cosy, ref_merged_lanelet

    def _create_navigator(self) -> Navigator:
        """
        creates and stores the Navigator of the current scenario

        """

        if not self.navigator:

            # Check cache first
            if self._use_cache_navigator:
                key = ObservationCollector.get_navigator_cache_key(self._scenario, self._planning_problem)
                if key in self._cache_navigator:
                    self.navigator = self._cache_navigator[key]
                    return self._cache_navigator[key]

            # not found in cache --> create new navigator
            route_planner = RoutePlanner(self._scenario, self._planning_problem,
                                         backend=RoutePlanner.Backend.NETWORKX_REVERSED, log_to_console=False, )

            route_candidates = route_planner.plan_routes()
            route = route_candidates.retrieve_best_route_by_orientation()

            navigator = Navigator(route)

            if self._use_cache_navigator:
                self._cache_navigator[key] = navigator

            self.navigator = navigator
            return navigator

    @staticmethod
    def get_navigator_cache_key(scenario: Scenario, planning_problem: PlanningProblem):
        return str(scenario.scenario_id) + '#' + str(planning_problem.planning_problem_id)


if __name__ == "__main__":
    import yaml
    from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

    config_file = PATH_PARAMS["configs"]["commonroad-v1"]
    with open(config_file, "r") as config_file:
        config = yaml.safe_load(config_file)
    configs = config["env_configs"]
    observation_collector = ObservationCollector(configs)
    print(observation_collector)
