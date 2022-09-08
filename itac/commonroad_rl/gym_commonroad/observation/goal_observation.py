import warnings
from collections import OrderedDict
from typing import Union, Dict, List, Tuple

import gym
import numpy as np
from commonroad.geometry.shape import ShapeGroup, Shape
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import State
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer
from commonroad_rl.gym_commonroad.observation.observation import Observation
from commonroad_rl.gym_commonroad.utils.navigator import Navigator
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

class GoalObservation(Observation):
    """
    This class contains all helper methods and the main observation method for observations related to the goal

    :param configs: the configuration specification
    """

    def __init__(self, configs: Dict, config_name: str = "goal_configs"):
        # Read configs
        configs = configs[config_name]
        self.relax_is_goal_reached: bool = configs.get("relax_is_goal_reached")

        self.observe_distance_goal_long: bool = configs.get("observe_distance_goal_long")
        self.observe_distance_goal_lat: bool = configs.get("observe_distance_goal_lat")
        self.observe_distance_goal_long_lane: bool = configs.get("observe_distance_goal_long_lane")
        self.observe_distance_goal_time: bool = configs.get("observe_distance_goal_time")
        self.observe_distance_goal_orientation: bool = configs.get("observe_distance_goal_orientation")
        self.observe_distance_goal_velocity: bool = configs.get("observe_distance_goal_velocity")
        self.observe_euclidean_distance: bool = configs.get("observe_euclidean_distance")

        self.observe_is_time_out = configs.get("observe_is_time_out")
        self.observe_is_goal_reached = configs.get("observe_is_goal_reached")

        # location for storing the past observations
        self.observation_history_dict: dict = dict()

    def build_observation_space(self) -> OrderedDict:
        observation_space_dict = OrderedDict()

        if self.observe_euclidean_distance:
            observation_space_dict["euclidean_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        if self.observe_distance_goal_long:
            observation_space_dict["distance_goal_long"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            observation_space_dict["distance_goal_long_advance"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                  dtype=np.float32)
        if self.observe_distance_goal_lat:
            observation_space_dict["distance_goal_lat"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            observation_space_dict["distance_goal_lat_advance"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                 dtype=np.float32)
        if self.observe_distance_goal_long_lane:
            observation_space_dict["distance_goal_long_lane"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        if self.observe_distance_goal_time:
            observation_space_dict["distance_goal_time"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_distance_goal_orientation:
            observation_space_dict["distance_goal_orientation"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                 dtype=np.float32)
        if self.observe_distance_goal_velocity:
            observation_space_dict["distance_goal_velocity"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        if self.observe_is_goal_reached:
            observation_space_dict["is_goal_reached"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)

        if self.observe_is_time_out:
            observation_space_dict["is_time_out"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)

        return observation_space_dict

    def observe(self, ego_state: State, goal: Union[GoalRegion, None] = None,
                scenario: Union[Scenario, None] = None, planning_problem: Union[PlanningProblem, None] = None,
                ego_lanelet_ids: Union[List[int], None] = None, navigator: Union[Navigator, None] = None,
                episode_length: int = None,local_ccosy: Union[None, CurvilinearCoordinateSystem] = None) \
            -> Union[np.array, Dict]:
        """ Create goal related observation for given state in an environment.

            :param ego_state: state from which to observe the environment
            :param goal: goal region
            :param scenario: current Scenario
            :param planning_problem: the current planning problem
            :return: ndarray of observation if flatten == True, observation dict otherwise
        """
        observation_dict = {}

        if self.observe_euclidean_distance:
            distance = GoalObservation._get_goal_euclidean_distance(ego_state.position, goal)
            observation_dict['euclidean_distance'] = np.array([distance])
        # observe distance goal long and lat and also the advance versions
        if self.observe_distance_goal_long or self.observe_distance_goal_lat:
            distance_goal_long, distance_goal_lat = self.get_long_lat_distance_to_goal(ego_state.position, navigator)

            observation_dict["distance_goal_long"] = np.array([distance_goal_long])
            observation_dict["distance_goal_lat"] = np.array([distance_goal_lat])

            if distance_goal_long is np.nan:
                distance_goal_long = self.observation_history_dict.get("distance_goal_long", 1e4)
                distance_goal_lat = self.observation_history_dict.get("distance_goal_lat", 1e4)
            (distance_goal_long_advance, distance_goal_lat_advance) = \
                self._get_long_lat_distance_advance_to_goal(distance_goal_long, distance_goal_lat)

            self.observation_history_dict["distance_goal_long_advance"] = distance_goal_long_advance
            self.observation_history_dict["distance_goal_lat_advance"] = distance_goal_lat_advance
            observation_dict["distance_goal_long_advance"] = np.array([distance_goal_long_advance])
            observation_dict["distance_goal_lat_advance"] = np.array([distance_goal_lat_advance])

            self.observation_history_dict["distance_goal_long"] = distance_goal_long
            self.observation_history_dict["distance_goal_lat"] = distance_goal_lat

        # observe the time to the goal state
        if self.observe_distance_goal_time:
            distance_goal_time = GoalObservation._get_goal_time_distance(ego_state.time_step, goal)
            observation_dict["distance_goal_time"] = np.array([distance_goal_time])
        # observe distance to the goal orientation from ego
        if self.observe_distance_goal_orientation:
            ego_state_orientation = ego_state.orientation if hasattr(ego_state, "orientation") else \
                np.arctan2(ego_state.velocity_y, ego_state.velocity)
            distance_goal_orientation = GoalObservation._get_goal_orientation_distance(ego_state_orientation, goal)
            observation_dict["distance_goal_orientation"] = np.array([distance_goal_orientation])
        # observe difference between ego velocity and goal velocity
        if self.observe_distance_goal_velocity:
            distance_goal_velocity = GoalObservation._get_goal_velocity_distance(ego_state.velocity, goal)
            observation_dict["distance_goal_velocity"] = np.array([distance_goal_velocity])

        # Get the longitudinal distance until the lane change must be finished
        if self.observe_distance_goal_long_lane:
            distance_goal_long_lane = self._get_long_distance_until_lane_change(ego_state, ego_lanelet_ids,
                                                                                navigator)
            self.observation_history_dict["distance_goal_long_lane"] = distance_goal_long_lane
            observation_dict["distance_goal_long_lane"] = np.array([distance_goal_long_lane])

        # Check if the ego vehicle has reached the goal
        if self.observe_is_goal_reached or self.observe_is_time_out:
            is_goal_reached = self._check_goal_reached(goal, ego_state, self.relax_is_goal_reached)
        if self.observe_is_goal_reached:
            observation_dict["is_goal_reached"] = np.array([is_goal_reached])
        # Check if maximum episode length exceeded
        if self.observe_is_time_out:
            is_time_out = GoalObservation._check_is_time_out(ego_state, goal, is_goal_reached, episode_length)
            if not is_time_out:
                # check if ego vehicle reaches the end of the road
                if not local_ccosy.cartesian_point_inside_projection_domain(ego_state.position[0], ego_state.position[1]):
                    is_time_out=True
            observation_dict["is_time_out"] = np.array([is_time_out])

        return observation_dict

    def draw(self, render_configs: Dict, render: MPRenderer, navigator: Union[None, Navigator]):
        """ Method to draw the observation """
        if render_configs["render_global_ccosy"]:
            # TODO: This functionality has been taken from commonroad-route-planner
            # As soon as the route-planner supports drawing only the ccosy, this part should be replaced
            draw_params = ParamServer(
                {"lanelet": {"center_bound_color": "#128c01",
                             "unique_colors": False,
                             "draw_stop_line": True,
                             "stop_line_color": "#ffffff",
                             "draw_line_markings": False,
                             "draw_left_bound": False,
                             "draw_right_bound": False,
                             "draw_center_bound": True,
                             "draw_border_vertices": False,
                             "draw_start_and_direction": False,
                             "show_label": False,
                             "draw_linewidth": 1,
                             "fill_lanelet": False,
                             "facecolor": "#128c01",
                             }
                 }
            )
            # TODO: temporary fix for missing draw method in Lanelet, remove after fixed in cr-io
            from commonroad.scenario.lanelet import LaneletNetwork
            LaneletNetwork.create_from_lanelet_list(navigator.merged_route_lanelets).draw(render, draw_params=draw_params)
            # for route_merged_lanelet in navigator.merged_route_lanelets:
            #     route_merged_lanelet.draw(render, draw_params=draw_params)

    @staticmethod
    def _get_goal_euclidean_distance(position: np.array, goal: GoalRegion) -> float:
        """
        calculates the euclidean distance of the current position to the goal

        :param position: current position
        :param goal: the goal of the current planning problem
        :return euclidean distance
        """
        if "position" not in goal.state_list[0].attributes:
            return 0.

        else:
            f_pos = goal.state_list[0].position
            if isinstance(f_pos, ShapeGroup):
                goal_position_list = np.array(
                    [GoalObservation._convert_shape_group_to_center(s.position) for s in goal.state_list])
            elif isinstance(f_pos, Shape):
                goal_position_list = np.array([s.position.center for s in goal.state_list])
            else:
                warnings.warn(f"Trying to calculate relative goal orientation but goal state position "
                              f"type ({type(f_pos)}) is not support, please set "
                              f"observe_distance_goal_euclidean = False or "
                              f"change state position type to one of the following: Polygon, Rectangle, Circle")
                return 0.
            goal_position_mean = np.mean(goal_position_list, axis=0)
            return np.linalg.norm(position - goal_position_mean)

    @staticmethod
    def _convert_shape_group_to_center(shape_group: ShapeGroup):
        position_list = [shape.center for shape in shape_group.shapes]
        return np.mean(np.array(position_list), axis=0)

    def _get_long_lat_distance_advance_to_goal(self, distance_goal_long: float,
                                               distance_goal_lat: float) -> Tuple[float, float]:
        """
        Get longitudinal and lateral distances to the goal over the planned route

        :param distance_goal_long: the current distance_goal_long observation
        :param distance_goal_lat: the current distance_goal_lat observation

        :return: The tuple of the longitudinal and the lateral distance advances
        """
        if "distance_goal_long" not in self.observation_history_dict or not self.observe_distance_goal_long:
            distance_goal_long_advance = 0.0
        else:
            distance_goal_long_advance = abs(self.observation_history_dict["distance_goal_long"]) - abs(
                distance_goal_long)

        if "distance_goal_lat" not in self.observation_history_dict or not self.observe_distance_goal_lat:
            distance_goal_lat_advance = 0.0
        else:
            distance_goal_lat_advance = abs(self.observation_history_dict["distance_goal_lat"]) - abs(distance_goal_lat)

        return distance_goal_long_advance, distance_goal_lat_advance

    @staticmethod
    def _get_goal_velocity_distance(velocity: float, goal: GoalRegion) -> float:
        """
        calculates the difference to the goal velocity

        calculates velocity - goal_velocity_interval_start    if velocity < goal_velocity_interval_start
                   velocity - goal_velocity_interval_end    if velocity > goal_velocity_interval_end
                   0                                          else

        :param velocity: velocity of current state
        :param goal: GoalRegion of current planning problem
        :return difference to the nearest goal velocity boundary
        """
        if "velocity" not in goal.state_list[0].attributes:
            return 0.

        else:
            velocity_start_list = np.array([s.velocity.start for s in goal.state_list])
            velocity_end_list = np.array([s.velocity.end for s in goal.state_list])
            goal_velocity_interval_start: float = np.squeeze(np.mean(velocity_start_list))
            goal_velocity_interval_end: float = np.squeeze(np.mean(velocity_end_list))
            if velocity < goal_velocity_interval_start:
                return velocity - goal_velocity_interval_start
            elif velocity > goal_velocity_interval_end:
                return velocity - goal_velocity_interval_end
            else:
                return 0.

    @staticmethod
    def _get_goal_orientation_distance(orientation: float, goal: GoalRegion) -> float:
        """
        calculate the distance of the current vehicle orientation to the goal

        calculates orientation - goal_orientation_interval_start    if orientation < goal_orientation_interval_start
                   orientation - goal_orientation_interval_end      if orientation > goal_orientation_interval_start
                   0                                                else

        :param orientation: orientation of current state
        :param goal: GoalRegion of current planning problem
        :return difference to the nearest goal orientation boundary using radians in interval [-pi,pi]
        """
        if "orientation" not in goal.state_list[0].attributes:
            return 0.

        else:
            orientation_start_list = np.array([s.orientation.start for s in goal.state_list])
            orientation_end_list = np.array([s.orientation.end for s in goal.state_list])
            goal_orientation_interval_start: float = np.mean(orientation_start_list) % (2 * np.pi)
            goal_orientation_interval_end: float = np.mean(orientation_end_list) % (2 * np.pi)

            orientation = orientation % (2 * np.pi)
            distance_start_right: float = (orientation - goal_orientation_interval_start) % (2 * np.pi)
            distance_start_left: float = (2 * np.pi) - distance_start_right
            distance_end_left: float = (goal_orientation_interval_end - orientation) % (2 * np.pi)
            distance_end_right: float = (2 * np.pi) - distance_end_left

            if (distance_end_left + distance_start_right) <= 2 * np.pi:
                return 0.0
            elif distance_start_left <= distance_end_right:
                return distance_start_left
            else:
                return -distance_end_right

    def _get_long_distance_until_lane_change(self, ego_state: State, ego_vehicle_lanelet_ids: List[int],
                                             navigator: Navigator) -> float:
        """
        Get the longitudinal distance until the lane change must be finished. It means that the ego vehicle is
        allowed to continue its was in the current lanelet and in its adjacent, but after this returned value,
        it must change the lane to the one which successor will lead to the goal

        :param ego_state: The current state
        :param ego_vehicle_lanelet_ids: The lanelet ids of the current state
        :param navigator: the navigator of the current planning problem
        :raises AssertionError: if ValueError of goal.navigator.get_lane_change_distance and no observations stored in
                history -> Ego vehicle started outside the global coordinate system
        :return: The longitudinal distance until the lane change must be finished
        """
        try:
            return navigator.get_lane_change_distance(ego_state, ego_vehicle_lanelet_ids)
            if not hasattr(ego_state, "orientation"):
                setattr(ego_state, "orientation", np.arctan2(ego_state.velocity_y, ego_state.velocity))
        except ValueError:
            assert self.distance_goal_long_lane, \
                "Ego vehicle started outside the global coordinate system"
            return self.observation_history_dict["distance_goal_long_lane"]

    @staticmethod
    def _get_goal_time_distance(time_step: float, goal: GoalRegion) -> float:
        """
        calculates the remaining time till the start of the goal time interval

        calculates time_step - goal_time_step_interval_start    if time_step < goal_time_step_interval_start
                   time_step - goal_time_interval_end           if time_step > goal_time_interval_end
                   0                                            else

        :param time_step: current time step
        :param goal: GoalRegion of current planning problem
        :return difference to the nearest goal time boundary
        """
        if "time_step" not in goal.state_list[0].attributes:
            return 0

        # time_step is mandatory for GoalRegion, doesn't need to check attribute
        time_start_list = np.array([s.time_step.start for s in goal.state_list])
        time_end_list = np.array([s.time_step.end for s in goal.state_list])
        goal_time_interval_start: float = np.mean(time_start_list)
        goal_time_interval_end: float = np.mean(time_end_list)
        if time_step < goal_time_interval_start:
            return time_step - goal_time_interval_start
        elif time_step > goal_time_interval_end:
            return time_step - goal_time_interval_end
        else:
            return 0

    @staticmethod
    def _check_goal_reached(goal: GoalRegion, ego_state: State, relax_is_goal_reached: bool = False) -> bool:
        """
        Check if goal is reached by ego vehicle.

        :param goal: GoalRegion of current planning problem
        :param ego_state: state of ego vehicle.
        :param relax_is_goal_reached: relaxes goal specification to position goal only
        :return: True if goal is reached
        """
        if ego_state.time_step == 0:
            return False

        if relax_is_goal_reached:
            for state in goal.state_list:
                if state.position.contains_point(ego_state.position):
                    return True
            return False
        else:
            return goal.is_reached(ego_state)

    @staticmethod
    def _check_is_time_out(ego_state: State, goal: GoalRegion, is_goal_reached: bool, episode_length=None) -> bool:
        """
        Check if maximum episode length exceeded

        :param goal: GoalRegion of current planning problem
        :param ego_state: state of ego vehicle
        :param is_goal_reached: flag whether the goal has been reached already
        :return: True if no more time left and the goal has not been reached
        """
        # TODO maybe store value and only calculate once
        if episode_length is None:
            episode_length = max(s.time_step.end for s in goal.state_list)
        return ego_state.time_step >= episode_length and not is_goal_reached

    @staticmethod
    def get_long_lat_distance_to_goal(position: np.array, navigator: Navigator) -> Tuple[float, float]:
        """
        Get longitudinal and lateral distances to the goal over the planned route

        :param position: the current position of the agent
        :param navigator: the navigator of the current planning problem
        :raises AssertionError: an AssertionError will be raised if goal.navigator.get_lon_lat_distance_to_goal
                raises a ValueError and there have not been any observations stored in the history

                this means that the ego vehicle started outside the global coordinate system

        :return: The tuple of the longitudinal and the lateral distances
        """
        try:
            return navigator.get_long_lat_distance_to_goal(position)
        except ValueError:
            return np.nan, np.nan


if __name__ == "__main__":
    import yaml
    from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

    config_file = PATH_PARAMS["configs"]["commonroad-v1"]
    with open(config_file, "r") as config_file:
        config = yaml.safe_load(config_file)
    configs = config["env_configs"]
    goal_observation = GoalObservation(configs)
    print(goal_observation)
