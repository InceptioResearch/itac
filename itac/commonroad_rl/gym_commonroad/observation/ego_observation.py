"""
Module EgoObservation
"""
from collections import defaultdict, OrderedDict
from typing import Union, Dict

import gym
import yaml
import numpy as np
from commonroad.common.solution import VehicleModel
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import State
from numpy import ndarray
from scipy import spatial

from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.gym_commonroad.action.vehicle import Vehicle
from commonroad_rl.gym_commonroad.observation.observation import Observation
from commonroad_rl.gym_commonroad.utils.scenario import approx_orientation_vector, angle_difference


class EgoObservation(Observation):
    """
    Ego-vehicle-related observation class
    """

    def __init__(self, configs: Dict, configs_name: str = "ego_configs"):
        """

        :param configs: dictionary to store all observation configurations
        :param configs_name: key of configs dictionary corresponding to this observation
        """
        # Read config
        ego_configs = configs[configs_name]
        self.observe_v_ego: bool = ego_configs.get("observe_v_ego")
        self.observe_a_ego: bool = ego_configs.get("observe_a_ego")
        self.observe_relative_heading: bool = ego_configs.get("observe_relative_heading")
        self.observe_steering_angle: bool = ego_configs.get("observe_steering_angle")
        self.observe_global_turn_rate: bool = ego_configs.get("observe_global_turn_rate")
        self.observe_remaining_steps: bool = ego_configs.get("observe_remaining_steps")
        self.observe_is_friction_violation: bool = ego_configs.get("observe_is_friction_violation")

        self.observation_dict = OrderedDict()
        self.observation_history_dict = defaultdict(list)
        try:
            self._is_PM_model = configs["vehicle_params"]["vehicle_model"] == 0
        except KeyError:
            pass

    def build_observation_space(self) -> OrderedDict:
        observation_space_dict = OrderedDict()

        if self.observe_v_ego:
            if self._is_PM_model:
                observation_space_dict["v_ego"] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
            else:
                observation_space_dict["v_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_a_ego:
            if self._is_PM_model:
                observation_space_dict["a_ego"] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
            else:
                observation_space_dict["a_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_steering_angle and not self._is_PM_model:
            observation_space_dict["steering_angle"] = gym.spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32)
        if self.observe_relative_heading and not self._is_PM_model:
            observation_space_dict["relative_heading"] = gym.spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32)
        if self.observe_global_turn_rate and not self._is_PM_model:
            observation_space_dict["global_turn_rate"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_is_friction_violation:
            observation_space_dict["is_friction_violation"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        if self.observe_remaining_steps:
            observation_space_dict["remaining_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        return observation_space_dict

    def observe(self, ego_lanelet: Lanelet, ego_vehicle: Vehicle, episode_length: int) -> Union[ndarray, Dict]:
        """
        Create ego-related observation for given state in an environment.
        """
        ego_state = ego_vehicle.state

        if self.observe_v_ego:
            if ego_vehicle.vehicle_model == VehicleModel.PM:
                self.observation_dict["v_ego"] = np.array([ego_state.velocity, ego_state.velocity_y])
            else:
                self.observation_dict["v_ego"] = np.array([ego_state.velocity])

        if self.observe_a_ego:
            if ego_vehicle.vehicle_model == VehicleModel.PM:
                self.observation_dict["a_ego"] = np.array([ego_state.acceleration, ego_state.acceleration_y])
            else:
                self.observation_dict["a_ego"] = np.array([ego_state.acceleration])

        if self.observe_steering_angle and ego_vehicle.vehicle_model != VehicleModel.PM:
            self.observation_dict["steering_angle"] = np.array([ego_state.steering_angle])

        if self.observe_relative_heading and ego_vehicle.vehicle_model != VehicleModel.PM:
            relative_heading = EgoObservation.get_lane_relative_heading(ego_state, ego_lanelet)
            self.observation_dict["relative_heading"] = relative_heading

        if self.observe_global_turn_rate and ego_vehicle.vehicle_model != VehicleModel.PM:
            self.observation_dict["global_turn_rate"] = np.array([ego_state.yaw_rate])

        if self.observe_is_friction_violation:
            is_friction_violation = self._check_friction_violation(ego_vehicle)
            self.observation_dict["is_friction_violation"] = np.array([is_friction_violation])

        if self.observe_remaining_steps:
            self.observation_dict["remaining_steps"] = np.array([episode_length - ego_state.time_step])

        return self.observation_dict

    def draw(self, render_configs: Dict, **kwargs):
        """ Method to draw the observation """

    @staticmethod
    def _check_friction_violation(ego_vehicle: Vehicle):
        return ego_vehicle.violate_friction

    @staticmethod
    def get_lane_relative_heading(ego_vehicle_state: State, ego_vehicle_lanelet: Lanelet) -> float:
        """
        Get the heading angle in the Frenet frame.

        :param ego_vehicle_state: state of ego vehicle
        :param ego_vehicle_lanelet: lanelet of ego vehicle
        :return: heading angle in frenet coordinate system relative to lanelet center vertices between -pi and pi
        """
        lanelet_angle = EgoObservation._get_orientation_of_polyline(ego_vehicle_state.position,
                                                                    ego_vehicle_lanelet.center_vertices)

        return angle_difference(approx_orientation_vector(lanelet_angle),
                                approx_orientation_vector(ego_vehicle_state.orientation))

    @staticmethod
    def _get_orientation_of_polyline(position: np.array, polyline: np.array) -> float:
        """
        Get the approximate orientation of the lanelet.

        :param position: position to calculate the orientation of lanelet
        :param polyline: polyline to calculate the orientation of
        :return: orientation (rad) of the lanelet
        """
        # TODO: This method could make use of the general relative orientation function or could be fully replaced by
        #  the functions moved to the route planner

        idx = spatial.KDTree(polyline).query(position)[1]
        if idx < position.shape[0] - 1:
            orientation = np.arccos((polyline[idx + 1, 0] - polyline[idx, 0]) /
                                    np.linalg.norm(polyline[idx + 1] - polyline[idx]))
            sign = np.sign(polyline[idx + 1, 1] - polyline[idx, 1])
        else:
            orientation = np.arccos((polyline[idx, 0] - polyline[idx - 1, 0]) /
                                    np.linalg.norm(polyline[idx] - polyline[idx - 1]))
            sign = np.sign(polyline[idx, 1] - polyline[idx - 1, 1])
        if sign >= 0:
            orientation = np.abs(orientation)
        else:
            orientation = -np.abs(orientation)
        # orientation = shift_orientation(orientation)
        return orientation


if __name__ == "__main__":
    CONFIG_FILE = PATH_PARAMS["configs"]["commonroad-v1"]
    with open(CONFIG_FILE, "r") as config_file:
        config = yaml.safe_load(config_file)
    ego_observation = EgoObservation(config["env_configs"])
    print(ego_observation)
