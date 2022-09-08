"""Class for dense reward"""
import warnings

import numpy as np

from commonroad_rl.gym_commonroad.action import Action
from commonroad_rl.gym_commonroad.reward.reward import Reward


class DenseReward(Reward):
    """Class for dense reward"""

    def __init__(self, configs: dict):
        self.reward_configs = configs["reward_configs_dense"]

        self.surrounding_configs = configs["surrounding_configs"]
        self.max_obs_dist: float = 0.0
        if self.surrounding_configs["observe_lane_circ_surrounding"]:
            self.max_obs_dist = self.surrounding_configs["lane_circ_sensor_range_radius"]
        elif self.surrounding_configs["observe_lane_rect_surrounding"]:
            self.max_obs_dist \
                = np.sqrt((self.surrounding_configs["lane_rect_sensor_range_length"] / 2) ** 2
                          + (self.surrounding_configs["lane_rect_sensor_range_width"] / 2) ** 2)
        elif self.surrounding_configs["observe_lidar_circle_surrounding"]:
            self.max_obs_dist = self.surrounding_configs["lidar_sensor_radius"]

        self.initial_goal_dist: float = -1.0

    def reset(self, observation_dict: dict, ego_action: Action):
        distance_goal_long = observation_dict["distance_goal_long"][0]
        distance_goal_lat = observation_dict["distance_goal_lat"][0]
        self.initial_goal_dist = np.sqrt(distance_goal_long ** 2 + distance_goal_lat ** 2)

        # Prevent cases where the ego vehicle starts in the goal region
        if self.initial_goal_dist < 1.0:
            warnings.warn("Ego vehicle starts in the goal region")
            self.initial_goal_dist = 1.0

    def calc_reward(self, observation_dict: dict, ego_action: Action) -> float:
        """
        Calculate the reward according to the observations

        :param observation_dict: current observations
        :param ego_action: Current ego_action of the environment
        :return: Reward of this step
        """

        # Calculate normalized distance to obstacles as a positive reward
        # Lane-based
        rel_pos = []
        if self.surrounding_configs["observe_lane_rect_surrounding"] or \
                self.surrounding_configs["observe_lane_circ_surrounding"]:
            rel_pos += observation_dict["lane_based_p_rel"].tolist()

        if len(rel_pos) == 0:
            r_obs_lane = 0.0
        else:
            # Minus 5 meters from each of the lane-based relative positions,
            # to get approximately minimal distances between vehicles,
            # instead of exact distances between centers of vehicles
            r_obs_lane = (self.reward_configs["reward_obs_distance_coefficient"] * (
                    np.sum(rel_pos) - 5.0 * len(rel_pos)) / (self.max_obs_dist * len(rel_pos)))

        # Lidar-based
        dist = []
        if self.surrounding_configs["observe_lidar_circle_surrounding"]:
            dist += observation_dict["lidar_circle_dist"].tolist()

        if len(dist) == 0:
            r_obs_lidar = 0.0
        else:
            r_obs_lidar = (self.reward_configs["reward_obs_distance_coefficient"] * np.sum(dist) / (
                    self.max_obs_dist * len(dist)))

        # Calculate normalized distance to goal as a negative reward
        dist_goal = np.sqrt(observation_dict["distance_goal_long"][0] ** 2
                            + observation_dict["distance_goal_lat"][0] ** 2)
        r_goal = -self.reward_configs["reward_goal_distance_coefficient"] * dist_goal / self.initial_goal_dist

        return r_obs_lane + r_obs_lidar + r_goal
