"""Class for hybrid reward"""
import logging
from math import exp

import numpy as np
from commonroad.common.solution import VehicleModel

from commonroad_rl.gym_commonroad.action import Action
from commonroad_rl.gym_commonroad.reward.reward import Reward

LOGGER = logging.getLogger(__name__)


class HybridReward(Reward):
    """Class for hybrid reward"""

    def __init__(self, configs: dict):
        self.goal_configs = configs["goal_configs"]
        self.reward_configs = configs["reward_configs_hybrid"]
        #        self.traffic_sign_configs = configs["traffic_sign_configs"]

        self.surrounding_configs = configs["surrounding_configs"]
        self.max_obs_dist: float = 0.0
        if self.surrounding_configs.get("observe_lane_circ_surrounding", False):
            self.max_obs_dist = self.surrounding_configs["lane_circ_sensor_range_radius"]
        elif self.surrounding_configs.get("observe_lane_rect_surrounding", False):
            self.max_obs_dist \
                = np.sqrt((self.surrounding_configs["lane_rect_sensor_range_length"] / 2) ** 2
                          + (self.surrounding_configs["lane_rect_sensor_range_width"] / 2) ** 2)
        elif self.surrounding_configs.get("observe_lidar_circle_surrounding", False):
            self.max_obs_dist = self.surrounding_configs["lidar_sensor_radius"]

        self.stopped_once = False

    def reset(self, observation_dict: dict, ego_action: Action):
        pass

    def calc_reward(self, observation_dict: dict, ego_action: Action) -> float:
        """
        Calculate the reward according to the observations

        :param observation_dict: current observations
        :param ego_action: Current ego_action of the environment
        :return: Reward of this step
        """
        reward = 0.0
        termination_reward = self.termination_reward(observation_dict)
        reward += termination_reward
        if termination_reward:
            # extra reward for being close the goal when terminating
            reward += self.long_distance_to_reference_path_reward(observation_dict)

        # Penalize reverse driving
        if self.reward_configs["reward_reverse_driving"]:
            reward += self.reverse_driving_penalty(ego_action)

        # Distance advancement
        if self.goal_configs["observe_distance_goal_long"] and self.goal_configs["observe_distance_goal_lat"]:
            reward += self.goal_distance_reward(observation_dict)

        # Closing in on goal-time
        if self.goal_configs["observe_distance_goal_time"]:
            reward += self.goal_time_reward(observation_dict, ego_action)
        if (
                (not self.goal_configs["observe_distance_goal_lat"]
                 or np.isclose(observation_dict["distance_goal_lat"][0], 0))
                and
                (not self.goal_configs["observe_distance_goal_long"]
                 or np.isclose(observation_dict["distance_goal_long"][0], 0))
                and
                (not self.goal_configs["observe_distance_goal_time"]
                 or observation_dict["distance_goal_time"] == 0)
        ):
            # Closing in on goal-orientation
            if self.goal_configs["observe_distance_goal_orientation"]:
                reward += self.goal_orientation_reward(observation_dict)

            # Closing in on goal-speed
            if self.goal_configs["observe_distance_goal_velocity"]:
                reward += self.goal_velocity_reward(observation_dict, ego_action)

        # Safe distance reward
        if self.reward_configs["reward_safe_distance_coef"]:
            reward += self.safe_distance_reward(observation_dict)

        # Traffic rule priority yield reward
        #      if self.reward_configs["yield_reward"]:
        #         reward += self.yield_reward(observation_dict)
        # Reference Path Reward
        if self.reward_configs['reward_lat_distance_reference_path']:
            reward += self.lat_distance_to_reference_path_reward(observation_dict)

        # TODO: was commented out in original reward, needs to be reworked/removed?
        # penalize running given stop sign
        #    if self.traffic_sign_configs["observe_stop_sign"]:
        #       if self.stopped_once and observation_dict["stop_sign_distance_long"] == -1.0:
        #           self.stopped_once = False
        #       if observation_dict["stop_sign"][0]:
        #           reward += self.traffic_sign_reward(observation_dict)

        # TODO: was commented out in original reward, needs to be reworked/removed?
        # reward += self.deviation_lane_center_penalty()

        # TODO: was commented out in original reward, needs to be reworked/removed?
        # # passenger comfort
        # reward += self.passenger_comfort_reward()

        # TODO: was commented out in original reward, needs to be reworked/removed?
        # # penalize large lateral velocity using PM model
        # if self.ego_action.vehicle.vehicle_model == VehicleModel.PM:
        #     reward += self.large_lat_velocity_penalty()

        return reward

    def termination_reward(self, observation_dict: dict) -> float:
        """Reward for the cause of termination"""
        # Reach goal
        if observation_dict["is_goal_reached"][0]:
            LOGGER.debug("GOAL REACHED!")
            return self.reward_configs["reward_goal_reached"]
        # Collision
        if observation_dict["is_collision"][0]:
            return self.reward_configs["reward_collision"]
        # Off-road
        if observation_dict["is_off_road"][0]:
            return self.reward_configs["reward_off_road"]
        # Friction violation
        if observation_dict["is_friction_violation"][0]:
            return self.reward_configs["reward_friction_violation"]
        # Exceed maximum episode length
        if observation_dict["is_time_out"][0]:
            return self.reward_configs["reward_time_out"]

        return 0.0

    def goal_distance_reward(self, observation_dict: dict) -> float:
        """Reward for getting closer to goal distance"""
        long_advance = observation_dict["distance_goal_long_advance"][0]
        lat_advance = observation_dict["distance_goal_lat_advance"][0]

        return self.reward_configs["reward_closer_to_goal_long"] * long_advance + \
               self.reward_configs["reward_closer_to_goal_lat"] * lat_advance

    @staticmethod
    def _time_to_goal_weight(ego_velocity, distance, goal_time_distance):
        """
        Weight function for the goal_time_reward.
        """
        goal_time_distance = abs(goal_time_distance)
        if np.isclose(distance, 0):
            return 1.
        if not np.isclose(ego_velocity, 0):
            time_to_goal = abs(distance / ego_velocity)
            if not np.isclose(time_to_goal, 0):
                return min(time_to_goal / goal_time_distance, goal_time_distance / time_to_goal)
        return 0.

    def goal_time_reward(self, observation_dict: dict, ego_action: Action) -> float:
        """Reward for getting closer to goal time"""
        # TODO: improve time reward?
        # Idea: take distance to goal and velocity into account by approximating time to goal at current velocity
        if observation_dict["distance_goal_time"][0] >= 0:
            return 0.0
        elif self.goal_configs["observe_euclidean_distance"]:
            return self._time_to_goal_weight(
                ego_action.vehicle.state.velocity,
                observation_dict["euclidean_distance"][0], observation_dict["distance_goal_time"][0]) * \
                   self.reward_configs["reward_get_close_goal_time"]
        else:
            return self.reward_configs["reward_get_close_goal_time"]

    def goal_velocity_reward(self, observation_dict: dict, ego_action: Action) -> float:
        """Reward for getting closer to goal velocity"""
        if ego_action.vehicle.vehicle_model == VehicleModel.PM:
            velocity = np.sqrt(ego_action.vehicle.state.velocity ** 2 + ego_action.vehicle.state.velocity_y ** 2)
        else:
            velocity = abs(ego_action.vehicle.state.velocity)

        return self.reward_configs["reward_close_goal_velocity"] \
               * np.exp(-1.0 * abs(observation_dict["distance_goal_velocity"][0])
                        / (-1 * observation_dict["distance_goal_velocity"][0] + velocity))

    def goal_orientation_reward(self, observation_dict: dict) -> float:
        """
        Reward for getting closer to goal orientation
        """
        return self.reward_configs["reward_close_goal_orientation"] \
               * np.exp(-1.0 * abs(observation_dict["distance_goal_orientation"][0]) / np.pi)

    def reverse_driving_penalty(self, ego_action: Action) -> float:
        """Penalty for driving backwards"""
        return self.reward_configs["reward_reverse_driving"] * (ego_action.vehicle.state.velocity < 0.0)

    def safe_distance_reward(self, observation_dict: dict) -> float:
        """Reward for keeping a safe distance to the leading vehicle. If the ego vehicle is changing lanes, not keeping
        a safe distance to the following vehicle is also penalized"""
        # TODO: fix safe distance reward

        reward = 0.0
        a_max = 11.5 # TODO: load a_max from vehicle parameters
        lane_change = self.surrounding_configs["observe_lane_change"] and observation_dict["lane_change"] > 0.
        # dist_lead = self.max_obs_dist
        # dist_follow = self.max_obs_dist
        # v_rel = [v_rel_left_follow, v_rel_same_follow, v_rel_right_follow, v_rel_left_lead, v_rel_same_lead,
        #          v_rel_right_lead]
        # p_rel = [p_rel_left_follow, p_rel_same_follow, p_rel_right_follow, p_rel_left_lead, p_rel_same_lead,
        #          p_rel_right_lead]
        if self.surrounding_configs["observe_lane_rect_surrounding"] \
                or self.surrounding_configs["observe_lane_circ_surrounding"]:
            dist_lead = observation_dict["lane_based_p_rel"][4]
            dist_follow = observation_dict["lane_based_p_rel"][1]
            v_rel_lead = observation_dict["lane_based_v_rel"][4]
            v_rel_follow = observation_dict["lane_based_v_rel"][1]
            v_ego = np.sqrt(np.sum(observation_dict["v_ego"] ** 2))
            v_lead = v_rel_lead + v_ego
            v_follow = v_ego - v_rel_follow
            safe_dist_lead = max((v_ego ** 2 - v_lead ** 2) / (2 * a_max), 4.)
            safe_dist_follow = max((v_follow ** 2 - v_ego ** 2) / (2 * a_max), 4.)
        # elif self.surrounding_configs["observe_lidar_circle_surrounding"]:
        #     [dist_leading, dist_following] = observation_dict["dist_lead_follow_rel"]
        else:
            raise NotImplementedError(f"Safe distance reward is only supported for lane-based observations currently!")

        reward += self._safe_distance_reward_function(dist_lead, safe_dist_lead)
        if lane_change:
            reward += self._safe_distance_reward_function(dist_follow, safe_dist_follow)

        assert isinstance(reward, float)
        return reward

    def _safe_distance_reward_function(self, distance: float, safe_distance: float) -> float:
        """Exponential reward function for keeping a safe distance"""
        if distance < safe_distance:
            return self.reward_configs["reward_safe_distance_coef"] * np.exp(-5. * distance / safe_distance)
        else:
            return 0.

    def traffic_sign_reward(self, observation_dict: dict) -> float:
        """Reward for obeying traffic sings"""
        reward = 0.0
        #        dis_stop_sign = observation_dict["stop_sign_distance_long"][0]
        # if distance between ego vehicle and stop sign shorter than 2m
        #        if 0 < dis_stop_sign < 2 and observation_dict["v_ego"][0] > 0:
        #            # penalize large velocity
        #            reward += self.reward_configs['reward_stop_sign_vel'] * np.exp(
        #               -5 * dis_stop_sign / 2) * abs(observation_dict["v_ego"][0])
        #            # reward += self.reward_configs['reward_stop_sign_vel'] * (1 - 0.5 * dis_stop_sign) * \
        #            # observation_dict["v_ego"][0]
        #            # positive reward for negative acceleration(the ego vehicle should decelerate)
        #            if observation_dict["a_ego"][0] < 0:
        #                reward += self.reward_configs['reward_stop_sign_acc']
        #        # Finally, add constant reward for stopping in front of the stop sign once:
        #        if not self.stopped_once and np.isclose(observation_dict["v_ego"], 0.0):
        #            reward += self.reward_configs["stop_sign_vel_zero"]
        #            self.stopped_once = True
        return reward

    def deviation_lane_center_penalty(self, observation_dict: dict) -> float:
        """Penalty for deviating from the lane center"""
        # reward = 0.0
        # # Deviation from lane center
        # # TODO: use min. road edge distance instead
        # if self.lanelet_configs["observe_lat_offset"]:
        #     reward += -self.reward_configs["reward_stay_in_road_center"] * \
        #               np.exp(np.abs(self.observation_dict["lat_offset"][0]))
        #
        # if self.lanelet_configs["observe_left_road_edge_distance"] or self.lanelet_configs[
        #     "observe_right_road_edge_distance"]:
        #     min_road_dist = np.min((self.observation_dict["left_road_edge_distance"][0],
        #                             self.observation_dict["right_road_edge_distance"][0]))
        #     reward += -self.reward_configs["reward_stay_in_road_center"] * np.exp(-min_road_dist)
        # # Degree of violation of friction constraint
        # if (
        #         self.observe_a_ego
        #         and self.observe_v_ego
        #         and self.observe_steering_angle
        # ):
        #     a_ego = self.observation_dict["a_ego"][0]
        #     v_ego = self.observation_dict["v_ego"][0]
        #     steering_ego = self.observation_dict["steering_angle"][0]
        #     l_wb = self.ego_vehicle.params.a + ego_vehicle.params.b
        #     a_max = self.ego_vehicle.params.longitudinal.a_max
        #     reward += self.reward_friction * (
        #             a_max
        #             - (a_ego ** 2 + (v_ego ** 2 * np.tan(steering_ego) / l_wb) ** 2)
        #             ** 0.5
        #     )
        #
        # return reward

    def passenger_comfort_reward(self) -> float:
        """Penalty for bad passenger comfort"""
        # if self.current_step >= 2:
        #     ego_vehicle = self.ego_action.vehicle
        #     jerk_long = (ego_vehicle.state.acceleration - ego_vehicle.previous_state.acceleration) \
        #         / self.scenario.dt
        #     jerk_lat = (ego_vehicle.state.acceleration_y - ego_vehicle.previous_state.acceleration_y) \
        #         / self.scenario.dt
        #     return -1.0 * self.reward_configs["reward_jerk_long"] * np.abs(jerk_long) \
        #         + self.reward_configs["reward_jerk_lat"] * np.abs(jerk_lat)

    def large_lat_velocity_penalty(self) -> float:
        """Penalty for large lateral velocity using PM model"""
        # longitudinal and lateral velocity
        # v_long = self.ego_action.vehicle.state.velocity
        # v_lat = self.ego_action.vehicle.state.velocity_y
        # # print("current_long_v:", v_long, "current_lat_v", v_lat)
        # return self.reward_configs['reward_lateral_velocity'] * np.abs(v_lat)

    def lat_distance_to_reference_path_reward(self, observation_dict, distance50pctreward=3) -> float:
        """
        :param: distance50pctreward: distance in m for which 50% reward is returned 

        postive reward for staying close to the reference path in lateral direction
        max reward: 0.999 (0 m to the reference pathin lat direction)
        max penalty: - 0.0 (inf / >20 m to the reference pathin lat direction)
        """
        if "distance_togoal_via_referencepath" not in observation_dict:
            return 0.0
        long_lat_pos = observation_dict["distance_togoal_via_referencepath"]

        lat = abs(long_lat_pos[1])

        # sigmoid_distance ~= 1 for distance 0m to reference path, ~=0.5 for distance50pctreward m, ~0.01 for 2 * distance50pctreward m
        sigmoid_distance = 1 - 1 / (1 + exp(distance50pctreward - lat))

        return self.reward_configs['reward_lat_distance_reference_path'] * sigmoid_distance

    def long_distance_to_reference_path_reward(self, observation_dict, distance50pctreward=5) -> float:
        """
        reward/penalty for distance to goal on the reference path in longitudinal direction
        max reward: 2.0, for long is 0.
        max reward: 1.0, when being more then distance50pctreward meters before goal
        max penalty: - log( abs(long_distance_past_goal) + 1), when past the goal
        """
        if "distance_togoal_via_referencepath" not in observation_dict:
            return 0.0
        long_lat_pos = observation_dict["distance_togoal_via_referencepath"]

        long = long_lat_pos[0]

        if distance50pctreward > long >= 0:
            return 2.0 - (long / distance50pctreward)
        elif long > distance50pctreward:
            # if before goal
            # sigmoid_distance ~= 1 for distance 0m to reference path, ~=0.5 for 5m, ~0.01 for 10m
            return min(1 / (long * distance50pctreward), 1.0) * self.reward_configs[
                'reward_long_distance_reference_path']

        elif long < 0:
            # if driven past the goal: penalty
            return - np.log(abs(long) + 1) * self.reward_configs['reward_long_distance_reference_path']
