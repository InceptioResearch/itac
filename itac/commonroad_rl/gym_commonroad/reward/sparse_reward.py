"""Class for sparse reward"""
import logging

from commonroad_rl.gym_commonroad.action import Action
from commonroad_rl.gym_commonroad.reward.reward import Reward

LOGGER = logging.getLogger(__name__)


class SparseReward(Reward):
    """Class for sparse reward"""

    def __init__(self, configs: dict):
        self.reward_configs = configs["reward_configs_sparse"]

    def calc_reward(self, observation_dict: dict, ego_action: Action) -> float:
        """
        Calculate the reward according to the observations

        :param observation_dict: current observations
        :param ego_action: Current ego_action of the environment
        :return: Reward of this step
        """
        reward = 0.0

        # Reach goal
        if observation_dict["is_goal_reached"][0]:
            LOGGER.debug("GOAL REACHED!")
            reward += self.reward_configs["reward_goal_reached"]
        # Collision
        if observation_dict["is_collision"][0]:
            reward += self.reward_configs["reward_collision"]
        # Off-road
        if observation_dict["is_off_road"][0]:
            reward += self.reward_configs["reward_off_road"]
        # Friction violation
        if observation_dict["is_friction_violation"][0]:
            reward += self.reward_configs["reward_friction_violation"]
        # Exceed maximum episode length
        if observation_dict["is_time_out"][0]:
            reward += self.reward_configs["reward_time_out"]

        return reward
