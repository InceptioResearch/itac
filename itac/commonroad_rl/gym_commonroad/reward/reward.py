"""Abstract class for rewards"""

__author__ = "Xiao Wang, Brian Liao, Niels Muendler, Peter Kocsis, Armin Ettenhofer"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

from abc import ABC, abstractmethod

from commonroad_rl.gym_commonroad.action import Action


class Reward(ABC):
    """Abstract class for rewards"""

    def reset(self, observation_dict: dict, ego_action: Action):
        pass

    @abstractmethod
    def calc_reward(self, observation_dict: dict, ego_action: Action) -> float:
        """
        Calculate the reward according to the observations

        :param observation_dict: current observations
        :param ego_action: Current ego_action of the environment
        :return: Reward of this step
        """
