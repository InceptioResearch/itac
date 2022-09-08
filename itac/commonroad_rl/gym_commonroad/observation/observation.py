# """
# Module containing the observation base class
# """
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Union, Dict

import numpy as np
from commonroad.scenario.obstacle import State


class Observation(ABC):
    """
    Abstract class to define an observation
    """

    @abstractmethod
    def observe(self, ego_state: State, **kwargs) -> Union[np.array, Dict]:
        """ Create observation for given state in an environment.

            :param ego_state: state from which to observe the environment
            :return: ndarray of observation if flatten == True, observation dict otherwise
        """
        pass

    @abstractmethod
    def draw(self, render_configs: Dict, **kwargs):
        """ Method to draw the observation """
        pass

    @abstractmethod
    def build_observation_space(self) -> OrderedDict:
        """ Method to build the observation space """
        pass
