import copy
import math
from typing import Union, List, Dict, Set, Tuple, Optional
import numpy as np
import warnings

from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Shape
from commonroad.scenario.trajectory import State

__author__ = "Christina Miller and Stefanie Manzinger"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "2022.1"
__maintainer__ = "Christina Miller"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"

from commonroad.visualization.drawable import IDrawable
from commonroad.visualization.param_server import ParamServer
from commonroad.visualization.renderer import IRenderer


class GoalRegion(IDrawable):
    def __init__(self, state_list: List[State],
                 lanelets_of_goal_position: Union[
                     None, Dict[int, List[int]]] = None):
        """
        Region, that has to be reached by the vehicle. Contains a list of
        goal states of which one has to be fulfilled
        to solve the scenario. If 'position' in a goal state is given as a
        list of lanelets, they are converted into a
        polygon. To reconstruct the lanelets later, the lanelet ids are
        stored in a dict in lanelets_of_goal_position.
        In no 'position' is given as lanelet, lanelets_of_goal_position is
        set to None.

        :param state_list: list of goal states (one of those has to be
        fulfilled)
        :param lanelets_of_goal_position: dict[index of state in state_list, list of lanelet ids].
        None, if no lanelet is given.
        """
        self.state_list = state_list
        self.lanelets_of_goal_position = lanelets_of_goal_position

    @property
    def state_list(self) -> List[State]:
        """List that contains all goal states"""
        return self._state_list

    @state_list.setter
    def state_list(self, state_list: List[State]):
        for state in state_list:
            self._validate_goal_state(state)
        self._state_list = state_list

    @property
    def lanelets_of_goal_position(self) -> Union[None, Dict[int, List[int]]]:
        """Dict that contains the index of the state in the state_list to which the lanelets belong. \
        None, if goal position is not a lanelet"""
        return self._lanelets_of_goal_position

    @lanelets_of_goal_position.setter
    def lanelets_of_goal_position(self, lanelets: Union[None, Dict[int, List[int]]]):
        if not hasattr(self, '_lanelets_of_goal_position'):
            if lanelets is not None:
                assert isinstance(lanelets, dict)
                assert all(isinstance(x, int) for x in lanelets.keys())
                assert all(isinstance(x, list) for x in lanelets.values())
                assert all(isinstance(x, int) for lanelet_list in lanelets.values() for x in lanelet_list)
            self._lanelets_of_goal_position = lanelets
        else:
            warnings.warn('<GoalRegion/lanelets_of_goal_position> lanelets_of_goal_position are immutable')

    def is_reached(self, state: State) -> bool:
        """
        Checks if a given state is inside the goal region.

        :param state: state with exact values
        :return: True, if state fulfills all requirements of the goal region. False if at least one requirement of the \
        goal region is not fulfilled.
        """
        is_reached_list = list()
        for goal_state in self.state_list:
            goal_state_tmp = copy.deepcopy(goal_state)
            goal_state_fields = set([slot for slot in goal_state.__slots__ if hasattr(goal_state, slot)])
            state_fields = set([slot for slot in goal_state.__slots__ if hasattr(state, slot)])
            state_new, state_fields, goal_state_tmp, goal_state_fields =\
                self._harmonize_state_types(state, goal_state_tmp, state_fields, goal_state_fields)

            if not goal_state_fields.issubset(state_fields):
                raise ValueError('The goal states {} are not a subset of the provided states {}!'
                                 .format(goal_state_fields, state_fields))
            is_reached = True
            if hasattr(goal_state, 'time_step'):
                is_reached = is_reached and self._check_value_in_interval(state_new.time_step, goal_state.time_step)
            if hasattr(goal_state, 'position'):
                is_reached = is_reached and goal_state.position.contains_point(state_new.position)
            if hasattr(goal_state, 'orientation'):
                is_reached = is_reached and self._check_value_in_interval(state_new.orientation, goal_state.orientation)
            if hasattr(goal_state, 'velocity'):
                is_reached = is_reached and self._check_value_in_interval(state_new.velocity, goal_state.velocity)
            is_reached_list.append(is_reached)
        return np.any(is_reached_list)

    def translate_rotate(self, translation: np.ndarray, angle: float):
        """
        translate and rotates the goal region with given translation and angle around the origin (0, 0)

        :param translation: translation vector [x_off, y_off] in x- and y-direction
        :param angle: rotation angle in radian (counter-clockwise)
        """
        for i, state in enumerate(self.state_list):
            self.state_list[i] = state.translate_rotate(translation, angle)

    @classmethod
    def _check_value_in_interval(cls, value: Union[int, float], desired_interval: Union[AngleInterval, Interval]) -> \
            bool:
        """
        Checks if an exact value is included in the desired interval. If desired_interval is not an interval,
        an exception is thrown.

        :param value: int or float value to test
        :param desired_interval: Desired interval in which value is tested
        :return: True, if value matches the desired_value, False if not.
        """
        if isinstance(desired_interval, (Interval, AngleInterval)):
            is_reached = desired_interval.contains(value)
        else:
            raise ValueError("<GoalRegion/_check_value_in_interval>: argument 'desired_interval' of wrong type. "
                             "Expected type: {}. Got type: {}.".format((type(Interval), type(AngleInterval)),
                                                                       type(desired_interval)))
        return is_reached

    @classmethod
    def _validate_goal_state(cls, state: State):
        """
        Checks if state fulfills the requirements for a goal state and raises Error if not.

        :param state: state to check
        """
        # mandatory fields:
        if not hasattr(state, 'time_step'):
            raise ValueError('<GoalRegion/_goal_state_is_valid> field time_step is mandatory. '
                             'No time_step attribute found.')

        # optional fields
        valid_fields = ['time_step', 'position', 'velocity', 'orientation']

        for attr in [attr for attr in state.__slots__ if hasattr(state, attr)]:
            if attr not in valid_fields:
                raise ValueError('<GoalRegion/_goal_state_is_valid> field error: allowed fields are '
                                 '[time_step, position, velocity, orientation]; "%s" detected' % attr)
            elif attr == 'position':
                if not isinstance(getattr(state, attr), Shape):
                    raise ValueError(
                        '<GoalRegion/_goal_state_is_valid> position needs to be an instance of '
                        '%s; got instance of %s instead' % (Shape, getattr(state, attr).__class__))
            elif attr == 'orientation':
                if not isinstance(getattr(state, attr), AngleInterval):
                    raise ValueError('<GoalRegion/_goal_state_is_valid> orientation needs to be an instance of %s; got '
                                     'instance of %s instead' % (AngleInterval, getattr(state, attr).__class__))
            else:
                if not isinstance(getattr(state, attr), Interval):
                    raise ValueError('<GoalRegion/_goal_state_is_valid> attributes must be instances of '
                                     '%s only (except from position and orientation); got "%s" for '
                                     'attribute "%s"' % (Interval, getattr(state, attr).__class__, attr))

    @staticmethod
    def _harmonize_state_types(state: State, goal_state: State,  state_fields: Set[str],
                               goal_state_fields: Set[str]):
        """
        Transforms states from value_x, value_y to orientation, value representation if required.
        :param state: state to check for goal
        :param goal_state: goal state
        :return:
        """
        state_new = copy.deepcopy(state)
        if {'velocity', 'velocity_y'}.issubset(state_fields) \
                and ({'orientation'}.issubset(goal_state_fields) or {'velocity'}.issubset(goal_state_fields))\
                and not {'velocity', 'velocity_y'}.issubset(goal_state_fields):

            if 'orientation' not in state_fields:
                state_new.orientation = math.atan2(state_new.velocity_y,
                                                   state_new.velocity)
                state_fields.add('orientation')

            state_new.velocity = np.linalg.norm(
                    np.array([state_new.velocity, state_new.velocity_y]))
            state_fields.remove('velocity_y')

        return state_new, state_fields, goal_state, goal_state_fields

    def draw(self, renderer: IRenderer,
             draw_params: Union[ParamServer, dict, None] = None,
             call_stack: Optional[Tuple[str, ...]] = tuple()):
        renderer.draw_goal_region(self, draw_params, call_stack)
