from typing import Union, List, Tuple, Dict, Optional
import numpy as np
import warnings

from commonroad.scenario.trajectory import State, Trajectory
from commonroad.planning.goal import GoalRegion
from commonroad.common.validity import is_natural_number

__author__ = "Christina Miller"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "2022.1"
__maintainer__ = "Christina Miller"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"

from commonroad.visualization.drawable import IDrawable
from commonroad.visualization.param_server import ParamServer
from commonroad.visualization.renderer import IRenderer


class PlanningProblem(IDrawable):
    def __init__(self, planning_problem_id: int, initial_state: State,
                 goal_region: GoalRegion):
        self.planning_problem_id = planning_problem_id
        self.initial_state = initial_state
        self.goal = goal_region

    @property
    def planning_problem_id(self) -> int:
        """Id of the planning problem"""
        return self._planning_problem_id

    @planning_problem_id.setter
    def planning_problem_id(self, problem_id: int):
        if not hasattr(self, '_planning_problem_id'):
            assert is_natural_number(problem_id), '<PlanningProblem/planning_problem_id>: Argument "problem_id" of ' \
                                                  'wrong type. Expected type: %s. Got type: %s.' \
                                                  % (int, type(problem_id))
            self._planning_problem_id = problem_id
        else:
            warnings.warn('<PlanningProblem/planning_problem_id> planning_problem_id is immutable')

    @property
    def initial_state(self) -> State:
        """Initial state of the ego vehicle"""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: State):
        mandatory_fields = ['position', 'velocity', 'orientation', 'yaw_rate', 'slip_angle', 'time_step']
        for field in mandatory_fields:
            if not hasattr(state, field):
                raise ValueError('<PlanningProblem/initial_state> fields [{}] are mandatory. '
                                 'No {} attribute found.'.format(', '.join(mandatory_fields), field))
        self._initial_state = state

    @property
    def goal(self) -> GoalRegion:
        """Region that has to be reached"""
        return self._goal_region

    @goal.setter
    def goal(self, goal_region: GoalRegion):
        assert(isinstance(goal_region, GoalRegion)), 'argument "goal_region" of wrong type. Expected type: %s. ' \
                                                     'Got type: %s.' % (GoalRegion, type(goal_region))
        self._goal_region = goal_region

    def goal_reached(self, trajectory: Trajectory) -> Tuple[bool, int]:
        """
        Checks if the goal region defined in the planning problem is reached by any state of a given trajectory

        :param trajectory: trajectory to test
        :return: Tuple: (True, index of first state in trajectory.state_list that reaches goal) if one state reaches
                 the goal. (False, -1) if no state reaches the goal.
        """
        for i, state in reversed(list(enumerate(trajectory.state_list))):
            if self.goal.is_reached(state):
                return True, i
        return False, -1

    def translate_rotate(self, translation: np.ndarray, angle: float):
        """
        translate and rotates the planning problem with given translation and
        angle around the origin (0, 0)

        :param translation: translation vector [x_off, y_off] in x- and y-direction
        :param angle: rotation angle in radian (counter-clockwise)
        """
        self.initial_state = self.initial_state.translate_rotate(translation,
                                                                 angle)
        self.goal.translate_rotate(translation, angle)

    def draw(self, renderer: IRenderer,
             draw_params: Union[ParamServer, dict, None] = None,
             call_stack: Optional[Tuple[str, ...]] = tuple()):
        renderer.draw_planning_problem(self, draw_params, call_stack)


class PlanningProblemSet(IDrawable):
    def __init__(self, planning_problem_list: Union[None, List[PlanningProblem]] = None):
        if planning_problem_list is None:
            planning_problem_list = []

        self._valid_planning_problem_list(planning_problem_list)

        self._planning_problem_dict = {
                planning_problem.planning_problem_id: planning_problem for
                planning_problem in
                                       planning_problem_list}

    @property
    def planning_problem_dict(self) -> Dict[int, PlanningProblem]:
        """Dict that contains all PlanningProblems that are added. Keys: Ids of planning problems"""
        return self._planning_problem_dict

    @planning_problem_dict.setter
    def planning_problem_dict(self, _dict):
        warnings.warn('<PlanningProblemSet/planning_problem_dict> planning_problem_dict is immutable')

    @staticmethod
    def _valid_planning_problem_list(planning_problem_list: List[PlanningProblem]):
        """
        Check if input list contains only PlanningProblem instances

        :param planning_problem_list: List[PlanningProblem]
        """
        assert isinstance(planning_problem_list, list), 'argument "planning_problem_list" of wrong type. ' \
                                                        'Expected type: %s. Got type: %s.' \
                                                        % (list, type(planning_problem_list))

        assert all(isinstance(p, PlanningProblem) for p in planning_problem_list), 'Elements of ' \
                                                                                   '"planning_problem_list" of wrong ' \
                                                                                   'type.'

    def add_planning_problem(self, planning_problem: PlanningProblem):
        """
        Adds the given planning problem to self.planning_problem_list

        :param planning_problem: Planning problem to add
        """
        assert isinstance(planning_problem, PlanningProblem), 'argument "planning_problem" of wrong type. ' \
                                                              'Expected type: %s. Got type: %s.' \
                                                              % (planning_problem, PlanningProblem)
        if planning_problem.planning_problem_id in self.planning_problem_dict.keys():
            raise ValueError(
                "Id {} is already used in PlanningProblemSet".format(planning_problem.planning_problem_id))

        self.planning_problem_dict[planning_problem.planning_problem_id] = planning_problem

    def find_planning_problem_by_id(self, planning_problem_id: int) -> PlanningProblem:
        """
        Searches in planning_problem_dict for a planning problem with the given id. Returns the planning problem or
        raises error, if id cannot be found.

        :param planning_problem_id: id to find
        :return: Planning problem with id planning_problem_id, Raises key error, if id not in the dict.
        """

        return self.planning_problem_dict[planning_problem_id]

    def translate_rotate(self, translation: np.ndarray, angle: float):
        """
        translate and rotates the planning problem set with given translation and angle around the origin (0, 0)

        :param translation: translation vector [x_off, y_off] in x- and y-direction
        :param angle: rotation angle in radian (counter-clockwise)
        """
        for planning_problem in self._planning_problem_dict.values():
            planning_problem.translate_rotate(translation, angle)

    def draw(self, renderer: IRenderer,
             draw_params: Union[ParamServer, dict, None] = None,
             call_stack: Optional[Tuple[str, ...]] = tuple()):
        renderer.draw_planning_problem_set(self, draw_params, call_stack)
