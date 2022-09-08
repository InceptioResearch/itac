

import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}')

from vehiclemodels import parameters_sedan, parameters_semi_trailer  # noqa
from commonroad.common.solution import CommonRoadSolutionWriter, Solution, PlanningProblemSolution, VehicleModel, VehicleType, CostFunction  # noqa
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet  # noqa
from commonroad.prediction.prediction import TrajectoryPrediction  # noqa
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType  # noqa
from commonroad.geometry.shape import Rectangle  # noqa
from commonroad.visualization.mp_renderer import MPRenderer  # noqa
from commonroad.common.file_reader import CommonRoadFileReader  # noqa
from commonroad.scenario.trajectory import Trajectory, State  # noqa
from commonroad.scenario.scenario import Scenario  # noqa

from SMP.motion_planner.motion_planner import MotionPlanner, MotionPlannerType  # noqa
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton  # noqa
from SMP.motion_planner.utility import plot_primitives  # noqa
from SMP.motion_planner.utility import create_trajectory_from_list_states, create_trajectory_from_list_states_semi_trailer  # noqa
from SMP.motion_planner.motion_planner_semi_trailer import TruckMotionPlanner, TruckMotionPlannerType  # NOQA
from SMP.maneuver_automaton.maneuver_automaton_semi_trailer import TruckManeuverAutomaton  # NOQA
