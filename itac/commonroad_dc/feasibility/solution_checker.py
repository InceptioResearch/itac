import math
from typing import Tuple, Dict

import numpy as np
from commonroad.common.solution import PlanningProblemSolution, TrajectoryType, Solution, VehicleModel
from commonroad.geometry.shape import Polygon, ShapeGroup
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State as StateTupleFactory, Trajectory, State
from commonroad_dc.pycrcc import CollisionChecker, CollisionObject

import commonroad_dc.feasibility.feasibility_checker as fc
from commonroad_dc.boundary import construction
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object, \
    create_collision_checker
from commonroad_dc.feasibility.feasibility_checker import FeasibilityException
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics


class SolutionCheckerException(Exception):
    """
    Main exception class for exceptions related to the solution checker.
    """
    pass


class CollisionException(SolutionCheckerException):
    """
    Exception class for exceptions related to the collisions.
    """
    pass


class GoalNotReachedException(SolutionCheckerException):
    """
    Exception class for exceptions related to the goal checks.
    """
    pass


class MissingSolutionException(SolutionCheckerException):
    """
    Exception class for exceptions related to the planning problem solutions.
    """
    pass


def _simulate_trajectory_if_input_vector(planning_problem_set: PlanningProblemSet,
                                         pp_solution: PlanningProblemSolution,
                                         dt: float) -> Tuple[VehicleDynamics, Trajectory]:
    vehicle_dynamics = VehicleDynamics.from_model(
        pp_solution.vehicle_model, pp_solution.vehicle_type)
    trajectory = pp_solution.trajectory
    if pp_solution.trajectory_type in [TrajectoryType.Input, TrajectoryType.PMInput]:
        initial_state = planning_problem_set.planning_problem_dict[
            pp_solution.planning_problem_id].initial_state
        trajectory = vehicle_dynamics.simulate_trajectory(
            initial_state, trajectory, dt)
    return vehicle_dynamics, trajectory


def _create_pp_solution_collision_object(planning_problem_set: PlanningProblemSet,
                                         pp_solution: PlanningProblemSolution,
                                         dt: float) -> CollisionObject:
    vehicle_dynamics, trajectory = _simulate_trajectory_if_input_vector(planning_problem_set,
                                                                        pp_solution, dt)
    trajectory_pred = TrajectoryPrediction(
        trajectory=trajectory, shape=vehicle_dynamics.shape, 
        vehicle_type=pp_solution.vehicle_type.name)
    collision_object = create_collision_object(trajectory_pred)
    return collision_object


def _construct_boundary_checker(scenario: Scenario) -> CollisionChecker:
    build = ['section_triangles', 'triangulation']
    boundary = construction.construct(scenario, build)
    road_boundary_shape_list = []
    initial_state = None
    for r in boundary['triangulation'].unpack():
        initial_state = StateTupleFactory(
            position=np.array([0, 0]), orientation=0.0, time_step=0)
        p = Polygon(np.array(r.vertices()))
        road_boundary_shape_list.append(p)
    road_bound = StaticObstacle(obstacle_id=scenario.generate_object_id(),
                                obstacle_type=ObstacleType.ROAD_BOUNDARY,
                                obstacle_shape=ShapeGroup(
                                    road_boundary_shape_list),
                                initial_state=initial_state)
    collision_checker = CollisionChecker()
    collision_checker.add_collision_object(create_collision_object(road_bound))
    return collision_checker


def _check_input_vector_feasibility(pp_solution: PlanningProblemSolution,
                                    initial_state: State,
                                    vehicle_dynamics: VehicleDynamics,
                                    dt: float) -> Tuple[bool, Trajectory]:
    try:
        feasible, simulated_trajectory = fc.input_vector_feasibility(initial_state, pp_solution.trajectory,
                                                                     vehicle_dynamics, dt)
        return feasible, simulated_trajectory

    except FeasibilityException as ex:
        msg = f'Solution is not feasible on planning problem solution {pp_solution.planning_problem_id}.'
        raise SolutionCheckerException(msg) from ex


def _check_trajectory_feasibility(pp_solution: PlanningProblemSolution,
                                  vehicle_dynamics: VehicleDynamics,
                                  dt: float) -> Tuple[bool, Trajectory]:
    try:
        feasible, reconstructed_inputs = fc.trajectory_feasibility(pp_solution.trajectory,
                                                                   vehicle_dynamics, dt)
        return feasible, reconstructed_inputs

    except FeasibilityException as ex:
        msg = f'Solution is not feasible on planning problem solution {pp_solution.planning_problem_id}.'
        raise SolutionCheckerException(msg) from ex


def _compute_input_vector_feasibility_cost(pp_solution: PlanningProblemSolution,
                                           initial_state: State,
                                           vehicle_dynamics: VehicleDynamics,
                                           dt: float) -> float:
    try:
        feasibility_ratio = fc.input_vector_feasibility_cost(initial_state, pp_solution.trajectory,
                                                                          vehicle_dynamics, dt)
        return feasibility_ratio

    except FeasibilityException as ex:
        msg = f'Solution is not feasible on planning problem solution {pp_solution.planning_problem_id}.'
        raise SolutionCheckerException(msg) from ex


def _compute_trajectory_feasibility_cost(pp_solution: PlanningProblemSolution,
                                         vehicle_dynamics: VehicleDynamics,
                                         dt: float) -> float:
    try:
        feasibility_ratio = fc.trajectory_feasibility_cost(pp_solution.trajectory,
                                                                        vehicle_dynamics, dt)
        return feasibility_ratio

    except FeasibilityException as ex:
        msg = f'Solution is not feasible on planning problem solution {pp_solution.planning_problem_id}.'
        raise SolutionCheckerException(msg) from ex


def solution_feasible(solution: Solution, dt: float,
                      planning_problem_set: PlanningProblemSet) -> Dict[int, Tuple[bool, Trajectory, Trajectory]]:
    """
    Checks whether the given solution is feasible.

    :param solution: Solution
    :param planning_problem_set: PlanningProblemSet
    :param dt: Scenario dt
    :return: planning problem id -> feasible, input or reconstructed input, trajectory or simulated trajectory.
        Raises FeasibilityException if there is any trajectory in the solution that is not feasible.
    """
    results = {}
    for pp_solution in solution.planning_problem_solutions:
        planning_problem = planning_problem_set.planning_problem_dict[
            pp_solution.planning_problem_id]
        initial_state = planning_problem.initial_state
        vehicle_dynamics = VehicleDynamics.from_model(
            pp_solution.vehicle_model, pp_solution.vehicle_type)

        if pp_solution.trajectory_type in [TrajectoryType.Input, TrajectoryType.PMInput]:
            feasible, simulated_trajectory = _check_input_vector_feasibility(pp_solution, initial_state,
                                                                             vehicle_dynamics, dt)
            results[pp_solution.planning_problem_id] = (
                feasible, pp_solution.trajectory, simulated_trajectory)
        else:
            feasible, inputs = _check_trajectory_feasibility(
                pp_solution, vehicle_dynamics, dt)
            results[pp_solution.planning_problem_id] = (
                feasible, inputs, pp_solution.trajectory)

    return results


def solution_feasible_cost(solution: Solution, dt: float,
                           planning_problem_set: PlanningProblemSet) -> Dict[int, float]:
    """
    Checks whether the given solution is feasible.

    :param solution: Solution
    :param planning_problem_set: PlanningProblemSet
    :param dt: Scenario dt
    :return: planning problem id -> feasible, input or reconstructed input, trajectory or simulated trajectory.
        Raises FeasibilityException if there is any trajectory in the solution that is not feasible.
    """
    results = {}
    for pp_solution in solution.planning_problem_solutions:
        planning_problem = planning_problem_set.planning_problem_dict[
            pp_solution.planning_problem_id]
        initial_state = planning_problem.initial_state
        vehicle_dynamics = VehicleDynamics.from_model(
            pp_solution.vehicle_model, pp_solution.vehicle_type)

        if pp_solution.trajectory_type in [TrajectoryType.Input, TrajectoryType.PMInput]:
            feasibility_ratio = _compute_input_vector_feasibility_cost(pp_solution, initial_state,
                                                                             vehicle_dynamics, dt)
            results[pp_solution.planning_problem_id] = (feasibility_ratio)
        else:
            feasibility_ratio = _compute_trajectory_feasibility_cost(
                pp_solution, vehicle_dynamics, dt)
            results[pp_solution.planning_problem_id] = (feasibility_ratio)

    return results


def obstacle_collision(scenario: Scenario,
                       planning_problem_set: PlanningProblemSet,
                       solution: Solution) -> bool:
    """
    Checks whether there is a collision between the ego vehicles of the solution and the scenario obstacles.

    :param scenario: Scenario
    :param planning_problem_set: PlanningProblemSet
    :param solution: Solution
    :return: False if there is no collision. Raises CollisionException if there are any obstacle collisions.
    """
    collision_checker = create_collision_checker(scenario)
    for pp_solution in solution.planning_problem_solutions:
        collision_object = _create_pp_solution_collision_object(planning_problem_set, pp_solution,
                                                                scenario.dt)
        if collision_checker.collide(collision_object):
            msg = f'There is a collision between the scenario obstacles and the ego vehicle in planning ' \
                  f'problem solution {pp_solution.planning_problem_id}'
            print(msg)
            return True

    return False


def boundary_collision(scenario: Scenario,
                       planning_problem_set: PlanningProblemSet,
                       solution: Solution) -> bool:
    """
    Checks whether the ego vehicles go out of lanelet boundaries.

    :param scenario: Scenario
    :param planning_problem_set: PlanningProblemSet
    :param solution: Solution
    :return: False if the ego vehicles stay in lanes. Raises CollisionException if there are any boundary collisions.
    """
    collision_checker = _construct_boundary_checker(scenario)
    for pp_solution in solution.planning_problem_solutions:
        collision_object = _create_pp_solution_collision_object(planning_problem_set, pp_solution,
                                                                scenario.dt)
        if collision_checker.collide(collision_object):
            msg = f'There is a collision between lanelet boundaries and the ego vehicle in planning ' \
                  f'problem solution {pp_solution.planning_problem_id}'
            print(msg)
            return True

    return False


def ego_collision(scenario: Scenario,
                  planning_problem_set: PlanningProblemSet,
                  solution: Solution) -> bool:
    """
    Checks whether the ego vehicles collide with each other in the solution.

    :param scenario: Scenario
    :param planning_problem_set: PlanningProblemSet
    :param solution: Solution
    :return: False if there is no collision. Raises CollisionException if there are any collisions between ego
        vehicles.
    """
    collision_checker = CollisionChecker()
    checked_pp_ids = []
    for pp_solution in solution.planning_problem_solutions:
        collision_object = _create_pp_solution_collision_object(planning_problem_set, pp_solution,
                                                                scenario.dt)
        if collision_checker.collide(collision_object):
            msg = f'There is a collision between ego vehicles in planning problem solutions {checked_pp_ids}'
            raise CollisionException(msg)

        collision_checker.add_collision_object(collision_object)
        checked_pp_ids.append(pp_solution.planning_problem_id)
    return False


def goal_reached(scenario: Scenario,
                 planning_problem_set: PlanningProblemSet,
                 solution: Solution) -> bool:
    """
    Checks whether the goal has been reached for each of the planning problem solutions.

    :param scenario: Scenario
    :param planning_problem_set: PlanningProblemSet
    :param solution: Solution
    :return: True if all reached goal. Raises GoalNotReachedException if there are any planning problem solutions
        that have not reached the goal position.
    """
    for pp_solution in solution.planning_problem_solutions:
        vehicle_dynamics, trajectory = _simulate_trajectory_if_input_vector(planning_problem_set,
                                                                            pp_solution,
                                                                            scenario.dt)
        planning_problem = planning_problem_set.planning_problem_dict[
            pp_solution.planning_problem_id]
        if not planning_problem.goal_reached(trajectory)[0]:
            msg = f'Ego vehicle has not reached the goal in planning planning problem solution ' \
                  f'{pp_solution.planning_problem_id}.'
            raise GoalNotReachedException(msg)

    return True


def solved_all_problems(planning_problem_set: PlanningProblemSet, solution: Solution) -> bool:
    """
    Checks whether the solution has solved all planning problems of the scenario.

    :param planning_problem_set: PlanningProblemSet
    :param solution: Solution
    :return: True if there is a solution for each planning problem. Raises MissingSolutionException if there are any
        missing planning problem solutions.
    """
    pp_ids = set(list(planning_problem_set.planning_problem_dict.keys()))
    solution_pp_ids = set(solution.planning_problem_ids)
    if not pp_ids == solution_pp_ids:
        msg = f'All planning problems of the scenario must be solved! Scenario planning problems: {pp_ids}, ' \
              f'Solved planning problems: {solution_pp_ids}'
        raise MissingSolutionException(msg)

    return True


def starts_at_correct_state(solution: Solution, planning_problem_set: PlanningProblemSet) -> bool:
    """
    Checks whether the given solution trajectories or input vectors start at the correct state/time step.

    Expected time steps are:
    - Equal to planning problem's initial time step, if input vector solution
    - Planning problem's initial state, if trajectory solution

    :param solution: Solution
    :param planning_problem_set: PlanningProblemSet
    :return: True if all solution trajectories start at correct time step
    """
    for pp_solution in solution.planning_problem_solutions:
        planning_problem = planning_problem_set.planning_problem_dict[
            pp_solution.planning_problem_id]
        is_input_vector = pp_solution.trajectory_type in [
            TrajectoryType.Input, TrajectoryType.PMInput]
        initial_state_pp = planning_problem.initial_state
        initial_state_sol = pp_solution.trajectory.state_list[0]
        ts = initial_state_sol.time_step
        expected_ts = [initial_state_pp.time_step]

        if is_input_vector:
            if ts not in expected_ts:
                msg = f'Planning Problem Solutionn with input vector does not ' \
                      f'start at correct time step!\nPlanning Problem ID: {pp_solution.planning_problem_id}' \
                      f'\nExpected time step: {expected_ts}\nActual time step: {ts}'
                raise SolutionCheckerException(msg)
        else:
            for attr in initial_state_pp.attributes:
                if not hasattr(initial_state_sol, attr):
                    continue

                solution_attr_tmp = getattr(initial_state_sol, attr)
                if pp_solution.vehicle_model == VehicleModel.PM:
                    if attr == "orientation":
                        solution_attr_tmp = math.atan2(
                            initial_state_sol.velocity_y, initial_state_sol.velocity)
                    elif attr == "velocity":
                        solution_attr_tmp = math.sqrt(
                            initial_state_sol.velocity_y**2 + initial_state_sol.velocity**2)

                if attr == "velocity":
                    # tolerance required for motion primitives
                    atol = 2.0
                else:
                    atol = 0.1

                if not np.allclose(getattr(initial_state_pp, attr), solution_attr_tmp, atol=atol):
                    msg = f'Planning Problem Solution trajectory does not ' \
                          f'start at the initial_state of the planning problem with ID: ' \
                          f'{pp_solution.planning_problem_id}' \
                          f'\nExpected {attr}={getattr(initial_state_pp, attr)}\n' \
                          f'Received {attr}={solution_attr_tmp}'
                    raise SolutionCheckerException(msg)

    return True


def valid_solution(scenario: Scenario,
                   planning_problem_set: PlanningProblemSet,
                   solution: Solution) -> Tuple[bool, Dict[int, Tuple[bool, Trajectory, Trajectory]]]:
    """
    Checks whether a solution is valid or not by checking
        - Solution has solved all planning problems of the scenario (solved_all_problems)
        - All planning problem solutions reached goal (goal_reached)
        - All planning problem solutions trajectories/input vectors start at correct time step (starts_at_correct_ts)
        - All planning problem solutions are feasible (solution_feasible)
        - There isn't a collision between the ego vehicles and the scenario obstacles (obstacle_collision)
        - The ego vehicles don't go out of lane boundaries (boundary_collision)
        - The ego vehicles don't collide with each other (ego_collision)

    It returns a dictionary that contains the inputs (or reconstructed inputs if trajectory solution), and
    trajectory (simulated trajectory if input vector solution).

    The valid_solution functions is being used as a basis for validity when a submission was made to
    `commonroad.in.tum.de website <https://commonroad.in.tum.de>`_ for evaluation. If the solution is not
    valid according to this function, then the solution will not be accepted.

    :param scenario: Scenario
    :param planning_problem_set: PlanningProblemSet
    :param solution: Solution
    :return: True if all checks pass successfully, and dictionary for simulated trajectories or reconstructed
        inputs. Raises SolutionCheckerException if any of the checks fail.
    """
    valid = all([
        solved_all_problems(planning_problem_set, solution),
        goal_reached(scenario, planning_problem_set, solution),
        starts_at_correct_state(solution, planning_problem_set),
        not obstacle_collision(scenario, planning_problem_set, solution),
        not boundary_collision(scenario, planning_problem_set, solution),
        not ego_collision(scenario, planning_problem_set, solution)
    ])
    results = solution_feasible(solution, scenario.dt, planning_problem_set)
    all_feasible = all([
        result[0]
        for pp_id, result in results.items()
    ])
    return valid and all_feasible, results
