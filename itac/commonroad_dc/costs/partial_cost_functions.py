from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.util import Interval
from commonroad.geometry.shape import ShapeGroup
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.route_matcher import SolutionProperties
from scipy.integrate import simps


class PartialCostFunctionException(Exception):
    pass


def euclidean_dist(x1: np.array, x2: np.array) -> float:
    """
    Returns the euclidean distance between two points.
    """
    return np.linalg.norm(x2 - x1)


def position_lanelets(scenario: Scenario, position: np.ndarray) -> List[Lanelet]:
    """
    Returns the list of lanelets that contains the given position
    """
    position_lanelet_ids = scenario.lanelet_network.find_lanelet_by_position(
        [position]
    )[0]
    position_lanelets = [
        scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        for lanelet_id in position_lanelet_ids
    ]
    return position_lanelets


def acceleration_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the acceleration cost for the given trajectory.
    """
    try:
        velocity = [state.velocity for state in trajectory.state_list]
        acceleration = np.diff(velocity) / scenario.dt
        acceleration_sq = np.square(acceleration)
        cost = simps(acceleration_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of acceleration cost!"
        raise PartialCostFunctionException(msg) from ex


def jerk_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the jerk cost for the given trajectory.
    """
    try:
        velocity = [state.velocity for state in trajectory.state_list]
        acceleration = np.diff(velocity) / scenario.dt
        jerk = np.diff(acceleration) / scenario.dt
        jerk_sq = np.square(jerk)
        cost = simps(jerk_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of jerk cost!"
        raise PartialCostFunctionException(msg) from ex


def jerk_lat_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the lateral jerk cost for the given trajectory.
    """
    try:
        lat_jerk = [state.lat_jerk for state in trajectory.state_list]
        jerk_sq = np.square(lat_jerk)
        cost = simps(jerk_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of lateraljerk cost!"
        raise PartialCostFunctionException(msg) from ex


def jerk_lon_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the longitudinal jerk cost for the given trajectory.
    """
    try:
        lon_jerk = [state.lon_jerk for state in trajectory.state_list]
        jerk_sq = np.square(lon_jerk)
        cost = simps(jerk_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of longitudinal jerk cost!"
        raise PartialCostFunctionException(msg) from ex


def steering_angle_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the steering angle cost for the given trajectory.
    """
    try:
        steering_angle = [
            state.steering_angle for state in trajectory.state_list]
        steering_angle_sq = np.square(steering_angle)
        cost = simps(steering_angle_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of steering angle cost!"
        raise PartialCostFunctionException(msg) from ex


def steering_rate_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the steering rate cost for the given trajectory.
    """
    try:
        steering_angle = [
            state.steering_angle for state in trajectory.state_list]
        steering_rate = np.diff(steering_angle) / scenario.dt
        steering_rate_sq = np.square(steering_rate)
        cost = simps(steering_rate_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of steering rate cost!"
        raise PartialCostFunctionException(msg) from ex


def compute_cost_by_attr(scenario: Scenario, trajectory: Trajectory, attr: str) -> float:
    attr_values = [getattr(state, attr) for state in trajectory.state_list]
    rate = np.diff(attr_values) / scenario.dt
    rate_sq = np.square(rate)
    cost = simps(rate_sq, dx=scenario.dt)
    return cost


def yaw_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the yaw cost for the given trajectory.
    """
    try:
        cost_all = 0
        for attr in ['yaw_angle_trailer', 'orientation']:
            if hasattr(trajectory.state_list[0], attr):
                t_cost = compute_cost_by_attr(
                    scenario, trajectory, attr)
                cost_all += t_cost
        return cost_all
    except Exception as ex:
        msg = f"An exception occurred during calculation of yaw cost!"
        raise PartialCostFunctionException(msg) from ex


def path_length_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the path length cost for the given trajectory.
    """
    try:
        velocity = [state.velocity for state in trajectory.state_list]
        cost = simps(velocity, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of path length cost!"
        raise PartialCostFunctionException(msg) from ex


def time_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the time cost for the given trajectory.
    """
    try:
        duration = (
            trajectory.state_list[-1].time_step -
            trajectory.state_list[0].time_step
        ) * scenario.dt
        return duration
    except Exception as ex:
        msg = f"An exception occurred during calculation of time cost!"
        raise PartialCostFunctionException(msg) from ex


def inverse_duration_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the inverse time cost for the given trajectory.
    """
    try:
        return 1 / min(
            time_cost(scenario, planning_problem, trajectory, properties), 0.1
        )  # in case trajectory has 0 ts
    except Exception as ex:
        msg = f"An exception occurred during calculation of inverse time cost!"
        raise PartialCostFunctionException(msg) from ex


def lane_center_offset_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the Lane Center Offset cost.

    TODO: Correct implementation in the future (priority low since depends on complicated calculation of ref path)
    """
    try:
        dists_to_lane_centers_sq = np.square(
            [s.lat_position for s in trajectory.state_list])
        cost = simps(dists_to_lane_centers_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of lane center offset cost!"
        raise PartialCostFunctionException(msg) from ex


def orientation_offset_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the Orientation Offset cost.

    """
    try:
        orientation_rel_lane_centers = np.square(
            [s.delta_orientation for s in trajectory.state_list])
        cost = simps(orientation_rel_lane_centers, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of orientation offset cost!"
        raise PartialCostFunctionException(msg) from ex


def velocity_offset_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the Velocity Offset cost.

    """
    try:
        goal_velocities = [
            goal_state.velocity.start
            if isinstance(goal_state.velocity, Interval)
            else goal_state.velocity
            for goal_state in planning_problem.goal.state_list
            if hasattr(goal_state, "velocity")
        ]
        goal_velocity = min(goal_velocities) if len(
            goal_velocities) > 0 else None

        velocity_diffs = []
        for state in trajectory.state_list:
            diff = goal_velocity - state.velocity if goal_velocity else 0
            velocity_diffs.append(diff)

        velocity_diffs_sq = np.square(velocity_diffs)
        cost = simps(velocity_diffs_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of velocity offset cost!"
        raise PartialCostFunctionException(msg) from ex


def longitudinal_velocity_offset_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the Velocity Offset cost.

    TODO: Correct implementation in the future (priority low since depends on complicated calculation of ref path)
    """
    try:
        goal_velocities = [
            goal_state.velocity.start
            if isinstance(goal_state.velocity, Interval)
            else goal_state.velocity
            for goal_state in planning_problem.goal.state_list
            if hasattr(goal_state, "long_velocity")
        ]
        goal_velocity = min(goal_velocities) if len(
            goal_velocities) > 0 else None

        velocity_diffs = []
        for state in trajectory.state_list:
            diff = goal_velocity - state.long_velocity if goal_velocity else 0
            velocity_diffs.append(diff)

        velocity_diffs_sq = np.square(velocity_diffs)
        cost = simps(velocity_diffs_sq, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of velocity offset cost!"
        raise PartialCostFunctionException(msg) from ex


def _get_shape_center(shape):
    # TODO make recursive later
    if not isinstance(shape, ShapeGroup):
        return shape.center
    else:
        x = np.array([shape.center[0] for shape in shape.shapes])
        y = np.array([shape.center[1] for shape in shape.shapes])
    return np.array([np.mean(x), np.mean(y)])


def distance_to_obstacle_cost(
    scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory,
        properties: Dict[SolutionProperties, Dict[int, Any]]
) -> float:
    """
    Calculates the Distance to Obstacle cost.
    """
    try:
        min_dists = []
        for state in trajectory.state_list:
            min_dists.append(
                np.min(properties[SolutionProperties.LonDistanceObstacles][state.time_step]))
        neg_min_dists = -0.2 * np.array(min_dists)
        exp_dists = np.array([np.math.exp(val) for val in neg_min_dists])
        cost = simps(exp_dists, dx=scenario.dt)
        return cost
    except Exception as ex:
        msg = f"An exception occurred during calculation of distance to obstacles cost!"
        raise PartialCostFunctionException(msg) from ex
