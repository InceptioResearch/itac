"""
Module for scenario related helper methods for the CommonRoad Gym environment
"""
import numpy as np
from typing import Tuple, Union, List, Set
from commonroad.common.solution import VehicleModel, VehicleType
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle
from commonroad.scenario.trajectory import State
from commonroad.scenario.scenario import ScenarioID, Scenario
from commonroad_dc.feasibility.feasibility_checker import trajectory_feasibility
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, VehicleParameters
from shapely.geometry import LineString


def parse_map_name(scenario_id: ScenarioID) -> str:
    return f"{scenario_id.map_name}-{str(scenario_id.map_id)}"


def make_valid_orientation(angle: float) -> float:
    # TODO: update this function in commonroad.common.util
    TWO_PI = 2.0 * np.pi
    angle = angle % TWO_PI
    if np.pi <= angle <= TWO_PI:
        angle = angle - TWO_PI
    assert -np.pi <= angle <= np.pi
    return angle


def get_road_edge(scenario) -> Tuple[dict, dict, dict, dict]:
    """
    TODO only used in preprocessing -> relocate there maybe??

    Get the road edge or solid white line of a lanelet.
    :return: Dictionary of left and right lanelet ids and road edge
    """
    left_road_edge_lanelet_id = {}
    right_road_edge_lanelet_id = {}
    left_road_edge = {}
    right_road_edge = {}
    for lanelet in scenario.lanelet_network.lanelets:
        if lanelet.lanelet_id not in right_road_edge_lanelet_id.keys():
            # Search the right most lanelet and use right vertices as right bound
            start = lanelet
            temp = []
            while start.adj_right_same_direction:
                temp.append(start.lanelet_id)
                start = scenario.lanelet_network.find_lanelet_by_id(start.adj_right)
            temp.append(start.lanelet_id)
            right_bound = LineString(start.right_vertices)
            right_road_edge_lanelet_id.update({k: start.lanelet_id for k in temp})
            right_road_edge[start.lanelet_id] = right_bound
        if lanelet.lanelet_id not in left_road_edge_lanelet_id.keys():
            start = lanelet
            temp = []
            # Search the left most lanelet and use right vertices as left bound
            while start.adj_left_same_direction:
                temp.append(start.lanelet_id)
                start = scenario.lanelet_network.find_lanelet_by_id(start.adj_left)
            temp.append(start.lanelet_id)
            left_bound = LineString(start.left_vertices)
            left_road_edge_lanelet_id.update({k: start.lanelet_id for k in temp})
            left_road_edge[start.lanelet_id] = left_bound
    return (
        left_road_edge_lanelet_id,
        left_road_edge,
        right_road_edge_lanelet_id,
        right_road_edge,
    )


def get_lane_marker(ego_vehicle_lanelet: Lanelet) -> Tuple[LineString, LineString]:
    """
    TODO remove, only used in old commonroad_env class

    Get the lane marker for the desired lanelet.

    :param ego_vehicle_lanelet: lanelet of ego vehicle
    :return: left and right lane marker
    """
    left_marker_line = LineString(ego_vehicle_lanelet.left_vertices)
    right_marker_line = LineString(ego_vehicle_lanelet.right_vertices)
    return left_marker_line, right_marker_line


def interpolate_steering_angles(
        state_list: List[State], parameters: VehicleParameters, dt: float
) -> List[State]:
    """
    Interpolates the not defined steering angles based on KS Model

    :param state_list: The list of the states
    :param parameters: The parameters of the vehicle
    :param dt: dt of the scenario
    :return: The state list with interpolated steering angles
    """
    if len(state_list) == 0:
        return state_list

    l_wb = parameters.a + parameters.b

    [orientations, velocities] = np.array(
        [[state.orientation, state.velocity] for state in state_list]
    ).T

    orientation_vectors = approx_orientation_vector(orientations)
    psi_dots = (
            angle_difference(orientation_vectors[:, :-1].T, orientation_vectors[:, 1:].T) / dt
    )
    avg_velocities = np.mean(np.array([velocities[:-1], velocities[1:]]), axis=0)
    avg_velocities[avg_velocities == 0.0] += np.finfo(float).eps

    steering_angles = np.arctan(psi_dots * l_wb / avg_velocities)
    if len(steering_angles) > 0:
        steering_angles = np.hstack((steering_angles, steering_angles[-1]))
    else:
        default_steering_angle = 0.0
        steering_angles = np.array([default_steering_angle])

    steering_angles = np.clip(
        steering_angles, parameters.steering.min, parameters.steering.max
    )

    def get_state_with_steering_angle(state: State, steering_angle: float):
        if hasattr(state, "steering_angle"):
            if state.steering_angle is None:
                state.steering_angle = steering_angle
        else:
            setattr(state, "steering_angle", steering_angle)
        return state

    return list(
        map(
            lambda state, steering_angle: get_state_with_steering_angle(
                state, steering_angle
            ),
            state_list,
            steering_angles,
        )
    )


def interpolate_steering_angles_of_obstacle(obstacle: DynamicObstacle, parameters: VehicleParameters, dt: float):
    """
    Interpolates the not defined steering angles of obstacle based on KS Model

    :param obstacle:
    :param parameters: The parameters of the vehicle
    :param dt: dt of the scenario
    """
    trajectory = obstacle.prediction.trajectory
    trajectory.state_list = interpolate_steering_angles(trajectory.state_list, parameters, dt)
    obstacle.initial_state.steering_angle = trajectory.state_list[0].steering_angle


def check_trajectory(
        obstacle: DynamicObstacle,
        vehicle_model: VehicleModel,
        vehicle_type: VehicleType,
        dt: float,
) -> bool:
    """
    Checks whether the trajectory of a given obstacle is feasible with a given vehicle model
    Note: Currently it is implemented for the KS model. As soon as the workaround is not needed,
    it can be rebased to fully use the Feasibility Checker

    :param obstacle: The obstacle which trajectory should be checked
    :param vehicle_model: The used vehicle model
    :param vehicle_type: THe type of the vehicle
    :param dt: Delta time of the simulation
    :return: True if the trajectory is feasible
    """
    trajectory = obstacle.prediction.trajectory
    vehicle_dynamics = VehicleDynamics.from_model(vehicle_model, vehicle_type)

    position_tolerance = 0.1
    orientation_tolerance = 2e-2

    e = np.array([position_tolerance, position_tolerance, orientation_tolerance])

    feasible, _ = trajectory_feasibility(trajectory, vehicle_dynamics, dt, e=e)
    return feasible


def approx_orientation_vector(orientation: Union[float, np.ndarray]) -> np.ndarray:
    """
    Approximate normed vector in a given orientation

    :param orientation: The orientation
    :return Normalized vector points to the defined orientation
    """
    return np.array([np.cos(orientation), np.sin(orientation)])


def angle_difference(vector_from: np.ndarray, vector_to: np.ndarray):
    """
    Returns angle between the two provided vectors, from v1 to v2

    :param vector_from: Vector from the angle should be measured
    :param vector_to: Vector to the angle should be measured
    :return: Signed relative angle between the two vectors
    """
    assert vector_from.ndim <= 2 and vector_to.ndim <= 2
    if vector_from.ndim == 1:
        vector_from = vector_from[None]
    if vector_to.ndim == 1:
        vector_to = vector_to[None]
    dot_product = np.einsum("ij,ij->i", vector_from, vector_to)
    determinant = np.einsum("ij,ij->i", vector_from, (np.flip(vector_to, axis=-1) * (np.array([1, -1]))))

    return np.arctan2(determinant, dot_product)