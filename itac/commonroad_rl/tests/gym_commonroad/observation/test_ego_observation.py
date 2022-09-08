import numpy as np
from commonroad.common.solution import VehicleModel
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.trajectory import State

from commonroad_rl.gym_commonroad.action.action import ContinuousAction
from commonroad_rl.gym_commonroad.observation import EgoObservation
from commonroad_rl.tests.common.marker import *


def construct_ego_observation(vehicle_model):
    # Ego-related observation settings
    return EgoObservation(
        configs={"ego_configs":
                     {"observe_v_ego": True,
                      "observe_a_ego": True,
                      "observe_relative_heading": True,
                      "observe_steering_angle": True,
                      "observe_global_turn_rate": True,
                      "observe_remaining_steps": True,
                      "observe_is_friction_violation": True},
                 "vehicle_params": {
                     "vehicle_type": 2,
                     "vehicle_model": vehicle_model
                 }})


def prepare_for_ego_test():
    # Initial state of KS vehicle model
    dummy_state_ks = {
        "position": np.array([0.0, 0.0]),
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "time_step": 0,
        "orientation": 0.0,
    }

    # Initial state of PM vehicle model
    dummy_state_pm = {
        "position": np.array([0.0, 0.0]),
        "time_step": 0,
    }

    # lanelet config
    lanelet = Lanelet(lanelet_id=0,
                      left_vertices=np.array([[0.0, 3.0], [10.0, 3.0]]),
                      center_vertices=np.array([[0.0, 0.0], [10.0, 0.0]]),
                      right_vertices=np.array([[0.0, -3.0], [10.0, -3.0]]))

    return dummy_state_pm, dummy_state_ks, lanelet


@pytest.mark.parametrize(
    ("ego_angle", "expected_output"),
    [
        (0, 0),
        (1, 1),
        (6, 6 - 2 * np.pi),
        (2 * np.pi, 0),
        (7, 7 - 2 * np.pi),
        (-1, -1),
        (-2 * np.pi, 0),
    ],
)
@unit_test
@functional
def test_get_lane_relative_heading(ego_angle, expected_output):
    dummy_state = {
        "velocity": 0.0,
        "position": np.array([0.0, 0.0]),
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "time_step": 0.0,
    }
    ego_vehicle_state = State(**dummy_state, orientation=ego_angle)
    ego_vehicle_lanelet = Lanelet(
        np.array([[0.0, 1.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, -1.0], [1.0, -1.0]]),
        0,
    )
    relative_angle = EgoObservation.get_lane_relative_heading(ego_vehicle_state, ego_vehicle_lanelet)
    assert np.isclose(relative_angle, expected_output)


dummy_state_pm, dummy_state_ks, ego_lanelet = prepare_for_ego_test()


@pytest.mark.parametrize(
    ("velocity", "steering_angle", "vehicle_model", "expected_output"),
    [
        (0, 0, 2, np.array([0., 0., 0., 0., 10.])),
        (10, 0.5, 2, np.array([10., 0., 0.5, 2.118, 10.])),
        (30, 0.5, 2, np.array([30., 0., 0.5, 6.355, 10.])),
        ([0, 0], None, 0, np.array([0., 0., 0., 0., 10.])),
        ([10, 0], None, 0, np.array([10., 0., 0., 0., 10.])),
        ([10, 2], None, 0, np.array([10., 0., 0., 0., 10.])),
    ]
)
@functional
@unit_test
def test_ego_state(velocity, steering_angle, vehicle_model, expected_output):
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": vehicle_model,  # 0: VehicleModel.PM; 2: VehicleModel.KS;
    }

    action_configs = {
        "action_type": "continuous", # discrete
        "action_base": "acceleration", # acceleration; jerk
        "long_steps": 5,
        "lat_steps": 5,

    }

    vehicle_action = ContinuousAction(vehicle_params, action_configs)
    if vehicle_params["vehicle_model"] == 2:  # VehicleModel.KS
        initial_state = State(**dummy_state_ks, velocity=velocity, steering_angle=steering_angle, )
    else:  # VehicleModel.PM
        initial_state = State(**dummy_state_pm, velocity=velocity[0], velocity_y=velocity[1])
    vehicle_action.reset(initial_state, dt=1.0)

    # Not to do anything, just continue the way with the given velocity
    action = np.array([0.0, 0.0])
    steps = 10
    for _ in range(steps):
        vehicle_action.step(action)

    ego_observation = construct_ego_observation(vehicle_model)
    ego_observation.observe(ego_lanelet, vehicle_action.vehicle, episode_length=20)
    observation = ego_observation.observation_dict
    if vehicle_model == 2:
        # Ego-related observations of KS model_type
        result = np.hstack((observation["v_ego"],
                            observation["a_ego"],
                            observation["steering_angle"],
                            observation["global_turn_rate"],
                            observation["remaining_steps"]))
    else:
        # Ego-related observations of PM model_type
        result = np.hstack((observation["v_ego"],
                            observation["a_ego"],
                            observation["remaining_steps"]))
    assert np.allclose(result, expected_output, rtol=0.01)


@pytest.mark.parametrize(
    ("vehicle_model", "action", "velocity", "steering_angle", "expected_output"),
    [
        # 0: VehicleModel.PM; 2: VehicleModel.KS
        (VehicleModel.PM, [0., 0.], [30., 0.], None, np.zeros(5)),
        (VehicleModel.PM, [0.5, 0.5], [30., 0.], None, np.zeros(5)),
        (VehicleModel.KS, [0., 0.], 30., 0.0, np.zeros(5)),
        (VehicleModel.KS, [0., 0.1], 30., 0.0, np.zeros(5)),
        (VehicleModel.KS, [0.02, 0.1], 30., 0.0, np.array([0, 0, 0, 0, 1])),
        (VehicleModel.KS, [0.05, 0.1], 30., 0.0, np.array([0, 0, 1, 1, 1])),
        (VehicleModel.KS, [0.2, 0.1], 30., 0.0, np.array([0, 1, 1, 1, 1])),
    ]
)
@functional
@unit_test
def test_check_friction_violation(vehicle_model, action, velocity, steering_angle, expected_output):
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": vehicle_model,  # 0: VehicleModel.PM; 2: VehicleModel.KS;
    }
    action_configs = {
        "action_type": "continuous", # discrete
        "action_base": "acceleration", # acceleration; jerk
        "long_steps": 5,
        "lat_steps": 5,

    }
    ego_observation = construct_ego_observation(vehicle_model)

    vehicle_action = ContinuousAction(vehicle_params, action_configs)
    if vehicle_params["vehicle_model"] == VehicleModel.KS:
        initial_state = State(**dummy_state_ks, velocity=velocity, steering_angle=steering_angle, )
    else:  # VehicleModel.PM
        initial_state = State(**dummy_state_pm, velocity=velocity[0], velocity_y=velocity[1])
    vehicle_action.reset(initial_state, dt=1.0)

    steps = 5
    result = np.zeros(steps)
    for i in range(steps):
        vehicle_action.step(action)
        ego_observation.observe(ego_lanelet, vehicle_action.vehicle, episode_length=20)
        result[i] = int(ego_observation.observation_dict["is_friction_violation"])
    assert np.all(result == expected_output)


@pytest.mark.parametrize(
    ("position", "polyline", "desired_orientation"),
    [
        (
                np.array([0, 0]),
                np.array([[0, 0], [1, 1], [0, 1]]) + 1,
                0.785398163
        ),
        (
                np.array([0, 0]),
                np.array([[100, -3], [0, 0], [-100, -3], [0, -103]]),
                3.111601,
        ),
        (
                np.array([0, 0]),
                np.array([[-7.3275, 12.5257],
                          [-7.5254, 9.1777],
                          [-11.278, 9.1652],
                          [-15.0305, 9.1526],
                          [-15.1272, 12.6073],
                          [-11.2273, 12.5665]]),
                -1.6298375
        ),
        (
                np.array([0, 0]),
                np.array([[-73.275, 12.5257],
                          [-75.254, 9.1777],
                          [-112.78, 9.1652],
                          [-150.305, 9.1526],
                          [-151.272, 12.6073],
                          [-112.273, 12.5665]]),
                -2.1046453
        ),
        (
                np.array([7, 12]),
                np.array([[-7.3275, 12.5257],
                          [-7.5254, 9.1777],
                          [-11.278, 9.1652],
                          [-15.0305, 9.1526],
                          [-15.1272, 12.6073],
                          [-11.2273, 12.5665]]),
                -1.6298375
        )
    ]
)
@functional
@unit_test
def test_get_orientation_of_polyline(position: np.ndarray, polyline: np.ndarray, desired_orientation: float):
    orientation = EgoObservation._get_orientation_of_polyline(position, polyline)
    assert np.isclose(orientation, desired_orientation)
