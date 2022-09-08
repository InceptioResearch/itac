"""
Unit tests of the module gym_commonroad.action.vehicle
"""
from commonroad_rl.gym_commonroad.action import *
from commonroad_rl.gym_commonroad.action.action import _rotate_to_curvi
from commonroad_rl.tests.common.marker import *

@pytest.mark.parametrize(
    ("action", "expected_position"),
    [
        (
                np.array([0, 2]),
                np.array([168.75, 0.0]),
        ),
        (
                np.array([1, 2]),
                np.array([96.875, 0.0]),
        ),
        (
                np.array([2, 2]),
                np.array([25.00, 0.0]),
        ),
        (
                np.array([3, 2]),
                np.array([-46.875, 0.0]),
        ),
        (
                np.array([4, 2]),
                np.array([-118.75, 0.0]),
        ),
        (
                np.array([2, 0]),
                np.array([25, 143.75]),
        ),
        (
                np.array([2, 1]),
                np.array([25, 71.875]),
        ),
        (
                np.array([2, 3]),
                np.array([25, -71.875]),
        ),
        (
                np.array([2, 4]),
                np.array([25, -143.75]),
        ),
    ],
)
@unit_test
@functional
def test_discrete_pm_planner(action, expected_position):
    """
    Tests the discrete point mass planner
    """
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": 0,
    }
    vehicle_action = DiscretePMAction(vehicle_params, long_steps=5, lat_steps=5)
    initial_state = State(
        **{
            "position": np.array([0., 0.]),
            "velocity": 5,
            "velocity_y": 0,
            "orientation": 0.0,
            "time_step": 0,
        })
    vehicle_action.reset(initial_state, dt=1.0)
    steps = 5
    for i in range(steps):
        vehicle_action.step(action)
    position = vehicle_action.vehicle.state.position

    assert np.allclose(position, expected_position, atol=0.01)


@pytest.mark.parametrize(
    ("action", "expected_position"),
    [
        (
                np.array([2, 2]),
                np.array([25.00, 0.0]),
        ),
        (
                np.array([0, 2]),
                np.array([162.00, 0.0]),
        ),
        (
                np.array([2, 0]),
                np.array([25.00, 137.00]),
        ),
        (
                np.array([2, 4]),
                np.array([25, -137.00]),
        ),
        (
                np.array([4, 2]),
                np.array([-112.00, 0.0]),
        ),
        (
                np.array([4, 4]),
                np.array([-76.646, -101.646]),
        ),
        (
                np.array([0, 0]),
                np.array([126.646, 101.646]),
        ),
        (
                np.array([1, 3]),
                np.array([112.553, -87.553]),
        ),
    ],
)
@unit_test
@functional
def test_discrete_jerk_planner(action, expected_position):
    """
    Tests the discrete point mass planner
    """
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": 0,
    }
    vehicle_action = DiscretePMJerkAction(vehicle_params, long_steps=5, lat_steps=5)
    initial_state = State(
        **{
            "position": np.array([0., 0.]),
            "velocity": 5,
            "velocity_y": 0,
            "orientation": 0.0,
            "time_step": 0,
            "acceleration": 0.0,
            "acceleration_y": 0.0,
        })
    vehicle_action.reset(initial_state, dt=1.0)
    steps = 5
    for i in range(steps):
        vehicle_action.step(action)
    position = vehicle_action.vehicle.state.position

    assert np.allclose(position, expected_position, atol=0.01)


@pytest.mark.parametrize(
    ("action", "expected_position"),
    [
        (
                [np.array([4, 2]), np.array([4, 2]), np.array([0, 2]), np.array([0, 2])],
                np.array([-43.499, 0.0]),
        ),
        (
                [np.array([0, 2]), np.array([0, 2]), np.array([4, 2]), np.array([4, 2])],
                np.array([83.500, 0.0]),
        ),
        (
                [np.array([0, 2]), np.array([0, 2]), np.array([4, 2]), np.array([3, 2])],
                np.array([86.000, 0.0]),
        ),
        (
                [np.array([2, 0]), np.array([2, 0]), np.array([2, 4]), np.array([2, 4])],
                np.array([20.00, 63.500]),
        ),
        (
                [np.array([0, 0]), np.array([2, 2]), np.array([4, 4]), np.array([2, 2])],
                np.array([68.790, 48.790]),
        ),
    ],
)
@unit_test
@functional
def test_discrete_jerk_planner_clipping(action, expected_position):
    """
    Tests the discrete point mass planner
    """
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": 0,
    }
    vehicle_action = DiscretePMJerkAction(vehicle_params, long_steps=5, lat_steps=5)
    initial_state = State(
        **{
            "position": np.array([0., 0.]),
            "velocity": 5,
            "velocity_y": 0,
            "orientation": 0.0,
            "time_step": 0,
            "acceleration": 0.0,
            "acceleration_y": 0.0,
        })
    vehicle_action.reset(initial_state, dt=1.0)
    for a in action:
        vehicle_action.step(a)
    position = vehicle_action.vehicle.state.position

    assert np.allclose(position, expected_position, atol=0.01)


@pytest.mark.parametrize(
    ("vector", "pos", "rotated_vector"),
    [
        (np.array([1., 0.0]), np.array([-1.5, 0.5]), np.array([0.811, 0.584])),
        (np.array([1., 0.0]), np.array([-0.5, 1.]), np.array([1., 0.])),
        (np.array([0.7071, 0.7071]), np.array([0.5, 0.5]), np.array([.973, -0.229])),
        (np.array([0., -1.]), np.array([0.5, -0.5]), np.array([-0.923, 0.382]))
    ]
)
@unit_test
@functional
def test_curvi_rotation(vector: np.ndarray, pos: np.ndarray, rotated_vector: np.ndarray):
    curvi = CurvilinearCoordinateSystem([np.array([-2, 0]), np.array([-1, 1]), np.array([0, 1]),
                                         np.array([1, 0]), np.array([0, -1])])

    res = _rotate_to_curvi(vector, curvi, pos)
    assert np.allclose(res, rotated_vector, atol=1.e-3)


@pytest.mark.parametrize(
    ("vehicle_model", "expected_value"),
    [
        (VehicleModel.PM, np.array([11.5, 11.5])),
        (VehicleModel.KS, np.array([0.4, 11.5])),
        (VehicleModel.YawRate, np.array([0.5, 11.5]))
    ]
)
@unit_test
@functional
def test_continuous_action_yaw_rate_rescale(vehicle_model, expected_value):
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": vehicle_model,  # 0: PM, 1: ST, 2: KS, 3: MB, 4: YawRate
    }
    action_configs = {
        "action_type": "continuous",
        "action_base": "acceleration" # acceleration; jerk
    }

    action = ContinuousAction(vehicle_params, action_configs)
    initial_state = State(
        **{
            "position": np.array([0., 0.]),
            "velocity": 23.,
            "velocity_y": 0,
            "orientation": 0.0,
            "time_step": 0,
            "acceleration": 0.0,
            "acceleration_y": 0.0,
        })


    with pytest.raises(AssertionError):
        action.rescale_action(np.array([1., 1.]))
    action.reset(initial_state, dt=1.0)

    scaled_action = action.rescale_action(np.array([1., 1.]))

    assert np.allclose(scaled_action, expected_value)
