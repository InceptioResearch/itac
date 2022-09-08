from commonroad_rl.gym_commonroad.action import *
from commonroad_rl.tests.common.marker import *

@pytest.mark.parametrize(
    ("steering_angle", "velocity", "expected_orientation"),
    [
        (0, 0, 0),
        (0, 30, 0),
        (0.5, 10, 2.1183441714035096),
        (0.5, 30, 3 * 2.1183441714035096 - 2 * np.pi),
        (-0.5, 10, -2.1183441714035096),
        (-0.5, 30, -3 * 2.1183441714035096 + 2 * np.pi),
    ],
)
@unit_test
@functional
def test_valid_vehicle_orientation(steering_angle, velocity, expected_orientation):
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": 2,  # 0: VehicleModel.PM; 2: VehicleModel.KS;
    }

    dummy_state = {
        "position": np.array([0.0, 0.0]),
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "time_step": 0,
        "orientation": 0.0,
    }
    initial_state = State(**dummy_state, steering_angle=steering_angle, velocity=velocity)
    dt = 1.0

    # Not to do anything, just continue the way with the given velocity
    action = np.array([0.0, 0.0])

    vehicle = ContinuousVehicle(vehicle_params)
    vehicle.reset(initial_state, dt)
    vehicle.set_current_state(vehicle.get_new_state(action, "acceleration"))

    resulting_orientation = vehicle.state.orientation
    assert np.isclose(resulting_orientation, expected_orientation)


@pytest.mark.parametrize(
    ("vehicle_model", "action", "expected_position"),
    [
        (
                VehicleModel.PM,
                np.array([2.3, 0.]),
                np.array([53.75, 0.0]),
        ),
        (
                VehicleModel.PM,
                np.array([2.3, 2.3]),
                np.array([53.75, 28.75]),
        ),
        (
                VehicleModel.KS,
                np.array([0, 2.3]),
                np.array([53.75, 0.0]),
        ),
        (
                VehicleModel.KS,
                np.array([0.04, 2.3]),
                np.array([26.11458402, 30.52582269]),
        ),
        (
                VehicleModel.YawRate,
                np.array([0., 2.3]),
                np.array([53.75, 0.0]),
        ),
        (
                VehicleModel.YawRate,
                np.array([0.2, 2.3]),
                np.array([42.98873942, 28.80964147]),
        ),
    ],
)
@unit_test
@functional
def test_continuous_vehicle(vehicle_model, action, expected_position):
    """
    Tests the different vehicle models
    """
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": vehicle_model.value,  # 0: PM, 1: ST, 2: KS, 3: MB, 4: YawRate
    }
    vehicle = ContinuousVehicle(vehicle_params)

    if vehicle_model == 0:
        initial_state = State(
            **{
                "position": np.array([0., 0.]),
                "velocity": 5,
                "velocity_y": 0,
                "time_step": 0,
            })
    else:
        initial_state = State(
            **{
                "position": np.array([0., 0.]),
                "steering_angle": 0.0,
                "orientation": 0.0,
                "velocity": 5.0,
                "time_step": 0,
            })

    vehicle.reset(initial_state, dt=1)
    steps = 5
    for _ in range(steps):
        vehicle.set_current_state(vehicle.get_new_state(action, "acceleration"))
    position = vehicle.state.position

    assert np.allclose(position, expected_position)

@unit_test
@nonfunctional
def test_yaw_rate_vehicle():
    vehicle_dynamics = YawRateDynamics(VehicleType.BMW_320i)
    a_max = vehicle_dynamics.parameters.longitudinal.a_max - 1e-6
    x0, y0 = 0., 0.
    v0 = 10.
    theta0 = 0.25 * np.pi
    dt = 0.04

    phi = 0.
    x = [x0, y0, 0., v0, theta0]
    while True:
        u = [a_max * np.sin(x[4] + phi) / x[3], -a_max * np.cos(x[4] + phi)] # [yaw rate, longitudinal acceleration]
        x = vehicle_dynamics.forward_simulation(x, u, dt, throw=True)
        if x[3] * np.cos(x[4]) <= 0.:
            break

    # max turning
    x = [x0, y0, 0., v0, theta0]
    while True:
        u = [0., -a_max]
        x = vehicle_dynamics.forward_simulation(x, u, dt, throw=True)
        if x[3] * np.cos(x[4]) <= 0.:
            break

    # max deceleration
    x = [x0, y0, 0., v0, theta0]
    while True:
        u = [a_max/x[3], 0.]
        x = vehicle_dynamics.forward_simulation(x, u, dt, throw=True)
        if x[3] * np.cos(x[4]) <= 0.:
            break
