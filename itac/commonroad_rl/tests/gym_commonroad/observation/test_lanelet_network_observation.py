import pickle
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario, ScenarioID
from commonroad.geometry.shape import Rectangle
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_route_planner.route_planner import RoutePlanner
from shapely.geometry import LineString, Point

from commonroad_rl.gym_commonroad.action.action import *
from commonroad_rl.gym_commonroad.observation import LaneletNetworkObservation, ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.gym_commonroad.utils.navigator import Navigator
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.path import *

xml_path = os.path.join(resource_root("test_laneletnetwork"))
pickle_path = os.path.join(output_root("test_laneletnetwork"), "pickles")


def prepare_for_test():
    pickle_xml_scenarios(
        input_dir=xml_path,
        output_dir=pickle_path
    )
    # Lanelet observation settings
    lanelet_observation = LaneletNetworkObservation(
        configs={"lanelet_configs":
                     {"observe_left_marker_distance": True,
                      "observe_right_marker_distance": True,
                      "observe_left_road_edge_distance": True,
                      "observe_right_road_edge_distance": True,
                      "observe_is_off_road": True,
                      "strict_off_road_check": True,
                      "observe_lat_offset": True
                      }}
    )
    # construct ego vehicle
    vehicle_type = VehicleType.BMW_320i
    vehicle_model = VehicleModel.KS
    vehicle = ContinuousAction({"vehicle_type": vehicle_type, "vehicle_model": vehicle_model},
                               {"action_base": "acceleration"})

    # Initial state of the vehicle
    dummy_state = {
        "position": np.array([0.0, 0.0]),
        "velocity": 30.0,
        "yaw_rate": 0.0,
        "steering_angle": 0.0,
        "time_step": 0,
        "orientation": 0.0,
    }

    # Load meta scenario and problem dict
    filename = "DEU_A9-2_1_T-1.pickle"

    meta_scenario_reset_dict_path = os.path.join(pickle_path, "meta_scenario", "meta_scenario_reset_dict.pickle")
    with open(meta_scenario_reset_dict_path, "rb") as f:
        meta_scenario_reset_dict = pickle.load(f)

    # Load scenarios and problems
    fn = os.path.join(pickle_path, "problem", filename)
    with open(fn, "rb") as f:
        problem_dict = pickle.load(f)

    # Set scenario and problem
    scenario_id = ScenarioID.from_benchmark_id(os.path.basename(fn).split(".")[0], "2020a")
    map_id = parse_map_name(scenario_id)
    reset_config = meta_scenario_reset_dict[map_id]
    scenario = restore_scenario(reset_config["meta_scenario"], problem_dict["obstacle"], scenario_id)

    # specify laneletpolygons
    lanelet_polygons = [(l.lanelet_id, l.convert_to_polygon()) for l in scenario.lanelet_network.lanelets]
    lanelet_polygons_sg = pycrcc.ShapeGroup()
    for l_id, poly in lanelet_polygons:
        lanelet_polygons_sg.add_shape(create_collision_object(poly))
    dummy_time_step = Interval(0.0, 0.0)
    return lanelet_observation, vehicle, dummy_state, scenario, reset_config, \
           lanelet_polygons, lanelet_polygons_sg, dummy_time_step


lanelet_observation, vehicle_action, dummy_state, scenario, road_edge, lanelet_polygons, \
lanelet_polygons_sg, dummy_time_step = prepare_for_test()


@pytest.mark.parametrize(
    ('goal_region', 'ego_state', 'sampling_points', 'static', 'expected_offset'),
    [
        (
                GoalRegion([State(time_step=Interval(1, 1), orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([3.0, 3.0])))]),
                State(**{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": np.pi / 2,
                    "position": np.array([0.0, 0.0]),
                    "velocity": 10
                }),
                [1],
                False,
                np.array([-6.])
        ),
        (
                GoalRegion([State(time_step=Interval(1, 1), orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([0.0, 0.0])))]),
                State(**{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": 0,
                    "position": np.array([0.0, 0.0]),
                    "velocity": 4
                }),
                [1],
                True,
                np.array([0])
        ),
        (
                GoalRegion([State(time_step=Interval(1, 1), orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([3.0, 3.0])))]),
                State(**{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi / 2.,
                    "position": np.array([0.0, 0.0]),
                    "velocity": 0.01
                }),
                [1],
                False,
                np.array([2.01])
        )
    ]
)
@functional
@unit_test
def test_get_relative_future_goal_offsets(goal_region: GoalRegion, ego_state: State, sampling_points: List[int],
                                          static: bool, expected_offset):
    """unittest for LaneletNetworkObservation._get_relative_future_goal_offsets method"""
    configs = {"lanelet_configs": {}}

    planning_problem = PlanningProblem(planning_problem_id=0, initial_state=ego_state,
                                       goal_region=goal_region)

    lanelet = Lanelet(lanelet_id=0,
                      left_vertices=np.array([[0.0, 3.0], [10.0, 3.0]]),
                      center_vertices=np.array([[0.0, 0.0], [10.0, 0.0]]),
                      right_vertices=np.array([[0.0, -3.0], [10.0, -3.0]]))

    scenario = Scenario(dt=0.1, scenario_id=ScenarioID("DEU_TEST-1_1_T-1"))
    scenario.lanelet_network.add_lanelet(lanelet)

    route_planner = RoutePlanner(
        scenario,
        planning_problem,
        backend=RoutePlanner.Backend.NETWORKX_REVERSED,
        log_to_console=False,
    )

    route_candidates = route_planner.plan_routes()
    route = route_candidates.retrieve_best_route_by_orientation()

    navigator = Navigator(route)

    observation = LaneletNetworkObservation(configs)
    calculated_offset = observation._get_relative_future_goal_offsets(ego_state, sampling_points, static, navigator)[0]
    assert np.isclose(calculated_offset, expected_offset)


@pytest.mark.parametrize(
    ("strict_off_road_check", "check_circle_radius", "action", "expected_output"),
    [
        (True, 0, np.array([0.0, 0.0]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (True, 0, np.array([0.05, 0.0]), [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]),
        (False, 0.1, np.array([0.05, 0.0]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (False, 0.5, np.array([0.05, 0.0]), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        (False, 1.0, np.array([0.05, 0.0]), [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        (False, 1.5, np.array([0.05, 0.0]), [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]),
    ],
)
@functional
@unit_test
def test_check_off_road(strict_off_road_check, check_circle_radius, action, expected_output):
    # manully reset the ego vehicle
    vehicle_action.reset(State(**dummy_state), dt=0.1)

    # specify off-road check mode
    lanelet_observation.strict_off_road_check = strict_off_road_check
    lanelet_observation.non_strict_check_circle_radius = check_circle_radius

    steps = 20
    result = np.zeros(steps)

    for i in range(steps):
        vehicle_action.step(action)
        result[i] = lanelet_observation._check_is_off_road(vehicle_action.vehicle, road_edge)

    assert np.all(result[-10:] == expected_output)


@functional
@unit_test
def test_is_off_road_end():
    vehicle_action.reset(State(position=np.array([387.5, 5.]),
                               orientation=0.,
                               velocity=0.,
                               time_step=0), dt=0.1)

    # specify off-road check mode
    lanelet_observation.strict_off_road_check = True

    # should not be off road if open end
    assert not lanelet_observation._check_is_off_road(vehicle_action.vehicle, road_edge)

    from commonroad_rl.tools.pickle_scenario.preprocessing import generate_reset_config
    reset_config = generate_reset_config(scenario, open_lane_ends=False)

    assert lanelet_observation._check_is_off_road(vehicle_action.vehicle, reset_config)


@module_test
@functional
def test_end_of_road_termination():
    env = gym.make("commonroad-v1",
                   meta_scenario_path=os.path.join(pickle_path, "meta_scenario"),
                   train_reset_config_path=os.path.join(pickle_path, "problem"))
    env.reset()
    # manually reset ego vehicle to the end of the road
    env.ego_action.reset(State(position=np.array([387.5, 5.]),
                               orientation=0.,
                               velocity=1.,
                               time_step=0), dt=0.1)
    action = np.array([0., 0.])
    obs, reward, done, info = env.step(action)

    assert info["is_time_out"] == 1


@pytest.mark.parametrize(
    ("action", "steps", "expected_output"),
    [
        # (np.array([0., 0.]), 0, np.array([1.7479, 1.7479, 5.2813, 5.4769, ])),
        (np.array([0., 0.]), 5, np.array([1.9025, 1.5932, 5.4359, 5.3222, ])),
        (np.array([0., 0.]), 10, np.array([2.0571, 1.4385, 5.5906, 5.1675, ])),
        (np.array([0., 0.05]), 5, np.array([1.7571, 1.7386, 5.2905, 5.4676, ])),
        (np.array([0., 0.05]), 10, np.array([0.8946, 2.6011, 4.4280, 6.3301, ])),
    ],
)
@functional
@unit_test
def test_get_distance_to_marker_and_road_edge(action, steps, expected_output):
    # manully reset the ego vehicle
    vehicle_action.reset(State(**dummy_state), dt=0.1)

    for i in range(steps):
        vehicle_action.step(action)

    # construct current lanelet of the ego vehicle
    ego_lanelet_ids = ObservationCollector.sorted_lanelets_by_state(scenario, vehicle_action.vehicle.state,
                                                                    lanelet_polygons, lanelet_polygons_sg)
    ego_lanelet_id = ego_lanelet_ids[0]
    ego_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
    lanelet_observation._get_distance_to_marker_and_road_edge(vehicle_action.vehicle.state, ego_lanelet, road_edge)

    result = np.hstack((lanelet_observation.observation_dict["left_marker_distance"],
                        lanelet_observation.observation_dict["right_marker_distance"],
                        lanelet_observation.observation_dict["left_road_edge_distance"],
                        lanelet_observation.observation_dict["right_road_edge_distance"]))

    assert np.allclose(result, expected_output, rtol=0.001)


@pytest.mark.parametrize(
    ("ego_vehicle_lanelet_id", "desired_left_edge", "desired_right_edge"),
    [
        (
                0,
                LineString([[0, 1], [3, 4]]),
                LineString([[112, 3], [23, 4]])
        ),
        (
                3,
                LineString([[0, 1], [3, 4]]),
                LineString([[112, 3], [23, 4]])
        ),
        (
                34,
                LineString([[0, 1], [3, 4], [2, 34]]),
                LineString([[112, 3], [23, 4], [2314, 34]])
        )
    ]
)
@functional
@unit_test
def test_get_road_edge(ego_vehicle_lanelet_id: int, desired_left_edge: LineString, desired_right_edge: LineString):
    road_edge = {
        "left_road_edge_lanelet_id_dict": np.zeros(ego_vehicle_lanelet_id + 1).astype(int),
        "right_road_edge_lanelet_id_dict": np.zeros(ego_vehicle_lanelet_id + 1).astype(int),
        "left_road_edge_dict": [desired_left_edge],
        "right_road_edge_dict": [desired_right_edge],
    }

    obs = LaneletNetworkObservation({"lanelet_configs": {}})
    left_road_edge, right_road_edge = obs._get_road_edge(road_edge, ego_vehicle_lanelet_id)
    assert all(np.isclose(left_road_edge.bounds, desired_left_edge.bounds))
    assert all(np.isclose(right_road_edge.bounds, desired_right_edge.bounds))


@pytest.mark.parametrize(
    ("position", "desired_lat_position", "arr"),
    [
        (
                np.array([136.1013, -52.64465]),
                0,
                np.array(
                    [
                        [136.1013, -52.64465],
                        [136.9463, -53.99465],
                        [138.9713, -55.94465],
                        [141.0563, -57.19465],
                        [144.0213, -58.59465],
                        [147.1013, -59.64465],
                        [152.3213, -60.39465],
                        [157.7213, -59.79465],
                        [163.6113, -58.04465],
                    ]
                ),
        ),
        (
                np.array([133.7763, -13.59465]),
                0.0008496,
                np.array(
                    [
                        [135.6863, 7.45535],
                        [135.0613, -1.19465],
                        [133.7763, -13.59465],
                        [132.1663, -27.94465],
                        [131.1563, -36.29465],
                    ]
                ),
        ),
        (
                np.array([135.2, 6]),
                -0.380153613,
                np.array(
                    [
                        [135.6863, 7.45535],
                        [135.0613, -1.19465],
                        [133.7763, -13.59465],
                        [132.1663, -27.94465],
                        [131.1563, -36.29465],
                    ]
                ),
        ),
        (
                np.array([137.1013, -54.64465]),
                -0.36073585,
                np.array(
                    [
                        [136.1013, -52.64465],
                        [136.9463, -53.99465],
                        [138.9713, -55.94465],
                        [141.0563, -57.19465],
                        [144.0213, -58.59465],
                        [147.1013, -59.64465],
                        [152.3213, -60.39465],
                        [157.7213, -59.79465],
                        [163.6113, -58.04465],
                    ]
                ),
        ),
    ]
)
@functional
@unit_test
def test_get_relative_offset(position: np.ndarray, desired_lat_position: float, arr: np.ndarray):
    curvi_cosy = Navigator.create_coordinate_system_from_polyline(arr)
    ego_vehicle_lat_position = LaneletNetworkObservation.get_relative_offset(curvi_cosy, position)
    assert np.isclose(ego_vehicle_lat_position, desired_lat_position), \
        f"differs: {np.array2string(np.array(ego_vehicle_lat_position), separator=',')} " \
        f"expected {np.array2string(np.array(desired_lat_position), separator=',')}: " \
        f"{np.isclose(np.array(ego_vehicle_lat_position), np.array(desired_lat_position), atol=1e-2)}"


@pytest.mark.parametrize(
    ("p", "l", "desired_distance"),
    [
        (
                Point(1, -1),
                LineString([[0, 0], [1, 0], [1, 1]]),
                1
        ),
        (
                Point(1, -1),
                LineString([[0, 0], [1, 0], [1, -1]]),
                0
        ),
        (
                Point(1, 2),
                LineString([[0, 0], [1, 0], [1, 1]]),
                1
        )
    ]
)
@functional
@unit_test
def test_get_distance_point_to_linestring(p: Point, l: LineString, desired_distance: float):
    observed_distance = LaneletNetworkObservation.get_distance_point_to_linestring(p, l)
    assert np.isclose(desired_distance, observed_distance)


@pytest.mark.parametrize(
    ("position", "left_marker", "right_marker", "left_edge", "right_edge", "desired_left_marker_distance",
     "desired_right_marker_distance", "desired_left_edge_distance", "desired_right_edge_distance"),
    [
        (
                np.array([0, 0]),
                LineString([[-1, 1.5], [0, 2], [1, 2]]),
                LineString([[-1, -1], [0, -1], [1, -2]]),
                LineString([[-1, 2], [0, 2.5], [1, 2.5]]),
                LineString([[-1, -1.5], [0, -1.5], [1, -1.5]]),
                1.7888543,
                1,
                2.2360679,
                1.5
        ),
        (
                np.array([1, 0.5]),
                LineString([[-1, 1.5], [0, 2], [1, 2]]),
                LineString([[-1, -1], [0, -1], [1, -2]]),
                LineString([[-1, 2], [0, 2.5], [1, 2.5]]),
                LineString([[-1, -1.5], [0, -1.5], [1, -1.5]]),
                1.5,
                1.80277563,
                2,
                2
        ),
        (
                np.array([1, 0.5]),
                LineString([[-0.5, 0.5], [0.5, 1.5]]),
                LineString([[-0.5, -0.5], [0.5, -0.5]]),
                LineString([[-0.5, 1.5], [0.5, 1.5]]),
                LineString([[-0.5, -1], [0.5, -1]]),
                1.06066017,
                1.11803398,
                1.11803398,
                1.58113883,
        ),
    ]
)
@functional
@unit_test
def test_get_distance_to_marker_and_road_edge(position: np.ndarray, left_marker: LineString, right_marker: LineString,
                                              left_edge: LineString, right_edge: LineString,
                                              desired_left_marker_distance: float, desired_right_marker_distance: float,
                                              desired_left_edge_distance: float, desired_right_edge_distance: float):
    ego_state = State(**{"time_step": 0, "position": position, "orientation": 0, "velocity": 0})

    distance_left_marker, distance_right_marker, distance_left_road_edge, distance_right_road_edge = \
        LaneletNetworkObservation.get_distance_to_marker_and_road_edge(ego_state, left_marker, right_marker,
                                                                       left_edge, right_edge)

    assert np.isclose(desired_right_edge_distance, distance_right_road_edge)
    assert np.isclose(desired_left_edge_distance, distance_left_road_edge)
    assert np.isclose(desired_right_marker_distance, distance_right_marker)
    assert np.isclose(desired_left_marker_distance, distance_left_marker)


@pytest.mark.parametrize(
    ("position", "curve", "expected_output"),
    [
        (np.array([0, 1.0]),
         np.array([[0.70710678, 0.70710678],
                   [0.57357644, 0.81915204],
                   [0.42261826, 0.90630779],
                   [0.25881905, 0.96592583],
                   [0.08715574, 0.9961947],
                   [0, 1.0],
                   [-0.08715574, 0.9961947],
                   [-0.25881905, 0.96592583],
                   [-0.42261826, 0.90630779],
                   [-0.57357644, 0.81915204],
                   [-0.70710678, 0.70710678]]), 1.07),
        (np.array([0., 2.]), np.array([[1.41421356, 1.41421356],
                                       [1.14715287, 1.63830409],
                                       [0.84523652, 1.81261557],
                                       [0.51763809, 1.93185165],
                                       [0.17431149, 1.9923894],
                                       [-0.17431149, 1.9923894],
                                       [-0.51763809, 1.93185165],
                                       [-0.84523652, 1.81261557],
                                       [-1.14715287, 1.63830409],
                                       [-1.41421356, 1.41421356]]), 0.5),
        (np.array([0., 0.]), np.array([[0, 0],
                                       [1, 1],
                                       [2, 2],
                                       [3, 3],
                                       [4, 4]]), 0.0)
    ]
)
@functional
@unit_test
def test_get_lane_curvature(position, curve, expected_output):
    ccosy = Navigator.create_coordinate_system_from_polyline(curve)
    curvature = LaneletNetworkObservation.get_lane_curvature(position, ccosy)
    # TODO: reduce tolerance after ccosy is fixed (extrapolation and resampling introduced curvature deviation)
    assert np.isclose(curvature, expected_output, atol=1e-2)
