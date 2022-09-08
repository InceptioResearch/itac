import numpy as np
from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Rectangle, Polygon, Circle, ShapeGroup
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario, ScenarioID
from commonroad.scenario.trajectory import State
from commonroad_route_planner.route_planner import RoutePlanner

from commonroad_rl.gym_commonroad.observation import GoalObservation
from commonroad_rl.gym_commonroad.utils.navigator import Navigator
from commonroad_rl.tests.common.marker import *

dummy_time_step = Interval(0.0, 0.0)


@pytest.mark.parametrize(
    ("ego_orientation", "expected_output"),
    [
        (-np.pi / 2., np.pi / 4.),
        (np.pi / 2., -np.pi / 4.),
        (-0.5, 0.),
        (0., 0.),
        (np.pi * 15. / 8., 0.),
        (0.5, 0.),
        (np.pi, 3 * np.pi / 4.),
    ],
)
@unit_test
@functional
def test_get_goal_orientation_distance(ego_orientation, expected_output):
    """Tests GoalObservation._get_goal_orientation_distance"""
    configs = {"goal_configs": {}}

    dummy_state = {
        "velocity": 0.0,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "time_step": 0.0,
        "position": np.array([0.0, 0.0])
    }
    ego_state = State(**dummy_state, orientation=ego_orientation)
    goal_state = State(time_step=dummy_time_step,
                       position=Rectangle(length=2.0, width=2.0, center=np.array([5.0, 0.0])),
                       orientation=AngleInterval(-np.pi / 4, np.pi / 4))

    goal_obs = GoalObservation(configs)

    min_goal_orientation_distance = goal_obs._get_goal_orientation_distance(ego_state.orientation,
                                                                            GoalRegion([goal_state]))

    assert np.isclose(min_goal_orientation_distance, expected_output)


# [10, 20]
@pytest.mark.parametrize(
    ("ego_time_step", "expected_output"),
    [
        (5, -5),
        (10, 0),
        (15, 0),
        (20, 0),
        (30, 10),
    ],
)
@unit_test
@functional
def test_get_goal_time_distance(ego_time_step, expected_output):
    """Tests GoalObservation._get_goal_time_distance"""

    dummy_state = {
        "velocity": 0.0,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "orientation": 0.0,
        "position": np.array([0.0, 0.0])
    }
    ego_state = State(**dummy_state, time_step=ego_time_step)
    goal_state = State(time_step=Interval(10, 20),
                       position=Rectangle(length=2.0, width=2.0, center=np.array([5.0, 0.0])),
                       orientation=AngleInterval(-np.pi / 2, np.pi / 2))

    min_goal_time_distance = GoalObservation._get_goal_time_distance(ego_state.time_step, GoalRegion([goal_state]))

    assert np.isclose(min_goal_time_distance, expected_output)


@pytest.mark.parametrize(
    ("ego_velocity", "expected_output"),
    [
        (5., -5.),
        (10., 0.),
        (15., 0.),
        (20., 0.),
        (30., 10.),
    ],
)
@unit_test
@functional
def test_get_goal_velocity_distance(ego_velocity, expected_output):
    """Tests GoalObservation._get_goal_velocity_distance"""

    dummy_state = {
        "time_step": dummy_time_step,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "orientation": 0.0,
        "position": np.array([0.0, 0.0])
    }
    ego_state = State(**dummy_state, velocity=ego_velocity)
    goal_state = State(time_step=Interval(10, 20),
                       position=Rectangle(length=2.0, width=2.0, center=np.array([5.0, 0.0])),
                       orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                       velocity=Interval(10., 20.))

    min_goal_velocity_distance = GoalObservation._get_goal_velocity_distance(ego_state.velocity,
                                                                             GoalRegion([goal_state]))

    assert np.isclose(min_goal_velocity_distance, expected_output)


@pytest.mark.parametrize(
    ("prev_advances", "lat_long_distance", "expected_output"),
    [
        ((5, 5), (2, 3), (3, 2)),
        ((5, 5), (7, 9), (-2, -4)),
        ((-13, -3), (-5, 3), (8, 0)),
        ((-3, -1), (-4, -2), (-1, -1)),
        ((-2, -2), (0, 4), (2, -2))
    ]
)
@unit_test
@functional
def test_get_long_lat_distance_advance_to_goal(prev_advances, lat_long_distance, expected_output):
    """Tests GoalObservation._get_long_lat_distance_advance_to_goal"""
    configs = {"goal_configs": {'observe_distance_goal_long': True,
                                'observe_distance_goal_lat': True}}

    dummy_state = {
        "time_step": Interval(1.0, 1.0),
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "orientation": 0.0,
        "position": np.array([0.0, 0.0])
    }
    ego_state = State(**dummy_state)

    goal_obs = GoalObservation(configs)
    goal_obs.observation_history_dict['distance_goal_long'] = prev_advances[0]
    goal_obs.observation_history_dict['distance_goal_lat'] = prev_advances[1]

    advance = goal_obs._get_long_lat_distance_advance_to_goal(*lat_long_distance)

    assert np.isclose(advance[0], expected_output[0])
    assert np.isclose(advance[1], expected_output[1])

    ego_state.time_step = 1
    goal_obs.observe_distance_goal_long = False
    goal_obs.observe_distance_goal_lat = False

    advance = goal_obs._get_long_lat_distance_advance_to_goal(*lat_long_distance)
    assert np.isclose(advance[0], 0)
    assert np.isclose(advance[1], 0)


@pytest.mark.parametrize(
    ('ego_position', 'goal_region', 'desired_distance'),
    [
        (
                np.array([0, 0]),
                GoalRegion([State(time_step=Interval(1, 1),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([0.0, 0.0])),
                                  orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  velocity=Interval(0., 0.))]),
                0.0
        ),
        (
                np.array([0, 0]),
                GoalRegion([State(time_step=Interval(1, 1),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([1.0, 1.0])),
                                  orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  velocity=Interval(0., 0.))]),
                np.sqrt(2.0)
        ),
        (
                np.array([0, 0]),
                GoalRegion([State(time_step=Interval(1, 1),
                                  position=Polygon(np.array([[0, 0], [0, 2], [2, 2], [2, 0]])),
                                  orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  velocity=Interval(0., 0.))]),
                np.sqrt(2.0)
        ),
        (
                np.array([0, 0]),
                GoalRegion([State(time_step=Interval(1, 1),
                                  position=Circle(1, np.array([1, 1])),
                                  orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  velocity=Interval(0., 0.))]),
                np.sqrt(2.0)
        ),
        (
                np.array([0, 0]), GoalRegion([
                    State(time_step=Interval(1, 1),
                          position=ShapeGroup([
                              Polygon(np.array([[-7.3275, 12.5257],
                                                [-7.5254, 9.1777],
                                                [-11.278, 9.1652],
                                                [-15.0305, 9.1526],
                                                [-15.1272, 12.6073],
                                                [-11.2273, 12.5665]])),
                              Polygon(np.array([[-55.7567, 6.6837],
                                                [-54.5595, 3.548],
                                                [-60.9422, 1.1307],
                                                [-67.3249, -1.2866],
                                                [-68.7141, 1.7041],
                                                [-62.2354, 4.1939]])),
                              Polygon(np.array([[-15.1272, 12.6073],
                                                [-15.0305, 9.1526],
                                                [-21.3058, 9.0022],
                                                [-27.581, 8.8518],
                                                [-27.8613, 12.1],
                                                [-21.4942, 12.3536]])),
                              Polygon(np.array([
                                  [-27.8613, 12.1],
                                  [-27.581, 8.8518],
                                  [-34.3544, 8.3423],
                                  [-41.5165, 7.3883],
                                  [-48.0403, 5.7701],
                                  [-54.5595, 3.548],
                                  [-55.7567, 6.6837],
                                  [-48.7239, 8.8909],
                                  [-41.8512, 10.351],
                                  [-34.9102, 11.5653]
                              ]))
                          ]),
                          orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                          velocity=Interval(0., 0.))
                ]),
                34.9101481
        )
    ]
)
@unit_test
@functional
def test_get_goal_euclidean_distance(ego_position: np.ndarray, goal_region: GoalRegion, desired_distance: float):
    actual_distance = GoalObservation._get_goal_euclidean_distance(ego_position, goal_region)
    assert np.isclose(actual_distance, desired_distance)

@pytest.mark.parametrize(
    ("shape_group", "actual_center"),
    [
        (
                ShapeGroup([
                    Polygon(np.array([[-7.3275, 12.5257],
                                      [-7.5254, 9.1777],
                                      [-11.278, 9.1652],
                                      [-15.0305, 9.1526],
                                      [-15.1272, 12.6073],
                                      [-11.2273, 12.5665]])),
                    Polygon(np.array([[-55.7567, 6.6837],
                                      [-54.5595, 3.548],
                                      [-60.9422, 1.1307],
                                      [-67.3249, -1.2866],
                                      [-68.7141, 1.7041],
                                      [-62.2354, 4.1939]])),
                    Polygon(np.array([[-15.1272, 12.6073],
                                      [-15.0305, 9.1526],
                                      [-21.3058, 9.0022],
                                      [-27.581, 8.8518],
                                      [-27.8613, 12.1],
                                      [-21.4942, 12.3536]])),
                    Polygon(np.array([
                        [-27.8613, 12.1],
                        [-27.581, 8.8518],
                        [-34.3544, 8.3423],
                        [-41.5165, 7.3883],
                        [-48.0403, 5.7701],
                        [-54.5595, 3.548],
                        [-55.7567, 6.6837],
                        [-48.7239, 8.8909],
                        [-41.8512, 10.351],
                        [-34.9102, 11.5653]
                    ]))
                ]),
                np.array([-33.93858901,   8.17866797])
        ),
        (
                ShapeGroup(shapes=[
                    Circle(4,np.array([0,0])),
                    Circle(100,np.array([-1,1])),
                    Polygon(np.array([
                        [4,4],
                        [8,4],
                        [8,8],
                        [6,6],
                        [4,8],
                    ]))
                ]),
                np.array([5/3, 6.555555/3])
        )
    ]
)
@unit_test
@functional
def test_convert_shape_group_to_center(shape_group: ShapeGroup, actual_center: np.ndarray):
    center = GoalObservation._convert_shape_group_to_center(shape_group)
    assert all(np.isclose(center, actual_center))

@pytest.mark.parametrize(
    ('ego_state', 'goal_region', 'is_reached'),
    [
        (
                State(**{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": 0.0,
                    "position": np.array([0.0, 0.0]),
                    "velocity": 0
                }),
                GoalRegion([State(time_step=Interval(1, 1),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([0.0, 0.0])),
                                  orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  velocity=Interval(0., 0.))]),
                True
        ),
        (
                State(**{
                    "time_step": 5,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": 0.0,
                    "position": np.array([0.0, 0.0]),
                    "velocity": 0
                }),
                GoalRegion([State(time_step=Interval(1, 1),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([0.0, 0.0])),
                                  orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  velocity=Interval(0., 0.))]),
                False
        ),
        (
                State(**{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": 0.0,
                    "position": np.array([4.0, 0.0]),
                    "velocity": 0
                }),
                GoalRegion([State(time_step=Interval(1, 1),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([0.0, 0.0])),
                                  orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  velocity=Interval(0., 0.))]),
                False
        ),
        (
                State(**{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": 3 * np.pi / 2,
                    "position": np.array([0.0, 0.0]),
                    "velocity": 0.
                }),
                GoalRegion([State(time_step=Interval(1, 1),
                                  position=Rectangle(length=2.0, width=2.0, center=np.array([0.0, 0.0])),
                                  orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                                  velocity=Interval(0., 0.))]),
                True
        )
    ]
)
@functional
@unit_test
def test_check_goal_reached(ego_state: State, goal_region: GoalRegion, is_reached: bool):
    """unittest for the GoalObservation._check_goal_reached method"""
    assert GoalObservation._check_goal_reached(goal_region, ego_state) == is_reached


@pytest.mark.parametrize(
    ('ego_state', 'goal_region', 'time_out'),
    [
        (
                State(**{
                    "time_step": 1
                }),
                GoalRegion([State(time_step=Interval(1, 1))]),
                True
        ),
        (
                State(**{
                    "time_step": 3
                }),
                GoalRegion([State(time_step=Interval(1, 1))]),
                True
        ),
        (
                State(**{
                    "time_step": 0
                }),
                GoalRegion([State(time_step=Interval(1, 1))]),
                False
        )
    ]
)
@functional
@unit_test
def test_check_is_time_out(ego_state: State, goal_region: GoalRegion, time_out: bool):
    """unittest for the GoalObservation._check_is_time_out method"""

    assert GoalObservation._check_is_time_out(ego_state, goal_region, False) == time_out
    assert GoalObservation._check_is_time_out(ego_state, goal_region, True) == False


@pytest.mark.parametrize(
    ("ego_position", "expected_output"),
    [
        (np.array([0.0, 0.0]), (4.0, 0.0)),
        (np.array([4.0, 0.0]), (0.0, 0.0)),
        (np.array([5.0, 0.0]), (0.0, 0.0)),
        (np.array([6.0, 0.0]), (0.0, 0.0)),
        (np.array([10.0, 0.0]), (-4.0, 0.0)),
        (np.array([0.0, 1.0]), (4.0, 0.0)),
        (np.array([4.0, 1.0]), (0.0, 0.0)),
        (np.array([5.0, 1.0]), (0.0, 0.0)),
        (np.array([6.0, 1.0]), (0.0, 0.0)),
        (np.array([10.0, 1.0]), (-4.0, 0.0)),
        (np.array([0.0, 2.0]), (4.0, -1.0)),
        (np.array([4.0, 2.0]), (0.0, -1.0)),
        (np.array([5.0, 2.0]), (0.0, -1.0)),
        (np.array([6.0, 2.0]), (0.0, -1.0)),
        (np.array([10.0, 2.0]), (-4.0, -1.0)),
        (np.array([0.0, -1.0]), (4.0, 0.0)),
        (np.array([4.0, -1.0]), (0.0, 0.0)),
        (np.array([5.0, -1.0]), (0.0, 0.0)),
        (np.array([6.0, -1.0]), (0.0, 0.0)),
        (np.array([10.0, -1.0]), (-4.0, 0.0)),
        (np.array([0.0, -2.0]), (4.0, 1.0)),
        (np.array([4.0, -2.0]), (0.0, 1.0)),
        (np.array([5.0, -2.0]), (0.0, 1.0)),
        (np.array([6.0, -2.0]), (0.0, 1.0)),
        (np.array([10.0, -2.0]), (-4.0, 1.0)),
    ],
)
@unit_test
@functional
def test_get_long_lat_distance_to_goal(ego_position, expected_output):
    """Tests the GoalObservation._get_long_lat_distance_to_goal function"""

    dummy_state = {
        "velocity": 0.0,
        "orientation": 0.0,
        "yaw_rate": 0.0,
        "slip_angle": 0.0,
        "time_step": 0.0,
    }
    ego_state = State(**dummy_state, position=ego_position)
    goal_state = State(time_step=dummy_time_step,
                       position=Rectangle(length=2.0, width=2.0, center=np.array([5.0, 0.0])))
    planning_problem = PlanningProblem(planning_problem_id=0, initial_state=ego_state,
                                       goal_region=GoalRegion([goal_state]))

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

    min_distance_long, min_distance_lat = GoalObservation.get_long_lat_distance_to_goal(ego_state.position, navigator)
    assert np.isclose(min_distance_long, expected_output[0]) and np.isclose(min_distance_lat, expected_output[1])
