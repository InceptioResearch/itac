import numpy as np
from typing import List, Union
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
from commonroad.visualization.mp_renderer import MPRenderer

dummy_time_step = Interval(0.0, 0.0)


@pytest.mark.parametrize(
    (
        "goal_region",
        "ego_state",
        "vehicle_cos_used",
        "distances_long",
        "expected_waypoints",
        "expected_orientations",
    ),
    [
        # first
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": 3 * np.pi / 2,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            Navigator.CosyVehicleObservation.LOCALCARTESIAN,
            [-100, -0.5, 0, 0.5, 5, 10.0, 20, 100],
            np.array(
                [
                    [-1, 0.0],
                    [-0.5, 0.0],
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [5, 0.0],
                    [10.0, 0.0],
                    [19.0, 0.0],
                    [19.0, 0.0],
                ]
            ),
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        ),
        # second
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            Navigator.CosyVehicleObservation.VEHICLEFRAME,
            [-100, -0.5, 0, 0.5, 5, 20, 100],
            - np.array(  # negative
                [
                    [-1, 0.0],
                    [-0.5, 0.0],
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [5, 0.0],
                    [19, 0.0],
                    [19, 0.0],
                ]
            ),
            np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]),
        ),
        # third
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            Navigator.CosyVehicleObservation.AUTOMATIC,
            [-100, -0.5, 0, 0.5, 5, 20, 100],
            -np.array(
                [
                    [-1, 0.0],
                    [-0.5, 0.0],
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [5, 0.0],
                    [18.99920, 0.0],
                    [18.99920, 0.0],
                ]
            ),
            np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]),
        ),
        # fourth
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 9.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            Navigator.CosyVehicleObservation.LOCALCARTESIAN,
            [-100, -0.5, 0, 0.5, 5, 20, 100],
            np.array(
                [
                    [-1, 0.0],
                    [-0.5, 0.0],
                    [0.0, 0.0],
                    [0.4993, 0.02508],
                    [4.081, 2.50],
                    [17.33516554, 8.50],
                    [19, 9],
                ]
            ),
            np.array(
                [
                    0,
                    0,
                    0.02506,
                    0.10440457,
                    0.87605805,
                    0.29145679,
                    0.29145679
                ]
            ),
        ),
        # fifth to test the coordinate convertion from rotation matrix
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": np.pi/2,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            
            Navigator.CosyVehicleObservation.VEHICLEFRAME,
            [-100, -0.5, 0, 0.5, 5, 20, 100],
            np.array(
                [
                
                    [0.0, 1],
                    [0.0, 0.5],
                    [0.0, 0.0],
                    [0.0, -0.5],
                    [0.0, -5],
                    [0.0, -19],
                    [0.0, -19],
                    
                ]
            ),
            np.array(
                [
                    -np.pi/2,-np.pi/2,-np.pi/2,-np.pi/2,-np.pi/2,-np.pi/2,-np.pi/2
                ]
            ),
        ),
    ],
)
@functional
@unit_test
def test_get_waypoints_of_reference_path(
    ego_state: State,
    goal_region: GoalRegion,
    vehicle_cos_used: Navigator.CosyVehicleObservation,
    distances_long: List[Union[float, int]],
    expected_waypoints: np.ndarray,
    expected_orientations: np.ndarray,
):
    """unittest for Navigator.get_waypoints_of_reference_path method"""

    _, _, navigator = helper_dummy_scenario(goal_region)

    # get waypoints
    pos, orients = navigator.get_waypoints_of_reference_path(
        ego_state, distances_ref_path=distances_long, observation_cos=vehicle_cos_used
    )
    print(help_allclose(expected_waypoints, pos))
    print(help_allclose(expected_orientations, orients))

    # observation = GoalObservation(configs)
    help_allclose(
        expected_waypoints, pos, what="positions of the waypoints points"
    )
    help_allclose(
        expected_orientations, orients, what="orientations of the waypoints points"
    )


@pytest.mark.parametrize(
    (
        "goal_region",
        "ego_state",
        "vehicle_cos_used",
        "distances_long",
        "expected_waypoints",
        "expected_orientations",
    ),
    [
        # first
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": 3 * np.pi / 2,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            Navigator.CosyVehicleObservation.LOCALCARTESIAN,
            [-100, -0.5, 0, 0.5, 5, 20, 100],
            np.array(
                [
                    [
                        [-1, 0.0],
                        [-0.5, 0.0],
                        [0.0, 0.0],
                        [0.5, 0.0],
                        [5.0, 0.0],
                        [9, 0.0],
                        [9, 0.0],
                    ],
                    [
                        [9.0, 0.0],
                        [9.0, 0.0],
                        [9.0, 0.0],
                        [9.5, 0.0],
                        [14.0, 0.0],
                        [19, 0.0],
                        [19, 0.0],
                    ],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        # second
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            Navigator.CosyVehicleObservation.VEHICLEFRAME,
            [-100, -0.5, 0, 0.5, 5, 20, 100],
            np.array(
                [
                    [
                        [1, 0.0],
                        [0.5, 0.0],
                        [0.0, 0.0],
                        [-0.5, 0.0],
                        [-5.0, 0.0],
                        [-9.0, 0.0],
                        [-9.0, 0.0],
                    ],
                    [
                        [-9.0, 0.0],
                        [-9.0, 0.0],
                        [-9.0, 0.0],
                        [-9.50, 0.0],
                        [-14, 0.0],
                        [-19, 0.0],
                        [-19, 0.0],
                    ],
                ]
            ),
            np.array(
                [
                    [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi],
                    [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi],
                ]
            ),
        ),
        # third
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            Navigator.CosyVehicleObservation.AUTOMATIC,
            [-100, -0.5, 0, 0.5, 5, 20, 100],
            np.array(
                [
                    [
                        [9.99300000e-01, 0.0],
                        [5.00000000e-01, 0.0],
                        [0.0, 0.0],
                        [-5.00000000e-01, 0.0],
                        [-5.0, 0.0],
                        [-8.99920, 0.0],
                        [-8.99920, 0.0],
                    ],
                    [
                        [-9.00070, 0.0],
                        [-9.00070, 0.0],
                        [-9.00070, 0.0],
                        [-9.50, 0.0],
                        [-1.40000000e01, 0.0],
                        [-1.89992000e01, 0.0],
                        [-1.89992000e01, 0.0],
                    ],
                ]
            ),
            np.array(
                [
                    [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi],
                    [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi],
                ]
            ),
        ),
        # fourth
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 9.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([1.0, 0.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            Navigator.CosyVehicleObservation.LOCALCARTESIAN,
            [-100, -0.5, 0, 0.5, 5, 20, 100],
            np.array(
                [
                    [
                        [-9.99300000e-01, 0.0],
                        [-5.00000000e-01, 0.0],
                        [0.0, 0.0],
                        [5.00000000e-01, 0.0],
                        [5.0, 0.0],
                        [8.99920, 0.0],
                        [8.99920, 0.0],
                    ],
                    [
                        [-9.99300000e-01, 6.0],
                        [-5.00000000e-01, 6.0],
                        [0.0, 6.0],
                        [5.00000000e-01, 6.0],
                        [5.0, 6.0],
                        [8.99920, 6.0],
                        [8.99920, 6.0],
                    ],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
@functional
@unit_test
def test_get_referencepath_multilanelets_waypoints(
    ego_state: State,
    goal_region: GoalRegion,
    vehicle_cos_used: Navigator.CosyVehicleObservation,
    distances_long: List[Union[float, int]],
    expected_waypoints: np.ndarray,
    expected_orientations: np.ndarray,
):
    """unittest for Navigator.get_referencepath_multilanelets_waypoints method"""

    _, _, navigator = helper_dummy_scenario(goal_region)

    # get waypoints
    pos, orients = navigator.get_referencepath_multilanelets_waypoints(
        ego_state,
        distances_per_lanelet=distances_long,
        lanelets_id_rel=[0, 1],
        observation_cos=vehicle_cos_used,
    )

    # observation = GoalObservation(configs)
    help_allclose(
        expected_waypoints, pos, what="positions of the waypoints points"
    )
    help_allclose(
        expected_orientations, orients, what="orientations of the waypoints paths"
    )


@pytest.mark.parametrize(
    (
        "goal_region",
        "ego_state",
        "expected_distance_lat", 
        "expected_distance_to_goal_long_on_ref",
        "expected_indomain"
    ),
    [
        # first
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": 3 * np.pi / 2,
                    "position": np.array([10.5, 5.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            np.array([5.0]),
            np.array([8.5]),
            1.0,
        ),
        # second
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([2.0, -0.5]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            np.array([-0.5]),
            np.array([17.0]),
            1.0,
        ),
        # third
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([20.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([1.0, 30.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            np.array([30.0]),
            np.array([18.0]),
            1.0,
        ),
        # fourth, before / ouside of domain
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([6.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([10.0, -1.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            np.array([-1.4141]),   # distance from closest point on ref (9.0, 0.0)
            np.array([-2.0]),  # distance from closest point on ref (9.0, 0.0) to goal (7.0, 0.0),
            0.7071,
        ),
        # fourth, before / ouside of domain
        (
            GoalRegion(
                [
                    State(
                        time_step=Interval(1, 1),
                        orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                        position=Rectangle(
                            length=2.0, width=2.0, center=np.array([6.0, 0.0])
                        ),
                    )
                ]
            ),
            State(
                **{
                    "time_step": 1,
                    "yaw_rate": 0.0,
                    "slip_angle": 0.0,
                    "orientation": -np.pi,
                    "position": np.array([10.0, 1.0]),
                    "velocity": 0.01,
                    "velocity_y": 0.0,
                }
            ),
            np.array([1.4141]),   # distance from closest point on ref (9.0, 0.0)
            np.array([-2.0]),  # distance from closest point on ref (9.0, 0.0) to goal (7.0, 0.0)
            0.7071,
        ),
    ],
)
@functional
@unit_test
def test_get_longlat_togoal_on_reference_path(
    ego_state: State,
    goal_region: GoalRegion,
    expected_distance_lat: np.ndarray,
    expected_distance_to_goal_long_on_ref: np.ndarray,
    expected_indomain
):
    """unittest for Navigator.get_referencepath_multilanelets_waypoints method"""

    _, _, navigator = helper_dummy_scenario(goal_region)

    # get waypoints
    min_distance_to_goal_long_on_ref, distance_lat, indomain = navigator.get_longlat_togoal_on_reference_path(
        state=ego_state
    )

    
    # observation = GoalObservation(configs)
    help_allclose(
        expected_distance_lat, distance_lat, what="(lat) distance"
    )
    help_allclose(
        expected_distance_to_goal_long_on_ref, min_distance_to_goal_long_on_ref, what="(longitudinal) distance"
    )

    help_allclose(
        indomain, expected_indomain, atol=1e-02, what="indomain"
    )


def help_allclose(a, b, what="", rtol=1e-03, atol=1e-05):
    """np.isclose(a, b) with lowered default tolerances"""
    assert np.allclose(
        a, b, rtol=1e-03, atol=1e-05
    ), f"{what} differs: {np.array2string(a, separator=',')} expected {np.array2string(b, separator=',')}: {np.isclose(a, b, rtol=1e-03, atol=1e-05)}"


def helper_dummy_scenario(goal_region):
    """create a simple scenario with a navigator"""

    initial_st = State(
        **{
            "time_step": 1,
            "yaw_rate": 0.0,
            "slip_angle": 0.0,
            "orientation": -np.pi,
            "position": np.array([0.0, 0.0]),
            "velocity": 0.01,
            "velocity_y": 0.0,
        }
    )

    planning_problem = PlanningProblem(
        planning_problem_id=0, initial_state=initial_st, goal_region=goal_region
    )

    lanelet0 = Lanelet(
        lanelet_id=0,
        left_vertices=np.array([[0.0, 3.0], [10.0, 3.0]]),
        center_vertices=np.array([[0.0, 0.0], [10.0, 0.0]]),
        right_vertices=np.array([[0.0, -3.0], [10.0, -3.0]]),
        successor=[2],
        adjacent_left=1,
        adjacent_left_same_direction=True,
    )
    lanelet1 = Lanelet(
        lanelet_id=1,
        left_vertices=np.array([[0.0, 9.0], [10.0, 9.0]]),
        center_vertices=np.array([[0.0, 6.0], [10.0, 6.0]]),
        right_vertices=np.array([[0.0, 3.0], [10.0, 3.0]]),
        adjacent_right=0,
        successor=[3],
        adjacent_right_same_direction=True,
    )
    lanelet3 = Lanelet(
        lanelet_id=3,
        left_vertices=np.array([[10.0, 9.0], [20.0, 12.0]]),
        center_vertices=np.array([[10.0, 6.0], [20.0, 9.0]]),
        right_vertices=np.array([[10.0, 3.0], [20.0, 6.0]]),
        predecessor=[1],
        adjacent_right=2,
        adjacent_right_same_direction=True,
    )
    lanelet2 = Lanelet(
        lanelet_id=2,
        left_vertices=np.array([[10.0, 3.0], [20.0, 3.0]]),
        center_vertices=np.array([[10.0, 0.0], [20.0, 0.0]]),
        right_vertices=np.array([[10.0, -3.0], [20.0, -3.0]]),
        predecessor=[0],
        adjacent_left=3,
        adjacent_left_same_direction=True,
    )

    scenario = Scenario(dt=0.1, scenario_id=ScenarioID("DEU_TEST-1_1_T-1"))
    scenario.lanelet_network.add_lanelet(lanelet0)
    scenario.lanelet_network.add_lanelet(lanelet3)
    scenario.lanelet_network.add_lanelet(lanelet1)
    scenario.lanelet_network.add_lanelet(lanelet2)

    route_planner = RoutePlanner(
        scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED
    )

    candidate_holder = route_planner.plan_routes()
    route = candidate_holder.retrieve_first_route()

    # print("ref_path", ref_path)
    navigator = Navigator(route=route)

    return planning_problem, scenario, navigator
