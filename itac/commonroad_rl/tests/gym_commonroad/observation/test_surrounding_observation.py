import pickle
from typing import List, Union, Optional

import commonroad_dc.pycrcc as pycrcc
import numpy as np
import yaml
from commonroad.geometry.shape import Polygon, Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import Obstacle, DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.scenario import ScenarioID, Scenario
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object, \
    create_collision_checker
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from shapely.geometry import Polygon as PolygonShapely
from commonroad_rl.gym_commonroad.action.vehicle import ContinuousVehicle
from commonroad_rl.gym_commonroad.observation import SurroundingObservation, ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.gym_commonroad.utils.navigator import Navigator
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios
from commonroad_rl.tests.common.evaluation import *
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.path import *

XML_PATH = os.path.join(resource_root("test_surroundings"))
PICKLE_PATH = os.path.join(output_root("test_surroundings"), "pickles")


def prepare_for_surrounding_test(filename, xml_scenarios_path=None, pickle_path=None):
    if xml_scenarios_path is None:
        xml_scenarios_path = XML_PATH
    if pickle_path is None:
        pickle_path = XML_PATH

    pickle_xml_scenarios(
        input_dir=xml_scenarios_path,
        output_dir=pickle_path
    )

    test_meta_path = os.path.join(pickle_path, "meta_scenario")
    test_problem_path = os.path.join(pickle_path, "problem")

    # Load meta scenario and problem dict
    meta_scenario_reset_dict_path = os.path.join(test_meta_path, "meta_scenario_reset_dict.pickle")
    with open(meta_scenario_reset_dict_path, "rb") as f:
        meta_scenario_reset_dict = pickle.load(f)

    # Load scenarios and problems
    fn = os.path.join(test_problem_path, filename)
    with open(fn, "rb") as f:
        problem_dict = pickle.load(f)

    # Set scenario and problem
    scenario_id = ScenarioID.from_benchmark_id(os.path.basename(fn).split(".")[0], "2020a")
    map_id = parse_map_name(scenario_id)
    reset_config = meta_scenario_reset_dict[map_id]
    scenario = restore_scenario(reset_config["meta_scenario"], problem_dict["obstacle"], scenario_id)

    connected_lanelet_dict = meta_scenario_reset_dict[map_id]["connected_lanelet_dict"]

    lanelet_polygons = [(l.lanelet_id, l.convert_to_polygon()) for l in scenario.lanelet_network.lanelets]
    lanelet_polygons_sg = pycrcc.ShapeGroup()
    for l_id, poly in lanelet_polygons:
        lanelet_polygons_sg.add_shape(create_collision_object(poly))

    return scenario, connected_lanelet_dict, lanelet_polygons, lanelet_polygons_sg


@pytest.mark.parametrize(("arr",),
                         [(np.array([[60.8363, -101.74465], [54.0463, -105.99465]]),),
                          (np.array([[60.8363, -101.74465], [54.05512039386607, -105.98912913758015]]),),
                          (np.array([[136.1013, -52.64465], [136.9463, -53.99465],
                                     [138.9713, -55.94465], [141.0563, -57.19465],
                                     [144.0213, -58.59465], [147.1013, -59.64465],
                                     [152.3213, -60.39465], [157.7213, -59.79465],
                                     [163.6113, -58.04465], ]),),
                          (np.array([[135.6863, 7.45535], [135.0613, -1.19465],
                                     [133.7763, -13.59465], [132.1663, -27.94465],
                                     [131.1563, -36.29465], ]),), ], )
@unit_test
@functional
def test_ccosy_contains_orig(arr):
    """
    Makes sure the original vertices are actually always included in the resulting polyline
    and the new polyline ends and start at the same position as the original one
    """
    res = Navigator.create_coordinate_system_from_polyline(arr)
    with does_not_raise():
        for x in arr:
            # try that no error is thrown
            res.convert_to_curvilinear_coords(x[0], x[1])


scenario_1, connected_lanelet_dict, lanelet_polygons, lanelet_polygons_sg = prepare_for_surrounding_test(
    "DEU_AAH-4_1000_T-1.pickle")

# TODO: fix following tests to use scenarios in test_surroundings

scenario_2, _, _, _ = prepare_for_surrounding_test(
    "DEU_A9-2_1_T-1.pickle",
    xml_scenarios_path=os.path.join(resource_root("test_laneletnetwork")),
    pickle_path=os.path.join(output_root("test_laneletnetwork"), "pickles")
)


@pytest.mark.parametrize((
        "ego_state", "observe_lane_rect_surrounding", "observe_lane_circ_surrounding", "p_rel_expected",
        "v_rel_expected"), [(State(**{"time_step": 50, "position": np.array([127.50756356637483, -50.69294785562317]),
                                      "orientation": 4.298126916546023, "velocity": 8.343911610829114}), True, False,
                             [50.1223503, 17.53095357, 50.1223503, 50.1223503, 50.1223503, 50.1223503],
                             [0.0, -0.3290005752341578, 0.0, 0.0, 0.0, 0.0]), (
                                    State(**{"time_step": 51,
                                             "position": np.array([127.39433877716928, -50.9499165494171]),
                                             "orientation": 4.296344007588243, "velocity": 8.558071157124918}), True,
                                    False,
                                    [50.1223503, 17.44662997, 50.1223503, 50.1223503, 50.1223503, 50.1223503],
                                    [0.0, -0.0348669273743365, 0.0, 0.0, 0.0, 0.0]), (
                                    State(**{"time_step": 52, "position": [127.2789061588074, -51.210826963166035],
                                             "orientation": 4.294731018487505, "velocity": 8.560948563466251}), True,
                                    False, [50.1223503, 17.3700263, 50.1223503, 50.1223503, 50.1223503, 50.1223503],
                                    [0.0, 0.05249202534491637, 0.0, 0.0, 0.0, 0.0]), (
                                    State(**{"time_step": 43,
                                             "position": np.array([128.2222609335789, -49.022624022869934]),
                                             "orientation": 4.319904886182895, "velocity": 7.082343029302184}), False,
                                    True,
                                    [50., 18.20347091, 50., 50., 47.40590347, 50.],
                                    [0.0, -2.025488953984791, 0.0, 0.0, 2.6653220542271576, 0.0],), (
                                    State(**{"time_step": 44,
                                             "position": np.array([128.13057639473206, -49.24300781209363]),
                                             "orientation": 4.315381265190625, "velocity": 7.291671591323439, }), False,
                                    True,
                                    [50., 18.05391643, 50., 50., 47.63507943, 50.],
                                    [0.0, -1.7668531163933068, 0.0, 0.0, 2.529247893570327, 0.0],), (
                                    State(**{"time_step": 45,
                                             "position": np.array([128.03441395825942, -49.47140451195109]),
                                             "orientation": 4.31163016190567, "velocity": 7.668769571559476}), False,
                                    True,
                                    [50., 17.92129415, 50., 50., 47.8514526, 50.],
                                    [0.0, -1.33718633344232, 0.0, 0.0, 2.2206628198837794, 0.0],), ], )
@module_test
@functional
def test_get_surrounding_obstacles_lane_based(ego_state, observe_lane_rect_surrounding, observe_lane_circ_surrounding,
                                              p_rel_expected, v_rel_expected):
    """
    Test for
    method at the initial step
    """

    surrounding_observation = SurroundingObservation(configs={
        "reward_configs_hybrid": {"reward_safe_distance_coef": -1.},
        "surrounding_configs": {"observe_lane_rect_surrounding": observe_lane_rect_surrounding,
                                "observe_lane_circ_surrounding": observe_lane_circ_surrounding,
                                "lane_rect_sensor_range_length": 100.,
                                "lane_rect_sensor_range_width": 7.,
                                "lane_circ_sensor_range_radius": 50.0,
                                "observe_is_collision": False,
                                "observe_lane_change": False,
                                "fast_distance_calculation": True}})

    ego_vehicle = ContinuousVehicle({"vehicle_type": 2, "vehicle_model": 2})
    ego_vehicle.reset(ego_state, dt=0.1)

    # find ego lanelet here because multiple observations need it
    ego_lanelet_ids = ObservationCollector.sorted_lanelets_by_state(scenario_1, ego_state, lanelet_polygons,
                                                                    lanelet_polygons_sg)
    ego_lanelet_id = ego_lanelet_ids[0]

    ego_lanelet = scenario_1.lanelet_network.find_lanelet_by_id(ego_lanelet_id)

    local_ccosy, _ = ObservationCollector.get_local_curvi_cosy(scenario_1.lanelet_network, ego_lanelet_id, None,
                                                               max_lane_merge_range=1000.)
    collision_checker = create_collision_checker(scenario_1)
    surrounding_observation.observe(scenario_1, ego_vehicle, ego_state.time_step, connected_lanelet_dict, ego_lanelet,
                                    collision_checker=collision_checker, local_ccosy=local_ccosy)

    p_rel = surrounding_observation.observation_dict["lane_based_p_rel"]
    v_rel = surrounding_observation.observation_dict["lane_based_v_rel"]

    # Check against ground truth
    assert np.allclose(np.array(p_rel), np.array(p_rel_expected), atol=1e-2), \
        f"differs: {np.array2string(np.array(p_rel), separator=',')} expected {np.array2string(np.array(p_rel_expected), separator=',')}: {np.isclose(np.array(p_rel), np.array(p_rel_expected), atol=1e-2)}"
    assert np.allclose(np.array(v_rel), np.array(v_rel_expected), atol=1e-2), \
        f"differs: {np.array2string(np.array(v_rel), separator=',')} expected {np.array2string(np.array(v_rel_expected), separator=',')}: {np.isclose(np.array(v_rel), np.array(v_rel_expected), atol=1e-2)}"


@unit_test
@functional
def test_obstacle_detection_lane_based():
    """
    Test for
    method at the initial step
    """

    surrounding_observation = SurroundingObservation(
        configs={
            "reward_configs_hybrid": {"reward_safe_distance_coef": -1.},
            "surrounding_configs": {"observe_lane_circ_surrounding": True,
                                    "lane_circ_sensor_range_radius": 100.,
                                    "observe_is_collision": False,
                                    "observe_lane_change": False,
                                    "fast_distance_calculation": False}
        }
    )

    ego_vehicle = ContinuousVehicle({"vehicle_type": 2, "vehicle_model": 4})
    ego_state = State(**{
        "position": np.array([-262.1121295, 9.5967584]),
        "orientation": -0.013548497271880677,
        "velocity": 37.07059064327494,
        "steering_angle": 5.203479437594755e-05,
        "yaw_rate": -0.010336290633383515,
        "acceleration": 11.422267557169176,
        "time_step": 56,
    })

    ego_vehicle.reset(ego_state, dt=0.1)
    from commonroad.common.file_reader import CommonRoadFileReader
    file_path = os.path.join(XML_PATH, "DEU_LocationDUpper-8_2_T-1.xml")
    scenario, _ = CommonRoadFileReader(file_path).open(lanelet_assignment=True)

    # find ego lanelet here because multiple observations need it
    ego_lanelet_id = 2
    ego_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
    local_ccosy, _ = ObservationCollector.get_local_curvi_cosy(scenario.lanelet_network, ego_lanelet_id, None,
                                                               max_lane_merge_range=1000.)
    collision_checker = create_collision_checker(scenario)
    surrounding_observation.observe(scenario, ego_vehicle, ego_state.time_step,
                                    connected_lanelet_dict={1: {1}, 2: {2}, 3: {3}}, ego_lanelet=ego_lanelet,
                                    collision_checker=collision_checker, local_ccosy=local_ccosy)
    detected_obstacles = surrounding_observation.detected_obstacles
    detected_ids = [obs.obstacle_id if obs is not None else None for obs in detected_obstacles]
    expected_ids = [17, None, 18, 14, 12, 15]

    assert detected_ids == expected_ids


@pytest.mark.parametrize(("ego_state", "prev_rel_pos", "p_rel_expected", "p_rel_rate_expected"), [(
        State(**{"time_step": 0, "position": np.array([130.95148999999998, -38.046040000000005]),
                 "orientation": 4.573031246397025, "velocity": 8.9459}),
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
         50.0, 50.0],
        [50.0, 50.0, 8.134482069004424, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
         50.0, 50.0, 50.0, 50.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],),
    # manually set to 0 at timestep 0
    (State(**{"time_step": 1, "position": np.array([130.91046689527747, -38.33872191154571]),
              "orientation": 4.5736457771412145, "velocity": 8.733597598338127}),
     [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
      50.0],
     [50.0, 50.0, 7.9603019648779325, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
      50.0, 50.0, 50.0, 50.0],
     [0.0, 0.0, 42.039698035122065, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0],), (State(**{"time_step": 2, "position": np.array([130.8706926599194, -38.625027045977326]),
                        "orientation": 4.575858469977114, "velocity": 8.568383732799687}),
               [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                50.0, 50.0, 50.0],
               [50.0, 50.0, 7.79517301, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                50.0, 50.0, 50.0, 50.0],
               [0.0, 0.0, 42.20482699, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0],), ], )
@unit_test
@functional
def test_get_surrounding_obstacles_lidar_circle(ego_state, prev_rel_pos, p_rel_expected, p_rel_rate_expected, ):
    """
    Tests the
    """

    surrounding_observation = SurroundingObservation(configs={
        "reward_configs_hybrid": {"reward_safe_distance_coef": -1.},
        "surrounding_configs": {"observe_lidar_circle_surrounding": True, "lidar_sensor_radius": 50.,
                                "lidar_circle_num_beams": 20,
                                "observe_is_collision": False,
                                "observe_lane_rect_surrounding": False, "observe_lane_circ_surrounding": False,
                                "observe_lane_change": False, "observe_relative_priority": True}})
    ego_vehicle = ContinuousVehicle({"vehicle_type": 2, "vehicle_model": 0})
    ego_vehicle.reset(ego_state, dt=0.1)

    surrounding_observation._prev_distances = prev_rel_pos
    # find ego lanelet here because multiple observations need it
    ego_lanelet_ids = ObservationCollector.sorted_lanelets_by_state(scenario_1, ego_state, lanelet_polygons,
                                                                    lanelet_polygons_sg)
    ego_lanelet_id = ego_lanelet_ids[0]

    ego_lanelet = scenario_1.lanelet_network.find_lanelet_by_id(ego_lanelet_id)

    local_ccosy, _ = ObservationCollector.get_local_curvi_cosy(scenario_1.lanelet_network, ego_lanelet_id, None,
                                                               max_lane_merge_range=1000.)
    collision_checker = create_collision_checker(scenario_1)
    surrounding_observation.observe(scenario_1, ego_vehicle, ego_state.time_step, connected_lanelet_dict, ego_lanelet,
                                    collision_checker=collision_checker, local_ccosy=local_ccosy)

    p_rel = surrounding_observation.observation_dict["lidar_circle_dist"]
    p_rel_rate = surrounding_observation.observation_dict["lidar_circle_dist_rate"]

    # Check against ground truth
    assert np.allclose(p_rel, p_rel_expected)
    assert np.allclose(p_rel_rate, np.array(p_rel_rate_expected) / scenario_1.dt)


@pytest.mark.parametrize(("obstacle_shapes", "ego_state", "actual_detection_points"), [(
        [PolygonShapely(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))],
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        np.array([[100, -3], [0, 0], [-100, -3], [0, -103]])), (
        [PolygonShapely(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])),
         PolygonShapely(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) + 2), ],
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        np.array([[100, -3], [0, 0], [-100, -3], [0, -103]])), (
        [PolygonShapely(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])),
         PolygonShapely(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) - 3), ],
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        np.array([[100, -3], [0, 0], [-2, -3], [0, -103]]))])
@unit_test
@functional
def test_get_distances_lidar_based(obstacle_shapes: List[Polygon], ego_state: State,
                                   actual_detection_points: np.ndarray):
    configs = {"reward_configs_hybrid": {"reward_safe_distance_coef": -1.},
               'surrounding_configs': {'lidar_circle_num_beams': 4,
                                       'lidar_sensor_radius': 10., 'observe_lidar_circle_surrounding': True}}

    surrounding_beams_ego_vehicle = []
    beam_start = ego_state.position
    for i in range(configs['surrounding_configs']['lidar_circle_num_beams']):
        theta = i * (2 * np.pi / configs['surrounding_configs']['lidar_circle_num_beams'])
        x_delta = configs['surrounding_configs']['lidar_sensor_radius'] * np.cos(theta)
        y_delta = configs['surrounding_configs']['lidar_sensor_radius'] * np.sin(theta)
        beam_length = np.sqrt(x_delta ** 2 + y_delta ** 2)
        beam_angle = ego_state.orientation + theta
        surrounding_beams_ego_vehicle.append((beam_start, beam_length, beam_angle))

    detected_obstacles = [None] * len(obstacle_shapes)
    s_obs = SurroundingObservation(configs)
    s_obs.max_obs_dist = 100.
    s_obs._ego_state = ego_state
    obstacle_distances, _ = s_obs._get_obstacles_with_surrounding_beams(obstacle_shapes, detected_obstacles,
                                                                        surrounding_beams_ego_vehicle)
    s_obs._current_time_step = 0
    dists, dist_rates, detection_points = s_obs._get_distances_lidar_based(surrounding_beams_ego_vehicle,
                                                                           obstacle_distances)

    assert all(np.isclose(dists, obstacle_distances))
    assert all(np.isclose(dist_rates, np.zeros(dist_rates.shape)))
    assert all(np.isclose(actual_detection_points, detection_points).reshape((-1)))


@pytest.mark.parametrize(
    ("connected_ego", "connected_successor", "connected_predecessor", "connected_left", "connected_right"),
    [({}, {}, {}, {}, {}), ({234}, {324}, {23}, {232}, {67}), ])
@unit_test
@functional
def test_get_nearby_lanelet_id(connected_ego: set, connected_successor: set, connected_predecessor: set,
                               connected_left: set, connected_right: set):
    ''' test for get_nearby_lanelet_id function of SurrouncingObservations'''
    ego_vehicle_lanelet = scenario_1.lanelet_network.find_lanelet_by_id(100)
    connected_lanelet_dict = {100: connected_ego,  # ego
                              103: connected_left,  # adj_left
                              104: connected_right,  # adj_right
                              101: connected_predecessor,  # predecessor
                              102: connected_successor,  # successor
                              }
    for id in set().union(connected_ego, connected_successor, connected_predecessor, connected_left, connected_right):
        if connected_lanelet_dict.get(id) is None:
            connected_lanelet_dict[id] = {}

    lanelet_dict, all_lanelets_set = SurroundingObservation.get_nearby_lanelet_id(connected_lanelet_dict,
                                                                                  ego_vehicle_lanelet)

    ego_all = set().union(connected_ego, connected_successor, connected_predecessor, {100})
    assert len(ego_all) == len(lanelet_dict['ego_all'])
    for ele in ego_all:
        assert ele in lanelet_dict['ego_all']

    assert 0 == len(lanelet_dict['left_all'])

    assert len(set().union(connected_right, {ego_vehicle_lanelet.adj_right})) == len(lanelet_dict['right_all'])
    for ele in connected_right:
        assert ele in lanelet_dict['right_all']


@pytest.mark.parametrize(("ego_state", "ego_lanelet_id", "desired_p_rel", "desired_v_rel"), [(
        State(**{"time_step": 0, "position": np.array([21, 3.8]), "orientation": 0, "velocity": 0}), 30626,
        [100.0, 100.0, 100.0, 100.0, 0.38061144, 100.0], [0.0, 0.0, 0.0, 0.0, 35.0, 0.0]), (
        State(**{"time_step": 0, "position": np.array([-39, -4]), "orientation": 0, "velocity": 0}), 30622,
        [100.0, 100.0, 100.0, 100.0, 65.750643, 100.0], [0.0, 0.0, 0.0, 0.0, 27.0, 0.0])])
@unit_test
@functional
def test_get_rel_v_p_lane_based(ego_state: State, ego_lanelet_id: int, desired_p_rel: List[float],
                                desired_v_rel: List[float]):
    """
    test for get_rel_v_p_lane_based method of SurroundingObservations
    """
    configs = {"reward_configs_hybrid": {"reward_safe_distance_coef": -1.},
               "surrounding_configs": {}}
    obs = SurroundingObservation(configs)
    obs.max_obs_dist = 100.0
    obs._local_ccosy = ObservationCollector.get_local_curvi_cosy(scenario_2.lanelet_network, ego_lanelet_id, None,
                                                                 max_lane_merge_range=5000.)[0]
    obs._ego_state = ego_state
    obstacles_lanelet_ids = [obstacle.initial_center_lanelet_ids.copy().pop() for obstacle in scenario_2.obstacles]
    obstacle_states = [obstacle.initial_state for obstacle in scenario_2.obstacles]
    lanelet_dict = SurroundingObservation.get_nearby_lanelet_id(connected_lanelet_dict,
                                                                scenario_2.lanelet_network.find_lanelet_by_id(
                                                                    ego_lanelet_id))[0]

    v_rel, p_rel, _, _, ego_vehicle_lat_position = obs._get_rel_v_p_lane_based(obstacles_lanelet_ids,
                                                                               obstacle_states,
                                                                               lanelet_dict,
                                                                               scenario_2.obstacles)

    assert all(np.isclose(v_rel, desired_v_rel).reshape(-1))
    assert all(np.isclose(p_rel, desired_p_rel).reshape(-1))


@pytest.mark.parametrize(("obs_state", "ego_state", "follow", "desired_v_rel", "desired_p_rel"), [(
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}), True, 0, 0), (
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        State(**{"time_step": 2, "position": np.array([0, -320]), "orientation": 0, "velocity": 0}), False, 0, 0), (
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        State(**{"time_step": 2, "position": np.array([0, -20]), "orientation": 0, "velocity": 4}), True, 4, 17.0)])
@unit_test
@functional
def test_get_rel_v_p_follow_leading(obs_state: State, ego_state: State, follow: bool, desired_v_rel: float,
                                    desired_p_rel: float):
    '''test for get_rel_v_p_follow_lead function of SurroundingObservation class'''
    distance_abs = np.linalg.norm(ego_state.position - obs_state.position)
    dummy_rel_vel = 0.
    dummy_dist = 100.

    v_rel_follow, p_rel_follow, o_follow, _, v_rel_lead, p_rel_lead, o_lead, _ = \
        SurroundingObservation.get_rel_v_p_follow_leading(
            1, distance_abs, dummy_dist, dummy_dist, None, None, obs_state, None, ego_state, None, None, None, None)
    v_rel_follow, p_rel_follow, o_follow, _, v_rel_lead, p_rel_lead, o_lead, _ = \
        SurroundingObservation.get_rel_v_p_follow_leading(
            -1, distance_abs, p_rel_follow, dummy_dist, v_rel_follow, None, obs_state, None, ego_state, o_follow, None,
            None, None)

    if follow:
        assert o_follow == obs_state
        assert np.isclose(v_rel_follow, desired_v_rel)
        assert np.isclose(p_rel_follow, desired_p_rel)
        assert o_lead == obs_state
        assert np.isclose(v_rel_lead, -desired_v_rel)
        assert np.isclose(p_rel_lead, desired_p_rel)
    else:
        assert v_rel_follow is None
        assert o_follow is None
        assert p_rel_follow is dummy_dist
        assert v_rel_lead is None
        assert o_lead is None
        assert p_rel_lead is dummy_dist


@pytest.mark.parametrize(("ego_state", "obstacle_shapes", 'desired_distances', 'lidar_circle_num_beams'), [(
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        [PolygonShapely(np.array([[0, 0], [1, 1], [0, 1]])), ], np.array([100., 3., 100., 100.]), 4), (
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        [PolygonShapely(np.array([[0, 0], [1, 1], [0, 1]])), PolygonShapely(np.array([[-3, 0], [-2, 1], [-3, 1]])),
         PolygonShapely(np.array([[4, -4], [5, -4], [5, -2], [4, -2]])), ],
        np.array([4., 100, 3., 3 * np.sqrt(2), 100, 100, 100., 100.]), 8), (
        State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}),
        [PolygonShapely(np.array([[0, 0], [1, 1], [0, 1]])), PolygonShapely(np.array([[-3, 0], [-2, 1], [-3, 1]])),
         PolygonShapely(np.array([[4, -4], [5, -4], [5, -2], [4, -2]])),
         PolygonShapely(np.array([[4, -5], [5, -5], [5, -7], [4, -6.9]])),
         PolygonShapely(np.array([[-1, -6], [1, -8], [-1, -8]]))],
        np.array([4., 100, 3., 3 * np.sqrt(2), 100, 100, 4., 100.]), 8)])
@unit_test
@functional
def test_get_obstacles_with_surrounding_beams(ego_state: State, obstacle_shapes: List[Polygon],
                                              desired_distances: np.ndarray, lidar_circle_num_beams: int):
    """
    test for get_obstacles_with_surrounding_beams function of SurroundingObservation class
    """
    configs = {"reward_configs_hybrid": {"reward_safe_distance_coef": -1.},
               "surrounding_configs": {"observe_lidar_circle_surrounding": True, "lidar_sensor_radius": 100}}

    lidar_sensor_radius = 100
    surrounding_beams_ego_vehicle = []
    beam_start = ego_state.position
    for i in range(lidar_circle_num_beams):
        theta = i * (2 * np.pi / lidar_circle_num_beams)
        x_delta = lidar_sensor_radius * np.cos(theta)
        y_delta = lidar_sensor_radius * np.sin(theta)
        beam_length = np.sqrt(x_delta ** 2 + y_delta ** 2)
        beam_angle = ego_state.orientation + theta
        surrounding_beams_ego_vehicle.append((beam_start, beam_length, beam_angle))

    obs = SurroundingObservation(configs)
    obs._ego_state = ego_state
    detected_obstacles = [None] * len(obstacle_shapes)
    obstacle_distances, _ = obs._get_obstacles_with_surrounding_beams(obstacle_shapes, detected_obstacles,
                                                                      surrounding_beams_ego_vehicle)
    print(obstacle_distances)
    assert all(np.isclose(obstacle_distances, desired_distances).reshape(-1))


@pytest.mark.parametrize(("obstacle_state", "obstacle_lanelet_ids", "desired_id"), [
    (State(**{"time_step": 2, "position": np.array([0, -3]), "orientation": 0, "velocity": 0}), [0], 0), (
            State(**{"time_step": 2, "position": np.array([-19, -2]), "orientation": 0, "velocity": 0}), [30622, 30624],
            30624), (
            State(**{"time_step": 2, "position": np.array([-19, -3]), "orientation": -np.pi / 4, "velocity": 0}),
            [30622, 30624], 30624)])
@unit_test
@functional
def test_get_occupied_lanelet_id(obstacle_state: State, obstacle_lanelet_ids: List[int], desired_id: Union[None, int]):
    '''test for the _get_occupied_lanelet_id function of SurroundingObservation'''

    lanelet_id = SurroundingObservation._get_occupied_lanelet_id(scenario_2, obstacle_lanelet_ids, obstacle_state)
    if desired_id is None:
        assert lanelet_id is None
    else:
        assert np.isclose(desired_id, lanelet_id)


@pytest.mark.parametrize(("lanelet_ids", "states", "all_lanelets_set", "filtered_lanelet_ids"),
                         [([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], {1, 2}, [1, 2]),
                          ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], {1, 2, 5}, [1, 2, 5]),
                          ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], {}, []),
                          ([30622, 30624], [1, 2], {30622, 30626}, [30622]), ])
@unit_test
@functional
def test_filter_obstacles_in_adj_lanelet(lanelet_ids: List[int], states: List[State], all_lanelets_set: set,
                                         filtered_lanelet_ids: List[int]):
    obstacles = [None] * len(states)
    obstacle_lanelet, obstacle_state, _ = \
        SurroundingObservation._filter_obstacles_in_adj_lanelet(lanelet_ids, states, obstacles, all_lanelets_set)
    assert len(obstacle_lanelet) == len(obstacle_state)
    assert len(obstacle_lanelet) == len(filtered_lanelet_ids)
    assert all(np.isclose(obstacle_lanelet, filtered_lanelet_ids).reshape(-1))


@pytest.mark.parametrize(("surrounding_area", "current_timestep", "desired_lanelet_ids"), [
    (create_collision_object(Polygon(np.array([[-100, -100], [-101, -100], [-101, -101]]))), 0, []),
    (create_collision_object(Polygon(np.array([[-39, -6], [387, -1.5], [387, 2.25], [-39.3, -2]]))), 0, [30622]),
    (create_collision_object(Polygon(np.array([[-39, -1.3], [387, 5.74], [387, 2.25], [-39.3, -2]]))), 0, []),
    (create_collision_object(Polygon(np.array([[-39, 4.87], [387, 9.27], [387, 5.74], [-39.3, 1.34]]))), 0, [30626])])
@unit_test
@functional
def test_get_obstacles_in_surrounding_area(surrounding_area: pycrcc.Shape, current_timestep: int,
                                           desired_lanelet_ids: List[int]):
    """
    current_timestep
    scenario
    obstacles
    """
    obs = SurroundingObservation({"reward_configs_hybrid": {}, "surrounding_configs": {}})
    obs._scenario = scenario_2
    obs._current_time_step = current_timestep

    lanelet_ids, obstacle_states, obstacles = obs._get_obstacles_in_surrounding_area(surrounding_area)
    assert len(lanelet_ids) == len(desired_lanelet_ids)
    assert all(np.isclose(desired_lanelet_ids, lanelet_ids).reshape(-1))


@pytest.mark.parametrize(("obs", "expected_types"), [
    (scenario_1.static_obstacles + scenario_1.dynamic_obstacles, [1, 1, 4, 1, 1, 1, 1, 1, 1, 1]),
    (scenario_2.static_obstacles + scenario_2.dynamic_obstacles, [1, 1]),
    ([None, None], [0, 0])
])
@unit_test
@functional
def test_get_obstacle_types(obs: List[Optional[Obstacle]], expected_types: List[int]):
    configs = {"reward_configs_hybrid": {"reward_safe_distance_coef": -1.},
               "surrounding_configs": {"observe_lidar_circle_surrounding": True}}
    observation = SurroundingObservation(configs)
    observation.observation_dict = dict()
    observation._get_vehicle_types(obs)

    detected_types = observation.observation_dict["vehicle_type"]
    print(detected_types)

    assert all(np.equal(detected_types, expected_types))


@pytest.mark.parametrize(("obs", "expected_lights"), [
    (scenario_1.dynamic_obstacles, 10 * [0]),
    (scenario_2.dynamic_obstacles, 2 * [0]),
    ([None, None], [0, 0])
])
@unit_test
@functional
def test_get_obstacle_lights(obs: List[Optional[Obstacle]], expected_lights: List[int]):
    configs = {"reward_configs_hybrid": {"reward_safe_distance_coef": -1.},
               "surrounding_configs": {"observe_lidar_circle_surrounding": True}}
    observation = SurroundingObservation(configs)
    observation.observation_dict = dict()
    observation._get_vehicle_lights(obs)

    detected_lights = observation.observation_dict["vehicle_signals"]

    assert all(np.equal(detected_lights, expected_lights))


@pytest.mark.parametrize("preprocess", [True])
@module_test
@functional
def test_continuous_collision_checking(preprocess):
    # construct a scenario with one dynamic obstacle
    scenario = Scenario(dt=1, scenario_id=ScenarioID("test"))
    obstacle_shape = Rectangle(width=2, length=5)
    state_list = [
        State(position=np.array([10., 0.]), velocity=10., orientation=0., time_step=1),
        State(position=np.array([20., 0.]), velocity=10., orientation=0., time_step=2)
    ]
    dynamic_obstacle = DynamicObstacle(scenario.generate_object_id(),
                                       ObstacleType.CAR,
                                       obstacle_shape,
                                       State(position=np.array([0., 0.]), velocity=10., orientation=0., time_step=0),
                                       TrajectoryPrediction(Trajectory(1, state_list), obstacle_shape))

    scenario.add_objects(dynamic_obstacle)

    # construct observation collector and collision checker
    config_file = PATH_PARAMS["configs"]["commonroad-v1"]
    with open(config_file, "r") as config_file:
        config = yaml.safe_load(config_file)

        # Assume default environment configurations
    configs = config["env_configs"]
    observation_collector = ObservationCollector(configs)
    observation_collector._scenario = scenario
    observation_collector._benchmark_id = "test"
    import time
    t1 = time.time()
    observation_collector._update_collision_checker()
    print(f"Elapsed time {time.time()-t1}")

    # ego vehicle
    vehicle_params = {
        "vehicle_type": 2,  # VehicleType.BMW_320i
        "vehicle_model": 0,  # 0: VehicleModel.PM; 2: VehicleModel.KS;
    }

    initial_state = State(position=np.array([5., -5.]), velocity=0., velocity_y=10., orientation=np.pi/2, time_step=0)
    dt = 1.0

    # Not to do anything, just continue the way with the given velocity
    vehicle = ContinuousVehicle(vehicle_params)
    vehicle.reset(initial_state, dt)
    next_state = State(position=np.array([5., 5.]), velocity=0., velocity_y=10., orientation=np.pi/2, time_step=1)
    vehicle.set_current_state(next_state)

    assert SurroundingObservation._check_collision(observation_collector._collision_checker, vehicle) == preprocess




