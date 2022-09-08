import pickle

import commonroad_dc.pycrcc as pycrcc
import numpy as np
from commonroad.scenario.scenario import ScenarioID
from commonroad.scenario.trajectory import State
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object

from commonroad_rl.gym_commonroad.action.vehicle import ContinuousVehicle
from commonroad_rl.gym_commonroad.observation import TrafficSignObservation, ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.path import *


xml_path = os.path.join(resource_root("test_traffic_sign"))
pickle_path = os.path.join(output_root("test_traffic_sign"), "pickles")

def prepare_for_traffic_sign_test():

    pickle_xml_scenarios(
        input_dir=xml_path,
        output_dir=pickle_path
    )

    # Traffic Sign observation settings
    traffic_sign_observation = TrafficSignObservation(
        configs={"traffic_sign_configs":
                     {"observe_stop_sign": True,
                      "observe_yield_sign": True,
                      "observe_priority_sign": True,
                      "observe_right_of_way_sign": True,
                      }})

    # specify resource path
    meta_scenario_path = os.path.join(pickle_path, "meta_scenario")
    problem_path = os.path.join(pickle_path, "problem")

    # Load meta scenario and problem dict
    filename = "DEU_AAH-3_33000_T-1.pickle"

    meta_scenario_reset_dict_path = os.path.join(meta_scenario_path, "meta_scenario_reset_dict.pickle")
    with open(meta_scenario_reset_dict_path, "rb") as f:
        meta_scenario_reset_dict = pickle.load(f)

    # Load scenarios and problems
    fn = os.path.join(problem_path, filename)
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

    return traffic_sign_observation, scenario, connected_lanelet_dict, lanelet_polygons, lanelet_polygons_sg


traffic_sign_observation, scenario_ts, connected_lanelet_dict_ts, lanelet_polygons_ts, lanelet_polygons_sg_ts = prepare_for_traffic_sign_test()


@pytest.mark.parametrize(
    ("ego_position", "traffic_sign_expected", "distance_expected"),
    [
        (np.array([51.3018, -17.9640]),  # Previous result calculated using wrong method or by calculating distance to
                                         # wrong (not closest) signs by hand
         ["stop_sign"],
         [7.7294],  # Used to be 5.8765
         ),
        (np.array([51.3108, -30.]),
         ["yield_sign"],
         [5.2505],  # Used to be 14.6937
         ),
        (np.array([84.0947, -55.1586]),
         ["priority_sign"],
         [17.0867]  # Used to be 14.8328
         ),
        (np.array([52.5147, -20.9587]),
         ["right_of_way_sign"],
         [1.0976],  # Used to be 0.0
         ),
    ],
)
@module_test
@functional
def test_get_traffic_sign_on_lanelets(ego_position, traffic_sign_expected, distance_expected):
    """
    Test for traffic sign observation
    """
    # set ego vehicle
    ego_state = State(**{"time_step": 0, "position": ego_position, "orientation": 0, "velocity": 0})
    ego_vehicle = ContinuousVehicle({"vehicle_type": 2, "vehicle_model": 2})
    ego_vehicle.reset(ego_state, dt=0.1)

    # find ego lanelet here because multiple observations need it
    ego_lanelet_ids = ObservationCollector.sorted_lanelets_by_state(scenario_ts, ego_state,
                                                                    lanelet_polygons_ts, lanelet_polygons_sg_ts)
    ego_lanelet_id = ego_lanelet_ids[0]

    ego_lanelet = scenario_ts.lanelet_network.find_lanelet_by_id(ego_lanelet_id)

    local_ccosy, _ = ObservationCollector.get_local_curvi_cosy(scenario_ts.lanelet_network,
                                                               ego_lanelet_id,
                                                               None,
                                                               max_lane_merge_range=1000.)
    # observe traffic sign on ego vehicle lanelet
    traffic_sign_observation.observe(scenario_ts, ego_vehicle, ego_lanelet, local_ccosy)

    traffic_sign_observed = []
    traffic_sign_distance = []
    if traffic_sign_observation.observation_dict["stop_sign"]:
        traffic_sign_observed.append("stop_sign")
    if traffic_sign_observation.observation_dict["yield_sign"]:
        traffic_sign_observed.append("yield_sign")
    if traffic_sign_observation.observation_dict["priority_sign"]:
        traffic_sign_observed.append("priority_sign")
    if traffic_sign_observation.observation_dict["right_of_way_sign"]:
        traffic_sign_observed.append("right_of_way_sign")

    # calculate the longitudinal distance from ego vehicle to traffic sign
    if len(traffic_sign_observed) != 0:
        for each_sign in traffic_sign_observed:
            dis = each_sign + "_distance_long"
            traffic_sign_distance.append(traffic_sign_observation.observation_dict[dis][0])
    print(traffic_sign_observed, traffic_sign_distance)

    # Check against ground truth
    assert traffic_sign_observed == traffic_sign_expected
    assert np.allclose(traffic_sign_distance, distance_expected, atol=0.001)


