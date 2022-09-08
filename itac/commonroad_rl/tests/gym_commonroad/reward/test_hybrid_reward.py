"""Unit tests for hybrid reward class"""
import math
import os
import pickle

import numpy as np
import yaml
from commonroad.common.solution import VehicleModel
from commonroad.scenario.scenario import ScenarioID

from commonroad_rl.gym_commonroad.reward.hybrid_reward import HybridReward
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.gym_commonroad.action import DiscretePMJerkAction
from commonroad_rl.gym_commonroad import constants
from commonroad_rl.gym_commonroad.observation import ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.tests.common.path import resource_root, output_root
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios

configs = {
    "reward_type": "hybrid_reward",
    "reward_configs_hybrid": {
        "reward_goal_reached": 50.,
        "reward_collision": -50.,
        "reward_off_road": -20.,
        "reward_friction_violation": -30.,
        "reward_time_out": -10.,
        "reward_closer_to_goal_long": 0.5,
        "reward_closer_to_goal_lat": 2.5,
        "reward_get_close_goal_time": 0.5,
        "reward_close_goal_orientation": 0.5,
        "reward_close_goal_velocity": 0.5
    },
    "goal_configs": {
        "observe_euclidean_distance": False,
        "observe_distance_goal_long": False
    },
    "surrounding_configs": {
        "observe_lane_change": False,
        "observe_lidar_circle_surrounding": True,
        "lidar_sensor_radius": 50.
    }
}


@unit_test
@functional
def test_termination_reward():
    reward = HybridReward(configs)
    observations = {
        "is_goal_reached": [True],
        "is_collision": [False],
        "is_off_road": [False],
        "is_friction_violation": [False],
        "is_time_out": [False]
    }

    assert reward.termination_reward(observations) == 50

    observations = {
        "is_goal_reached": [False],
        "is_collision": [False],
        "is_off_road": [True],
        "is_friction_violation": [False],
        "is_time_out": [False]
    }

    assert reward.termination_reward(observations) == -20


@unit_test
@functional
def test_goal_distance_reward():
    reward = HybridReward(configs)
    observations = {
        "distance_goal_long_advance": [3.],
        "distance_goal_lat_advance": [1.]
    }

    assert math.isclose(reward.goal_distance_reward(observations), 4.)

    observations = {
        "distance_goal_long_advance": [0.5],
        "distance_goal_lat_advance": [0.]
    }

    assert math.isclose(reward.goal_distance_reward(observations), .25)


@unit_test
@functional
def test_goal_time_reward():
    reward = HybridReward(configs)
    vehicle_params = {
        "vehicle_type": 2,  # 1: FORD_ESCORT; 2: BMW_320i; 3: VW_VANAGON
        "vehicle_model": 0  # 0: PM, 1: ST, 2: KS, 3: MB, 4: YawRate
    }
    ego_action: DiscretePMJerkAction = DiscretePMJerkAction(vehicle_params, 5, 5)
    observations = {
        "distance_goal_time": [-1],
    }

    assert math.isclose(reward.goal_time_reward(observations, ego_action), 0.5)

    observations = {
        "distance_goal_time": [1],
    }
    assert math.isclose(reward.goal_time_reward(observations, ego_action), 0)

    observations = {
        "distance_goal_time": [0],
    }
    assert math.isclose(reward.goal_time_reward(observations, ego_action), 0)


@unit_test
@functional
def test_goal_velocity_reward():
    reward = HybridReward(configs)
    observations = {
        "distance_goal_time": [0],
        "distance_goal_long": [0],
        "distance_goal_lat": [0],
        "distance_goal_velocity": [-1.]
    }

    class Dummy(object):
        pass

    ego_action = Dummy()
    setattr(ego_action, "vehicle", Dummy())
    setattr(ego_action.vehicle, "state", Dummy())
    setattr(ego_action.vehicle, "vehicle_model", VehicleModel.PM)
    setattr(ego_action.vehicle.state, "velocity", 2.)
    setattr(ego_action.vehicle.state, "velocity_y", 0.5)

    assert math.isclose(reward.goal_velocity_reward(observations, ego_action), 0.3606, abs_tol=1e-2)


@unit_test
@functional
def test_goal_orientation_reward():
    reward = HybridReward(configs)
    observations = {
        "distance_goal_time": [0],
        "distance_goal_long": [0],
        "distance_goal_lat": [0],
        "distance_goal_orientation": [1.]
    }
    assert math.isclose(reward.goal_orientation_reward(observations), (math.exp(-1.0 / np.pi) * 0.5))


@pytest.mark.parametrize(
    ("observations", "expected_reward"),
    [(
            {
                "v_ego": np.array([23., 0.]),
                "lane_based_v_rel": np.array([0., 0., 0., 0., -11.5, 0.]),
                "lane_based_p_rel": np.array([0., 0., 0., 0., 10., 0.]),
                "lane_change": np.array([0.])
            },
            -0.05510302167
    ),
        (
        {
            "v_ego": np.array([23.]),
            "lane_based_v_rel": np.array([0., 0., 0., 0., -11.5, 0.]),
            "lane_based_p_rel": np.array([0., 0., 0., 0., 10., 0.]),
            "lane_change": np.array([0.])
        },
        -0.05510302167
    ),
     (
         {
             "v_ego": np.array([23.]),
             "lane_based_v_rel": np.array([0., 0., 0., 0., 1., 0.]),
             "lane_based_p_rel": np.array([0., 0., 0., 0., 10., 0.]),
             "lane_change": np.array([0.])
         },
         0.
     ),
     (
         {
             "v_ego": np.array([11.5]),
             "lane_based_v_rel": np.array([0., -11.5, 0., 0., 0., 0.]),
             "lane_based_p_rel": np.array([0., 10., 0., 0., 10., 0.]),
             "lane_change": np.array([1.])
         },
         -0.05510302167
     )],
)
@unit_test
@functional
def test_safe_distance_reward(observations, expected_reward):

    configs["surrounding_configs"]["observe_lane_change"] = True
    configs["surrounding_configs"]["observe_lane_rect_surrounding"] = True
    configs["surrounding_configs"]["lane_rect_sensor_range_length"] = 100.
    configs["surrounding_configs"]["lane_rect_sensor_range_width"] = 8.
    configs["reward_configs_hybrid"]["reward_safe_distance_coef"] = -1.

    reward = HybridReward(configs)

    assert math.isclose(reward.safe_distance_reward(observations), expected_reward)


@pytest.mark.parametrize(
    "benchmark_id",
    ["FRA_Miramas-5_1_T-1",
     "USA_US101-14_1_T-1"],
)
@module_test
@nonfunctional
def test_reward_calculation(benchmark_id):
    config_path = constants.PATH_PARAMS["configs"]["commonroad-v1"]
    resource_path = resource_root("test_reward")

    xml_scenarios_path = os.path.join(resource_path)
    output_base_path = os.path.join(output_root("test_reward"))

    pickle_path = os.path.join(output_base_path, "pickles")

    # Pickle CommonRoad scenarios
    pickle_xml_scenarios(
        input_dir=xml_scenarios_path,
        output_dir=pickle_path
    )

    meta_scenario_path = os.path.join(pickle_path, "meta_scenario")
    problem_path = os.path.join(pickle_path, "problem")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = config["env_configs"]
    config["vehicle_params"]["vehicle_model"] = 0
    reward = HybridReward(config)
    ego_action: DiscretePMJerkAction = DiscretePMJerkAction(config["vehicle_params"], 5, 5)
    observation_collector = ObservationCollector(config)

    with open(os.path.join(meta_scenario_path, "meta_scenario_reset_dict.pickle"), "rb") as f:
        meta_scenario_reset_dict = pickle.load(f)

    # benchmark_id = "FRA_Miramas-5_1_T-1"
    with open(os.path.join(problem_path, f"{benchmark_id}.pickle"), "rb") as f:
        problem_dict: dict = pickle.load(f)

    problem = list(problem_dict["planning_problem_set"].planning_problem_dict.values())[0]

    scenario_id = ScenarioID.from_benchmark_id(benchmark_id, "2020a")
    map_id = parse_map_name(scenario_id)
    reset_config = meta_scenario_reset_dict[map_id]
    scenario = restore_scenario(reset_config["meta_scenario"], problem_dict["obstacle"], scenario_id)
    ego_action.reset(problem.initial_state, scenario.dt)
    observation_collector.reset(scenario, problem, reset_config, benchmark_id)
    observation_collector.observe(ego_action.vehicle)
    reward.reset(observation_collector.observation_dict, ego_action)
    reward.calc_reward(observation_collector.observation_dict, ego_action)
