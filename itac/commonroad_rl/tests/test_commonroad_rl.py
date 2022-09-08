__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Peter Kocsis"
__email__ = "peter.kocsis@tum.de"
__status__ = "Integration"

"""
Integration tests of the CommonRoad-RL repository
"""
import logging
import os
import shutil

from commonroad_rl.train_model import run_stable_baselines, run_stable_baselines_argsparser
from commonroad_rl.generate_solution import solve_scenarios
from commonroad_rl.tests.common.marker import *
from commonroad_rl.tests.common.path import resource_root, output_root
from commonroad_rl.tools.pickle_scenario.xml_to_pickle import pickle_xml_scenarios

logging.root.setLevel(logging.DEBUG)

resource_path = resource_root("test_commonroad_rl")
output_path = output_root("test_commonroad_rl")


def run_overfit(test_batch, goal_relaxation, num_of_steps, env_id):
    xml_scenarios_path = os.path.join(resource_path, test_batch)
    output_base_path = os.path.join(output_path, test_batch)
    pickle_path = os.path.join(output_base_path, "pickles")
    log_path = os.path.join(output_base_path, "logs")
    solution_path = os.path.join(output_base_path, "solutions")

    # Pickle CommonRoad scenarios
    pickle_xml_scenarios(
        input_dir=xml_scenarios_path,
        output_dir=pickle_path
    )

    # Overfit model
    meta_scenario_path = os.path.join(pickle_path, "meta_scenario")
    train_reset_config_path = os.path.join(pickle_path, "problem")
    test_reset_config_path = os.path.join(pickle_path, "problem")
    shutil.copytree(test_reset_config_path, os.path.join(pickle_path, "problem_test"))
    visualization_path = os.path.join(output_path, "images")
    print(meta_scenario_path)
    algo = "ppo2"

    args_str = (
        f"--algo {algo} --env {env_id} --seed 1 --eval-freq 1000 --log-folder {log_path} --n-timesteps {num_of_steps}"
        f" --info_keywords is_collision is_time_out is_off_road --env-kwargs"
        f' reward_type:"hybrid_reward"'
        f' meta_scenario_path:"{meta_scenario_path}"'
        f' train_reset_config_path:"{train_reset_config_path}"'
        f' test_reset_config_path:"{test_reset_config_path}"'
        f' visualization_path:"{visualization_path}" '
    )

    if env_id == "commonroad-v1":
        args_str += "goal_configs:{'relax_is_goal_reached':" + f"{goal_relaxation}" + "} "
        # TODO: force other necessary settings for this scenario
        args_str += "surrounding_configs:{'observe_lane_circ_surrounding':" + f"{True}" + "}"
        args_str += " surrounding_configs:{'observe_lidar_circle_surrounding':" + f"{False}" + "}"
        args_str += " reward_configs_hybrid:{'reward_get_close_coefficient':2.," \
                    "'reward_goal_reached':1000.,'reward_collision':-1000.}"
        args_str += " vehicle_params:{'vehicle_type':2,'vehicle_model':0}"

    args = run_stable_baselines_argsparser().parse_args(args_str.split(sep=" "))
    run_stable_baselines(args)  # , save_path=os.path.join(output_path, "model"))

    # Solve scenarios
    model_path = os.path.join(log_path, algo, f"{env_id}_1")
    cost_function = "JB1"

    results = solve_scenarios(
        test_path=pickle_path,
        model_path=model_path,
        algo=algo,
        solution_path=solution_path,
        cost_function=cost_function,
        env_id=env_id,
    )

    assert all(results), f"not all overfit scenarios solved: {results}"


@pytest.mark.parametrize(
    ("env_id", "test_batch", "goal_relaxation", "num_of_steps"),
    [("commonroad-v1", "DEU_A9-2_1_T-1", False, 6000)]
)
@functional
@integration_test
def test_overfit_model(env_id, test_batch, goal_relaxation, num_of_steps):
    run_overfit(test_batch, goal_relaxation, num_of_steps, env_id)


# TODO: add more difficult batch
@pytest.mark.parametrize(
    ("env_id", "test_batch", "goal_relaxation", "num_of_steps"),
    [("commonroad-v1", "DEU_A99-1_1_T-1", False, 30000)],
)
@slow
@functional
@integration_test
def test_overfit_model_slow(env_id, test_batch, goal_relaxation, num_of_steps):
    run_overfit(test_batch, goal_relaxation, num_of_steps, env_id)
