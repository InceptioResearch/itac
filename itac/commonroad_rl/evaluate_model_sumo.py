"""
Module for playing trained model using Stable baselines
"""
import argparse
import csv
import time
import logging
import os
import yaml
import glob
import multiprocessing

os.environ["KMP_WARNINGS"] = "off"
os.environ["KMP_AFFINITY"] = "none"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
logging.getLogger("tensorflow").disabled = True
from stable_baselines.common import set_global_seeds

from commonroad_rl.evaluate_model import create_environments
from commonroad_rl.train_model import LoggingMode
from commonroad_rl.utils_run.utils import load_model_and_vecnormalize
from commonroad_rl.gym_commonroad.observation.goal_observation import GoalObservation
from traci.exceptions import FatalTraCIError, TraCIException

import copy
import os
import pickle
from typing import Tuple, Dict
import numpy as np

from sumocr.sumo_config.default import DefaultConfig
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.interface.sumo_simulation import SumoSimulation
from sumocr.maps.sumo_scenario import ScenarioWrapper

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv

try:
    from cbf_iss.cbf.cbf_wrapper import CBFWrapper
except ImportError:
    CBFWrapper = None
try:
    from mpi4py import MPI
except ImportError:
    print("ImportFailure MPI")
    MPI = None

LOGGER = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluates PPO2 trained model with SUMO interactive scenarios",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env_id", type=str, default="commonroad-v1", help="environment ID")
    parser.add_argument("--algo", type=str, default="ppo2")
    parser.add_argument("--num_processes", "-n", type=int, default=1)
    parser.add_argument("--test_path", "-path", type=str, default="", help="Path to SUMO scenario folders")
    parser.add_argument("--evaluation_path", type=str, default="evaluation_sumo")
    parser.add_argument("--model_path", "-model", type=str, default="")
    parser.add_argument("--viz_path", "-viz", type=str, default="")
    parser.add_argument("--hyperparam_filename", "-hyperparam_f", type=str, default="model_hyperparameters.yml")
    parser.add_argument("--config_filename", "-config_f", type=str, default="environment_configurations.yml")
    parser.add_argument("--logging_mode", default=LoggingMode.INFO, type=LoggingMode, choices=list(LoggingMode))
    parser.add_argument("--render", "-r", action="store_true", help="Whether to render the env during simulation")

    return parser


def simulate_scenario(conf: DefaultConfig,
                      scenario_wrapper: ScenarioWrapper,
                      planning_problem_set: PlanningProblemSet = None,
                      model=None,
                      env=None, render=False) -> Tuple[Scenario, Dict[int, EgoVehicle], Dict]:
    """
    Simulates an interactive scenario with specified mode

    :param conf: config of the simulation
    :param scenario_wrapper: scenario wrapper used by the Simulator
    :param planning_problem_set: planning problem set of the scenario
    :return: simulated scenario and dictionary with items {planning_problem_id: EgoVehicle}
    """

    # initialize simulation
    sumo_sim = SumoSimulation()
    # TODO: fix in scenario converter
    conf.dt = scenario_wrapper.initial_scenario.dt
    sumo_sim.initialize(conf, scenario_wrapper, planning_problem_set=planning_problem_set)

    # simulation with plugged in planner
    if isinstance(env.venv.envs[0].env.env, CommonroadEnv):
        cr_env = env.venv.envs[0].env.env
    elif isinstance(env.venv.envs[0].env, CBFWrapper):
        cr_env = env.venv.envs[0].env.env.env

    assert isinstance(cr_env, CommonroadEnv)

    def run_simulation():
        step = 0
        done = False
        infos = dict()
        while not done:
            # retrieve the CommonRoad scenario at the current time step, e.g. as an input for a prediction module
            current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
            # TODO: use commonroad-io Scenario.assign_obstacles_to_lanelets after next release
            current_scenario.assign_obstacles_to_lanelets(time_steps=[0])
            for idx, ego_vehicle in enumerate(sumo_sim.ego_vehicles.values()):
                # retrieve the current state of the ego vehicle
                planning_problem = list(planning_problem_set.planning_problem_dict.values())[idx]
                planning_initital_state = copy.deepcopy(ego_vehicle.current_state)
                # TODO: check if slip_angle matters
                planning_initital_state.time_step = 0
                planning_initital_state.slip_angle = 0.

                observations = env.reset(
                    scenario=current_scenario,
                    planning_problem=PlanningProblem(
                        planning_problem_id=planning_problem.planning_problem_id,
                        initial_state=planning_initital_state,
                        goal_region=planning_problem.goal
                    )
                )
                # fix is_time_out here since the current_step inside CommonRoadEnv is incorrect
                is_goal_reached = cr_env.observation_dict["is_goal_reached"][0]
                is_time_out = step >= cr_env.observation_collector.episode_length and not is_goal_reached
                cr_env.observation_dict["is_time_out"] = np.array([is_time_out])
                done, reason, termination_info = cr_env.termination.is_terminated(
                    cr_env.observation_dict,
                    cr_env.ego_action
                )

                infos.update(termination_info)

                # update infos (filter out invalid collision)
                if isinstance(env.venv.envs[0].env, CBFWrapper) and \
                        (termination_info["is_collision"] == 1 or termination_info["is_off_road"] == 1):
                    # rerun simulation to filter out invalid collision and offroad
                    # (should reproduce the same result since reproducable)
                    simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()
                    obs = env.reset(
                        scenario=simulated_scenario,
                        planning_problem=planning_problem
                    )
                    done = False
                    while not done:
                        action = model.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)
                    infos = info[0]

                if render:
                    env.render(mode=step + 1)

                if done:
                    return infos # terminated by collision

                observation_keys = []
                for key, value in cr_env.observation_dict.items():
                    observation_keys.extend([key]*value.size)

                obs_distance_goal_long_advance_idx = observation_keys.index("distance_goal_long_advance")
                obs_distance_goal_lat_advance_idx = observation_keys.index("distance_goal_lat_advance")
                obs_distance_goal_time_idx = observation_keys.index("distance_goal_time")
                obs_remaining_steps_idx = observation_keys.index("remaining_steps")

                if step > 0:
                    # fix distance_goal_long_advance
                    distance_goal_long_advance = abs(last_distance_goal_long) - abs(
                        cr_env.observation_dict["distance_goal_long"])
                    distance_goal_lat_advance = abs(last_distance_goal_lat) - abs(
                        cr_env.observation_dict["distance_goal_lat"])

                    remaining_steps = cr_env.observation_dict["remaining_steps"] - step
                    distance_goal_time = np.array([GoalObservation._get_goal_time_distance(step, planning_problem.goal)])

                    observations[0][obs_distance_goal_long_advance_idx] = distance_goal_long_advance
                    observations[0][obs_distance_goal_lat_advance_idx] = distance_goal_lat_advance
                    observations[0][obs_remaining_steps_idx] = remaining_steps
                    observations[0][obs_distance_goal_time_idx] = distance_goal_time

                    # normalize
                    tmp_normalized_obs = env.normalize_obs(observations)
                    for idx in [obs_distance_goal_long_advance_idx, obs_distance_goal_lat_advance_idx, obs_remaining_steps_idx, obs_distance_goal_time_idx]:
                        observations[0][idx] = tmp_normalized_obs[0][idx]

                assert cr_env.scenario is current_scenario
                action = model.predict(observations, deterministic=True)

                last_distance_goal_long = cr_env.observation_dict["distance_goal_long"]
                last_distance_goal_lat = cr_env.observation_dict["distance_goal_lat"]
                _, _, done, info = env.step(action)
                infos = info[0]

                if done:
                    return infos # terminated by other reasons (offroad, timeout, goal reached etc)

                # get ego state
                next_state = cr_env.ego_action.vehicle.state

                # update the ego vehicle with new trajectory with only 1 state for the current step
                next_state.time_step = 1
                trajectory_ego = [next_state]
                ego_vehicle.set_planned_trajectory(trajectory_ego)

            if done:
                break
            sumo_sim.simulate_step()
            step += 1

    info = run_simulation()

    # retrieve the simulated scenario in CR format
    simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

    # stop the simulation
    sumo_sim.stop()

    ego_vehicles = {list(planning_problem_set.planning_problem_dict.keys())[0]:
                        ego_v for _, ego_v in sumo_sim.ego_vehicles.items()}

    return simulated_scenario, ego_vehicles, info


def simulate_with_planner(interactive_scenario_path: str,
                          create_ego_obstacle: bool = False,
                          model=None,
                          env=None,
                          render=False) \
        -> Tuple[Scenario, PlanningProblemSet, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with a plugged in motion planner

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param create_ego_obstacle: indicates whether to create obstacles from the planned trajectories as the ego vehicles
    :return: Tuple of the simulated scenario, planning problem set, and list of ego vehicles
    """
    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    scenario_with_planner, ego_vehicles, info = simulate_scenario(
        conf,
        scenario_wrapper,
        planning_problem_set=planning_problem_set,
        model=model,
        env=env,
        render=render
    )

    scenario_with_planner.scenario_id = scenario.scenario_id

    if create_ego_obstacle:
        for pp_id, planning_problem in planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            scenario_with_planner.add_objects(obstacle_ego)

    return scenario_with_planner, planning_problem_set, ego_vehicles, info


def load_sumo_configuration(interactive_scenario_path: str) -> DefaultConfig:
    with open(os.path.join(interactive_scenario_path, "simulation_config.p"), "rb") as input_file:
        conf = pickle.load(input_file)

    return conf


def simulate_batch(i, sumo_files, viz_path, evaluation_path, args):
    env, normalize = create_environments(
        args.env_id,
        args.model_path,
        viz_path,
        args.hyperparam_filename,
        args.config_filename,
        args.logging_mode.value,
        play=False,
        test_env=True,
        cache_navigators=True, # SUMO scenarios have to cache navigators since reset is called for each step
        action_configs={"continuous_collision_checking": False} # SUMO scenario don't support continuous collision check
    )
    model, env = load_model_and_vecnormalize(args.model_path, args.algo, normalize, env)

    fd_result = open(os.path.join(evaluation_path, f"{i}_results.csv"), "w")
    csv_writer = csv.writer(fd_result)

    num_valid_collisions, num_collisions, num_valid_off_road, num_offroad, \
    num_goal_reaching, num_timeout, total_scenarios = 0, 0, 0, 0, 0, 0, 0

    for i, sumo_fn in enumerate(sumo_files):
        # if os.getlogin() == "wangx":
        #     scenario_id = os.path.splitext(os.path.basename(sumo_fn))[0].replace("T-1", "I-1")
        #     sumo_fn = os.path.join(
        #         "/home/wangx/data/highD-dataset-v1.0/10m_goal_no_downsample_lane_change/sumo",
        #         scenario_id
        #     )
        try:
            scenario_with_planner, _, ego_vehicles, info = simulate_with_planner(
                sumo_fn, model=model, env=env, render=args.render
            )
        except (FatalTraCIError, TraCIException, AttributeError) as e:
            # copy erronous scenario
            os.makedirs(os.path.join(args.model_path, "error"), exist_ok=True)
            os.system(f"cp -r {sumo_fn} {os.path.join(args.model_path, 'error/.')}")
            LOGGER.info(f"scenario {sumo_fn}: {e}")
            continue

        # log collision rate, off-road rate, and goal-reaching rate
        total_scenarios += 1
        num_valid_collisions += info.get("valid_collision", info["is_collision"])
        num_collisions += info["is_collision"]
        num_timeout += info["is_time_out"]
        num_valid_off_road += info.get("valid_off_road", info["is_off_road"])
        num_offroad += info["is_off_road"]
        num_goal_reaching += info["is_goal_reached"]

        termination_reason = "other"
        if info.get("is_time_out", 0) == 1:
            termination_reason = "time_out"
        elif info.get("is_off_road", 0) == 1:
            if "valid_off_road" in info and info["valid_off_road"] == 1:
                termination_reason = "valid_off_road"
            else:
                termination_reason = "off_road"
        elif info.get("is_collision", 0) == 1:
            if "valid_collision" in info and info["valid_collision"] == 1:
                termination_reason = "valid_collision"
            else:
                termination_reason = "collision"
        elif info.get("is_goal_reached", 0) == 1:
            termination_reason = "goal_reached"

        if "scenario_name" not in info or "current_episode_time_step" not in info:
            print(f"info={info}")
            continue
        csv_writer.writerow((info["scenario_name"], info["current_episode_time_step"], termination_reason))

    fd_result.close()

    return num_valid_collisions, num_collisions, num_valid_off_road, num_offroad, \
           num_goal_reaching, num_timeout, total_scenarios


def main():
    t1 = time.time()
    args = get_parser().parse_args()

    LOGGER.setLevel(args.logging_mode.value)
    handler = logging.StreamHandler()
    handler.setLevel(args.logging_mode.value)
    LOGGER.addHandler(handler)

    LOGGER.info("Start")

    # TODO: take all sumo folders that have a pickle file in problem_test
    sumo_paths = sorted(glob.glob(os.path.join(args.test_path, "*.pickle")))

    # if os.getlogin() == "xiao":
    sumo_paths = [x[0] for x in os.walk(args.test_path)][1:]

    # create evaluation folder in model_path
    evaluation_path = os.path.join(args.model_path, args.evaluation_path)
    os.makedirs(evaluation_path, exist_ok=True)
    if args.viz_path == "":
        args.viz_path = os.path.join(evaluation_path, "img")

    LOGGER.info(f"Number of scenarios: {len(sumo_paths)}")

    set_global_seeds(1)

    if args.num_processes == 1:
        t1 = time.time()
        total_num_valid_collisions, total_num_collisions, total_num_valid_off_road, total_num_offroad, \
        total_num_goal_reaching, total_num_timeout, total_scenarios = \
            simulate_batch(0, sumo_paths, args.viz_path, evaluation_path, args)
        LOGGER.info(f"Elapsed time for simulating {len(sumo_paths)} scenarios: {time.time() - t1}s")
    else:
        num_files_per_process = len(sumo_paths) // args.num_processes
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = pool.starmap(
                simulate_batch,
                [
                    (
                        i,
                        sumo_paths[i * num_files_per_process: (i + 1) * num_files_per_process],
                        args.viz_path,
                        evaluation_path,
                        args
                    )
                    for i in range(args.num_processes)
                ])

        total_num_valid_collisions, total_num_collisions, total_num_valid_off_road, total_num_offroad, \
        total_num_goal_reaching, total_num_timeout, total_scenarios = 0, 0, 0, 0, 0, 0, 0
        for res in results:
            num_valid_collisions, num_collisions, num_valid_off_road, num_offroad, \
            num_goal_reaching, num_timeout, num_scenarios = res
            total_num_valid_collisions += num_valid_collisions
            total_num_valid_off_road += num_valid_off_road
            total_num_collisions += num_collisions
            total_num_offroad += num_offroad
            total_num_goal_reaching += num_goal_reaching
            total_num_timeout += num_timeout
            total_scenarios += num_scenarios

    # save evaluation results
    with open(os.path.join(evaluation_path, "results.csv"), 'w') as fd_result:
        fd_result.write("benchmark_id, time_steps, termination_reason\n")
        for i in range(args.num_processes):
            path = os.path.join(evaluation_path, f"{i}_results.csv")
            with open(path, 'r') as f:
                fd_result.write(f.read())
            os.remove(path)

    with open(os.path.join(evaluation_path, "overview.yml"), "w") as f:
        yaml.dump({
            "total_scenarios": total_scenarios,
            "num_valid_collisions": total_num_valid_collisions,
            "num_collisions": total_num_collisions,
            "num_timeout": total_num_timeout,
            "num_valid_off_road": total_num_valid_off_road,
            "num_off_road": total_num_offroad,
            "num_goal_reached": total_num_goal_reaching,
            "percentage_goal_reached": 100.0 * total_num_goal_reaching / total_scenarios,
            "percentage_offroad": 100.0 * total_num_offroad / total_scenarios,
            "percentage_collisions": 100.0 * total_num_collisions / total_scenarios,
            "percentage_valid_off_road": 100.0 * total_num_valid_off_road / total_scenarios,
            "percentage_valid_collisions": 100.0 * total_num_valid_collisions / total_scenarios,
            "percentage_timeout": 100.0 * total_num_timeout / total_scenarios
        }, f)

    print(f"Elapsed time for {len(sumo_paths)} scenarios: {time.time()-t1}s")


if __name__ == "__main__":
    main()
