"""
Module for playing trained model using Stable baselines
"""
import argparse
import csv
import glob
import logging
import os
os.environ["KMP_WARNINGS"] = "off"
os.environ["KMP_AFFINITY"] = "none"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").disabled = True
import pickle
from typing import Union
import numpy as np
import yaml
from gym import Env
from stable_baselines.common import set_global_seeds

from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.train_model import LoggingMode
from commonroad_rl.utils_run.utils import get_wrapper_class, make_env, load_model_and_vecnormalize
from commonroad_rl.utils_run.vec_env import CommonRoadVecEnv

try:
    from mpi4py import MPI
except ImportError:
    print("ImportFailure MPI")
    MPI = None

LOGGER = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluates PPO2 trained model with specified test scenarios",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env_id", type=str, default="commonroad-v1", help="environment ID")
    parser.add_argument("--algo", type=str, default="ppo2")
    parser.add_argument("--test_path", "-i", type=str, help="Path to pickled test scenarios",
                        default=PATH_PARAMS["test_reset_config"])
    parser.add_argument("--model_path", "-model", type=str, help="Path to trained model",
                        default=PATH_PARAMS["log"] + "/ppo2/commonroad-v0_3")
    parser.add_argument("--evaluation_path", "-eval_path", type=str, help="Folder to store evaluation data",
                        default="evaluation")
    parser.add_argument("--viz_path", "-viz", type=str, default="")
    parser.add_argument("--num_scenarios", "-n", default=-1, type=int, help="Maximum number of scenarios to draw")
    parser.add_argument("--multiprocessing", "-mpi", action="store_true")
    parser.add_argument("--combine_frames", "-1", action="store_true",
                        help="Combine rendered environments into one picture")
    parser.add_argument("--skip_timesteps", "-st", type=int, default=1,
                        help="Only render every nth frame (including first and last)")
    parser.add_argument("--no_render", "-nr", action="store_true", help="Whether store render images")
    parser.add_argument("--no_csv", action="store_true", help="Not store statistic results to speed up MPI processes")
    parser.add_argument("--hyperparam_filename", "-hyperparam_f", type=str, default="model_hyperparameters.yml")
    parser.add_argument("--config_filename", "-config_f", type=str, default="environment_configurations.yml")
    parser.add_argument("--log_action_curve", action="store_true", help="Store action curve plot for analysis")
    parser.add_argument("--log_step_info", action="store_true", help="Store all info dict in step function")
    # parser.add_argument("--step_info_kwargs", type=str, nargs="+", default=(),
    #                     help="(tuple) extra information to log, from the information return of environment.step")
    parser.add_argument("--logging_mode", default=LoggingMode.INFO, type=LoggingMode, choices=list(LoggingMode))

    return parser


def create_environments(env_id: str, model_path: str, viz_path: str,
                        hyperparam_filename: str = "model_hyperparameters.yml",
                        config_filename: str = "environment_configurations.yml",
                        logging_mode: LoggingMode = LoggingMode.INFO.value, test_path=None, meta_path=None,
                        play: bool = True, test_env: bool = True, **kwargs) -> CommonRoadVecEnv:
    """
    Create CommonRoad vectorized environment environment

    :param config_filename:
    :param env_id: Environment gym id
    :param test_path: Path to the test files
    :param meta_path: Path to the meta-scenarios
    :param model_path: Path to the trained model
    :param viz_path: Output path for rendered images
    :param hyperparam_filename: The filename of the hyperparameters
    :param env_kwargs: Keyword arguments to be passed to the environment
    """
    # Get environment keyword arguments including observation and reward configurations
    config_fn = os.path.join(model_path, config_filename)
    # TODO: config is loaded in CommonRoadEnv, duplicated here => remove
    # with open(config_fn, "r") as f:
    #     env_kwargs = yaml.load(f, Loader=yaml.Loader)

    # env_kwargs.update({"logging_mode": logging_mode,
    #                    "meta_scenario_path": meta_path,
    #                    "test_reset_config_path": test_path,
    #                    "visualization_path": viz_path,
    #                    "play": play,
    #                    "test_env": test_env})

    env_kwargs = {"logging_mode": logging_mode,
                  "visualization_path": viz_path,
                  "play": play,
                  "test_env": test_env,
                  "config_file": config_fn}

    env_kwargs.update(kwargs)
    if meta_path is not None:
        env_kwargs.update({"meta_scenario_path": meta_path})
    if test_path is not None:
        env_kwargs.update({"test_reset_config_path": test_path})

    # Load model hyperparameters:
    hyperparam_fn = os.path.join(model_path, hyperparam_filename)
    with open(hyperparam_fn, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.Loader)

    env_wrapper = get_wrapper_class(hyperparams)

    # Create environment
    # note that CommonRoadVecEnv is inherited from DummyVecEnv
    env = CommonRoadVecEnv([make_env(env_id, 0, wrapper_class=env_wrapper, env_kwargs=env_kwargs, subproc=False)])

    def on_reset_callback(env: Union[Env, CommonroadEnv], elapsed_time: float):
        # reset callback called before resetting the env
        if env.observation_dict["is_goal_reached"][-1]:
            LOGGER.debug("Goal reached")
        else:
            LOGGER.debug("Goal not reached")
        # env.render()

    env.set_on_reset(on_reset_callback)

    return env, hyperparams["normalize"]


def main():
    args = get_parser().parse_args()

    if MPI is None:
        args.multiprocessing = False

    LOGGER.setLevel(args.logging_mode.value)
    handler = logging.StreamHandler()
    handler.setLevel(args.logging_mode.value)
    LOGGER.addHandler(handler)

    meta_path = os.path.join(args.test_path, "meta_scenario")

    # mpi for parallel processing
    rank = 0
    size = 1
    comm = None
    if args.multiprocessing:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        test_path = os.path.join(args.test_path, "problem_test", str(rank))

    else:
        test_path = os.path.join(args.test_path, "problem_test")

    # create evaluation folder in model_path
    evaluation_path = os.path.join(args.model_path, args.evaluation_path)
    os.makedirs(evaluation_path, exist_ok=True)
    if args.viz_path == "":
        args.viz_path = os.path.join(evaluation_path, "img")

    env, normalize = create_environments(
        args.env_id, args.model_path, args.viz_path, args.hyperparam_filename, args.config_filename,
        args.logging_mode.value, test_path=test_path, meta_path=meta_path)

    LOGGER.info(f"Testing a maximum of {args.num_scenarios} scenarios")

    model, env = load_model_and_vecnormalize(args.model_path, args.algo, normalize, env)

    set_global_seeds(1)
    num_valid_collisions, num_collisions, num_valid_off_road, num_off_road, num_goal_reaching, \
        num_timeout, total_scenarios = 0, 0, 0, 0, 0, 0, 0

    if args.log_action_curve:
        accelerations = {}
        jerks = {}

    # In case there a no scenarios at all
    try:
        obs = env.reset()
    except IndexError:
        args.num_scenarios = 0

    if not args.no_csv:
        fd_result = open(os.path.join(evaluation_path, f"{rank}_results.csv"), "w")
        csv_writer = csv.writer(fd_result)

    count = 0
    info_dict = {}
    # for kwargs in args.step_info_kwargs:
    #     info_dict[kwargs] = []
    while count != args.num_scenarios:
        done = False
        if not args.no_render:
            env.render()
        benchmark_id = env.venv.envs[0].benchmark_id
        LOGGER.debug(benchmark_id)
        if args.log_action_curve:
            accelerations[benchmark_id] = []
            jerks[benchmark_id] = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if not args.no_render:
                env.render()
            if args.log_step_info:
                for kwargs in info[0]:
                    if kwargs not in info_dict:
                        info_dict[kwargs] = [info[0][kwargs]]
                    else:
                        info_dict[kwargs].append(info[0][kwargs])
            # for kwargs in args.step_info_kwargs:
            #     info_dict[kwargs].append(info[0][kwargs])
            LOGGER.debug(f"Step: {env.venv.envs[0].current_step}, \tReward: {reward}, \tDone: {done}")

            if args.log_action_curve:
                dt = env.venv.envs[0].scenario.dt
                ego_vehicle = env.venv.envs[0].ego_action.vehicle
                jerk_x = (ego_vehicle.state.acceleration - ego_vehicle.previous_state.acceleration) / dt
                jerk_y = (ego_vehicle.state.acceleration_y - ego_vehicle.previous_state.acceleration_y) / dt
                accelerations[benchmark_id].append(
                    [ego_vehicle.state.acceleration, ego_vehicle.state.acceleration_y])
                jerks[benchmark_id].append([jerk_x, jerk_y])
        # log collision rate, off-road rate, and goal-reaching rate
        info = info[0]

        total_scenarios += 1
        num_valid_collisions += info.get("valid_collision", info["is_collision"])
        num_collisions += info["is_collision"]
        num_timeout += info["is_time_out"]
        num_valid_off_road += info.get("valid_off_road", info["is_off_road"])
        num_off_road += info["is_off_road"]
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

        out_of_scenarios = info["out_of_scenarios"]
        if not args.no_csv:
            csv_writer.writerow((info["scenario_name"], info["current_episode_time_step"], termination_reason))

        if out_of_scenarios:
            break
        count += 1
    if not args.no_csv:
        fd_result.close()

    if args.log_action_curve:
        for key in accelerations.keys():
            accelerations[key] = np.array(accelerations[key])
            jerks[key] = np.array(jerks[key])
        with open(f"{args.model_path.split('/')[-1]}_actions.pkl", 'wb') as f:
            pickle.dump({"accelerations": accelerations, "jerks": jerks}, f)

    if args.log_step_info:
        with open(f"{os.path.abspath(args.model_path)}_step_info_{rank}.pkl", 'wb') as f:
            pickle.dump(info_dict, f)
    if args.multiprocessing:
        data = (total_scenarios, num_valid_collisions, num_collisions, num_valid_off_road, num_off_road, \
                num_timeout, num_goal_reaching)
        data = comm.gather(data)
        if rank == 0:
            g_num_scenarios, g_num_valid_collisions, g_num_collisions, g_num_valid_off_road, \
                g_num_off_road, g_num_timeout, g_num_goal_reaching = zip(*data)
            total_scenarios = sum(g_num_scenarios)
            num_valid_collisions = sum(g_num_valid_collisions)
            num_collisions = sum(g_num_collisions)
            num_valid_off_road = sum(g_num_valid_off_road)
            num_off_road = sum(g_num_off_road)
            num_timeout = sum(g_num_timeout)
            num_goal_reaching = sum(g_num_goal_reaching)
        else:
            return
    if not args.no_csv:
        # save evaluation results
        with open(os.path.join(evaluation_path, "results.csv"), 'w') as fd_result:
            fd_result.write("benchmark_id, time_steps, termination_reason\n")
            for i in range(size):
                path = os.path.join(evaluation_path, f"{i}_results.csv")
                with open(path, 'r') as f:
                    fd_result.write(f.read())
                os.remove(path)

        with open(os.path.join(evaluation_path, "overview.yml"), "w") as f:
            yaml.dump({
                "total_scenarios": total_scenarios,
                "num_collisions": num_collisions,
                "num_valid_collisions": num_valid_collisions,
                "num_valid_off_road": num_valid_off_road,
                "num_timeout": num_timeout,
                "num_off_road": num_off_road,
                "num_goal_reached": num_goal_reaching,
                "percentage_goal_reached": 100.0 * num_goal_reaching / total_scenarios,
                "percentage_off_road": 100.0 * num_off_road / total_scenarios,
                "percentage_collisions": 100.0 * num_collisions / total_scenarios,
                "percentage_valid_off_road": 100.0 * num_valid_off_road / total_scenarios,
                "percentage_valid_collisions": 100.0 * num_valid_collisions / total_scenarios,
                "percentage_timeout": 100.0 * num_timeout / total_scenarios
            }, f)

    if not args.no_csv:
        # Reorganize the rendered images according to result of the scenario
        # Flatten directory
        img_path = args.viz_path
        for d in os.listdir(img_path):
            dir_path = os.path.join(img_path, d)
            if not os.path.isdir(dir_path):
                continue

            for f in os.listdir(dir_path):
                os.rename(os.path.join(dir_path, f), os.path.join(img_path, f))

            os.rmdir(dir_path)

        # Split into different termination reasons
        with open(os.path.join(evaluation_path, "results.csv"), "r") as f:
            os.mkdir(os.path.join(img_path, "time_out"))
            os.mkdir(os.path.join(img_path, "off_road"))
            os.mkdir(os.path.join(img_path, "collision"))
            os.mkdir(os.path.join(img_path, "goal_reached"))
            os.mkdir(os.path.join(img_path, "other"))

            reader = csv.reader(f)
            reader.__next__()
            for [scenario_id, _, t_reason] in reader:
                if args.no_render:
                    dest_path = os.path.join(img_path, t_reason)
                else:
                    os.mkdir(os.path.join(img_path, t_reason, scenario_id))
                    dest_path = os.path.join(img_path, t_reason, scenario_id)
                for file_path in glob.glob(os.path.join(img_path, scenario_id + '*')):
                    os.rename(file_path, os.path.join(dest_path, os.path.basename(file_path)))


if __name__ == "__main__":
    main()
