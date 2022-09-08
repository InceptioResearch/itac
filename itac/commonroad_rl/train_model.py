"""
Module for training an agent using stable baselines
"""
import os

os.environ["KMP_WARNINGS"] = "off"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import logging

logging.getLogger("tensorflow").disabled = True
import gym
import time
import uuid
import copy
import yaml
import difflib
import argparse
import importlib
import numpy as np
from enum import Enum
from pprint import pformat
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.her import HERGoalEnvWrapper
from stable_baselines.gail import ExpertDataset
from stable_baselines import logger
# NUM_PARALLEL_EXEC_UNITS = 4
# os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "none"  # "granularity=fine,verbose,compact,1,0"

SB_LOGGER = logger.Logger(folder=None, output_formats=[logger.HumanOutputFormat(sys.stdout)])
LOGGER = logging.getLogger(__name__)

import commonroad_rl.gym_commonroad
from commonroad_rl.gym_commonroad.constants import ROOT_STR, PATH_PARAMS
from commonroad_rl.utils_run.callbacks import (
    SaveVecNormalizeCallback,
    MultiEnvsEvalCallback,
)
from commonroad_rl.utils_run.hyperparams_opt import optimize_hyperparams
from commonroad_rl.utils_run.observation_configs_opt import optimize_observation_configs
from commonroad_rl.utils_run.reward_configs_opt import optimize_reward_configs
from commonroad_rl.utils_run.noise import LinearNormalActionNoise
from commonroad_rl.utils_run.utils import (
    StoreDict,
    linear_schedule,
    get_wrapper_class,
    get_latest_run_id,
    make_env,
    ALGOS,
)

# numpy warnings because of tensorflow
import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings(action="ignore", category=UserWarning, module="gym")
warnings.simplefilter(action="ignore", category=FutureWarning)
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Optional dependencies
try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import (
    VecFrameStack,
    VecNormalize,
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)
from stable_baselines.common.noise import (
    AdaptiveParamNoiseSpec,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines.common.schedules import constfn
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback


class LoggingMode(Enum):
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    ERROR = logging.ERROR


def run_stable_baselines_argsparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, default="commonroad-v1", help="environment ID")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)
    parser.add_argument("-i", "--trained-agent", type=str, default="",
                        help="Path to a pretrained agent to continue training")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo2", type=str,
                        required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="Set the number of timesteps", default=int(1e6), type=int)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1, type=int)
    parser.add_argument("--eval-freq", default=10000, type=int,
                        help="Evaluate the agent every n steps (if negative, no evaluation)")
    parser.add_argument("--eval_timesteps", help="Number of timesteps to use for evaluation", default=1000, type=int)
    parser.add_argument("--eval_episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--save-freq", default=-1, type=int,
                        help="Save the model every n steps (if negative, no checkpoint)")
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--configs-path", type=str, default="",
                        help="Path to file for overwriting environment configurations")
    parser.add_argument("--hyperparams-path", type=str, default="",
                        help="Path to file for overwriting model hyperparameters")
    parser.add_argument("--optimize-observation-configs", action="store_true", default=False,
                        help="Optimize observation configurations")
    parser.add_argument("--optimize-reward-configs", action="store_true", default=False,
                        help="Optimize reward configurations")
    parser.add_argument("--guided", action="store_true", default=False,
                        help="Guided optimization of some reward parameters")
    parser.add_argument("--optimize-hyperparams", action="store_true", default=False,
                        help="Optimize model hyperparameters")
    parser.add_argument("--n-trials", help="Number of trials for optimization", type=int, default=5)
    parser.add_argument("--n-jobs", help="Number of parallel jobs for optimization", type=int, default=1)
    parser.add_argument("--sampler", type=str, default="tpe", choices=["random", "tpe", "skopt"],
                        help="Sampler for optimization")
    parser.add_argument("--pruner", help="Pruner for optimization", type=str,
                        default="median", choices=["halving", "median", "none"])
    parser.add_argument("--logging_mode", default=LoggingMode.INFO, type=LoggingMode, choices=list(LoggingMode))
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[],
                        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)")
    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict,
                        help="Overwrite model hyperparameters (e.g. learning_rate:0.01 train_freq:10)")
    parser.add_argument("-uuid", "--uuid", choices=["top", "none", "true"], type=str, default="none",
                        help="Ensure that the run has a unique ID")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict,
                        help='Overwrite environment configurations '
                             '(e.g. observe_heading:"True" reward_type:"\'default_reward\'" '
                             'meta_scenario_path:"str(\'./DEU_LocationA-11_11_T-1/meta_scenario\')")')
    parser.add_argument("--n_envs", help="Number of parallel training processes", type=int, default=1)
    parser.add_argument("--info_keywords", type=str, nargs="+", default=(),
                        help="(tuple) extra information to log, from the information return of environment.step, "
                             "see stable_baselines/bench/monitor.py")

    return parser


def construct_logger(logging_mode: int, save_path: str, logger):
    logger.setLevel(logging_mode)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_mode)
    file_handler = logging.FileHandler(os.path.join(save_path, "console_copy.txt"))
    file_handler.setLevel(logging_mode)

    formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug(f"Logging to console and {save_path + '/console_copy.txt'}")


def create_parameter_schedule(hyperparams: dict):
    # Create learning rate schedules
    for key in ["learning_rate", "cliprange", "cliprange_vf"]:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constfn(float(hyperparams[key]))
        else:
            raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
    return hyperparams


def construct_save_path(args):
    log_path = os.path.join(args.log_folder, args.algo)
    if args.uuid == "top":
        return args.log_folder
    elif args.uuid == "true":
        uuid_str = f"_{uuid.uuid4()}"
    else:
        uuid_str = ""
    return os.path.join(log_path, f"{args.env}_{get_latest_run_id(log_path, args.env) + 1}{uuid_str}")


def create_vec_normalized_env(exp_folder: str, env: VecNormalize):
    if isinstance(env, VecNormalize):
        env = env.venv
    if os.path.exists(os.path.join(exp_folder, "vecnormalize.pkl")):
        LOGGER.info(f"Loading running average from {os.path.join(exp_folder, 'vecnormalize.pkl')}")
        return VecNormalize.load(os.path.join(exp_folder, 'vecnormalize.pkl'), env)
    else:
        raise FileNotFoundError(f"vecnormalize.pkl not found in {exp_folder}")


def parse_noise(hyperparams, n_actions, algo):
    noise_type = hyperparams["noise_type"].strip()
    noise_std = hyperparams["noise_std"]
    if "adaptive-param" in noise_type:
        assert algo == "ddpg", "Parameter is not supported by SAC"
        hyperparams["param_noise"] = AdaptiveParamNoiseSpec(
            initial_stddev=noise_std, desired_action_stddev=noise_std
        )
    elif "normal" in noise_type:
        if "lin" in noise_type:
            hyperparams["action_noise"] = LinearNormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions),
                final_sigma=hyperparams.get("noise_std_final", 0.0) * np.ones(n_actions),
                max_steps=args.n_timesteps,
            )
        else:
            hyperparams["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
            )
    elif "ornstein-uhlenbeck" in noise_type:
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )
    else:
        raise RuntimeError(f"Unknown noise type {noise_type}")

    del hyperparams["noise_type"]
    del hyperparams["noise_std"]
    hyperparams = del_key_from_hyperparams(hyperparams, "noise_std_final")
    LOGGER.info(f"Applying {noise_type} noise with std {noise_std}")

    return hyperparams


def continue_learning(hyperparams, args, env, tensorboard_log, callbacks, normalize, save_path, env_id, rank):

    # Policy should not be changed
    del hyperparams["policy"]
    hyperparams = del_key_from_hyperparams(hyperparams, "n_envs")
    exp_folder = os.path.dirname(args.trained_agent)

    # TODO: The following statements have no effect, probably the model definition should be moved after these lines
    if normalize:
        env = create_vec_normalized_env(exp_folder, env)

    LOGGER.info(f"Loading from {args.trained_agent}")
    model = ALGOS[args.algo].load(args.trained_agent, env=env, tensorboard_log=tensorboard_log,
                                  verbose=args.logging_mode.value, **hyperparams)

    # Arguments to the learn function
    kwargs = {}
    if args.log_interval > -1:
        kwargs = {"log_interval": args.log_interval}
    if len(callbacks) > 0:
        kwargs["callback"] = callbacks

    try:
        model.learn(args.n_timesteps, **kwargs)
    except KeyboardInterrupt:
        pass

    # Only save worker of rank 0 when using mpi
    if rank == 0:
        LOGGER.info(f"Saving model to {save_path}")
        model.save(os.path.join(save_path, str(env_id)))
        if normalize:
            # Important: save the running average, for testing the agent we need that normalization
            LOGGER.info(f"Saving vectorized and normalized environment wrapper to {save_path}")
            model.get_vec_normalize_env().save(os.path.join(save_path, "vecnormalize.pkl"))


def optimize_parameters(hyperparams, args, save_path, create_env, sampling_setting_reward_configs,
                        env_kwargs, tensorboard_log, sampling_setting_observation_configs):
    def create_model(hyperparams, configs):
        """
        Helper to create a model with different hyperparameters
        """
        hyperparams = del_key_from_hyperparams(hyperparams, "n_envs")

        return ALGOS[args.algo](env=create_env(args.n_envs, eval_env=False, **configs),
                                tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

    if args.optimize_hyperparams:
        LOGGER.debug("Optimizing model hyperparameters")
        log_path = os.path.join(save_path, "model_hyperparameter_optimization")
        os.makedirs(log_path, exist_ok=True)
        optimized_hyperparams = optimize_hyperparams(args.algo,
                                                     args.env,
                                                     create_model,
                                                     create_env,
                                                     n_timesteps=args.n_timesteps,
                                                     eval_freq=args.eval_freq,
                                                     n_eval_episodes=args.eval_episodes,
                                                     n_trials=args.n_trials,
                                                     hyperparams=hyperparams,
                                                     configs=env_kwargs,
                                                     n_jobs=args.n_jobs,
                                                     seed=args.seed,
                                                     sampler_method=args.sampler,
                                                     pruner_method=args.pruner,
                                                     verbose=args.logging_mode.value,
                                                     log_path=log_path,
                                                     best_model_save_path=log_path)
        LOGGER.info(f"Saving optimized model hyperparameters to {log_path}")
        report_name = f"report_{args.algo}_{args.env}-{args.n_trials}-" \
                      f"trials-{args.n_timesteps}-steps-{args.sampler}-{args.pruner}.yml"
        with open(os.path.join(log_path, report_name), "w") as f:
            yaml.dump(optimized_hyperparams, f)

    if args.optimize_reward_configs:
        LOGGER.debug("Optimizing reward configurations")

        log_path = os.path.join(save_path, "reward_configuration_optimization")
        os.makedirs(log_path, exist_ok=True)
        optimized_reward_configs = optimize_reward_configs(args.algo,
                                                           args.env,
                                                           create_model,
                                                           create_env,
                                                           sampling_setting=sampling_setting_reward_configs,
                                                           n_timesteps=args.n_timesteps,
                                                           eval_freq=args.eval_freq,
                                                           n_eval_episodes=args.eval_episodes,
                                                           n_trials=args.n_trials,
                                                           hyperparams=hyperparams,
                                                           configs=env_kwargs,
                                                           n_jobs=args.n_jobs,
                                                           seed=args.seed,
                                                           sampler_method=args.sampler,
                                                           pruner_method=args.pruner,
                                                           verbose=args.logging_mode.value,
                                                           log_path=log_path,
                                                           best_model_save_path=log_path,
                                                           guided=args.guided)

        LOGGER.info(f"Saving optimized reward configurations to {log_path}")
        report_name = f"report_{args.algo}_{args.env}-{args.n_trials}-trials" \
                      f"-{args.n_timesteps}-steps-{args.sampler}-{args.pruner}.yml"
        with open(os.path.join(log_path, report_name), "w") as f:
            yaml.dump(optimized_reward_configs, f)

    if args.optimize_observation_configs:
        LOGGER.debug("Optimizing observation configurations")

        log_path = os.path.join(save_path, "observation_configuration_optimization")
        os.makedirs(log_path, exist_ok=True)
        optimized_observation = optimize_observation_configs(args.algo,
                                                             args.env,
                                                             create_model,
                                                             create_env,
                                                             sampling_setting=sampling_setting_observation_configs,
                                                             n_timesteps=args.n_timesteps,
                                                             eval_freq=args.eval_freq,
                                                             n_eval_episodes=args.eval_episodes,
                                                             n_trials=args.n_trials,
                                                             hyperparams=hyperparams,
                                                             configs=env_kwargs,
                                                             n_jobs=args.n_jobs,
                                                             seed=args.seed,
                                                             sampler_method=args.sampler,
                                                             pruner_method=args.pruner,
                                                             verbose=args.logging_mode.value,
                                                             log_path=log_path,
                                                             best_model_save_path=log_path)

        LOGGER.info(f"Saving optimized observation configurations to {log_path}")
        report_name = f"report_{args.algo}_{args.env}-{args.n_trials}-trials" \
                      f"-{args.n_timesteps}-steps-{args.sampler}-{args.pruner}.yml"
        with open(os.path.join(log_path, report_name), "w") as f:
            yaml.dump(optimized_observation, f)


def train_from_scratch(hyperparams, args, env, tensorboard_log, callbacks, save_path, env_id, normalize, rank):
    hyperparams = del_key_from_hyperparams(hyperparams, "n_envs")
    ### add gail model
    if args.algo == 'gail':
        expert_path = os.path.join(args.save_path, "../expert/R_G3_T3_36.npz")
        dataset = ExpertDataset(expert_path, verbose=1)
        tb_log_name = hyperparams['tb_log_name']
        hyperparams = del_key_from_hyperparams(hyperparams, "mode")
        hyperparams = del_key_from_hyperparams(hyperparams, "rule")
        hyperparams = del_key_from_hyperparams(hyperparams, "tb_log_name")
        model = ALGOS[args.algo](env=env, expert_dataset=dataset, tensorboard_log=tensorboard_log,
                             seed=args.seed, verbose=1, **hyperparams)
    else:
        model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log,
                             seed=args.seed, verbose=args.logging_mode.value, **hyperparams)

    # Arguments to the learn function
    kwargs = {}
    if args.log_interval > -1:
        kwargs = {"log_interval": args.log_interval}
    if len(callbacks) > 0:
        kwargs["callback"] = callbacks
    ### add gail model 
    if args.algo == 'gail':
        kwargs['tb_log_name'] = tb_log_name

    try:
        model.learn(args.n_timesteps, **kwargs)
    except KeyboardInterrupt:
        pass

    # Only save worker of rank 0 when using mpi
    if rank == 0:
        LOGGER.info(f"Saving model to {save_path}")
        model.save(os.path.join(save_path, str(env_id)))
        if normalize:
            # Important: save the running average, for testing the agent we need that normalization
            LOGGER.info(f"Saving vectorized and normalized environment wrapper to {save_path}")
            model.get_vec_normalize_env().save(os.path.join(save_path, "vecnormalize.pkl"))


def del_key_from_hyperparams(hyperparams: dict, key: str):
    if key in hyperparams.keys():
        del hyperparams[key]

    return hyperparams


def run_stable_baselines(args):
    """
    Run training with stable baselines
    For help, see README.md
    """
    t1 = time.time()
    args.info_keywords = tuple(args.info_keywords)

    # Check support for algo
    if ALGOS[args.algo] is None:
        raise ValueError(f"{args.algo} requires MPI to be installed")

    # TODO: check subfolder existence for commonroad env

    # Set save paths if not specified already
    save_path = args.save_path if args.save_path is not None else construct_save_path(args)
    os.makedirs(save_path, exist_ok=True)
    construct_logger(args.logging_mode.value, save_path, LOGGER)

    # Configure Stable Baselines logger
    # For details of formats, see https://github.com/hill-a/stable-baselines/blob/
    # b3f414f4f2900403107357a2206f80868af16da3/stable_baselines/logger.py#L572
    logger.configure(folder=save_path)

    if args.env == "cr-monitor-v0":
        import gym_monitor
        import crmonitor

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2 ** 32 - 1)

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        valid_extension = args.trained_agent.endswith(".pkl") or args.trained_agent.endswith(".zip")
        assert valid_extension and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .zip/.pkl file"

    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        LOGGER.info(f"Using MPI for multiprocessing with {MPI.COMM_WORLD.Get_size()} workers")
        rank = MPI.COMM_WORLD.Get_rank()
        LOGGER.info(f"Worker rank: {rank}")

        args.seed += rank
        if rank != 0:
            # Do not log anything for non-master nodes
            LOGGER.setLevel(logging.WARNING)
            args.tensorboard_log = ""

    tensorboard_log = None if args.tensorboard_log == "" else os.path.join(args.tensorboard_log, env_id)

    is_atari = False
    if "NoFrameskip" in env_id:
        is_atari = True

    LOGGER.info(f"Environment id: {env_id}")
    LOGGER.info(f"Seed: {args.seed}")
    LOGGER.info(f"Using {args.n_envs} environments")
    LOGGER.info(f"Learning with {args.n_timesteps} timesteps")

    env_kwargs = {}
    if "commonroad" in env_id or env_id == "cr-monitor-v0":
        # Get environment keyword arguments including observation and reward configurations
        with open(PATH_PARAMS["configs"][env_id], "r") as config_file:
            configs = yaml.safe_load(config_file)
            env_configs = configs["env_configs"]
            sampling_setting_reward_configs = configs["sampling_setting_reward_configs"]
            sampling_setting_observation_configs = configs["sampling_setting_observation_configs"]
            env_kwargs.update(env_configs)

    # Overwrite environment configurations if needed, first from file then from command arguments -> TODO: remove???
    if os.path.isfile(args.configs_path):
        with open(args.configs_path, "r") as configs_file:
            env_kwargs.update(yaml.safe_load(configs_file))

    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    # Save environment configurations for later inspection
    LOGGER.info(f"Saving environment configurations into {save_path}")
    with open(os.path.join(save_path, "environment_configurations.yml"), "w") as output_file:
        yaml.dump(env_kwargs, output_file)
    LOGGER.debug(f"Environment configurations:")
    LOGGER.debug(pformat(env_kwargs))

    # Load model hyperparameters from yaml file
    with open(os.path.join(ROOT_STR, f"commonroad_rl/hyperparams/{args.algo}.yml"), "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(f"Model hyperparameters not found for {args.algo}-{env_id}")

    # Overwrite model hyperparameters if needed, first from file then from command arguments
    if os.path.isfile(args.hyperparams_path):
        with open(args.hyperparams_path, "r") as hyperparams_file:
            tmp = yaml.safe_load(hyperparams_file)
            hyperparams.update(tmp)
    if args.hyperparams is not None:
        hyperparams.update(args.hyperparams)

    # Save model hyperparameters for later inspection
    LOGGER.info(f"Saving model hyperparameters into {save_path}")
    with open(os.path.join(save_path, "model_hyperparameters.yml"), "w") as f:
        yaml.dump(hyperparams, f)
    LOGGER.debug(f"Model hyperparameters loaded and set for {args.algo}-{env_id}")
    LOGGER.debug(pformat(hyperparams))

    # HER is only a wrapper around an algo
    algo = args.algo
    if algo == "her":
        algo = hyperparams["model_class"]
        assert algo in {"sac", "ddpg", "dqn", "td3"}, f"{algo} is not compatible with HER"
        # Retrieve the model class
        hyperparams["model_class"] = ALGOS[hyperparams["model_class"]]
        if hyperparams["model_class"] is None:
            raise ValueError(f"{algo} requires MPI to be installed")

    # Create learning rate schedules for ppo2 and sac
    if algo in ["ppo2", "sac", "td3"]:
        hyperparams = create_parameter_schedule(hyperparams)

    # Check if normalize
    normalize = False
    normalize_kwargs = {}
    if "normalize" in hyperparams.keys():
        normalize = hyperparams["normalize"]
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams["normalize"]

    # Convert to python object if needed
    if "policy_kwargs" in hyperparams.keys() and isinstance(hyperparams["policy_kwargs"], str):
        hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])

    # Delete keys so the dict can be pass to the model constructor
    hyperparams = del_key_from_hyperparams(hyperparams, "n_timesteps")

    # Obtain a class object from a wrapper name string in hyperparams and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    hyperparams = del_key_from_hyperparams(hyperparams, "env_wrapper")

    callbacks = []
    if args.save_freq > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq // args.n_envs, 1)
        callbacks.append(CheckpointCallback(save_freq=args.save_freq, save_path=save_path,
                                            name_prefix="rl_model", verbose=args.logging_mode.value))
        callbacks.append(SaveVecNormalizeCallback(
            save_freq=args.save_freq,
            save_path=save_path,
            name_prefix="vecnormalize",
            verbose=args.logging_mode.value,
        ))

    def create_env(n_envs, eval_env=False, **kwargs):
        """
        Create the environment and wrap it if necessary

        :param n_envs: (int)
        :param eval_env: (bool) Whether is it an environment used for evaluation or not
        :return: (Union[gym.Env, VecEnv])
        :return: (gym.Env)
        """

        # Set verbosity
        env_kwargs["logging_mode"] = args.logging_mode.value

        # Update environment keyword arguments from optimization sampler if specified
        env_kwargs.update(kwargs)

        # Differentiate keyword arguments for test environments
        env_kwargs_test = copy.deepcopy(env_kwargs)
        if "commonroad" in args.env or args.env == "cr-monitor-v0":
            env_kwargs_test["test_env"] = True

        # Do not log eval env (issue with writing the same file)
        # log_dir = None if eval_env else save_path
        log_dir = os.path.join(save_path, "test") if eval_env else save_path

        if is_atari:
            LOGGER.debug("Using Atari wrapper")
            new_env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
            # Frame-stacking with 4 frames
            new_env = VecFrameStack(new_env, n_stack=4)
        elif algo in ["dqn", "ddpg"]:
            if hyperparams.get("normalize", False):
                LOGGER.warning("WARNING: normalization not supported yet for DDPG/DQN")

            new_env = DummyVecEnv([make_env(env_id,
                                            rank,
                                            args.seed,
                                            wrapper_class=env_wrapper,
                                            log_dir=log_dir,
                                            logging_path=save_path,
                                            env_kwargs=env_kwargs_test if eval_env else env_kwargs,
                                            info_keywords=args.info_keywords)])
            new_env.seed(args.seed)
            if env_wrapper is not None:
                new_env = env_wrapper(new_env)
        elif algo != 'gail':
            if n_envs == 1:
                new_env = DummyVecEnv([make_env(env_id,
                                                0,
                                                args.seed,
                                                wrapper_class=env_wrapper,
                                                log_dir=log_dir,
                                                logging_path=save_path,
                                                env_kwargs=env_kwargs_test if eval_env else env_kwargs,
                                                info_keywords=args.info_keywords)])
            else:
                # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
                # On most env, SubprocVecEnv does not help and is quite memory hungry
                new_env = SubprocVecEnv([make_env(env_id,
                                                  i,
                                                  args.seed + n_envs if eval_env else args.seed,
                                                  log_dir=log_dir,
                                                  logging_path=save_path,
                                                  wrapper_class=env_wrapper,
                                                  env_kwargs=env_kwargs_test if eval_env else env_kwargs,
                                                  subproc=True,
                                                  info_keywords=args.info_keywords) for i in range(n_envs)],
                                        start_method="spawn")
            if normalize:
                if len(normalize_kwargs) > 0:
                    LOGGER.debug(f"Normalization activated: {normalize_kwargs}")
                else:
                    LOGGER.debug("Normalizing observations and rewards")
                new_env = VecNormalize(new_env, **normalize_kwargs)
        # Optional Frame-stacking
        if hyperparams.get("frame_stack", False):
            n_stack = hyperparams["frame_stack"]
            new_env = VecFrameStack(new_env, n_stack)
            LOGGER.info(f"Stacking {n_stack} frames")
            del hyperparams["frame_stack"]
            output_file.close()
        if args.algo == "her":
            # Wrap the env if need to flatten the dict obs
            if isinstance(new_env, VecEnv):
                new_env = _UnvecWrapper(new_env)
            new_env = HERGoalEnvWrapper(new_env)
        ### add gail model
        if args.algo == "gail":
            env1 = gym.make("commonroad-v1",
                            **env_configs)
            print('env1.observation_space', env1.observation_space)        
                     
            env_kwargs["mode"] = hyperparams["mode"]
            env_kwargs["rule"] = hyperparams["rule"]
            env_kwargs["env"] = env1
            new_env = DummyVecEnv([make_env(env_id,
                                        0,
                                        args.seed,
                                        log_dir=log_dir,
                                        logging_path=save_path,
                                        env_kwargs=env_kwargs_test if eval_env else env_kwargs,
                                        info_keywords=args.info_keywords)])
        return new_env

    # HINT: Three main options to be chosen from
    # HINT: 1. Continue training with a pretrained agent
    # HINT: 2. Optimize model hyperparameters and/or configurations of observations and rewards
    # HINT: 3. Start training from scratch
    if args.optimize_hyperparams or args.optimize_observation_configs or args.optimize_reward_configs:
        optimize_parameters(hyperparams, args, save_path, create_env, sampling_setting_reward_configs,
                            env_kwargs, tensorboard_log, sampling_setting_observation_configs)
    else:
        # Create training environments
        env = create_env(args.n_envs)

        # Create testing environments if needed, do not normalize reward
        if args.eval_freq > 0:
            # Account for the number of parallel environments
            eval_freq = max(args.eval_freq // args.n_envs, 1)

            # Do not normalize the rewards of the eval env
            old_kwargs = None
            if normalize:
                if len(normalize_kwargs) > 0:
                    old_kwargs = normalize_kwargs.copy()
                    normalize_kwargs["norm_reward"] = False
                else:
                    normalize_kwargs = {"norm_reward": False}

            save_vec_normalize = SaveVecNormalizeCallback(
                save_freq=1,
                save_path=save_path,
                name_prefix="vecnormalize",
                verbose=args.logging_mode.value,
            )

            eval_callback = MultiEnvsEvalCallback(
                create_env(args.n_envs, eval_env=True),
                log_path=save_path,
                best_model_save_path=save_path,
                eval_freq=eval_freq,
                n_eval_timesteps=args.eval_timesteps,
                callback_on_new_best=save_vec_normalize,
                verbose=args.logging_mode.value,
            )
            callbacks.append(eval_callback)

            # Restore original kwargs
            if old_kwargs is not None:
                normalize_kwargs = old_kwargs.copy()

        LOGGER.info(f"Elapsed time for preparing steps: {time.time() - t1} s")

        # Parse noise string for DDPG and SAC and TD3
        if algo in ["ddpg", "sac", "td3"] and hyperparams.get("noise_type") is not None:
            hyperparams = parse_noise(hyperparams, env.action_space.shape[0], algo)

        # Load a trained model and continue training
        if os.path.isfile(args.trained_agent):
            continue_learning(hyperparams, args, env, tensorboard_log, callbacks, normalize, save_path, env_id, rank)
        else:
            # Train an agent from scratch
            train_from_scratch(hyperparams, args, env, tensorboard_log, callbacks, save_path, env_id, normalize, rank)

    LOGGER.info(f"Elapsed time: {time.time() - t1} s")


if __name__ == "__main__":
    args = run_stable_baselines_argsparser().parse_args(sys.argv[1:])
    run_stable_baselines(args)
