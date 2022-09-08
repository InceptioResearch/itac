import argparse
import glob
import importlib
import os
import re
import logging
LOGGER = logging.getLogger(__name__)

import gym
import yaml
import numpy as np
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
try:
    import mpi4py
except ImportError:
    mpi4py = None

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.policies import FeedForwardPolicy as BasePolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.bench import Monitor
from stable_baselines import logger
from stable_baselines import PPO2, A2C, ACER, ACKTR, HER, SAC, TD3, GAIL

# DDPG and TRPO require MPI to be installed
if mpi4py is None:
    DDPG, TRPO = None, None
else:
    from stable_baselines import DDPG, TRPO

from stable_baselines.common.vec_env import (
    DummyVecEnv,
    VecNormalize as SBVecNormalize,
    VecFrameStack,
    SubprocVecEnv,
)
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds, BaseRLModel

from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
ALGOS = {
    "a2c": A2C,
    "acer": ACER,
    "acktr": ACKTR,
    "ddpg": DDPG,
    "her": HER,
    "sac": SAC,
    "ppo2": PPO2,
    "trpo": TRPO,
    "td3": TD3,
    "gail": GAIL,
}
# import DQN
try:
    from stable_baselines import DQN
except:
    print("Can't import DQN, skipped!")
else:
    ALGOS["dqn"] = DQN


# ================== Custom Policies =================


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs, layers=[64], layer_norm=True, feature_extraction="mlp")


class CustomMlpPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs, layers=[16], feature_extraction="mlp")


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs, layers=[256, 256], feature_extraction="mlp")


register_policy("CustomSACPolicy", CustomSACPolicy)
register_policy("CustomDQNPolicy", CustomDQNPolicy)
register_policy("CustomMlpPolicy", CustomMlpPolicy)


class CRVecNormalize(SBVecNormalize):
    def __init__(self, vec_normalize: SBVecNormalize):
        self.vec_normalize = vec_normalize
        # SBVecNormalize.__init__(self, SBVecNormalize)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.vec_normalize, name)

    def reset(self, **kwargs):
        """
        Reset all environments
        """
        obs = self.venv.reset(**kwargs)
        self.old_obs = obs
        self.ret = np.zeros(self.num_envs)
        if self.training:
            self._update_reward(self.ret)
        return self.normalize_obs(obs)


def load_model_and_vecnormalize(model_path: str, algo: str, normalize: bool, env: gym.Env) -> BaseRLModel:
    """
    Load trained model and corresponding vecnormalize.pkl
    :param model_path: Path to folder containing the trained model
    :param algo: The used RL algorithm
    :param normalize: If the env was normalized during training
    :param env: The gym.Env used during training
    :return: best_model.zip if exists else the last model and corresponding VecNormalize wrapped Env
    """
    # Load the trained agent
    files = os.listdir(model_path)
    if "best_model.zip" in files:
        model_path = os.path.join(model_path, "best_model.zip")
        if normalize:
            vec_normalize_path = model_path.replace("best_model.zip", "vecnormalize.pkl")
    else:
        # No best_model.zip, find last model
        files = sorted(glob.glob(os.path.join(model_path, "rl_model*.zip")))

        def extract_number(f):
            s = re.findall("\d+", f)
            return int(s[-1]) if s else -1, f

        model_path = max(files, key=extract_number)
        vec_normalize_path = model_path.replace("rl_model", "vecnormalize").replace(".zip", ".pkl")

    if os.path.exists(vec_normalize_path):
        LOGGER.info(f"Loading saved running average from {vec_normalize_path}")
        env = CRVecNormalize(SBVecNormalize.load(vec_normalize_path, env))
    else:
        raise FileNotFoundError(f"vecnormalize.pkl not found in {vec_normalize_path}")
        
    # During testing the vecnormalize should not update the moving average
    env.training = False
    LOGGER.info(f"Loading model from {model_path}")
    model = ALGOS[algo].load(model_path)

    return model, env


def flatten_dict_observations(env):
    assert isinstance(env.observation_space, gym.spaces.Dict)
    keys = env.observation_space.spaces.keys()
    return gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))


def get_wrapper_class(hyperparams):
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - utils_run.wrappers.DoneOnSuccessWrapper:
            reward_offset: 1.0
        - utils_run.wrappers.TimeFeatureWrapper
        - utils_run.wrappers.IncreaseTimeStepWrapper:
            num_steps: 10


    :param hyperparams: (dict)
    :return: a subclass of gym.Wrapper (class object) you can use to
             create another Gym env giving an original env.
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams.keys():
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env):
            """
            :param env: (gym.Env)
            :return: (gym.Env)
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def make_env(
        env_id,
        rank=0,
        seed=0,
        log_dir=None,
        logging_path=None,
        wrapper_class=None,
        env_kwargs=None,
        subproc=False,
        info_keywords=(),
):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str) path to monitor information
    :param logging_path: (str) path to console_copy.txt
    :param wrapper_class: (type) a subclass of gym.Wrapper to wrap the original env with
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    :param subproc: (bool) Whether env is used in a SubprocVecEnv, used to specify scenario path
    :param info_keywords: (tuple) extra information to log, from the information return of environment.step
    """
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    if env_kwargs is None:
        env_kwargs = {}

    def _init():
        if env_id == "cr-monitor-v0":
            pass
        if "commonroad" in env_id:
            pass

        set_global_seeds(seed + rank)
        if subproc and ("commonroad" in env_id or env_id == "cr-monitor-v0"):
            train_reset_config_path = env_kwargs.pop("train_reset_config_path", PATH_PARAMS["train_reset_config"])
            test_reset_config_path = env_kwargs.pop("test_reset_config_path", PATH_PARAMS["test_reset_config"])
            env = gym.make(env_id,
                           train_reset_config_path=os.path.join(train_reset_config_path, str(rank)),
                           test_reset_config_path=os.path.join(test_reset_config_path, str(rank)),
                           logging_path=logging_path,
                           **env_kwargs)
        else:
            env = gym.make(env_id, logging_path=logging_path, **env_kwargs)

        # We allow a Gym env wrapper (a subclass of gym.Wrapper)
        if wrapper_class:
            env = wrapper_class(env)

        env.seed(seed + rank)
        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        env = Monitor(env, log_file, info_keywords=info_keywords)

        return env

    return _init


def create_test_env(
        env_id,
        n_envs=1,
        is_atari=False,
        stats_path=None,
        seed=0,
        log_dir="",
        should_render=True,
        hyperparams=None,
        env_kwargs=None,
):
    """
    Create environment for testing a trained agent

    :param env_id: (str)
    :param n_envs: (int) number of processes
    :param is_atari: (bool)
    :param stats_path: (str) path to folder containing saved running averaged
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param should_render: (bool) For Pybullet env, display the GUI
    :param env_wrapper: (type) A subclass of gym.Wrapper to wrap the original
                        env with
    :param hyperparams: (dict) Additional hyperparams (ex: n_stack)
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    :return: (gym.Env)
    """
    # HACK to save logs
    if log_dir is not None:
        os.environ["OPENAI_LOG_FORMAT"] = "csv"
        os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        logger.configure()

    if hyperparams is None:
        hyperparams = {}

    if env_kwargs is None:
        env_kwargs = {}

    # Create the environment and wrap it if necessary
    env_wrapper = get_wrapper_class(hyperparams)
    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif n_envs > 1:
        # start_method = 'spawn' for thread safe
        env = SubprocVecEnv(
            [
                make_env(
                    env_id,
                    i,
                    seed,
                    log_dir,
                    wrapper_class=env_wrapper,
                    env_kwargs=env_kwargs,
                )
                for i in range(n_envs)
            ]
        )
    # Pybullet envs does not follow gym.render() interface
    elif "Bullet" in env_id:
        # HACK: force SubprocVecEnv for Bullet env
        env = SubprocVecEnv(
            [
                make_env(
                    env_id,
                    0,
                    seed,
                    log_dir,
                    wrapper_class=env_wrapper,
                    env_kwargs=env_kwargs,
                )
            ]
        )
    else:
        env = DummyVecEnv(
            [
                make_env(
                    env_id,
                    0,
                    seed,
                    log_dir,
                    wrapper_class=env_wrapper,
                    env_kwargs=env_kwargs,
                )
            ]
        )

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print("with params: {}".format(hyperparams["normalize_kwargs"]))
            env = VecNormalize(env, training=False, **hyperparams["normalize_kwargs"])

            if os.path.exists(os.path.join(stats_path, "vecnormalize.pkl")):
                env = VecNormalize.load(
                    os.path.join(stats_path, "vecnormalize.pkl"), env
                )
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                # Legacy:
                env.load_running_average(stats_path)

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print("Stacking {} frames".format(n_stack))
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0

        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def get_trained_models(log_folder):
    """

    :param log_folder: (str) Root log folder
    :return: (dict) Dict representing the trained agent
    """
    algos = os.listdir(log_folder)
    trained_models = {}
    for algo in algos:
        for ext in ["zip", "pkl"]:
            for env_id in glob.glob("{}/{}/*.{}".format(log_folder, algo, ext)):
                # Retrieve env name
                env_id = env_id.split("/")[-1].split(".{}".format(ext))[0]
                trained_models["{}-{}".format(algo, env_id)] = (algo, env_id)
    return trained_models


def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if (
                env_id == "_".join(file_name.split("_")[:-1])
                and ext.isdigit()
                and int(ext) > max_run_id
        ):
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False):
    """

    :param stats_path: (str)
    :param norm_reward: (bool)
    :param test_mode: (bool)
    :return: (dict, str)
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml"), "r") as f:
                hyperparams = yaml.load(
                    f, Loader=yaml.UnsafeLoader
                )  # pytype: disable=module-attr
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {
                    "norm_obs": hyperparams["normalize"],
                    "norm_reward": norm_reward,
                }
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path


def find_saved_model(algo, log_path, env_id, load_best=False):
    """

    :param algo: (str)
    :param log_path: (str) Path to the directory with the saved model
    :param env_id: (str)
    :param load_best: (bool)
    :return: (str) Path to the saved model
    """
    model_path, found = None, False
    for ext in ["pkl", "zip"]:
        model_path = "{}/{}.{}".format(log_path, env_id, ext)
        found = os.path.isfile(model_path)
        if found:
            break

    if load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(
            "No model found for {} on {}, path: {}".format(algo, env_id, model_path)
        )
    return model_path


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
            setattr(namespace, self.dest, arg_dict)

