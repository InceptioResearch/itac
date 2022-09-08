"""
A utility function to be called when optimizing reward configurations
"""
import logging
import os
from collections import defaultdict
from pprint import pformat

import numpy as np
import optuna
import yaml
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.her import HERGoalEnvWrapper

from commonroad_rl.plot_learning_curves import ts2collision, ts2reward, ts2goal, ts2off_road, ts2max_time
from commonroad_rl.utils_run.callbacks import RewardConfigsTrialEvalCallback
from commonroad_rl.utils_run.plot_util import default_xy_fn
from commonroad_rl.utils_run.plot_util import load_results as load_results_baselines, default_split_fn

__author__ = "Brian Liao, Johannes Kaiser"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

# Methods that retrieve the individual parameters of monitor.csv
PLOT_DICT = {
    # "Mean Reward": ts2ep_rew_mean,
    "Total Reward": ts2reward, # ts2xy,
    # "Total Robustness reward": ts2monitor_reward,
    # "Total Sparse reward": ts2gym_reward,
    # "Step Robustness reward": ts2monitor_reward_step,
    # "Step Sparse reward": ts2gym_reward_step,
    "Goal-Reaching Rate": ts2goal,
    "Collision Rate": ts2collision,
    "Off-Road Rate": ts2off_road,
    "Max Time Reached": ts2max_time,
}

""" 
Dict used for the guided optimization
Structure: 
Failure rate
    | reward type
    |   | [list of considered rewards for the failure and reward type]
    ...
"""
GUIDED = {
    "Off-Road Rate":{
        "reward_configs_dense":[],
        "reward_configs_hybrid":["reward_off_road"],
        "reward_configs_sparse":[]
    },
    "Max Time Reached":{
        "reward_configs_dense":[],
        "reward_configs_hybrid":["reward_time_out"],
        "reward_configs_sparse":[]
    },
    "Collision Rate":{
        "reward_configs_dense":[],
        "reward_configs_hybrid":["reward_collision"],
        "reward_configs_sparse":[]
    },
    "Goal-Reaching Rate":{
        "reward_configs_dense":[],
        "reward_configs_hybrid":["reward_goal_reached"],
        "reward_configs_sparse":[]
    }
}

"""
Dict of all considered reward types for the guided optimization with 
respective [direction of improvement, rate aimed for]
ex. reward_off_road improves the goal reching rate when it is decreased
    the goal off_road_rate is 0 when considering reward_off_road 
"""

# first entry of array is direction of improvement of related rewards with scaling
# second entry of array is goal value
SIGN_OF_IMPROVEMENT = {
    "reward_off_road" : [-10, 0],
    "reward_time_out" : [-10, 0],
    "reward_collision" : [-10, 0],
    "reward_goal_reached" : [10, 1],
}


# Value the rewards are normalized to
SUM_OF_REWARD_VALUES = 300


def parse_sampler(sampler_method: str, n_startup_trials: int, seed: int):
    """
    This function loads an enviromenet based on the env_configs

    :param sampler_method: string of [random, tpe, skopt]
    :param n_startup_trials: int 
    :param seed: int seeding the sampling for reproducing results
    """
    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if sampler_method == "random":
        return RandomSampler(seed=seed)
    elif sampler_method == "tpe":
        return TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == "skopt":
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        return SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
    else:
        raise ValueError(f"[reward_configs_opt.py] Unknown sampler: {sampler_method}")


def parse_pruner(pruner_method: str, n_startup_trials: int, n_evaluations: int, n_trials: int):
    """
    This function loads an enviromenet based on the env_configs

    :param pruner_method: string of [halving, median, none]
    :param n_startup_trials: int 
    :param n_evaluations: int number of evaluations of the model after each eval_freq
    :param  n_trials: int number of trials to be executed
    """
    if pruner_method == "halving":
        return SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner_method == "median":
        return MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    elif pruner_method == "none":
        # Do not prune
        return MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    else:
        raise ValueError(f"[reward_configs_opt.py] Unknown pruner: {pruner_method}")


def average(lst):
    return sum(lst)/len(lst)


def smooth_results(allresults, xy_fn=default_xy_fn, split_fn=default_split_fn, group_fn=default_split_fn):
    """
    Snoothes the result over the last few evaluations

    :param allresults: (float)
    :param xy_fn: function Result -> x,y           - function that converts results objects into tuple of x and y values.
                                              By default, x is cumsum of episode lengths, and y is episode rewards

    :param split_fn: function Result -> hashable   - function that converts results objects into keys to split curves into
                                              sub-panels by.
                                              That is, the results r for which split_fn(r) is different will be put
                                              on different sub-panels.
                                              By default, the portion of r.dirname between last / and -<digits> is
                                              returned. The sub-panels are
                                              stacked vertically in the figure.

    :param group_fn: function Result -> hashable   - function that converts results objects into keys to group curves by.
                                              That is, the results r for which group_fn(r) is the same will be put
                                              into the same group.
                                              Curves in the same group have the same color (if average_group is
                                              False), or averaged over
                                              (if average_group is True). The default value is the same as default
                                              value for split_fn
    :return: (list) ordered list of averaged results 
    """
    
    result_list = []

    if split_fn is None: split_fn = lambda _: ''
    if group_fn is None: group_fn = lambda _: ''
    sk2r = defaultdict(list)  # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    assert len(sk2r) > 0

    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2c = defaultdict(int)
        sresults = sk2r[sk]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = xy_fn(result)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))

            if len(x) >= 10:
                avg_y = average(y[-int(len(y)/3):])
            elif len(x) != 0:
                avg_y = y[-1]
            else:
                avg_y = 0

            result_list.append(avg_y)
    return average(result_list)


def group_fn(results):
    return os.path.basename(results.dirname)


def optimize_reward_configs(
        algo,
        env_id,
        model_fn,
        env_fn,
        sampling_setting,
        n_timesteps=5000,
        eval_freq=500,
        n_eval_episodes=5,
        n_trials=10,
        hyperparams=None,
        configs=None,
        n_jobs=1,
        sampler_method="random",
        pruner_method="halving",
        seed=13,
        verbose=1,
        log_path=None,
        best_model_save_path=None,
        guided=False,
):
    """

    :param algo: (str)
    :param env_id: (str)
    :param model_fn: (func) function that is used to instantiate the model
    :param env_fn: (func) function that is used to instantiate the env
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param hyperparams: (dict) model hyperparameters
    :param configs: (dict) environment configurations to be optimized
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param seed: (int)
    :param sampling_setting: (dict) sampling intervals for correspinding items
    :param verbose: (int)
    :param log_path: (str) folder for saving evaluation results during optimization
    :param best_model_save_path (str) folder for saving the best model
    :param guided: (bool) if True, the optimization is performed in a guided manner
    :return: (dict) detailed result of the optimization
    """
    # TODO: remove; duplicate in RewardConfigsTrialEvalCallback
    # LOGGER.setLevel(verbose)
    #
    # if not len(LOGGER.handlers):
    #     formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
    #     stream_handler = logging.StreamHandler()
    #     stream_handler.setLevel(verbose)
    #     stream_handler.setFormatter(formatter)
    #     LOGGER.addHandler(stream_handler)
    #
    #     if log_path is not None:
    #         file_handler = logging.FileHandler(filename=os.path.join(log_path, "console_copy.txt"))
    #         file_handler.setLevel(verbose)
    #         file_handler.setFormatter(formatter)
    #         LOGGER.addHandler(file_handler)

    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(log_path, "console_copy2.txt"))
    file_handler.setLevel(logging.INFO)
    LOGGER.addHandler(file_handler)

    LOGGER.info(f"Optimizing environment configurations for {env_id}")

    # Enviornment configurations including observation space settings and reward weights
    if configs is None:
        configs = {}

    n_startup_trials = 10
    n_evaluations = n_timesteps // eval_freq

    LOGGER.info(f"Optimizing with {n_trials} trials using {n_jobs} parallel jobs, "
                f"each with {n_timesteps} maximal time steps and {n_evaluations} evaluations")

    sampler = parse_sampler(sampler_method, n_startup_trials, seed)
    pruner = parse_pruner(pruner_method, n_startup_trials, n_evaluations, n_trials)
    LOGGER.debug(f"Sampler: {sampler_method} - Pruner: {pruner_method}")

    # Create study object on environment configurations from Optuna
    reward_configs_study = optuna.create_study(sampler=sampler, pruner=pruner)

    trial_dict = {}
    based_on_dict = {}

    def objective_reward_configs(trial):
        """
        Optimization objective for environment configurations
        """
        LOGGER.info(f"Trial dict: {trial_dict}")
        LOGGER.info(f"Based_on dict: {based_on_dict}")
        trial.model_class = None

        if algo == "her":
            trial.model_class = hyperparams["model_class"]

        # Hack to use DDPG/TD3 noise sampler
        if algo in ["ddpg", "td3"] or trial.model_class in ["ddpg", "td3"]:
            trial.n_actions = env_fn(n_envs=1).action_space.shape[0]

        # Get keyword arguments for model hyperparameters and environment configurations
        kwargs_hyperparams = hyperparams.copy()
        kwargs_configs = configs.copy()

        # Sample environment configurations and keep model hyperparameters
        sampled_reward_configs = sample_commonroad_reward_configs(trial, sampling_setting, trial_dict, based_on_dict, log_path, LOGGER, guided)
        kwargs_configs.update(sampled_reward_configs)

        # Save data for later inspection
        tmp_path = os.path.join(log_path, "trial_" + str(trial.number))
        os.makedirs(tmp_path, exist_ok=True)
        with open(os.path.join(tmp_path, "sampled_reward_configurations.yml"), "w") as f:
            yaml.dump(kwargs_configs, f)
            LOGGER.info("Saving sampled reward configurations into " + tmp_path)
        LOGGER.debug("[reward_configs_opt.py] Sampled reward configurations:")
        LOGGER.debug(pformat(kwargs_configs))

        # Create model and environments for optimization TODO: save model in tmp_path
        model = model_fn(kwargs_hyperparams, kwargs_configs)
        # TODO: why n_envs=1 for eval_env --> EvalCallback only allows one eval_env --> Use MultiEvalCallback instead??
        eval_env = env_fn(n_envs=1, eval_env=True, **kwargs_configs)

        # Account for parallel envs
        model_env = model.get_env()
        if isinstance(model_env, VecEnv):
            eval_freq_ = max(eval_freq // model_env.num_envs, 1)

        LOGGER.info(f"Evaluating with {n_eval_episodes} episodes after every {eval_freq_} time steps")

        reward_configs_eval_callback = RewardConfigsTrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq_,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=True,
            verbose=verbose,
        )

        if algo == "her":
            # Wrap the env if need to flatten the dict obs
            if isinstance(eval_env, VecEnv):
                eval_env = _UnvecWrapper(eval_env)
            eval_env = HERGoalEnvWrapper(eval_env)

        try:
            # TODO: what is model.runner?
            model.learn(n_timesteps, callback=reward_configs_eval_callback)
            # Free memory
            model.save(os.path.join(tmp_path, "trial_" + str(trial.number)))
            model.env.close()
            eval_env.close()
            
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()

        past_results = load_results_baselines(os.path.dirname(log_path))
        own_trial_dict = {}
        for i, (k, xy_fn) in enumerate(PLOT_DICT.items()):
            output = smooth_results(past_results, group_fn=group_fn, xy_fn=xy_fn)
            own_trial_dict[k] = output
        trial_dict["trial_" + str(trial.number)] = own_trial_dict

        is_pruned = reward_configs_eval_callback.is_pruned
        cost = reward_configs_eval_callback.cost
        del model.env, eval_env
        del model
        if is_pruned:
            raise optuna.exceptions.TrialPruned()
        LOGGER.info(f"Cost of {trial.number} resulted in {cost}")
        return cost

    try:
        LOGGER.info(f"Trying to optimize environment configurations with {n_trials} trials and {n_jobs} jobs")
        reward_configs_study.optimize(objective_reward_configs, n_trials=n_trials, n_jobs=n_jobs)
        LOGGER.info(f"final trial dict: {trial_dict}")
    except KeyboardInterrupt:
        pass

    LOGGER.info(f"Number of finished trials: {len(reward_configs_study.trials)}")
    LOGGER.info(f"Best value: {reward_configs_study.best_trial.value}")
    LOGGER.info("Best reward configurations: ")
    LOGGER.info(pformat(reward_configs_study.best_trial.params))

    return reward_configs_study.best_trial.params


def sample_commonroad_reward_configs(trial, sampling_setting, trial_dict, based_on_dict, log_path, logger, guided):
    # Read in reward items and intervals, and generate random values
    reward_configs = {}
    for key, values in sampling_setting.items():
        if len(values) == 1:
            method, interval = next(iter(values.items()))
            if method == "categorical":
                reward_configs[key] = trial.suggest_categorical(key, interval)
            elif method == "uniform":
                reward_configs[key] = trial.suggest_uniform(key, interval[0], interval[1])
            elif method == "loguniform":
                reward_configs[key] = trial.suggest_loguniform(key, interval[0], interval[1])
            else:
                print(f"[reward_configs_opt.py] Sampling method {method} not supported")
        elif len(values) > 1:
            # TODO: commonroad-v1
            reward_configs[key] = dict()
            for subkey, subvalue in values.items():
                method, interval = next(iter(subvalue.items()))
                if method == "categorical":
                    reward_configs[key][subkey] = trial.suggest_categorical(subkey, interval)
                elif method == "uniform":
                    reward_configs[key][subkey] = trial.suggest_uniform(subkey, interval[0], interval[1])
                elif method == "loguniform":
                    reward_configs[key][subkey] = trial.suggest_loguniform(subkey, interval[0], interval[1])
                else:
                    print(f"[reward_configs_opt.py] Sampling method {method} not supported")
    
    if guided: # guided optimization
        logger.info("guided trial")
        compare_trial = trial.number - 1
        updated = False
        while not updated:
            if compare_trial == -1:
                # First trial
                # Do as normal, as first trial or first trial incomplete
                # Trial dict based on is none
                based_on_dict["trial_" + str(trial.number)] = None
                # updated allows to stop looking for a comparisson trial
                updated = True
            elif "trial_" + str(compare_trial) in trial_dict:  # one trial before current is found

                previous_trial = "trial_" + str(compare_trial)
                # based on dict is updated
                based_on_dict["trial_" + str(trial.number)] = previous_trial
                previous_rates = trial_dict[previous_trial]
                previous_rewards_path = os.path.join(os.path.join(log_path, previous_trial), "sampled_reward_configurations.yml")
                previous_rewards = yaml.load(open(previous_rewards_path))

                if based_on_dict[previous_trial] != None:   # second trial before current is found
                    second_to_prev_trial = based_on_dict[previous_trial]
                    second_to_prev_rates = trial_dict[second_to_prev_trial]
                    second_to_prev_rewards_path = os.path.join(os.path.join(log_path, second_to_prev_trial), "sampled_reward_configurations.yml")
                    second_to_prev_rewards = yaml.load(open(second_to_prev_rewards_path))
                    for failure, value in GUIDED.items():   # for each type of failure: off_road etc.
                        for key, subkeys in value.items():  # for each reward type
                            for subkey in subkeys:          # for each reward
                                # the new reward is the sum of the past reward, a change based on the failure rate, a change based on the change of failure rate with respect to the last change in reward
                                # SIGN_OF_IMPROVEMENT[subkey][1] is used to guide the results in different directions: goal reaching rate --> 1; off road rate --> 0
                                # SIGN_OF_IMPROVEMENT[subkey][0] to specify the direction of improvement goal_reaching rate getting bigger forces the model to reach the goal, off_road reward getting smaller forces the model to stay on the road
                                change_in_reward_value = previous_rewards[key][subkey] - second_to_prev_rewards[key][subkey] + 0.000001 #change reward
                                change_in_failure = abs(SIGN_OF_IMPROVEMENT[subkey][1] - second_to_prev_rates[failure]) - abs(SIGN_OF_IMPROVEMENT[subkey][1] - previous_rates[failure]) #positive if improvement
                                # If we get close to the optimum drop failure rate based change and rely on derivative based change
                                if abs(SIGN_OF_IMPROVEMENT[subkey][1] - previous_rates[failure]) > 0.1:
                                    scaling =  abs(SIGN_OF_IMPROVEMENT[subkey][1] - previous_rates[failure])
                                else:
                                    scaling = 0
                                # allow for bigger jumps when far away from optimum, go to local minimum afterwards
                                reward_configs[key][subkey] = float(previous_rewards[key][subkey] + int(SIGN_OF_IMPROVEMENT[subkey][0]) * (- change_in_failure / np.sign(change_in_reward_value) + scaling * 3))
                                logger.info(f"reward {subkey} set to {float(previous_rewards[key][subkey] + int(SIGN_OF_IMPROVEMENT[subkey][0]) * (- change_in_failure / np.sign(change_in_reward_value) + scaling * 3))}")
                                logger.debug(f" failure {failure}")
                                logger.debug(f"rates {previous_rates[failure]}")
                                logger.debug(f"adapted rates {abs(SIGN_OF_IMPROVEMENT[subkey][1] - previous_rates[failure])}")
                                logger.debug(f"new rewards {float(previous_rewards[key][subkey] + int(SIGN_OF_IMPROVEMENT[subkey][0]) * (- change_in_failure / np.sign(change_in_reward_value) + scaling * 3))}")
                                logger.debug(f"change {float(int(SIGN_OF_IMPROVEMENT[subkey][0]) * (- change_in_failure / np.sign(change_in_reward_value) + scaling * 3))}")
                else:   # Only on comparisson trial is found (Only one finished yet)
                    for failure, value in GUIDED.items():
                        for key, subkeys in value.items():
                            for subkey in subkeys:
                                # No derivative if only one trial can be found
                                reward_configs[key][subkey] = float(previous_rewards[key][subkey] + int(SIGN_OF_IMPROVEMENT[subkey][0]) * abs(SIGN_OF_IMPROVEMENT[subkey][1] - previous_rates[failure]) * 3)
                                logger.info(f"reward {subkey} set to {float(previous_rewards[key][subkey] + int(SIGN_OF_IMPROVEMENT[subkey][0]) * abs(SIGN_OF_IMPROVEMENT[subkey][1] - previous_rates[failure]) * 3)}")
                                logger.debug(f" failure {failure}")
                                logger.debug(f"rates {previous_rates[failure]}")
                                logger.debug(f"new rewards {float(previous_rewards[key][subkey] + int(SIGN_OF_IMPROVEMENT[subkey][0]) * abs(SIGN_OF_IMPROVEMENT[subkey][1] - previous_rates[failure]) * 3)}")
                #  updated allows to stop looking for a comparisson trial
                updated = True
            else:
                compare_trial -= 1  # Try to find trial with one number less, allows for multithreading, when process 5 is started, we can still compare to results of trial 1
        logger.info(f"reward configs before norm {reward_configs}")
        # Norming
        total_reward_of_optimized = 0
        for failure, value in GUIDED.items():   # for each type of failure: off_road etc.
                for key, subkeys in value.items():  # for each reward type
                    for subkey in subkeys:
                        total_reward_of_optimized = total_reward_of_optimized + abs(reward_configs[key][subkey])
        for failure, value in GUIDED.items():   # for each type of failure: off_road etc.
                for key, subkeys in value.items():  # for each reward type
                    for subkey in subkeys:
                        reward_configs[key][subkey] = (reward_configs[key][subkey]/total_reward_of_optimized)*SUM_OF_REWARD_VALUES
    logger.info(f"reward configs {reward_configs}")
    return reward_configs


REWARD_CONFIGS_SAMPLER = {
    "commonroad-v0": sample_commonroad_reward_configs,
    "commonroad-v1": sample_commonroad_reward_configs
}
