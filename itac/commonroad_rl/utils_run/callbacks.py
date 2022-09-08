import logging
import os
import warnings
from typing import Union, Optional, Callable, Tuple, List

import gym
import numpy as np
from stable_baselines.common.callbacks import BaseCallback, EvalCallback, EventCallback
from stable_baselines.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    sync_envs_normalization,
)

LOGGER = logging.getLogger(__name__)
# from commonroad_rl.train_model import construct_logger
# dirty fix, TODO: investivate why can not import from train_model.py
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


def construct_path_with_trial_number(path: str, trial_numer):
    if path is not None:
        if trial_numer == "best_model":
            return os.path.join(path, trial_numer)
        else:
            return os.path.join(path, "trial_" + str(trial_numer))
    else:
        return path


class HyperparamsTrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial during model hyperparameter optimization.
    """

    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, log_path=None,
                 best_model_save_path=None, deterministic=True, verbose=1):

        best_model_save_path = construct_path_with_trial_number(best_model_save_path, trial.number)
        log_path = construct_path_with_trial_number(log_path, trial.number)

        super(HyperparamsTrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.cost = 0.0

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(HyperparamsTrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report last?
            # report num_timesteps or elasped time?
            self.cost = -1 * self.best_mean_reward
            self.trial.report(self.cost, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class ObservationConfigsTrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial during observation configuration optimization.
    """

    def __init__(
            self,
            eval_env,
            trial,
            n_eval_episodes=5,
            eval_freq=10000,
            log_path=None,
            best_model_save_path=None,
            deterministic=True,
            verbose=1,
    ):
        best_model_save_path = construct_path_with_trial_number(best_model_save_path, trial.number)
        log_path = construct_path_with_trial_number(log_path, trial.number)

        super(ObservationConfigsTrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.cost = 0.0

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(ObservationConfigsTrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            self.cost = -1 * self.best_mean_reward
            self.trial.report(self.cost, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class RewardConfigsTrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial during reward configaration optimization.
    """

    def __init__(
            self,
            eval_env,
            trial,
            n_eval_episodes=5,
            eval_freq=1000,
            log_path=None,
            best_model_save_path=None,
            deterministic=True,
            verbose=1,
    ):
        # Set save path and logger
        map_verbose_to_logging = {2: logging.DEBUG, 1: logging.INFO}
        self.log_path = construct_path_with_trial_number(log_path, trial.number)
        # Save best model into `($best_model_save_path)/trial_($trial_number)/best_model.zip`
        self.best_model_save_path = construct_path_with_trial_number(best_model_save_path, trial.number)
        construct_logger(map_verbose_to_logging.get(verbose, logging.ERROR), self.log_path, LOGGER)

        super(RewardConfigsTrialEvalCallback, self).__init__(eval_env=eval_env,
                                                             n_eval_episodes=n_eval_episodes,
                                                             eval_freq=eval_freq,
                                                             deterministic=deterministic,
                                                             verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.lowest_mean_cost = np.inf
        self.last_mean_cost = np.inf
        self.cost = 0.0

        # Log evaluation information into `($log_path)/trial_($trial_number)/evaluations.npz`
        self.evaluation_timesteps = []
        self.evaluation_costs = []
        self.evaluation_lengths = []

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            def evaluate_policy_configs(model, env, n_eval_episode=10, render=False, deterministic=True, callback=None):
                """
                Runs policy for `n_eval_episodes` episodes and returns cost for optimization.
                This is made to work only with one env.

                :param model: (BaseRLModel) The RL agent you want to evaluate.
                :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`, this must contain only one environment.
                :param n_eval_episode: (int) Number of episode to evaluate the agent
                :param deterministic: (bool) Whether to use deterministic or stochastic actions
                :param render: (bool) Whether to render the environment or not
                :param callback: (callable) callback function to do additional checks, called after each step.
                :return: ([float], [int]) list of episode costs and lengths
                """
                if isinstance(env, VecEnv):
                    assert env.num_envs == 1, "You must pass only one environment when using this function"

                episode_costs = []
                episode_lengths = []
                for _ in range(n_eval_episode):
                    obs = env.reset()
                    done, info, state = False, None, None

                    # Record required information
                    # Since vectorized environments get reset automatically after each episode,
                    # we have to keep a copy of the relevant states here.
                    # See https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html for more details.
                    episode_length, episode_cost = 0, 0.
                    episode_is_time_out = []
                    episode_is_collision = []
                    episode_is_off_road = []
                    episode_is_goal_reached = []
                    episode_is_friction_violation = []
                    while not done:
                        action, state = model.predict(obs, state=state, deterministic=deterministic)
                        obs, reward, done, info = env.step(action)

                        episode_length += 1
                        episode_is_time_out.append(info[-1]["is_time_out"])
                        episode_is_collision.append(info[-1]["is_collision"])
                        episode_is_off_road.append(info[-1]["is_off_road"])
                        episode_is_goal_reached.append(info[-1]["is_goal_reached"])
                        episode_is_friction_violation.append(info[-1]["is_friction_violation"])

                        if callback is not None:
                            callback(locals(), globals())
                        if render:
                            env.render()

                    # Calculate cost for optimization from state information
                    normalized_episode_length = episode_length / info[-1]["max_episode_time_steps"]
                    if episode_is_time_out[-1]:
                        episode_cost += 10.0  # * (1 / normalized_episode_length)
                    if episode_is_collision[-1]:
                        episode_cost += 10.0  # * (1 / normalized_episode_length)
                    if episode_is_off_road[-1]:
                        episode_cost += 10.0  # * (1 / normalized_episode_length)
                    if episode_is_friction_violation[-1]:
                        episode_cost += (
                                10.0 * episode_is_friction_violation[-1] / episode_length
                        )  # * (1 / normalized_episode_length)
                    if episode_is_goal_reached[-1]:
                        episode_cost -= 10.0  # * normalized_episode_length

                    episode_costs.append(episode_cost)
                    episode_lengths.append(episode_length)

                return episode_costs, episode_lengths

            sync_envs_normalization(self.training_env, self.eval_env)
            episode_costs, episode_lengths = evaluate_policy_configs(
                self.model,
                self.eval_env,
                n_eval_episode=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
            )

            mean_cost, std_cost = np.mean(episode_costs), np.std(episode_costs)
            mean_length, std_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_cost = mean_cost

            LOGGER.info("Evaluating at learning time step: {}".format(self.num_timesteps))
            LOGGER.info("Cost mean: {:.2f}, std: {:.2f}".format(mean_cost, std_cost))
            LOGGER.info("Length mean: {:.2f}, std: {:.2f}".format(mean_length, std_length))

            if self.log_path is not None:
                self.evaluation_timesteps.append(self.num_timesteps)
                self.evaluation_costs.append(episode_costs)
                self.evaluation_lengths.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluation_timesteps,
                    episode_costs=self.evaluation_costs,
                    episode_lengths=self.evaluation_lengths,
                )

            if mean_cost < self.lowest_mean_cost:
                self.lowest_mean_cost = mean_cost
                if self.best_model_save_path is not None:
                    self.model.save(self.best_model_save_path)
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            self.eval_idx += 1
            self.cost = self.lowest_mean_cost
            self.trial.report(self.cost, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class MultiEnvsEvalCallback(EventCallback):
    """
    Callback for evaluating an agent with multiple environments, i.e. SubprocVecEnv
    """

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            log_path: str = None,
            best_model_save_path: str = None,
            eval_freq: int = 1000,
            n_eval_timesteps: int = 1000,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 2,
    ):
        # Set basics
        super(MultiEnvsEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_timesteps = n_eval_timesteps
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_lengths = []

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        self.eval_env = eval_env

        # Store best model in `best_model.zip`
        self.best_model_save_path = construct_path_with_trial_number(best_model_save_path, "best_model")
        # Write logs in `evaluations.npz`
        self.log_path = construct_path_with_trial_number(log_path, "evaluations")

        # Set save path and logger
        if not len(LOGGER.handlers):
            map_verbose_to_logging = {2: logging.DEBUG, 1: logging.INFO}
            construct_logger(map_verbose_to_logging.get(verbose, logging.ERROR), log_path, LOGGER)

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn(f"Training and eval env are not of the same type {self.training_env} != {self.eval_env}")

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            def evaluate_policy_multi_envs(
                    model: "BaseRLModel",
                    env: Union[gym.Env, VecEnv],
                    n_eval_timesteps: int = 1000,
                    deterministic: bool = True,
                    render: bool = False,
                    callback: Optional[Callable] = None,
            ) -> Tuple[List[float], List[int]]:
                """
                Runs policy on `env.num_envs` environments until `n_eval_timesteps` timesteps have been collected asynchronously,,
                and returns lists of timesteps rewards and timesteps lengths.
                """

                episode_rewards = []
                episode_lengths = []

                # Reset observation
                # Pad observation for recrrent policies
                # See https://github.com/hill-a/stable-baselines/issues/1015
                obs = env.reset()
                is_recurrent = model.policy.recurrent
                if is_recurrent:
                    zero_completed_obs = np.zeros((model.n_envs,) + model.observation_space.shape)
                    zero_completed_obs[0, :] = obs
                    obs = zero_completed_obs

                dones, states = [False] * env.num_envs, [None] * env.num_envs
                episode_rewards_tmp = [0.0] * env.num_envs
                episode_lengths_tmp = [0] * env.num_envs
                while True:
                    # Predict actions and take a step
                    actions, states = model.predict(obs, state=states, deterministic=deterministic)
                    new_obs, rewards, dones, _info = env.step(actions)

                    if is_recurrent:
                        obs[0, :] = new_obs
                    else:
                        obs = new_obs

                    # Collect results for each evaluating environment
                    for i in range(env.num_envs):
                        if dones[i]:
                            episode_rewards_tmp[i] += rewards[i]
                            episode_lengths_tmp[i] += 1
                            episode_rewards.append(episode_rewards_tmp[i])
                            episode_lengths.append(episode_lengths_tmp[i])
                            episode_rewards_tmp[i] = 0.0
                            episode_lengths_tmp[i] = 0
                            LOGGER.debug(f"Collected {len(episode_rewards)} evaluation episodes")
                        else:
                            episode_rewards_tmp[i] += rewards[i]
                            episode_lengths_tmp[i] += 1

                    if callback is not None:
                        callback(locals(), globals())
                    if render:
                        env.render()

                    # Break if n_eval_timesteps enough results are obtained
                    if sum(episode_lengths) >= n_eval_timesteps:
                        break

                return episode_rewards, episode_lengths

            # Sync training and evaluating envs if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy_multi_envs(self.model,
                                                                          self.eval_env,
                                                                          self.n_eval_timesteps,
                                                                          self.deterministic,
                                                                          self.render)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_lengths.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_lengths)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            LOGGER.info("Eval num_timesteps={}, episode_reward={:.2f} +/- {:.2f}".format(
                self.num_timesteps, mean_reward, std_reward))
            LOGGER.info("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward > self.best_mean_reward:
                LOGGER.info("New best mean reward!")
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    self.model.save(self.best_model_save_path)
                if self.callback is not None:
                    return self._on_event()

        return True


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix=None, verbose=1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        # Set save path and logger
        if not len(LOGGER.handlers):
            map_verbose_to_logging = {2: logging.DEBUG, 1: logging.INFO}
            construct_logger(map_verbose_to_logging.get(verbose, logging.ERROR), save_path, LOGGER)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")

            LOGGER.info(f"Saving vectorized and normalized environment wrapper to {self.save_path}")
            self.model.get_vec_normalize_env().save(path)
        return True
