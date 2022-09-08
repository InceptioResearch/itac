"""
Module for CommonRoad vectorized environment used in Stable Baselines
"""
import time
from typing import Callable

import numpy as np
from gym import Env
from stable_baselines.common.vec_env import DummyVecEnv


class CommonRoadVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.on_reset = None
        self.start_times = np.array([])

    def set_on_reset(self, on_reset_callback: Callable[[Env, float], None]):
        self.on_reset = on_reset_callback

    def reset(self, **kwargs):
        self.start_times = np.array([time.time()] * self.num_envs)
        # copied from DummyVecEnv to enable reset with kwargs
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset(**kwargs)
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def step_wait(self):
        out_of_scenarios = False
        for env_idx in range(self.num_envs):
            (obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx],) = self.envs[env_idx].step(
                np.squeeze(self.actions[env_idx]))
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs

                # Callback
                elapsed_time = time.time() - self.start_times[env_idx]
                self.on_reset(self.envs[env_idx], elapsed_time)
                self.start_times[env_idx] = time.time()

                # If one of the environments doesn't have anymore scenarios it will throw an Exception on reset()
                try:
                    obs = self.envs[env_idx].reset()
                except IndexError:
                    out_of_scenarios = True
            self._save_obs(env_idx, obs)
            self.buf_infos[env_idx]["out_of_scenarios"] = out_of_scenarios
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()
