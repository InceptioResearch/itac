"""
Module for the CommonRoad Gym environment
"""
import os
import pathlib

import gym
import glob
import yaml
import pickle
import random
import logging
import warnings
import numpy as np

from typing import Tuple, Union

# import from commonroad-drivability-checker
from commonroad.geometry.shape import Rectangle

# import from commonroad-io
from commonroad.scenario.scenario import ScenarioID, Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import ParamServer, write_default_params
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType

# import from commonroad-rl
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.gym_commonroad.observation import ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.gym_commonroad.action import action_constructor
from commonroad_rl.gym_commonroad.reward import reward_constructor
from commonroad_rl.gym_commonroad.reward.reward import Reward
from commonroad_rl.gym_commonroad.reward.termination import Termination

import matplotlib
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


class CommonroadEnv(gym.Env):
    """
    Description:
        This environment simulates the ego vehicle in a traffic scenario using commonroad environment. The task of
        the ego vehicle is to reach the predefined goal without going off-road, collision with other vehicles, and
        finish the task in specific time frame. Please consult `commonroad_rl/gym_commonroad/README.md` for details.
    """

    metadata = {"render.modes": ["human"]}

    # For the current configuration check the ./configs.yaml file
    def __init__(
            self,
            meta_scenario_path=PATH_PARAMS["meta_scenario"],
            train_reset_config_path=PATH_PARAMS["train_reset_config"],
            test_reset_config_path=PATH_PARAMS["test_reset_config"],
            visualization_path=PATH_PARAMS["visualization"],
            logging_path=None,
            test_env=False,
            play=False,
            config_file=PATH_PARAMS["configs"]["commonroad-v1"],
            logging_mode=1,
            **kwargs,
    ) -> None:
        """
        Initialize environment, set scenario and planning problem.
        """
        # Set logger if not yet exists
        LOGGER.setLevel(logging_mode)

        if not len(LOGGER.handlers):
            formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging_mode)
            stream_handler.setFormatter(formatter)
            LOGGER.addHandler(stream_handler)

            if logging_path is not None:
                file_handler = logging.FileHandler(filename=os.path.join(logging_path, "console_copy.txt"))
                file_handler.setLevel(logging_mode)
                file_handler.setFormatter(formatter)
                LOGGER.addHandler(file_handler)

        LOGGER.debug("Initialization started")

        # Default configuration
        if isinstance(config_file, (str, pathlib.Path)):
            with pathlib.Path(config_file).open("r") as config_file:
                config = yaml.safe_load(config_file)

        # Assume default environment configurations
        self.configs = config.get("env_configs", config)

        # Overwrite environment configurations if specified
        if kwargs is not None:
            for k, v in kwargs.items():
                assert k in self.configs, f"Configuration item not supported: {k}"
                # TODO: update only one term in configs
                if isinstance(v, dict):
                    self.configs[k].update(v)
                else:
                    self.configs.update({k: v})

        # Make environment configurations as attributes
        self.vehicle_params: dict = self.configs["vehicle_params"]
        self.action_configs: dict = self.configs["action_configs"]
        self.render_configs: dict = self.configs["render_configs"]
        self.reward_type: str = self.configs["reward_type"]

        # change configurations when using point mass vehicle model
        if self.vehicle_params["vehicle_model"] == 0:
            self.observe_heading = False
            self.observe_steering_angle = False
            self.observe_global_turn_rate = False
            self.observe_distance_goal_long_lane = False

        # Flag for popping out scenarios
        self.play = play

        # Load scenarios and problems
        self.meta_scenario_path = meta_scenario_path
        self.all_problem_dict = dict()
        self.planning_problem_set_list = []

        # Accelerator structures
        # self.cache_goal_obs = dict()

        if isinstance(meta_scenario_path, (str, pathlib.Path)):
            meta_scenario_reset_dict_path = pathlib.Path(self.meta_scenario_path) / "meta_scenario_reset_dict.pickle"
            with meta_scenario_reset_dict_path.open("rb") as f:
                self.meta_scenario_reset_dict = pickle.load(f)
        else:
            self.meta_scenario_reset_dict = meta_scenario_path

        self.train_reset_config_path = train_reset_config_path

        def load_reset_config(path):
            path = pathlib.Path(path)
            problem_dict = {}
            for p in path.glob("*.pickle"):
                with p.open("rb") as f:
                    problem_dict[p.stem] = pickle.load(f)
            return problem_dict

        if not test_env and not play:
            if isinstance(train_reset_config_path, (str, pathlib.Path)):
                self.all_problem_dict = load_reset_config(train_reset_config_path)
            else:
                self.all_problem_dict = train_reset_config_path
            self.is_test_env = False
            LOGGER.info(f"Training on {train_reset_config_path} with {len(self.all_problem_dict.keys())} scenarios")
        else:
            if isinstance(test_reset_config_path, (str, pathlib.Path)):
                self.all_problem_dict = load_reset_config(test_reset_config_path)
            else:
                self.all_problem_dict = test_reset_config_path
            LOGGER.info(f"Testing on {test_reset_config_path} with {len(self.all_problem_dict.keys())} scenarios")

        self.visualization_path = visualization_path

        self.termination = Termination(self.configs)
        self.terminated = False
        self.termination_reason = None

        self.ego_action, self.action_space = action_constructor(self.action_configs, self.vehicle_params)

        # Observation space
        self.observation_collector = ObservationCollector(self.configs)

        # Reward function
        self.reward_function: Reward = reward_constructor.make_reward(self.configs)

        # TODO initialize reward class

        LOGGER.debug(f"Meta scenario path: {meta_scenario_path}")
        LOGGER.debug(f"Training data path: {train_reset_config_path}")
        LOGGER.debug(f"Testing data path: {test_reset_config_path}")
        LOGGER.debug("Initialization done")

        # ----------- Visualization -----------
        self.cr_render = None
        self.draw_params = None

    @property
    def observation_space(self):
        return self.observation_collector.observation_space

    @property
    def observation_dict(self):
        return self.observation_collector.observation_dict

    def seed(self, seed=Union[None, int]):
        self.action_space.seed(seed)

    def reset(self, benchmark_id=None, scenario: Scenario = None,
              planning_problem: PlanningProblem = None) -> np.ndarray:
        """
        Reset the environment.
        :param benchmark_id: benchmark id used for reset to specific scenario
        :param reset_renderer: parameter used for reset the renderer to default

        :return: observation
        """
        self._set_scenario_problem(benchmark_id, scenario=scenario, planning_problem=planning_problem)
        self.ego_action.reset(self.planning_problem.initial_state, self.scenario.dt)
        self.observation_collector.reset(self.scenario, self.planning_problem, self.reset_config, self.benchmark_id,
                                         clone_collision_checker=scenario is None or planning_problem is None)
        self.reset_renderer()
        # TODO: remove self._set_goal()
        self._set_initial_goal_reward()

        self.terminated = False

        initial_observation = self.observation_collector.observe(self.ego_action.vehicle)
        self.reward_function.reset(self.observation_dict, self.ego_action)
        self.termination.reset(self.observation_dict, self.ego_action)

        self.v_ego_mean = self.ego_action.vehicle.state.velocity
        # TODO: tmp store all observations in info for paper draft, remove afterwards
        self.observation_list = [self.observation_dict]

        return initial_observation

    @property
    def current_step(self):
        return self.observation_collector.time_step

    @current_step.setter
    def current_step(self, time_step):
        raise ValueError(f"<CommonroadEnv> Set current_step is prohibited!")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Propagate to next time step, compute next observations, reward and status.

        :param action: vehicle acceleration, vehicle steering velocity
        :return: observation, reward, status and other information
        """
        if self.action_configs['action_type'] == "continuous":
            action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        # Make action and observe result
        self.ego_action.step(action, local_ccosy=self.observation_collector.local_ccosy)
        observation = self.observation_collector.observe(self.ego_action.vehicle)

        # Check for termination
        done, reason, termination_info = self.termination.is_terminated(self.observation_dict, self.ego_action)
        if reason is not None:
            self.termination_reason = reason

        if done:
            self.terminated = True

        # Calculate reward
        reward = self.reward_function.calc_reward(self.observation_dict, self.ego_action)

        self.v_ego_mean += self.ego_action.vehicle.state.velocity
        self.observation_list.append(self.observation_list)
        assert str(self.scenario.scenario_id) == self.benchmark_id
        info = {
            "scenario_name": self.benchmark_id,
            "chosen_action": action,
            "current_episode_time_step": self.current_step,
            "max_episode_time_steps": self.observation_collector.episode_length,
            "termination_reason": self.termination_reason,
            "v_ego_mean": self.v_ego_mean / self.current_step,
            "observation_list": self.observation_list
        }
        info.update(termination_info)

        if self.configs["surrounding_configs"]["observe_lane_circ_surrounding"] \
                or self.configs["surrounding_configs"]["observe_lane_rect_surrounding"]:
            info["ttc_follow"], info["ttc_lead"] = CommonroadEnv.get_ttc_lead_follow(self.observation_dict)

        return observation, reward, done, info

    def reset_renderer(self, renderer: Union[MPRenderer, None] = None,
                       draw_params: Union[ParamServer, dict, None] = None) -> None:
        if renderer:
            self.cr_render = renderer
        else:
            self.cr_render = MPRenderer()

        if draw_params:
            self.draw_params = draw_params
        else:
            self.draw_params = ParamServer({
                "scenario": {"time_begin": self.current_step,
                             "lanelet_network": {
                                 "lanelet": {
                                     "show_label": False,
                                     "fill_lanelet": True},
                                 "traffic_sign": {
                                     "draw_traffic_signs": False,
                                     "show_traffic_signs": "all",
                                     "show_label": False,
                                     'scale_factor': 0.1},
                                 "intersection": {
                                     "draw_intersections": False}
                             },
                             "dynamic_obstacle": {"show_label": False}}
            })

    def render(self, mode: str = "human", **kwargs) -> None:
        """
        Generate images for visualization.

        :param mode: default as human for visualization
        :return: None
        """
        # Render only every xth timestep, the first and the last
        if not (self.current_step % self.render_configs["render_skip_timesteps"] == 0 or self.terminated):
            return

        # update timestep in draw_params
        self.draw_params.update({"scenario": {"time_begin": self.current_step}})

        # Draw scenario, goal, sensing range and detected obstacles
        self.scenario.draw(self.cr_render, self.draw_params)

        # Draw certain objects only once
        if (not self.render_configs["render_combine_frames"] or self.current_step == 0) and not isinstance(mode, int):
            self.planning_problem.draw(self.cr_render)

        self.observation_collector.render(self.render_configs, self.cr_render)

        # Draw ego vehicle # draw icon
        ego_obstacle = DynamicObstacle(
            obstacle_id=self.scenario.generate_object_id(),
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=Rectangle(length=self.ego_action.vehicle.parameters.l,
                                     width=self.ego_action.vehicle.parameters.w),
            initial_state=self.ego_action.vehicle.state
        )
        ego_obstacle.draw(self.cr_render, draw_params=ParamServer({
            "time_begin": self.current_step,
            "dynamic_obstacle": {
                "draw_icon": True,
                "vehicle_shape": {
                    "occupancy": {
                        "shape": {
                            "rectangle": {
                                "opacity": 1.0,
                                "facecolor": "red",
                                "edgecolor": "red",
                                "linewidth": 0.5,
                                "zorder": 20
                            }
                        }
                    }
                }
            }})
                          )
        #self.ego_action.vehicle.collision_object.draw(self.cr_render,
        #                                              draw_params={"facecolor": "green", "zorder": 30})

        # Save figure, only if frames should not be combined or simulation is over
        os.makedirs(os.path.join(self.visualization_path, str(self.scenario.scenario_id)), exist_ok=True)
        if not self.render_configs["render_combine_frames"] or self.terminated:
            if isinstance(mode, int):
                filename = os.path.join(self.visualization_path, str(self.scenario.scenario_id),
                                        self.file_name_format % mode) + ".png"
            else:
                filename = os.path.join(self.visualization_path, str(self.scenario.scenario_id),
                                        self.file_name_format % self.current_step) + ".png"
            if self.render_configs["render_follow_ego"]:
                # TODO: works only for highD, implement range relative to ego orientation
                # follow ego
                x, y = self.ego_action.vehicle.state.position
                range = self.render_configs["render_range"]
                self.cr_render.plot_limits = [x - range[0], x + range[0], y - range[1], y + range[1]]
            self.cr_render.render(show=False, filename=filename, keep_static_artists=True)

        # =================================================================================================================
        #
        #                                    reset functions
        #
        # =================================================================================================================

    def _set_scenario_problem(self, benchmark_id=None, scenario: Scenario = None,
                              planning_problem: PlanningProblem = None) -> None:
        """
        Select scenario and planning problem.

        :return: None
        """
        if self.play:
            # pop instead of reusing
            LOGGER.debug(f"Number of scenarios left {len(list(self.all_problem_dict.keys()))}")
            self.benchmark_id = random.choice(list(self.all_problem_dict.keys()))
            problem_dict = self.all_problem_dict.pop(self.benchmark_id)
        else:
            if benchmark_id is not None:
                self.benchmark_id = benchmark_id
                problem_dict = self.all_problem_dict[benchmark_id]
            elif scenario is None or planning_problem is None:
                self.benchmark_id, problem_dict = random.choice(list(self.all_problem_dict.items()))

        if scenario is None or planning_problem is None:
            # Set reset config dictionary
            scenario_id = ScenarioID.from_benchmark_id(self.benchmark_id, "2020a")
            map_id = parse_map_name(scenario_id)
            self.reset_config = self.meta_scenario_reset_dict[map_id]
            # meta_scenario = self.problem_meta_scenario_dict[self.benchmark_id]
            self.scenario = restore_scenario(self.reset_config["meta_scenario"], problem_dict["obstacle"], scenario_id)
            self.planning_problem: PlanningProblem = random.choice(
                list(problem_dict["planning_problem_set"].planning_problem_dict.values())
            )
        else:
            # TODO: calculate reset_config online
            from commonroad_rl.tools.pickle_scenario.preprocessing import generate_reset_config
            self.reset_config = generate_reset_config(scenario, open_lane_ends=True)
            self.scenario = scenario
            self.planning_problem = planning_problem
            self.benchmark_id = str(scenario.scenario_id)

        # Set name format for visualization
        self.file_name_format = self.benchmark_id + "_ts_%03d"

    def _set_initial_goal_reward(self) -> None:
        """
        Set ego vehicle and initialize its status.

        :return: None
        """
        self.goal = self.observation_collector.goal_observation

        # Compute initial distance to goal for normalization if required
        if self.reward_type == "dense_reward":  # or "hybrid_reward":
            self.observation_collector._create_navigator()
            distance_goal_long, distance_goal_lat = self.goal.get_long_lat_distance_to_goal(
                self.ego_action.vehicle.state.position, self.observation_collector.navigator
            )
            self.initial_goal_dist = np.sqrt(distance_goal_long ** 2 + distance_goal_lat ** 2)

            # Prevent cases where the ego vehicle starts in the goal region
            if self.initial_goal_dist < 1.0:
                warnings.warn("Ego vehicle starts in the goal region")
                self.initial_goal_dist = 1.0

    @staticmethod
    def get_ttc_lead_follow(observation_dict):
        idx_follow = 1
        idx_lead = 4

        def get_ttc(p_rel, v_rel):
            if np.isclose(v_rel, 0.):
                return np.inf
            else:
                return p_rel / -v_rel

        # lane_based_v_rel = v_lead - v_follow
        # ttc: (s_lead - s_follow) / (v_follow - v_lead)
        ttc_follow = get_ttc(observation_dict["lane_based_p_rel"][idx_follow],
                             observation_dict["lane_based_v_rel"][idx_follow])
        ttc_lead = get_ttc(observation_dict["lane_based_p_rel"][idx_lead],
                           observation_dict["lane_based_v_rel"][idx_lead])

        return ttc_follow, ttc_lead
