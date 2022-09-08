"""
Module to check if all training/testing scenarios have the same goal-configuration and automatically configure
the goal_observations in configs.yaml
"""

__author__ = "Armin Ettenhofer"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Armin Ettenhofer"
__email__ = "armin.ettenhofer@tum.de"
__status__ = "Released"

import argparse
import logging
import os
import pickle
import re
from typing import List, Set

from commonroad.scenario.trajectory import State

import commonroad_rl.gym_commonroad.constants as constants

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)


def get_args():
    """Scan arguments"""
    parser = argparse.ArgumentParser(
        description="Analyzes goal_definitions of scenarios and configures goal_observations for model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--problem_dir", "-i", help="Path to problems", type=str,
                        default=os.path.join(constants.PATH_PARAMS["pickles"], "problem"))
    parser.add_argument("--configure", "-c", help="Adjust the goal_observations in the configuration",
                        action="store_true")
    parser.add_argument("--config_path", "-cp", help="Path to model configuration file", type=str,
                        default=constants.PATH_PARAMS["configs"]["commonroad-v1"])
    return parser.parse_args()


def main(problem_dir: str = os.path.join(constants.PATH_PARAMS["pickles"], "problem"), configure: bool = False,
         config_path: str = constants.PATH_PARAMS["configs"]["commonroad-v1"]) -> bool:
    """
    Analyzes goal-requirements of scenarios and sets goal-observations in config

    :param problem_dir: Path to problem directory
    :param configure: If the goal_observations should be configured
    :param config_path: Path to the environment configuration file
    :return: If all scenarios have the same goal definition
    """

    # Check arguments
    assert os.path.isdir(problem_dir), "The problem directory doesn't exist"
    if configure:
        assert os.path.isfile(config_path), "The path to the config file is invalid!"

    # Find all pickled scenarios
    _, _, scenarios_pickled = next(os.walk(problem_dir))
    scenarios_pickled = [filename for filename in scenarios_pickled if os.path.splitext(filename)[1] == ".pickle"]
    if len(scenarios_pickled) == 0:
        LOGGER.warning("The problem directory does not contain any problems!")
        return True

    # Analyze the goal_states
    goal_states = analyze_problems(base_path=problem_dir, filenames=scenarios_pickled)

    # Evaluate the results
    matching: bool = goal_states.pop("matching")
    position: bool = goal_states["position"]
    velocity: bool = goal_states["velocity"]
    orientation: bool = goal_states["orientation"]
    time_step: bool = goal_states["time_step"]

    if matching:
        print("The goal states of all problems have the same attributes.")
    else:
        print("Not all goal states of the problems match.\n")
    print("The recommended goal_observation configuration is:")
    print(f"Position: {position}, Velocity: {velocity}, Orientation: {orientation}, Time_step: {time_step}\n")

    # Set config.yaml if desired
    if not configure:
        return matching

    set_configuration(config_path=config_path, goal_observation=goal_states)
    return matching


def analyze_problems(base_path: str, filenames: [str]) -> dict:
    """
    Checks if the goal states of all problems have the same definitions and returns the resulting definition

    :param base_path: Path to problem directory
    :param filenames: Names of all the problems to be analyzed
    :return: Goal definition
    """
    print(f"Analyzing {len(filenames)} files.\n")

    init = False
    matching = True
    attributes: Set[str] = set()

    # Look at every goal state in every planning problem in every scenario
    for filename in filenames:
        with open(os.path.join(base_path, filename), 'rb') as file:
            planning_problem_dict: dict = pickle.load(file)["planning_problem_set"].planning_problem_dict

        for _, planning_problem in planning_problem_dict.items():
            goal_state_list: List[State] = planning_problem.goal.state_list
            for goal_state in goal_state_list:
                if init:
                    goal_attributes = set(goal_state.attributes)
                    if goal_attributes != attributes:
                        # Print the first not matching file
                        if matching:
                            print(f"\'{os.path.splitext(filename)[0]}\' and \'{os.path.splitext(filenames[0])[0]}\'"
                                  f" don't have the same goal_state attributes.\n")
                        matching = False
                        attributes |= goal_attributes

                # Initialize with the first goal_state
                else:
                    init = True
                    attributes = set(goal_state.attributes)

    config = {"position": False, "velocity": False, "orientation": False, "time_step": False, "matching": matching}
    for attribute in attributes:
        config[attribute] = True

    return config


def set_configuration(config_path: str, goal_observation: dict):
    """
    Sets the goal_observations in the config according to the input goal_observation

    :param config_path: Path to the config-file (.yaml file)
    :param goal_observation: How the goal observations should be configured
    """
    print("Setting the goal_observations in the configs.yaml\n")
    with open(config_path, "r") as file:
        config = file.read()

    if goal_observation["position"]:
        config = re.sub("observe_distance_goal_long:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_long: True", config)
        config = re.sub("observe_distance_goal_lat:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_lat: True", config)
    else:
        config = re.sub("observe_distance_goal_long:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_long: False", config)
        config = re.sub("observe_distance_goal_lat:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_lat: False", config)

    if goal_observation["velocity"]:
        config = re.sub("observe_distance_goal_velocity:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_velocity: True", config)
    else:
        config = re.sub("observe_distance_goal_velocity:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_velocity: False", config)

    if goal_observation["orientation"]:
        config = re.sub("observe_distance_goal_orientation:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_orientation: True", config)
    else:
        config = re.sub("observe_distance_goal_orientation:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_orientation: False", config)

    if goal_observation["time_step"]:
        config = re.sub("observe_distance_goal_time:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_time: True", config)
    else:
        config = re.sub("observe_distance_goal_time:( ){0,5}([Tt]rue|[Ff]alse)",
                        "observe_distance_goal_time: False", config)

    with open(config_path, "w") as file:
        file.write(config)


if __name__ == "__main__":
    args = get_args()
    main(
        problem_dir=args.problem_dir,
        configure=args.configure,
        config_path=args.config_path
    )
