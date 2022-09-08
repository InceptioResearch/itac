"""
Module for reading and writing scenario related objects
"""

import os
import re
import random
from pathlib import Path
from typing import List, Tuple

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario, ScenarioID


def get_project_root():
    """
    Get the root path of project.
    :return: String of the project root path
    """
    return Path(__file__).parent.parent.parent.parent


def get_all_scenario(scenario_path: str) -> List[str]:
    """
    Get the list of all scenarios_backup in the path directory.

    :param scenario_path: the string of the directory
    :return: list of scenarios_backup names
    """
    scenario_list = []
    for file in os.listdir(scenario_path):
        if file.endswith(".xml"):
            scenario_list.append(str(os.path.basename(file)).split(".")[0])
    return scenario_list


def read_scenario(scenario_path: str, scenario_name: str = None) -> Tuple[Scenario, PlanningProblemSet]:
    """
    Get the desired scenario and planning problem set. If the scenario name is not given, return a random scenario with
    its problem set.

    :param scenario_path: path of directory
    :param scenario_name: given scenario name
    :return: scenario and planning problem set
    """
    if scenario_name is None:
        scenario_name = random.choice(get_all_scenario(scenario_path))
    else:
        if not scenario_name.endswith(".xml"):
            scenario_name += ".xml"
    scenario_full_name = os.path.join(scenario_path, scenario_name)
    scenario, planning_problem_set = CommonRoadFileReader(scenario_full_name).open()
    return scenario, planning_problem_set


def read_planning_problem(planning_problem_set: PlanningProblemSet, problem_id: int = None) -> PlanningProblem:
    """
    Get the desired planning problem. If the id is not given, get a random problem.

    :param planning_problem_set: Given planning problem set.
    :param problem_id: The id of the problem
    :return: the planning problem
    """
    if problem_id is None:
        all_ids = list(planning_problem_set.planning_problem_dict.keys())
        planning_problem = planning_problem_set.planning_problem_dict[random.choice(all_ids)]
    else:
        planning_problem = planning_problem_set.planning_problem_dict[problem_id]
    return planning_problem


def scenario_id_to_location(benchmark_id: str) -> str:
    location = re.split(r"([-_])", benchmark_id)[2]
    return location


def restore_scenario(meta_scenario: Scenario, obstacle_list: List[DynamicObstacle], scenario_id: ScenarioID):
    # scenario_new = copy.deepcopy(meta_scenario)
    # scenario_new.add_objects(obstacle_list)
    # return scenario_new
    # TODO: try remove and re-add obstacles ( 2times faster than using deepcopy, 1329ms vs 3514ms)
    meta_scenario.scenario_id = scenario_id
    meta_scenario.remove_obstacle(meta_scenario.obstacles)
    meta_scenario.add_objects(obstacle_list)
    return meta_scenario
