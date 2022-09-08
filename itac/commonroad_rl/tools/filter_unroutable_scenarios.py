"""
Script to filter out scenarios where no route can be found
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

from commonroad.planning.planning_problem import PlanningProblem
from commonroad_route_planner.route_planner import RoutePlanner

import commonroad_rl.gym_commonroad.constants as constants
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)


def get_args():
    """Scan arguments"""
    parser = argparse.ArgumentParser(
        description="Filters out non-routable scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pickles", "-i", help="Path to problems", type=str,
                        default=constants.PATH_PARAMS["pickles"])
    parser.add_argument("--output", "-o", help="Path to output dir", type=str,
                        default=os.path.join(constants.PATH_PARAMS["pickles"], "unroutable-problems"))
    return parser.parse_args()


def main(pickles: str, output: str) -> None:
    """
    Filter out non-routable scenarios
    """
    pickles = os.path.expanduser(pickles)
    output = os.path.expanduser(output)
    os.makedirs(name=output, exist_ok=True)
    problem = os.path.join(pickles, "problem")
    meta_scenario_path = os.path.join(pickles, "meta_scenario")

    with open(os.path.join(meta_scenario_path, "problem_meta_scenario_dict.pickle"), "rb") as f:
        problem_meta_scenario_dict = pickle.load(f)

    for file in os.listdir(problem):
        if not os.path.isfile(os.path.join(problem, file)) or not file.endswith(".pickle"):
            continue

        with open(os.path.join(problem, file), "rb") as f:
            problem_dict: dict = pickle.load(f)

        planning_problem: PlanningProblem = list(problem_dict["planning_problem_set"].planning_problem_dict.values())[0]
        benchmark_id = os.path.splitext(file)[0]
        meta_scenario = problem_meta_scenario_dict[benchmark_id]
        scenario = restore_scenario(meta_scenario, problem_dict["obstacle"])

        rp = RoutePlanner(scenario, planning_problem)
        routes = rp.plan_routes()

        if routes.num_route_candidates == 0:
            os.rename(os.path.join(problem, file), os.path.join(output, file))


if __name__ == "__main__":
    args = get_args()
    main(
        pickles=args.pickles,
        output=args.output
    )
