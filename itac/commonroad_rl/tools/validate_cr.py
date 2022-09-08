#! /usr/bin/env python

__author__ = "Niels Mündler"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

__desc__ = """
Validates a Commonroad scenario syntactically
"""

from argparse import ArgumentParser
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle
from commonroad.planning.planning_problem import PlanningProblem

from commonroad_route_planner.route_planner import RoutePlanner

from lxml import etree
from shapely.validation import explain_validity
from sys import exit


def check_adjacency(lanelet_network: LaneletNetwork):
    """
    Check that adjacencies are assigned correctly and bilaterally
    """
    errors = 0
    for l in lanelet_network.lanelets:
        if l.adj_left is not None:
            l_left = lanelet_network.find_lanelet_by_id(l.adj_left)
            if l.adj_left_same_direction and l_left.adj_right != l.lanelet_id:
                print(
                    f"Left of lanelet {l.lanelet_id} is {l.adj_left} facing the same direction but right of {l_left.lanelet_id} is {l_left.adj_right}"
                )
                errors += 1
            if not l.adj_left_same_direction and l_left.adj_left != l.lanelet_id:
                print(
                    f"Left of lanelet {l.lanelet_id} is {l.adj_left} facing the opposite direction but left of {l_left.lanelet_id} is {l_left.adj_left}"
                )
                errors += 1
        if l.adj_right is not None:
            l_right = lanelet_network.find_lanelet_by_id(l.adj_right)
            if l.adj_right_same_direction and l_right.adj_left != l.lanelet_id:
                print(
                    f"Right of lanelet {l.lanelet_id} is {l.adj_right} facing the same direction but left of {l_right.lanelet_id} is {l_right.adj_left}"
                )
                errors += 1
            if not l.adj_right_same_direction and l_right.adj_right != l.lanelet_id:
                print(
                    f"Left of lanelet {l.lanelet_id} is {l.adj_right} facing the opposite direction but left of {l_right.lanelet_id} is {l_right.adj_left}"
                )
                errors += 1
    return errors


def check_valid_lanelet_polygon(lanelet_network: LaneletNetwork):
    errors = 0
    for lanelet in lanelet_network.lanelets:
        try:
            assert lanelet.convert_to_polygon().shapely_object.is_valid
        except AssertionError as e:
            print(f"Lanelet {lanelet.lanelet_id} has invalid geometry")
            print(explain_validity(lanelet.convert_to_polygon().shapely_object))
            errors += 1
    return errors


def check_successors(lanelet_network: LaneletNetwork):
    """
    Check that adjacencies are assigned correctly and bilaterally
    """
    errors = 0
    for l in lanelet_network.lanelets:
        if l.successor is not None:
            for i in l.successor:
                suc = lanelet_network.find_lanelet_by_id(i)
                if not suc.predecessor or not l.lanelet_id in suc.predecessor:
                    print(
                        f"Lanelet {i} is successor of {l.lanelet_id} but does not have {l.lanelet_id} as predecessor"
                    )
                    errors += 1
        if l.predecessor is not None:
            for i in l.predecessor:
                pred = lanelet_network.find_lanelet_by_id(i)
                if not pred.successor or not l.lanelet_id in pred.successor:
                    print(
                        f"Lanelet {i} is predecessor of {l.lanelet_id} but does not have {l.lanelet_id} as successor"
                    )
                    errors += 1
    return errors


def check_obstacle_off_road(scenario: Scenario):
    errors = 0
    for o in scenario.obstacles:
        off_road = True
        if isinstance(o, DynamicObstacle):
            for s in o.prediction.trajectory.state_list:
                if scenario.lanelet_network.find_lanelet_by_position([s.position]):
                    off_road = False
                    break
        elif isinstance(o, StaticObstacle):
            off_road = not scenario.lanelet_network.find_lanelet_by_position(
                [o.initial_state.position]
            )
        try:
            assert not off_road
        except AssertionError:
            print(f"Obstacle {o.obstacle_id} is off the road at all times")
            errors += 1
    return errors


def check_path_to_goal(scenario: Scenario, planning_problem: PlanningProblem):
    errors = 0
    route_planner = RoutePlanner(
        scenario,
        planning_problem,
        backend=RoutePlanner.Backend.NETWORKX_REVERSED,
        log_to_console=False,
    )
    if len(route_planner.get_route_candidates().route_candidates) == 0:
        # no path found
        print(f"No path from initial position to goal region")
        errors += 1
    return errors


def validate(xml_path: Path, xsd_path: Path) -> 0:
    # test whether a scenario file passes the schema test
    try:
        assert xml_path.exists()
        etree.clear_error_log()
        xmlschema = etree.XMLSchema(etree.parse(str(xsd_path)))
        tmp = etree.parse(str(xml_path))
        xmlschema.assertValid(tmp)
    except AssertionError:
        print("File not found")
        return 1
    except etree.DocumentInvalid as e:
        print("File invalid: {}".format(e))
        return 1
    except Exception as e:
        print("Unknown error: {}".format(e))
        return 1
    return 0


def get_parser():
    parser = ArgumentParser(description=__desc__)
    parser.add_argument(
        "input", help="Path to input file(s) (.xml commonroad scenario)", nargs="+"
    )
    parser.add_argument(
        "-s",
        help="Path to schema file (.xsd commonroad specification)",
        default="./XML_commonRoad_XSD_2020a.xsd",
        dest="spec",
    )
    return parser


def main():
    args = get_parser().parse_args()

    xsd_path = Path(args.spec)
    assert xsd_path.exists(), f"file path {xsd_path} doesn't exist"

    total_errors = 0
    for input_file in args.input:
        xml_path = Path(input_file)
        print(xml_path)
        errors = 0

        # Check XML validity
        errors += validate(xml_path, xsd_path)

        # Additionally check if adjacent right/left is always assigned respectively
        scenario, planning_problem_set = CommonRoadFileReader(str(xml_path)).open()
        lanelet_network = scenario.lanelet_network
        errors += check_adjacency(lanelet_network)

        # check for invalid lanelet polygons like self-intersections or similar
        errors += check_valid_lanelet_polygon(scenario.lanelet_network)

        # Check that no obstacle is off the road all the time
        errors += check_obstacle_off_road(scenario)

        # # Check that there exists a route from the initial position
        # # to the goal
        # for pp in planning_problem_set.planning_problem_dict.values():
        #     errors += check_path_to_goal(scenario, pp)

        total_errors += errors

    if total_errors > 0:
        print(f"{total_errors} Errors where detected")
        exit(-1)
    else:
        print("No errors")
        exit(0)


if __name__ == "__main__":
    main()
