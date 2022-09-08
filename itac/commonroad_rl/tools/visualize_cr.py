#! /usr/bin/env python

__author__ = "Niels Mündler"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

__desc__ = """
Visualize a commonroad scenario.
Can also be used to check for problems in the lanelet network and to create animations.
"""

from matplotlib import pyplot as plt

from argparse import ArgumentParser
from pathlib import Path
from math import ceil, log10

from commonroad.visualization.plot_helper import (
    set_non_blocking,
    redraw_obstacles,
    draw_object,
)
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet


def reduce_to_successors_of(scenario: Scenario, first_lanelet_id: int):
    """
    Reduces the lanelet network of the given scenario to only successors of the lanelet of the given id.
    Useful to check for successor/predecessor errors in the lanelet network

    :param scenario: scenario for which to show lanelet network
    :param first_lanelet_id: lanelet of which to only show successors
    :return: None
    """
    _frontier = [first_lanelet_id]
    lanelet_ids = set()
    lanelets = set()
    while _frontier:
        c = scenario.lanelet_network.find_lanelet_by_id(_frontier.pop(0))
        lanelets.add(c)
        lanelet_ids.add(c.lanelet_id)
        for succ in c.successor:
            if succ not in lanelet_ids:
                _frontier.append(succ)

    scenario.lanelet_network = LaneletNetwork.create_from_lanelet_list(list(lanelets))


if __name__ == "__main__":
    parser = ArgumentParser(description=__desc__)
    parser.add_argument(
        "input", type=Path, help="Path to input file (.xml commonroad scenario)"
    )
    parser.add_argument(
        "-n",
        default=0,
        type=int,
        help="Number of time steps to show (default: 0 - show all frames)",
        dest="number",
    )
    parser.add_argument(
        "-s",
        default=3,
        type=int,
        help="Number of frames to skip each time step",
        dest="speed",
    )
    parser.add_argument(
        "-b", default=0, type=int, help="First frame to be drawn", dest="begin"
    )
    parser.add_argument(
        "-o",
        default="",
        type=str,
        help="Output directory for saved animation",
        dest="output",
    )
    parser.add_argument(
        "-l", action="store_true", help="Draw lanelet labels", dest="labels"
    )
    parser.add_argument(
        "-t",
        "--suffix",
        default="png",
        help="Filetype, suffix of the files to be written",
        dest="suffix",
    )
    parser.add_argument(
        "--successors_of",
        default="0",
        type=int,
        help="Only draw all successors of the lanelet with specified id",
    )
    args = parser.parse_args()

    input_d = args.input
    if not input_d.exists():
        print("Error: path {} does not exist".format(input_d))
        exit(-1)
    scenario, planning_problem_set = CommonRoadFileReader(input_d).open()

    WRITE = args.output != ""
    if WRITE:
        output_d = Path(args.output).joinpath(input_d.stem)
        output_d.mkdir(parents=True, exist_ok=True)

    # fast drawing according to https://commonroad.in.tum.de/static/docs/commonroad-io/user/visualization.html#speed-up-plotting-for-real-time-applications
    set_non_blocking()  # ensures interactive plotting is activated

    figsize = [10, 10]
    fig = plt.figure(figsize=figsize)
    plt.gca().set_aspect("equal")
    handles = {}  # collects handles of obstacles for fast updating of figures

    # inital plot including the lanelet network
    draw_params = {
        "scenario": {
            "lanelet_network": {
                "traffic_sign": {
                    "draw_traffic_signs": False,
                    "show_traffic_signs": "all",
                    "show_label": False,
                },
                "intersection": {"draw_intersections": True},
                "lanelet": {
                    "draw_stop_line": True,
                    "show_label": False
                }
            },
            "dynamic_obstacle": {
                "trajectory": {
                    "unique_colors": True,
                },
                "show_label": False,
            },
        },
        "time_begin": args.begin * args.speed,
    }

    if args.number != 0:
        number_frames = args.number
    else:
        number_frames = max(
            1,
            1,
            *[
                o.prediction.occupancy_set[-1].time_step
                for o in scenario.dynamic_obstacles
            ],
            *[
                state.time_step.end
                for pp in planning_problem_set.planning_problem_dict.values()
                for state in pp.goal.state_list
            ],
        )
        number_frames = number_frames // args.speed + 1
    numdigits = ceil(log10(number_frames + 1))
    suffix = args.suffix

    if args.successors_of != 0:
        # the difference between the function "reduce to successors of"
        # and "all lanelets by merging successors from lanelet"
        # may yield yet another problem
        m, s = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
            scenario.lanelet_network.find_lanelet_by_id(args.successors_of),
            scenario.lanelet_network,
            max_length=1000,
        )
        scenario.lanelet_network = LaneletNetwork.create_from_lanelet_list(
            list(scenario.lanelet_network.find_lanelet_by_id(i) for l in s for i in l)
        )
        reduce_to_successors_of(scenario, args.successors_of)

    # draw_object(scenario, handles=handles, draw_params=draw_params)
    draw_object(scenario, draw_params=draw_params)
    draw_object(planning_problem_set)
    fig.canvas.draw()
    plt.autoscale()
    plt.savefig(f"{input_d.stem}.svg")
    plt.show()
    input("Press enter to start animation and start writing frames")
    # redraw
    for i in range(args.begin + 1, args.begin + number_frames):
        draw_params["time_begin"] = i * args.speed
        redraw_obstacles(
            scenario, handles=handles, figure_handle=fig, draw_params=draw_params
        )
        if WRITE:
            plt.savefig(
                output_d.joinpath(
                    "{:0{numdigits}d}.{}".format(i, suffix, numdigits=numdigits)
                )
            )
        else:
            plt.show()


# visualization manual
# https://commonroad-io.readthedocs.io/en/latest/user/visualization/#draw-params