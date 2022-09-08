"""
A short tool to assess the number and time spanned by the files resulting from the conversion
Pass the path to the problem pickles as first command line argument.
"""

__author__ = "Niels Mündler"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

from pathlib import Path
import pickle
from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "pickles_path",
        default="pickles/problem",
        help="Path to pickled problem scenarios",
    )
    parser.add_argument(
        "-t",
        "--time_step",
        type=float,
        default=0.04,
        help="Time step length of the pickled scenario. Default is 0.04 seconds per step = 30 fps at the original recording",
        dest="time_step",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    pickles_path = Path(args.pickle_path)
    if not pickles_path.is_dir():
        print("Error: path {} is not a directory".format(pickles_path))
        exit(-1)

    overall_time = 0
    read_scenarios = 0
    for problem_path in pickles_path.iterdir():
        print(problem_path)
        try:
            with problem_path.open("rb") as file:
                pickled_problem = pickle.load(file)
                planning_problem_set = pickled_problem["planning_problem_set"]
                read_scenarios += 1
                for p in planning_problem_set.planning_problem_dict.values():
                    overall_time += max(s.time_step.end for s in p.goal.state_list)
        except Exception as e:
            print(e)
            pass
    print(f"Read {read_scenarios} scenarios")
    print(f"Overall training timesteps: {overall_time}")
    seconds = args.time_step * overall_time
    print(f"This is equivalent to {seconds} seconds")
    hours = (seconds / 60) / 60
    print(f"This is equivalent to {hours} hours")
