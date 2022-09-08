"""
Loads a set of pickled scenarios and measures loading time and memory consumption
"""
import os
import pickle
import glob
from time import time
from argparse import ArgumentParser
from memory_profiler import profile


def load_scenarios(path):
    storage = []

    meta_scenario_reset_dict_path = os.path.join(
        path, "meta_scenario", "meta_scenario_reset_dict.pickle"
    )
    with open(meta_scenario_reset_dict_path, "rb") as f:
        storage.append(pickle.load(f))

    problem_meta_scenario_dict_path = os.path.join(
        path, "meta_scenario", "problem_meta_scenario_dict.pickle"
    )
    with open(problem_meta_scenario_dict_path, "rb") as f:
        storage.append(pickle.load(f))

    print(f"Loading {path}...")
    fns = glob.glob(os.path.join(path, "problem_train", "*.pickle"))
    for fn in fns:
        with open(fn, "rb") as f:
            storage.append(pickle.load(f))

    return len(storage) - 2


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to pickle training data")
    parser.add_argument(
        "-m",
        "--memory",
        action="store_true",
        help="Do memory profiling of the pickle loading",
    )
    args = parser.parse_args()

    _begin = time()
    if args.memory:
        load_scenarios = profile(load_scenarios)
    num = load_scenarios(args.path)
    _end = time()
    print(f"Took {_end-_begin}s to load {num} scenarios")
