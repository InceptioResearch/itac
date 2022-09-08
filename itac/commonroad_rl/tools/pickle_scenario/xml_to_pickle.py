import os
import glob
import shutil
import pickle
import argparse
import multiprocessing as mp

from time import time
from typing import List

from commonroad.common.file_reader import CommonRoadFileReader
import commonroad_dc.pycrcc as pycrcc
from commonroad_rl.tools.pickle_scenario.preprocessing import generate_reset_config
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name


def get_args():
    parser = argparse.ArgumentParser(description="Converts CommonRoad xml files to pickle files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", "-i", type=str, default="/data/highD-dataset-v1.0/cr_scenarios")
    parser.add_argument("--output_dir", "-o", type=str, default="/data/highD-dataset-v1.0/pickles")
    parser.add_argument("--duplicate", "-d", action="store_true",
                        help="Duplicate one scenario file to problem_train and problem_test, for overfitting")
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of multiple processes to convert dataset, default=1')

    return parser.parse_args()


def is_initial_collision(planning_problem_set, env_reset):

    initial_state = list(planning_problem_set.planning_problem_dict.values())[0].initial_state
    boundary_collision_object = env_reset["boundary_collision_object"]
    ego_collision_object = pycrcc.RectOBB(
        4.508 / 2,
        1.610 / 2,
        initial_state.orientation if hasattr(initial_state, "orientation") else 0.0,
        initial_state.position[0],
        initial_state.position[1],
        )

    return ego_collision_object.collide(boundary_collision_object)


def process_single_file(rank: int, fns: List[str], duplicate: bool, output_dir: str, open_lane_ends: bool):

    meta_scenario_reset_dict = dict()
    processed_location_list = []

    for i, fn in enumerate(fns):
        print(f"{i + 1}/{len(fns)}", end="\r")
        scenario, planning_problem_set = CommonRoadFileReader(fn).open(lanelet_assignment=True)
        problem_dict = {"obstacle": scenario.obstacles, "planning_problem_set": planning_problem_set}
        map_name_id = parse_map_name(scenario.scenario_id)
        if map_name_id not in processed_location_list:
            processed_location_list.append(map_name_id)
            env_reset = generate_reset_config(scenario, open_lane_ends)
            if is_initial_collision(planning_problem_set, env_reset):
                continue
            meta_scenario_reset_dict[map_name_id] = env_reset

        if duplicate:
            with open(os.path.join(output_dir, "problem_train", f"{scenario.scenario_id}.pickle"), "wb") as f:
                pickle.dump(problem_dict, f)
            shutil.copytree(os.path.join(output_dir, "problem_train"), os.path.join(output_dir, "problem_test"))
        else:
            with open(os.path.join(output_dir, "problem", f"{scenario.scenario_id}.pickle"), "wb") as f:
                pickle.dump(problem_dict, f)

    os.makedirs(os.path.join(output_dir, f"meta_scenario_{rank}"), exist_ok=True)
    with open(os.path.join(output_dir, f"meta_scenario_{rank}", "meta_scenario_reset_dict.pickle"), "wb") as f:
        pickle.dump(meta_scenario_reset_dict, f)


def pickle_xml_scenarios(input_dir: str, output_dir: str, duplicate: bool = False, num_processes: int = 1,
                         open_lane_ends: bool = True):
    # makedir
    os.makedirs(output_dir, exist_ok=True)

    meta_scenario_path = "meta_scenario"
    fns = glob.glob(os.path.join(input_dir, "*.xml"))

    os.makedirs(os.path.join(output_dir, meta_scenario_path), exist_ok=True)
    if not duplicate:
        os.makedirs(os.path.join(output_dir, "problem"), exist_ok=True)
    else:
        os.makedirs(os.path.join(output_dir, "problem_train"), exist_ok=True)

    _start_time = time()
    num_scenarios_per_process = len(fns) // num_processes

    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(
                process_single_file,
                [
                    (
                        i,
                        fns[i * num_scenarios_per_process : (i + 1) * num_scenarios_per_process],
                        duplicate,
                        output_dir, open_lane_ends
                    )
                    for i in range(num_processes)
                ]
            )
    else:
        process_single_file(0, fns, duplicate, output_dir, open_lane_ends)
    meta_scenario_reset_dict = {}
    for i in range(num_processes):
        with open(os.path.join(output_dir, f"{meta_scenario_path}_{i}", "meta_scenario_reset_dict.pickle"), "rb") as f:
            meta_scenario_reset_dict.update(pickle.load(f))
        shutil.rmtree(os.path.join(output_dir, f"{meta_scenario_path}_{i}"))

    with open(os.path.join(output_dir, meta_scenario_path, "meta_scenario_reset_dict.pickle"), "wb") as f:
        pickle.dump(meta_scenario_reset_dict, f)

    print(len(meta_scenario_reset_dict.keys()))
    print("Took {}s".format(time() - _start_time))


if __name__ == "__main__":
    # get arguments
    args = get_args()

    pickle_xml_scenarios(
        args.input_dir,
        args.output_dir,
        duplicate=args.duplicate,
        num_processes=args.num_processes,
        open_lane_ends=True
    )
