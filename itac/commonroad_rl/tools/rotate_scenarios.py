"""
A tool to manipulate scenarios (rotate and translate) and save the modified version.
"""

__author__ = "Michael Feil"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Michael Feil"
__email__ = "michael.feil@tum.de"
__status__ = "Released"

import argparse
import logging
import os

import numpy as np
from typing import Tuple, List
import glob
import copy
import multiprocessing

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)


def get_args() -> Tuple[argparse.ArgumentParser, np.ndarray]:
    """Scan arguments"""
    parser = argparse.ArgumentParser(
        description="Analyzes goal_definitions of scenarios and configures goal_observations for model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem_dir_in",
        "-i",
        help="Path to input xml",
        type=str,
        default="/home/rl_students_ss21/data/inD/xml",
    )
    parser.add_argument(
        "--problem_dir_out",
        "-o",
        help="Path to folder of input xml",
        default="/home/rl_students_ss21/data/inD/rotated_xml",
    )
    parser.add_argument(
        "--rotations",
        "-r",
        help="list of rotations [0, 2Pi]",
        nargs="+",
        type=float,
        default=[np.pi / 2],
    )
    parser.add_argument(
        "--translations_x",
        "-tx",
        help="list of translations in x",
        nargs="+",
        type=float,
        default=[0],
    )
    parser.add_argument(
        "--translations_y",
        "-ty",
        help="list of translations in y",
        nargs="+",
        type=float,
        default=[0],
    )
    args = parser.parse_args()

    rot = np.asarray(args.rotations)
    tx = np.asarray(args.translations_x)
    ty = np.asarray(args.translations_y)

    if not (len(rot) == len(tx) == len(ty)):
        raise Exception(
            "length of rotations, translations_x, translations_y must be equal"
        )
    for i, angle in enumerate(rot):
        if angle > 2 * np.pi or angle < 0:
            LOGGER.warning(f"angle {i} is outside [0, {2*np.pi}]: {angle}")
    # check no angle is duplicated, i.e. no scenario is compute twice the same way.
    rot = rot % (2 * np.pi)
    rot_tx_ty = np.vstack((rot, tx, ty))
    unique = np.unique(rot_tx_ty, axis=1)
    if not np.array_equal(rot_tx_ty, unique):
        raise Exception(
            f"each set of (rotations, translations_x, translations_y) must be unique: \n {unique} \n {rot_tx_ty}"
        )
    args.rot = rot.tolist()
    return args, rot_tx_ty


def main() -> None:
    """
    Run rotation/translation of xmls with imports goal-requirements of scenarios and sets goal-observations in config
    argparse arguemnts
        --problem_dir_in
        --problem_dir_out
        optinal for rotation/translation also a three list of arguments:
            --translations_x
            --translations_y
            --rotations
    """
    args, rot_tx_ty = get_args()

    # Check arguments
    assert os.path.isdir(
        args.problem_dir_in
    ), f"The problem directory doesn't exist {args.problem_dir_in}"
    os.makedirs(args.problem_dir_out, exist_ok=True)
    xml_scenarios = glob.glob(args.problem_dir_in + "**/*.xml")

    # split on workers
    no_splits = 1 if len(xml_scenarios) < 100 else multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=no_splits) as pool:
        # each process uses a set of files
        pool.starmap(
            rotate_scenarios,
            [
                (xml_scenario_paths, rot_tx_ty, args.problem_dir_out)
                for xml_scenario_paths in np.array_split(xml_scenarios, no_splits)
            ],
        )


def rotate_scenarios(
    xml_scenario_paths: List[str], rot_tx_ty: np.ndarray, outdir: str
) -> None:
    """
    Run rotation/translation of xmls with imports goal-requirements of scenarios and sets goal-observations in config

    :param xml_scenario_paths: List of filepaths
    :param outdir: str, directory for output
    :param rot_tx_ty: np.ndarray, array of shape (n,3) with shape[1] is [rotation in rad, translation_x in m, translation_y in m],

    :return: None
    """
    try:
        print(
            f"start rotating {len(xml_scenario_paths)} sceanrios on {rot_tx_ty.shape[1]} settings"
        )
        for xml_scenario_path in xml_scenario_paths:
            # open scenario
            reader = CommonRoadFileReader(xml_scenario_path)
            scenario, planning_problem_set = reader.open()

            if len(rot_tx_ty.T) == 1:
                write_rotate_and_translate_scenario(
                    scenario,
                    planning_problem_set,
                    reader,
                    outdir,
                    np.squeeze(rot_tx_ty),
                    use_copy=False,
                )
            else:
                for rot_tx_ty_setting in rot_tx_ty.T:
                    print(rot_tx_ty_setting)
                    write_rotate_and_translate_scenario(
                        scenario,
                        planning_problem_set,
                        reader,
                        outdir,
                        rot_tx_ty_setting,
                    )
    except Exception as ex:
        LOGGER.exception(ex)


def write_rotate_and_translate_scenario(
    scenario: Scenario,
    planning_problem_set: PlanningProblemSet,
    reader: CommonRoadFileReader,
    outdir: str,
    rot_tx_ty_setting: List[float, float, float],
    use_copy=True,
) -> None:
    """
    creates rotated/translated xml for scenario and planning_problem_set in the folder outdir.
    Rotated by rot_tx_ty_setting[0]
    Translated by rot_tx_ty_setting[1:]

    :param scenario: Scenario to be rotated/translated
    :param planning_problem_set: PlanningProblemSet to be rotated/translated
    :param reader: CommonRoadFileReader, Instance
    :param outdir: str, directory for output
    :param rot_tx_ty_setting: List[rotation in rad, translation_x in m, translation_y in m],
    :param use_copy: default True, set to True if scenario used multiple times

    :return: None
    """
    if use_copy:
        scenario = copy.deepcopy(scenario)
        planning_problem_set = copy.deepcopy(planning_problem_set)
    scenario.translate_rotate(rot_tx_ty_setting[1:], rot_tx_ty_setting[0])
    planning_problem_set.translate_rotate(rot_tx_ty_setting[1:], rot_tx_ty_setting[0])

    writer = CommonRoadFileWriter(
        scenario,
        planning_problem_set,
        author=reader._get_author(),
        affiliation=reader._get_affiliation(),
        source=reader._get_source(),
    )

    new_filename = os.path.join(
        outdir,
        f"{scenario.benchmark_id}_{rot_tx_ty_setting[0]}_{rot_tx_ty_setting[1]}_{rot_tx_ty_setting[2]}.xml",
    )

    writer.write_to_file(
        filename=new_filename, overwrite_existing_file=OverwriteExistingFile.ALWAYS
    )


if __name__ == "__main__":
    main()
