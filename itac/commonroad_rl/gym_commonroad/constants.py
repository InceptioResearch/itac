"""
Module for CommonRoad Gym environment related constants
"""
# Lanelet parameters
import os
from commonroad_rl.gym_commonroad.utils.scenario_io import get_project_root

# Visualization parameters
DRAW_PARAMS = {
    "draw_shape": True,
    "draw_icon": True,
    "draw_bounding_box": True,
    "trajectory_steps": 2,
    "show_label": False,
    "occupancy": {
        "draw_occupancies": 0,
        "shape": {
            "rectangle": {
                "opacity": 0.2,
                "facecolor": "#fa0200",
                "edgecolor": "#0066cc",
                "linewidth": 0.5,
                "zorder": 18,
            }
        },
    },
    "shape": {
        "rectangle": {
            "opacity": 1.0,
            "facecolor": "#fa0200",
            "edgecolor": "#831d20",
            "linewidth": 0.5,
            "zorder": 20,
        }
    },
}

# Path
ROOT_STR = str(get_project_root())
ITAC_ROOT = os.getenv('ITAC_ROOT')

PATH_PARAMS = {
    "visualization": ITAC_ROOT + '/outputs/img',
    "pickles": ITAC_ROOT + '/scenarios/highway_test_pickle',
    "meta_scenario": ITAC_ROOT + '/scenarios/highway_test_pickle/meta_scenario',
    "train_reset_config": ITAC_ROOT + '/scenarios/highway_test_pickle/problem',
    "test_reset_config": ITAC_ROOT + '/scenarios/highway_test_pickle/problem',
    "log": ITAC_ROOT + '/outputs/log',
    "commonroad_solution": ITAC_ROOT + "/outputs/cr_solution",
    "configs": {"commonroad-v1": ITAC_ROOT + "/itac/commonroad_rl/gym_commonroad/configs.yaml",
                "cr_monitor-v0": ITAC_ROOT + "/itac/commonroad_rl/gym_commonroad/configs.yaml"}
}

# PATH_PARAMS = {
#     "visualization": ROOT_STR + "/img",
#     "pickles": ROOT_STR + "/pickles",
#     "meta_scenario": ROOT_STR + "/pickles/meta_scenario",
#     "train_reset_config": ROOT_STR + "/pickles/problem_train",
#     "test_reset_config": ROOT_STR + "/pickles/problem_test",
#     "log": ROOT_STR + "/log",
#     "commonroad_solution": ROOT_STR + "/cr_solution",
#     "configs": {"commonroad-v1": ROOT_STR + "/commonroad_rl/gym_commonroad/configs.yaml",
#                 "cr_monitor-v0": ROOT_STR + "/commonroad_rl/gym_commonroad/configs.yaml"}
# }
