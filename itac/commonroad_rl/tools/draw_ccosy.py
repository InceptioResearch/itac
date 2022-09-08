"""
Module for drawing the ccosy
"""

__author__ = "Brian Liao, Niels Muendler, Peter Kocsis"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

import argparse
import random
from typing import List

import commonroad_dc.pycrcc as pycrcc
import commonroad_dc.pycrccosy as pycrccosy
# import pycrcc
import imageio
# import commonroad_ccosy.visualization.draw_dispatch

import numpy as np

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad_dc.geometry.util import resample_polyline
from commonroad_dc.collision.visualization import draw_dispatch as crdc_draw_dispatch
from matplotlib import gridspec

from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv, restore_scenario
from commonroad_rl.gym_commonroad.observation.goal_observation import GoalObservation
from commonroad_rl.gym_commonroad.action.vehicle import Vehicle

import matplotlib
# try:
#     matplotlib.use("TkAgg")
# except:
#     matplotlib.use("AGG")
import matplotlib.pyplot as plt

VEHICLE_PARAMS = {
    "vehicle_type": 2,  # VehicleType.BMW_320i
    "vehicle_model": 0,
}


def argsparser():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", "-d", action="store_true")
    parser.add_argument("--animate", "-a", action="store_true")
    return parser.parse_args()


def get_lanelet_orientation_at_state(lanelet: Lanelet, state: State):
    """
    Approximates the lanelet orientation with the two closest point to the given state

    :param lanelet: Lanelet on which the orientation at the given state should be calculated
    :param state: State where the lanelet's orientation should be calculated
    :return: An orientation in interval [-pi,pi]
    """
    # TODO optimize more for speed

    center_vertices = lanelet.center_vertices

    position_diff = []
    for idx in range(len(center_vertices) - 1):
        vertex1 = center_vertices[idx]
        position_diff.append(np.linalg.norm(state.position - vertex1))

    closest_vertex_index = position_diff.index(min(position_diff))

    vertex1 = center_vertices[closest_vertex_index, :]
    vertex2 = center_vertices[closest_vertex_index + 1, :]
    direction_vector = vertex2 - vertex1
    return np.arctan2(direction_vector[1], direction_vector[0])


def draw_ccosy(
        scenario: Scenario,
        planning_problem: PlanningProblem,
        ccosy_list: List[pycrccosy.CurvilinearCoordinateSystem],
        ego_state: State = None,
):
    """
    Draws all ccosy to the current plot

    :param scenario: The scenario to be plotted
    :param planning_problem: The planning problem to be plotted
    :param ccosy_list: The list of the ccosies to be plotted
    :param ego_state: The state of the ego vehicle to be plotted (optional)
    """
    for global_cosy in ccosy_list:
        draw_object(
            global_cosy.get_segment_list()
        )
    draw_object(
        scenario,
        draw_params={
            "time_begin": 0,
            "scenario": {
                "lanelet_network": {
                    "lanelet": {
                        "show_label": False,
                        "fill_lanelet": False,
                    }
                },
            },
            "planning_problem_set": {
                "planning_problem": {
                    "goal_region": {
                        "draw_shape": True,
                        "shape": {
                            "polygon": {
                                "opacity": 0.5,
                                "linewidth": 0.5,
                                "facecolor": "#f1b514",
                                "edgecolor": "#302404",
                                "zorder": 15,
                            },
                            "rectangle": {
                                "opacity": 0.5,
                                "linewidth": 0.5,
                                "facecolor": "#f1b514",
                                "edgecolor": "#302404",
                                "zorder": 15,
                            },
                            "circle": {
                                "opacity": 0.5,
                                "linewidth": 0.5,
                                "facecolor": "#f1b514",
                                "edgecolor": "#302404",
                                "zorder": 15,
                            },
                        },
                    },
                },
            },
        },
    )
    draw_object(planning_problem)

    ego_vehicle = Vehicle.create_vehicle(VEHICLE_PARAMS)
    ego_params = ego_vehicle.params

    # Draw ego
    if ego_state is None:
        ego_state = planning_problem.initial_state
    collision_object = pycrcc.RectOBB(
        ego_params.l / 2,
        ego_params.w / 2,
        ego_state.orientation,
        ego_state.position[0],
        ego_state.position[1],
    )
    crdc_draw_dispatch.draw_object(
        collision_object, draw_params={"collision": {"facecolor": "red"}}
    )
    plt.gca().set_aspect("equal")


def animate_goal_obs(goal_obs: GoalObservation, title: str):
    """
    Creates animation about the ccosy and the ego vehicle moving over the ccosies
    Next to the scenario the goal related observations are plotted as well

    :param goal_obs: The GoalObservation object handles the goal related observations
    :param title: The title of the plot
    """
    # Create states over the route which will be animated
    states_to_test = []
    merged_lanelets = goal_obs.get_reference_lanelet_route()
    for lanelet in merged_lanelets:
        polyline = lanelet.center_vertices
        polyline = resample_polyline(polyline, step=2.0)

        # Add some point before and after the lane for better visualization
        normal_vector = polyline[1, :] - polyline[0, :]
        points_before_the_lane = [
            (polyline[0, :] - step * normal_vector) for step in reversed(range(1, 10))
        ]

        normal_vector = polyline[-1, :] - polyline[-2, :]
        points_after_the_lane = [
            (polyline[-1, :] + step * normal_vector) for step in range(1, 10)
        ]

        polyline = np.vstack((points_before_the_lane, polyline, points_after_the_lane))

        states_to_test.extend(
            [
                State(
                    position=position,
                    orientation=get_lanelet_orientation_at_state(
                        lanelet, State(position=position)
                    ),
                )
                for position in polyline
            ]
        )

    plots = []
    long_distances = []
    lat_distances = []
    long_lane_distances = []
    num_states = len(states_to_test)

    fig = plt.figure(figsize=(25, 10))
    fig.suptitle(title)

    # Plot each steps
    for state_to_test in states_to_test:
        long_dist, lat_dist = goal_obs.get_long_lat_distance_to_goal(
            state_to_test.position
        )
        long_lane_change_dist = goal_obs.get_long_distance_until_lane_change(
            state_to_test
        )

        long_distances.append(long_dist)
        lat_distances.append(lat_dist)
        long_lane_distances.append(long_lane_change_dist)

        fig.clf()
        # Left side of the animation: The scenario
        outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1
        )
        ax = plt.Subplot(fig, inner[0])
        fig.add_subplot(ax)
        draw_ccosy(
            goal_obs.scenario,
            goal_obs.planning_problem,
            goal_obs.navigator.ccosy_list,
            state_to_test,
        )

        # Right side of the animation: The plots
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[1], wspace=0.1, hspace=0.1
        )
        ax = plt.Subplot(fig, inner[0])
        ax.set_xlim([0, num_states - 1])
        ax.set_ylim([min([min(long_distances), 0]) - 0.1, max(long_distances) + 0.1])
        fig.add_subplot(ax)
        plt.plot(long_distances, label="d_long")
        ax.legend()

        ax = plt.Subplot(fig, inner[1])
        ax.set_xlim([0, num_states - 1])
        ax.set_ylim([min([min(lat_distances), 0]) - 0.1, max(lat_distances) + 0.1])
        fig.add_subplot(ax)
        plt.plot(lat_distances, label="d_lat")
        ax.legend()

        ax = plt.Subplot(fig, inner[2])
        ax.set_xlim([0, num_states - 1])
        ax.set_ylim(
            [min([min(long_lane_distances), 0]) - 0.1, max(long_lane_distances) + 0.1]
        )
        fig.add_subplot(ax)
        plt.plot(long_lane_distances, label="d_long_lane")
        ax.legend()

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plots.append(image)

    # Create gif
    imageio.mimsave(f"./goal_dist_{title}.gif", plots, fps=5)


def main(arguments):
    import commonroad_rl
    print(commonroad_rl.__file__)
    cr_env = CommonroadEnv()

    all_problems = list(cr_env.all_problem_dict.items())
    num_all_problems = len(all_problems)
    for i, (benchmark_id, problem_dict) in enumerate(all_problems):

        meta_scenario = cr_env.problem_meta_scenario_dict[benchmark_id]
        obstacle_list = problem_dict["obstacle"]
        scenario = restore_scenario(meta_scenario, obstacle_list)

        planning_problems = list(
            problem_dict["planning_problem_set"].planning_problem_dict.values()
        )
        num_planning_problems = len(planning_problems)
        if num_planning_problems > 1:
            print(
                f"There are {num_planning_problems} planning problems in the scenario {benchmark_id}"
            )

        planning_problem = random.choice(planning_problems)

        try:
            goal_obs = GoalObservation(scenario, planning_problem)

            state_to_test = planning_problem.initial_state
            long_lat_dist = goal_obs.get_long_lat_distance_to_goal(
                state_to_test.position
            )
            long_lane_change_dist = goal_obs.get_long_distance_until_lane_change(
                state_to_test
            )

            print(
                f"{i + 1}/{num_all_problems} DONE {benchmark_id} - "
                f"Long-lat distance: {long_lat_dist}, Lane change dist: {long_lane_change_dist}"
            )
        except ValueError as ex:
            print(
                f"{i + 1}/{num_all_problems} ------------- ERROR ------------ in {benchmark_id}:{ex}"
            )
            raise ex

        if arguments.animate:
            animate_goal_obs(goal_obs, benchmark_id)

        if arguments.draw:
            fig = plt.figure(figsize=(25, 10))
            fig.suptitle(benchmark_id)
            draw_ccosy(scenario, planning_problem, goal_obs.navigator.ccosy_list)
            plt.show()


if __name__ == "__main__":
    args = argsparser()
    main(args)
