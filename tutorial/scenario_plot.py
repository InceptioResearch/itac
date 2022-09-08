# path of the file to be opened
import sys
sys.path.append('../itac')
from itac import *

import cvxpy as cp
import math
import time
import signal
import argparse


def my_handler(signum, frame):
    exit(0)


signal.signal(signal.SIGINT, my_handler)  # 读取Ctrl+c信号


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="input commonroad xml")
    parser.add_argument("--input",
                        help="commonroad xml",
                        default="incep_map_commonroad_osbstacles.xml")
    scenario, planning_problem_set = CommonRoadFileReader(
        parser.parse_args().input).open()

    plt.figure(figsize=(50, 50))
    plt.ion()
    for t in range(0, 300):
        rnd = MPRenderer()
        window_size = 150
        rnd.plot_limits = [-window_size,
                           window_size, -window_size, window_size]
        scenario.draw(rnd, draw_params={
            'time_begin': t, 'focus_obstacle_id': scenario.dynamic_obstacles[0].obstacle_id})

        planning_problem_set.draw(rnd)
        rnd.render()
        plt.pause(scenario.dt)

