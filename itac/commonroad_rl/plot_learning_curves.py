"""
Module for plotting learning curves
"""
import os

os.environ["KMP_WARNINGS"] = "off"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging

logging.getLogger("tensorflow").disabled = True

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

import numpy as np
import math
import argparse
from commonroad_rl.utils_run.plot_util import smooth
from commonroad_rl.utils_run.plot_util import plot_results as plot_results_baselines
from commonroad_rl.utils_run.plot_util import load_results as load_results_baselines

try:
    from gym_monitor.util import *
except:
    pass

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

LATEX = False
LABELPAD = 15
FIGSIZE = (14, 12)

if LATEX:
    # use Latex font
    FONTSIZE = 28
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": 'lmodern',
        # blank entries should cause plots
        "font.sans-serif": [],  # ['Avant Garde'],              # to inherit fonts from the document
        # 'text.latex.unicode': True,
        "font.monospace": [],
        "axes.labelsize": FONTSIZE,  # LaTeX default is 10pt font.
        "font.size": FONTSIZE - 10,
        "legend.fontsize": FONTSIZE,  # Make the legend/label fonts
        "xtick.labelsize": FONTSIZE,  # a little smaller
        "ytick.labelsize": FONTSIZE,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts
            r"\usepackage[T1]{fontenc}",  # plots will be generated
            r"\usepackage[detect-all,locale=DE]{siunitx}",
        ]  # using this preamble
    }
    matplotlib.rcParams.update(pgf_with_latex)


def argsparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="log")
    parser.add_argument("--model_path", "-model", type=str, nargs="+", default=(),
                        help="(tuple) Relative path of the to be plotted model from the log folder")
    parser.add_argument("--no_render", "-nr", action="store_true", help="Whether to render images")
    parser.add_argument("-t", "--title", help="Figure title", type=str, default="result")
    # TODO: integrate sliding window size
    parser.add_argument("--smooth", action="store_true",
                        help="Smooth learning curves (average around a sliding window)")

    return parser.parse_args()


def ts2reward(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l) * 1e-6
    y_var = results.monitor.r.values

    return x_var, y_var


def ts2goal(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values) * 1e-6
    y_var = results.monitor.is_goal_reached.values

    return x_var, y_var


def ts2collision(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l) * 1e-6
    if hasattr(results.monitor, "valid_collision"):
        y_var = results.monitor.valid_collision
    else:
        y_var = results.monitor.is_collision

    return x_var, y_var


def ts2off_road(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l) * 1e-6
    if hasattr(results.monitor, "valid_off_road"):
        y_var = results.monitor.valid_off_road.values
    else:
        y_var = results.monitor.is_off_road.values

    return x_var, y_var


def ts2max_time(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    x_var = np.cumsum(results.monitor.l.values * 1e-6)
    y_var = smooth(results.monitor.is_time_out.values, radius=1)

    return x_var, y_var


def ts2goal_time(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l * 1e-6)
    y_var = [0]
    for i, is_goal_reached in enumerate(results.monitor.is_goal_reached.values):
        if is_goal_reached:
            y_var.append(results.monitor.current_episode_time_step.values[i])
        else:
            # y_var.append(results.monitor.max_episode_time_steps[i])
            y_var.append(y_var[-1])

    return x_var, np.array(y_var[1:])


def ts2friction_violation(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    # TODO: Implement xaxis
    x_var = np.cumsum(results.monitor.l.values * 1e-3)
    y_var = smooth(results.monitor.is_friction_violation.values, radius=50)

    return x_var, y_var


def ts2v_ego(results):
    # TODO: plot percentile
    # x_var = np.cumsum(results.monitor.l.values) * 1e-3
    # y_var = np.abs(results.monitor.v_ego_mean.values)
    n_percentile = np.zeros((11))
    v_ego_mean = np.array(results.monitor.v_ego_mean.values)
    for p in range(11):
        n_percentile[p] = np.percentile(v_ego_mean, p * 10)

    return np.array(range(11)), n_percentile


def ts2u1(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values) * 1e-6
    y_var = np.abs(results.monitor.u_cbf_1_sum.values) / results.monitor.l.values
    return x_var, y_var


def ts2u2(results):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """

    x_var = np.cumsum(results.monitor.l.values) * 1e-6
    y_var = np.abs(results.monitor.u_cbf_1_sum.values) / results.monitor.l.values
    return x_var, y_var


PLOT_DICT = {
    "Total Reward": ts2reward,
    "Goal-Reaching Rate": ts2goal,
    # "Collision Rate": ts2collision,
    # "Off-Road Rate": ts2off_road,
    # "Time-Out Rate": ts2max_time,
    # "Goal Reaching Time": ts2goal_time,
    # "Mean ego velocity": ts2v_ego,
    # "Total Robustness reward": ts2monitor_reward,
    # "Total Sparse reward": ts2gym_reward,
    # "Min Robustness": ts2min_robustness,
    # "Max Robustness": ts2max_robustness,
    # "Avg. Step Robustness reward": ts2monitor_reward_step,
    # "Avg. Step Sparse reward": ts2gym_reward_step,
    # "True traffic rule violation": ts2rule_violation,
    # "Valid traffic rule violation": ts2valid_rule_violation,
    # "Active step robustness reward": ts2active_step_robustness,
    # "Active total robustness reward": ts2active_total_robustness,
    # "Step robustness reward vs num violation": violation2step_robustness_tmp,
    "$|u_1 - u_\mathrm{RL1}|$   [rad/$\mathrm{s}^2$]": ts2u1,
    "$|u_2 - u_\mathrm{RL2}|$   [m/$\mathrm{s}^2$]": ts2u2
    # "Friction violation": ts2friction_violation,
}


def group_fn(results):
    return os.path.basename(results.dirname)


def main():
    args = argsparser()

    log_dir = args.log_folder
    model_paths = tuple(args.model_path)

    num_of_columns = 2
    num_of_rows = math.ceil(len(PLOT_DICT) / num_of_columns)

    for idx, model in enumerate(model_paths):
        fig, axarr = plt.subplots(num_of_rows, num_of_columns, sharex=False, squeeze=True, figsize=FIGSIZE, dpi=100)
        results = load_results_baselines(os.path.join(log_dir, model))

        for i, (k, xy_fn) in enumerate(PLOT_DICT.items()):
            legend = i == 0
            plot_line = False  # i >1
            set_y_lim = "Rate" in k
            try:
                fig, axarr = plot_results_baselines(
                    results, fig, axarr,
                    nrows=num_of_rows, ncols=num_of_columns, xy_fn=xy_fn,
                    idx_row=i // num_of_columns, idx_col=i % num_of_columns,
                    average_group=False, resample=args.smooth,
                    group_fn=group_fn,
                    xlabel="Training Steps * 1e6", ylabel=k,
                    # xlabel="\\textbf{Training Steps * 1000}",
                    # ylabel="\\textbf{" + k + "}",
                    labelpad=LABELPAD,
                    legend_outside=True,
                    legend=legend, plot_line=plot_line, set_y_lim=set_y_lim)
            except AttributeError:
                continue
        format = "pdf"
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(log_dir, model, f"{model}.{format}"), format=format, bbox_inches='tight')
        LOGGER.info(f"Saved {model}.{format} to {log_dir}/{model}")  # , figure, os.path.join(log_dir, model))


if __name__ == "__main__":
    main()
