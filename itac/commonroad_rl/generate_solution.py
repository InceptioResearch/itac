"""
Module for solving CommonRoad scenarios using trained models
"""
import argparse
import logging
import os

from commonroad_dc.feasibility.feasibility_checker import FeasibilityException

os.environ["KMP_WARNINGS"] = "off"
os.environ["KMP_AFFINITY"] = "none"
logging.getLogger("tensorflow").disabled = True
from typing import Union, List
import gym
import yaml
import warnings
import numpy as np
from commonroad.common.solution import (PlanningProblemSolution, Solution, CostFunction, CommonRoadSolutionWriter,
                                        TrajectoryType, )
from commonroad.scenario.scenario import ScenarioID, Scenario
from commonroad.common.solution import VehicleModel
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.trajectory import Trajectory, State
from commonroad_dc.feasibility.solution_checker import valid_solution, _simulate_trajectory_if_input_vector, \
    GoalNotReachedException, CollisionException
from gym import Env

from commonroad_rl.utils_run.vec_env import CommonRoadVecEnv
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.utils_run.utils import load_model_and_vecnormalize


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluates PPO2 trained model with specified test scenarios",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--algo", type=str, default="ppo2")
    parser.add_argument("--test_path", "-i", type=str, help="Path to pickled test scenarios",
                        default=PATH_PARAMS["test_reset_config"], )
    parser.add_argument("--model_path", "-model", type=str, help="Path to trained model", required=True)
    parser.add_argument("--multiprocessing", "-mpi", action="store_true")
    parser.add_argument("--solution_path", "-sol", type=str,
                        help="Path to the desired directory of the generated solution files",
                        default=PATH_PARAMS["commonroad_solution"], )
    parser.add_argument("--cost_function", "-cost", type=str, default="JB1")
    parser.add_argument("--hyperparam_filename", "-f", type=str, default="model_hyperparameters.yml")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Debug mode (overrides verbose mode)")
    parser.add_argument("--config_filename", "-config_f", type=str, default="environment_configurations.yml", )

    return parser


def create_solution(commonroad_env: CommonroadEnv, output_directory: str, cost_function: str,
                    computation_time: Union[float, None] = None, processor_name: Union[str, None] = "auto",
                    input_trajectory: bool = True) -> bool:
    """
    Creates a CommonRoad solution file from the terminated environment

    :param commonroad_env: The terminated environment
    :param output_directory: The directory where the solution file will be written
    :param cost_function: The cost function to be used during the solution generation
    :param computation_time: The elapsed time during solving the scenario
    :param processor_name: The name of the used processor
    :return: True if the solution is valid and the solution file is written
    """
    os.makedirs(output_directory, exist_ok=True)
    list_state = list()
    ego_vehicle = commonroad_env.ego_action.vehicle
    for state in ego_vehicle.state_list:
        model_type = ego_vehicle.vehicle_model
        if model_type == VehicleModel.PM:
            if input_trajectory:
                if state.time_step == 0:
                    # skipping first accelerations since they are not executed
                    continue
                kwarg = {"time_step": state.time_step-1, # shift time step for input states
                         "acceleration": state.acceleration, "acceleration_y": state.acceleration_y}
            else:
                kwarg = {"position": state.position, "velocity": state.velocity, "velocity_y": state.velocity_y,
                         "time_step": state.time_step, "orientation": np.arctan2(state.velocity_y, state.velocity)}
        elif model_type == VehicleModel.SEMI_TRAILER:
            kwarg = {"position": state.position, "velocity": state.velocity, "steering_angle": state.steering_angle,
                     "orientation": state.orientation, "hitch_angle":state.hitch_angle,"position_trailer":state.position_trailer,"yaw_angle_trailer":state.yaw_angle_trailer, "time_step": state.time_step}
        else:
           kwarg = {"position": state.position, "velocity": state.velocity, "steering_angle": state.steering_angle,
                     "orientation": state.orientation, "time_step": state.time_step}
        list_state.append(State(**kwarg))

    trajectory = Trajectory(initial_time_step=list_state[0].time_step, state_list=list_state)

    planning_problem_solution = PlanningProblemSolution(
            planning_problem_id=commonroad_env.planning_problem.planning_problem_id, vehicle_model=model_type,
            vehicle_type=ego_vehicle.vehicle_type, cost_function=CostFunction[cost_function], trajectory=trajectory, )
    scenario_id = ScenarioID.from_benchmark_id(commonroad_env.benchmark_id, "2020a")
    solution = Solution(scenario_id=scenario_id,  # TODO: wrong usage, fix in commonroad environment to use scenario_id
                        planning_problem_solutions=[planning_problem_solution], computation_time=computation_time,
                        processor_name=processor_name, )

    # # =================================== DEBUG ============================================================
    # planned_list = []
    # for state in ego_vehicle.state_list:
    #     kwargs = {"position": state.position, "velocity": state.velocity, "velocity_y": state.velocity_y,
    #               "time_step": state.time_step, "orientation": np.arctan2(state.velocity_y, state.velocity),
    #                  "acceleration": state.acceleration, "acceleration_y": state.acceleration_y}
    #     planned_list.append(State(**kwargs))
    # write_solution_file(commonroad_env.scenario, PlanningProblemSet([commonroad_env.planning_problem]),
    #                     solution, output_directory, planned_list=planned_list)
    # # ===================================  END ============================================================
    # Check the solution
    LOGGER.debug(f"Check solution of {commonroad_env.benchmark_id}")
    solution_valid, results = valid_solution(commonroad_env.scenario,
                                             PlanningProblemSet([commonroad_env.planning_problem]), solution, )
    if solution_valid:
        # write solution to a xml file
        csw = CommonRoadSolutionWriter(solution=solution)
        csw.write_to_file(output_path=output_directory, overwrite=True)
        LOGGER.info(f"Solution feasible, {commonroad_env.benchmark_id} printed to {output_directory}")
    else:
        LOGGER.info(f"Unable to create solution, invalid trajectory!")
    return solution_valid


def solve_scenarios(test_path: str, model_path: str, algo: str, solution_path: str, cost_function: str,
                    multiprocessing: bool = False, hyperparam_filename: str = "model_hyperparameters.yml",
                    config_filename: str = "environment_configurations.yml", env_id: str = "commonroad-v1",
                    render: bool = False
                    ) -> List[bool]:
    """
    Solve a batch of scenarios using a trained model

    :param test_path: Path to the test files
    :param model_path: Path to the trained model
    :param algo: the used RL algorithm
    :param solution_path: Path to the folder where the solution files will be written
    :param cost_function: The cost function to be used during the solution generation
    :param multiprocessing: Indicates whether using multiprocessing or not (default is False)
    :param hyperparam_filename: The filename of the hyperparameters (default is model_hyperparameters.yml)
    :param config_filename: The environment configuration file name (default is environment_configurations.yml)
    :return: List of boolean values which indicates whether a scenario has been successfully solved or not
    """

    # mpi for parallel processing
    if multiprocessing:
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None
        if MPI is not None:
            rank = MPI.COMM_WORLD.Get_rank()
            test_path = os.path.join(test_path, str(rank))

    # Get environment keyword arguments including observation and reward configurations
    config_fn = os.path.join(model_path, config_filename)
    with open(config_fn, "r") as f:
        env_kwargs = yaml.load(f, Loader=yaml.Loader)

    env_kwargs["termination_configs"].update({"terminate_on_friction_violation": False})
    env_kwargs["goal_configs"].update({"relax_is_goal_reached": False})
    env_kwargs.update(
        {"meta_scenario_path": os.path.join(test_path, 'meta_scenario'),
         "test_reset_config_path": os.path.join(test_path, 'problem_test')}
    )

    def env_fn():
        return gym.make(env_id, play=True, **env_kwargs)

    env = CommonRoadVecEnv([env_fn])
    results = []

    def on_reset_callback(env: Union[Env, CommonroadEnv], elapsed_time: float):
        if env.observation_dict["is_goal_reached"][0] and not env.observation_dict["is_collision"][0]:
            LOGGER.info("Goal reached")
            os.makedirs(solution_path, exist_ok=True)
            solution_valid = create_solution(env, solution_path, cost_function, computation_time=elapsed_time)
        else:
            goal = env.planning_problem.goal
            state = env.ego_action.vehicle.state
            goal.is_reached(state)
            LOGGER.info("Goal not reached")
            LOGGER.info(f"Termination reason: {env.termination_reason}")
            solution_valid = False

        results.append(solution_valid)

    env.set_on_reset(on_reset_callback)

    # Load model hyperparameters:
    hyperparam_fn = os.path.join(model_path, hyperparam_filename)
    with open(hyperparam_fn, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.Loader)

    normalize = hyperparams["normalize"]

    model, env = load_model_and_vecnormalize(model_path, algo, normalize, env)

    obs = env.reset()
    if render:
        env.render()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        try:
            obs, reward, done, info = env.step(action)
            if render:
                env.render()
            LOGGER.info(f"Step: {env.venv.envs[0].current_step}, \tReward: {reward}, \tDone: {done}")
            if info[0].get("out_of_scenarios", False):
                break
        except IndexError as e:
            # If the environment is done, it will be reset.
            # However the reset throws an exception if there are no more scenarios to be solved.
            LOGGER.info(f"Cannot choose more scenarios to be solved, msg: {e}")
            break
    return results


def write_solution_file(scenario: Scenario,
                        planning_problem_set: PlanningProblemSet,
                        solution: Solution, output_directory: str, planned_list: List[State]) -> bool:
    # print(f"Check validity of solution for {scenario.benchmark_id}")
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    try:
        solution_valid, results = valid_solution(scenario, planning_problem_set, solution)
    except FeasibilityException as e:
        warnings.warn(str(e))
        planned_input_list = solution.planning_problem_solutions[0].trajectory.state_list
        # draw inputs
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(list(range(len(planned_input_list))),
                 [state.steering_angle_speed for state in planned_input_list], color="black", label="planned")
        plt.plot([0, len(planned_input_list)], [veh_params.steering.v_min, veh_params.steering.v_min],
                 color="red", label="bounds")
        plt.plot([0, len(planned_input_list)], [veh_params.steering.v_max, veh_params.steering.v_max],
                 color="red", label="bounds")
        plt.legend()
        plt.ylabel("steering angle velocity")
        plt.subplot(2, 1, 2)
        plt.plot(list(range(len(planned_input_list))),
                 [state.acceleration for state in planned_input_list], color="black", label="planned")
        plt.plot([0, len(planned_input_list)], [-veh_params.longitudinal.a_max, -veh_params.longitudinal.a_max],
                 color="red", label="bounds")
        plt.plot([0, len(planned_input_list)], [veh_params.longitudinal.a_max, veh_params.longitudinal.a_max],
                 color="red", label="bounds")
        plt.ylabel("acceleration")
        plt.show()

        return

    except (CollisionException, GoalNotReachedException) as e:
        warnings.warn(str(e))
        csw = CommonRoadSolutionWriter(solution=solution)
        csw.write_to_file(output_path=output_directory, overwrite=True)
        planning_problem_solution = solution.planning_problem_solutions[0]
        _, reconstructed_trajectory = _simulate_trajectory_if_input_vector(planning_problem_set,
                                                                           planning_problem_solution,
                                                                           scenario.dt)
        for state in reconstructed_trajectory.state_list:
            state.orientation = np.arctan2(state.velocity_y, state.velocity)

        # plt.figure(figsize=(20, 20))
        # # plt.subplot(2, 1, 1)
        # draw_object(scenario, plot_limits=[-50, 50, -50, 50])
        # # draw_object(planning_problem_set)
        # planned_state_list = planned_list
        # plt.plot(*np.array([state.position for state in planned_state_list]).T,
        #          color='black', marker='o', markersize=0.2, zorder=20, linewidth=0.5, label='planned trajectories')
        # plt.plot(*np.array([state.position for state in reconstructed_trajectory.state_list]).T,
        #          color='blue', marker='o', markersize=0.2, zorder=20, linewidth=0.5,
        #          label='reconstructed trajectories')
        # plt.legend()
        # plt.gca().set_aspect('equal')
        # plt.subplot(2, 1, 2)
        # plt.title("orientation")
        # plt.plot([state.orientation for state in planned_state_list], color="black")
        # plt.plot([state.orientation for state in reconstructed_trajectory.state_list], color="blue")

        # plt.savefig("trajectories.png", dpi=300)
        # plt.show()
        return

    if solution_valid:
        csw = CommonRoadSolutionWriter(solution=solution)
        csw.write_to_file(output_path=output_directory, overwrite=True)
    else:
        # draw planned and reconstructed inputs/trajectories
        planning_problem_solution = solution.planning_problem_solutions[0]
        if not planning_problem_solution.trajectory_type is TrajectoryType.Input:
            # draw inputs
            warnings.warn("Planned inputs and reconstructed inputs deviate too much!")
            planned_input_list = planned_list
            reconstructed_input_list = results[1][1].state_list
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(list(range(len(planned_input_list))),
                     [state.steering_angle_speed for state in planned_input_list], color="black", label="planned")
            plt.plot(list(range(len(reconstructed_input_list))),
                     [state.steering_angle_speed for state in reconstructed_input_list], color="blue",
                     label="reconstructed")
            plt.legend()
            plt.ylabel("steering angle velocity")
            plt.subplot(2, 1, 2)
            plt.plot(list(range(len(planned_input_list))),
                     [state.acceleration for state in planned_input_list], color="black", label="planned")
            plt.plot(list(range(len(reconstructed_input_list))),
                     [state.acceleration for state in reconstructed_input_list], color="blue", label="reconstructed")
            plt.ylabel("acceleration")
            plt.show()


def load_and_check_solution(scenario_file_path: str, solution_file_path: str) -> bool:
    """
    Function to check whether a solution file is valid for a scenario or not

    :param scenario_file_path: Path to the scenario
    :param solution_file_path: Path to the solution
    :return: True if the solution is valid
    """
    from commonroad.common.file_reader import CommonRoadFileReader
    from commonroad.common.solution import CommonRoadSolutionReader

    scenario, pp = CommonRoadFileReader(scenario_file_path).open()
    solution = CommonRoadSolutionReader.open(solution_file_path)
    solution_valid, results = valid_solution(scenario, pp, solution)
    if solution_valid is True:
        LOGGER.info(f"Solution feasible")
    else:
        LOGGER.info(f"Unable to create solution, invalid trajectory!")
    return solution_valid


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.INFO)
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    solve_scenarios(args.test_path, args.model_path, args.algo, args.solution_path, args.cost_function,
                    multiprocessing=args.multiprocessing, hyperparam_filename=args.hyperparam_filename,
                    config_filename=args.config_filename, )
