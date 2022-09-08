from typing import Tuple

import numpy as np
from commonroad.common.solution import TrajectoryType
from commonroad.scenario.trajectory import Trajectory, State
from scipy.optimize import minimize

from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, PointMassDynamics, VehicleDynamicsException


class FeasibilityException(Exception):
    pass


class StateTransitionException(FeasibilityException):
    pass


class FeasibilityObjectiveException(StateTransitionException):
    pass


class FeasibilityCriteriaException(StateTransitionException):
    pass


class TrajectoryFeasibilityException(FeasibilityException):
    pass


class InputVectorFeasibilityException(FeasibilityException):
    pass


def _approx_orientation_vector(o: float) -> np.array:
    """ Converts given orientation to approximate normed vector """
    return np.clip(np.array([np.cos(o), np.sin(o)]), -1.0, 1.0)  # clip because of the floating point errors


def _angle_vector_diff(v1: np.array, v2: np.array) -> float:
    """ Returns angle between the two provided orientation vectors """
    dot = np.clip(v1.dot(v2), -1.0, 1.0)  # clip because of the floating point errors
    return np.arccos(dot)


def _angle_diff(o1: float, o2: float) -> float:
    """ Returns the difference between two angles by converting them to approx orientation vector. """
    diff = _angle_vector_diff(
        _approx_orientation_vector(o1),
        _approx_orientation_vector(o2)
    )
    return diff


def _adjust_state_bounds(x: np.ndarray, vehicle: VehicleDynamics, ftol: float = 1e-8):
    """
    Adjusts the state bounds by subtracting/adding small error in order to use the state in state
    transition feasibility.

    When vehicle model's state is at the state bounds in terms of velocity and steering angle, there
    will be more than one valid input that outputs the same next state. By subtracting/adding small error
    to the values on bounds, we try to reduce the number of valid inputs (to a single hopefully).
    (See vehicle model documentation)
    """
    if isinstance(vehicle, PointMassDynamics):
        velocity_x_adjustment = np.sign(x[2]) * ftol * 10
        if x[2] == vehicle.parameters.longitudinal.v_max or x[2] == vehicle.parameters.longitudinal.v_min:
            x[2] -= velocity_x_adjustment

        velocity_y_adjustment = np.sign(x[3]) * ftol * 10
        if x[3] == vehicle.parameters.longitudinal.v_max or x[3] == vehicle.parameters.longitudinal.v_min:
            x[3] -= velocity_y_adjustment
        return x

    steering_angle_adjustment = np.sign(x[2]) * ftol * 10
    if x[2] == vehicle.parameters.steering.max or x[2] == vehicle.parameters.steering.min:
        x[2] -= steering_angle_adjustment

    velocity_adjustment = np.sign(x[3]) * ftol * 10
    if x[3] == vehicle.parameters.longitudinal.v_max or x[3] == vehicle.parameters.longitudinal.v_min:
        x[3] -= velocity_adjustment

    return x


def position_orientation_objective(u: np.array, x0: np.array, x1: np.array, dt: float,
                                   vehicle_dynamics: VehicleDynamics, ftol: float = 1e-8,
                                   e: np.array = np.array([2e-2, 2e-2, 3e-2])) -> float:
    """
    Position-Orientation objective function to be minimized for the state transition feasibility.

    Simulates the next state using the inputs and calculates the norm of the difference between the
    simulated next state and actual next state. Position, velocity and orientation state fields will
    be used for calculation of the norm.

    :param u: input values
    :param x0: initial state values
    :param x1: next state values
    :param dt: delta time
    :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
    :param ftol: ftol parameter used by the optimizer
    :param e: error margin, function will return norm of the error vector multiplied with 100 as cost
        if the input violates the friction circle constraint or input bounds.
    :return: cost
    """
    try:
        x0_adjusted = _adjust_state_bounds(x0, vehicle_dynamics, ftol)
        x1_adjusted = _adjust_state_bounds(x1, vehicle_dynamics, ftol)
        x1_sim = vehicle_dynamics.forward_simulation(x0_adjusted, u, dt, throw=False)

        # if the input violates the constraints
        if x1_sim is None:
            return np.linalg.norm(e * 100)

        if isinstance(vehicle_dynamics, PointMassDynamics):
            diff = np.subtract(x1_adjusted, x1_sim)
            cost = np.linalg.norm(diff)
            return cost

        pos_diff = np.subtract(x1_adjusted[:2], x1_sim[:2])
        # steering_diff = _angle_diff(x1_adjusted[2], x1_sim[2])
        vel_diff = x1_adjusted[3] - x1_sim[3]
        orient_diff = _angle_diff(x1_adjusted[4], x1_sim[4])
        diff = np.append(pos_diff, np.array([vel_diff, orient_diff]))
        cost = np.linalg.norm(diff)
        if not isinstance(vehicle_dynamics, PointMassDynamics):
            diff_tmp = diff[:3]
            if np.any(np.greater(np.abs(diff_tmp), e) ):
                cost += np.linalg.norm(np.abs(diff_tmp[np.greater(np.abs(diff_tmp), e)]) - e[np.greater(np.abs(diff_tmp), e)])
        return cost

    except VehicleDynamicsException as ex:
        msg = f'An exception occurred during the calculation of position-orientation objective!\n' \
              f'x0: {x0}\nx1: {x1}\nu: {u}\nVehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}'
        raise FeasibilityObjectiveException(msg) from ex


def position_orientation_feasibility_criteria(x: np.array, x_sim: np.array, vehicle_dynamics: VehicleDynamics,
                                              e: np.array = np.array([2e-2, 2e-2, 3e-2]), d: int = 4) -> bool:
    """
    Position-Orientation feasibility criteria to be checked between the real next state and the simulated
    next state in the state transition feasibility testing after a valid input has been found.

    Checks whether the position and orientation difference is within acceptable between actual state and
    simulated state.

    :param x: real next state
    :param x_sim: simulated next state
    :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
    :param e: error margin, function will return False if the positional difference between the simulated
        next state and the actual next state is bigger then error margin.
    :param d: decimal points where the difference values are rounded up to in order to avoid floating point
        errors set it based on the error margin, i.e e=0.02, d=3
    :return: True if the positional difference is below error margin.
    """
    if not len(e) == 3:
        msg = f'Invalid error vector for position-orientation feasibility criteria! e: {e}'
        raise FeasibilityCriteriaException(msg)

    if not len(x) >= 4:
        msg = f'Invalid init state vector for position-orientation feasibility criteria! ' \
              f'Init state vector: {x}'
        raise FeasibilityCriteriaException(msg)

    if not len(x_sim) >= 4:
        msg = f'Invalid simulated state vector for position-orientation feasibility criteria! ' \
              f'Simulated state vector: {x_sim}'
        raise FeasibilityCriteriaException(msg)

    if isinstance(vehicle_dynamics, PointMassDynamics):
        diff = np.round(np.abs(np.subtract(x, x_sim)), d)
        return all(np.less(diff, np.concatenate((e[:2], e[:2]))))

    pos_diff = np.subtract(x[[0, 1]], x_sim[[0, 1]])
    orient_diff = _angle_diff(x[4], x_sim[4])
    diff = np.append(pos_diff, orient_diff)
    abs_diff = np.abs(diff)
    round_diff = np.round(abs_diff, d)
    return all(np.less(round_diff, e))


def state_transition_feasibility(x0: State,
                                 x1: State,
                                 vehicle_dynamics: VehicleDynamics,
                                 dt: float,
                                 objective=position_orientation_objective,
                                 criteria=position_orientation_feasibility_criteria,
                                 ftol: float = 1e-8,
                                 e: np.array = np.array([2e-2, 2e-2, 3e-2]),
                                 d: int = 4,
                                 maxiter: int = 100,
                                 disp: bool = False) -> Tuple[bool, State]:
    """
    Checks if the state transition is feasible between given two state according to the vehicle dynamics.

    Tries to find a valid input for the state transition by minimizing the objective function, and then
    checks if the state simulated by using the reconstructed input is feasible.

    By default, the trajectory feasibility checker will use position-orientation objective function as the
    objective and position-orientation feasibility criteria function will be used for feasibility criteria.

    Objectives can be changed by passing a function with the signature `fun(u: np.array, x0: np.array,
    x1: np.array, dt: float, vehicle_dynamics: VehicleDynamics, ftol: float = 1e-8, e: np.array -> float`

    Feasibility criteria can be changed by passing a function with the signature `fun(x: np.array,
    x_sim: np.array, vehicle_dynamics: VehicleDynamics, e: np.array = np.array([2e-2, 2e-2, 3e-2]),
    d: int = 4) -> bool`

    :param x0: initial state
    :param x1: next state
    :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
    :param dt: delta time
    :param objective: callable `fun(u, x0, x1, dt, vehicle_dynamics) -> float`, objective function to be
        minimized in order to find a valid input for state transition
    :param criteria: callable `fun(x1, x_sim, vehicle_dynamics) -> bool`, feasibility criteria to be checked
        between the real next state and the simulated next state
    :param ftol: ftol passed to the minimizer function
    :param e: error margin passed to the feasibility criteria function
    :param d: decimal points where the difference values are rounded up to in order to avoid floating point
        errors set it based on the error margin, i.e e=0.02, d=4
    :param maxiter: maxiter passed to the minimizer function
    :param disp: disp passed to the minimizer function
    :return: True if feasible, and the constructed input as State
    """
    try:
        x0_vals, x0_ts = vehicle_dynamics.state_to_array(x0)
        x1_vals, x1_ts = vehicle_dynamics.state_to_array(x1)
        u0 = np.array([0, 0])

        # Minimize difference between simulated state and next state by varying input u
        u0 = minimize(objective, u0, args=(x0_vals, x1_vals, dt, vehicle_dynamics, ftol, e),
                      options={'disp': disp, 'maxiter': maxiter, 'ftol': ftol},
                      method='SLSQP', bounds=vehicle_dynamics.input_bounds).x

        # Get simulated state using the found inputs
        x1_sim = vehicle_dynamics.forward_simulation(x0_vals, u0, dt, throw=False)
        if x1_sim is None:
            msg = f'Minimizer was not able to reconstruct a valid input for the given states!\n' \
                  f'x0: {x0}\nx1: {x1}\nVehicle: {type(vehicle_dynamics)}\nReconstructed input:{u0}\n' \
                  f'dt: {dt}, ftol: {ftol}, e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}'
            if disp: print(msg)
            return False, vehicle_dynamics.array_to_input(u0, x0_ts)

        # Check the criteria for the feasibility
        feasible = criteria(x1_vals, x1_sim, vehicle_dynamics, e, d)

        return feasible, vehicle_dynamics.array_to_input(u0, x0_ts)

    except (FeasibilityObjectiveException, FeasibilityCriteriaException) as ex:
        msg = f'An exception occurred within the objective of feasibility criteria functions!\n' \
              f'x0: {x0}\nx1: {x1}\nVehicle: {type(vehicle_dynamics)}\n' \
              f'dt: {dt}, ftol: {ftol}, e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}'
        raise StateTransitionException(msg) from ex

    except Exception as ex:  # catch any other exception (in order to debug if there is an unexpected error)
        msg = f'An exception occurred during state transition feasibility checking!\n' \
              f'x0: {x0}\nx1: {x1}\nVehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}, ' \
              f'e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}'
        raise Exception(msg) from ex


def trajectory_feasibility(trajectory: Trajectory,
                           vehicle_dynamics: VehicleDynamics,
                           dt: float,
                           objective=position_orientation_objective,
                           criteria=position_orientation_feasibility_criteria,
                           ftol: float = 1e-8,
                           e: np.array = np.array([2e-2, 2e-2, 3e-2]),
                           d: int = 4,
                           maxiter: int = 100,
                           disp: bool = False) -> Tuple[bool, Trajectory]:
    """
    Checks if the given trajectory is feasible for the vehicle model by checking if the state transition is
    feasible between each consecutive state of the trajectory.

    The state_transition_feasibility function will be applied to consecutive states of a given trajectory,
    and the reconstructed inputs will be returned as Trajectory object. If the trajectory was not feasible,
    reconstructed inputs up to infeasible state will be returned.

    ATTENTION: Reconstructed inputs are just approximated inputs for the forward simulation between
    consecutive states n and n+1. Simulating full trajectory from the initial state by using the
    reconstructed inputs can result in a different (but similar) trajectory compared to the real one.
    The reason for this is the small differences between the approximate inputs and the real inputs adding
    up as we simulate further from the initial state.

    By default, the trajectory feasibility checker will use position-orientation objective function as the
    objective and position-orientation feasibility criteria function will be used for feasibility criteria.

    Objectives can be changed by passing a function with the signature `fun(u: np.array, x0: np.array,
    x1: np.array, dt: float, vehicle_dynamics: VehicleDynamics, ftol: float = 1e-8, e: np.array) -> float`

    Feasibility criteria can be changed by passing a function with the signature `fun(x: np.array,
    x_sim: np.array, vehicle_dynamics: VehicleDynamics, e: np.array = np.array([2e-2, 2e-2, 3e-2]),
    d: int = 4) -> bool`

    :param trajectory: trajectory
    :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
    :param dt: delta time
    :param objective: callable `fun(u, x0, x1, dt, vehicle_dynamics) -> float`, objective function to be
        minimized in order to find a valid input for state transition
    :param criteria: callable `fun(x1, x_sim, vehicle_dynamics) -> bool`, feasibility criteria to be
        checked between the real next state and the simulated next state
    :param ftol: ftol passed to the minimizer function
    :param e: error margin passed to the feasibility criteria function
    :param d: decimal points where the difference values are rounded up to in order to avoid floating
        point errors set it based on the error margin, i.e e=0.02, d=4
    :param maxiter: maxiter passed to the minimizer function
    :param disp: disp passed to the minimizer function
    :return: True if feasible, and list of constructed inputs as Trajectory object
    """
    trajectory_type = TrajectoryType.get_trajectory_type(trajectory, vehicle_dynamics.vehicle_model)
    if trajectory_type in [TrajectoryType.Input, TrajectoryType.PMInput]:
        raise FeasibilityException('Invalid trajectory type!')

    try:
        reconstructed_inputs = []
        for x0, x1 in zip(trajectory.state_list[:-1], trajectory.state_list[1:]):
            feasible, reconstructed_input = state_transition_feasibility(x0, x1, vehicle_dynamics, dt, objective,
                                                                         criteria, ftol, e, d, maxiter, disp)
            reconstructed_inputs.append(reconstructed_input)
            if not feasible:
                input_vector = Trajectory(initial_time_step=reconstructed_inputs[0].time_step,
                                          state_list=reconstructed_inputs)
                return False, input_vector

        input_vector = Trajectory(initial_time_step=reconstructed_inputs[0].time_step,
                                  state_list=reconstructed_inputs)
        return True, input_vector

    except StateTransitionException as ex:
        msg = f'An error occurred during feasibility checking!\n' \
              f'Vehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}, ' \
              f'e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}'
        raise TrajectoryFeasibilityException(msg) from ex

    except Exception as ex:  # catch any other exception (in order to debug if there is an unexpected error)
        msg = f'An exception occurred during trajectory feasibility checking!\n' \
              f'Vehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}, ' \
              f'e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}'
        raise Exception(msg) from ex


def input_vector_feasibility(initial_state: State, input_vector: Trajectory,
                             vehicle_dynamics: VehicleDynamics, dt: float) -> Tuple[bool, Trajectory]:
    """
    Checks whether the given input vector (as Trajectory object) is feasible according to the input and
    state constraints.

    The input bounds and friction circle constraint of corresponding vehicle model is being used as
    criteria of validity. During the process of feasibility checking, the trajectory will be simulated
    for the given initial state and input vector. If there is an infeasible input, all the trajectory
    states simulated up to that input will be returned instead.

    For example, if we have initial state wth time step 0, and valid input vector that contains 20 inputs,
    the trajectory will be completely simulated and returned together with the feasibility result. If we
    have an input vector that is valid up to 5th input, then the trajectory will be simulated up to 5th
    time step, but 6th time step will not be simulated since the input is not feasible.

    :param initial_state: initial state which the input vector will be applied
    :param input_vector: input vector s Trajectory object
    :param vehicle_dynamics: the vehicle dynamics model to be used for input constraint checks
    :param dt: delta time
    :return: True if feasible, and simulated trajectory.
    """
    trajectory_type = TrajectoryType.get_trajectory_type(input_vector, vehicle_dynamics.vehicle_model)
    if not (trajectory_type in [TrajectoryType.Input, TrajectoryType.PMInput]):
        raise FeasibilityException('Invalid trajectory type!')

    try:
        states = [vehicle_dynamics.convert_initial_state(initial_state)]
        for inp in input_vector.state_list:
            within_bounds = vehicle_dynamics.input_within_bounds(inp)
            violates_friction = vehicle_dynamics.violates_friction_circle(states[-1], inp)

            if not within_bounds or violates_friction:
                trajectory = Trajectory(initial_time_step=initial_state.time_step, state_list=states)
                return False, trajectory

            next_state = vehicle_dynamics.simulate_next_state(states[-1], inp, dt)
            states.append(next_state)

        trajectory = Trajectory(initial_time_step=initial_state.time_step, state_list=states)
        return True, trajectory

    except VehicleDynamicsException as ex:
        msg = f'An error occurred during input vector feasibility checking!\n' \
              f'Vehicle: {type(vehicle_dynamics)}\ndt: {dt}\n' \
              f'Initial State: {initial_state}'
        raise InputVectorFeasibilityException(msg) from ex

    except Exception as ex:
        msg = f'An exception occurred during input vector feasibility checking!\n' \
              f'Vehicle: {type(vehicle_dynamics)}'
        raise Exception(msg) from ex

def trajectory_feasibility_cost(trajectory: Trajectory,
                           vehicle_dynamics: VehicleDynamics,
                           dt: float,
                           objective=position_orientation_objective,
                           criteria=position_orientation_feasibility_criteria,
                           ftol: float = 1e-8,
                           e: np.array = np.array([2e-2, 2e-2, 3e-2]),
                           d: int = 4,
                           maxiter: int = 100,
                           disp: bool = False) -> float:
    """
    Checks if the given trajectory is feasible for the vehicle model by checking if the state transition is
    feasible between each consecutive state of the trajectory.

    The state_transition_feasibility function will be applied to consecutive states of a given trajectory,
    and the reconstructed inputs will be returned as Trajectory object. If the trajectory was not feasible,
    reconstructed inputs up to infeasible state will be returned.

    ATTENTION: Reconstructed inputs are just approximated inputs for the forward simulation between
    consecutive states n and n+1. Simulating full trajectory from the initial state by using the
    reconstructed inputs can result in a different (but similar) trajectory compared to the real one.
    The reason for this is the small differences between the approximate inputs and the real inputs adding
    up as we simulate further from the initial state.

    By default, the trajectory feasibility checker will use position-orientation objective function as the
    objective and position-orientation feasibility criteria function will be used for feasibility criteria.

    Objectives can be changed by passing a function with the signature `fun(u: np.array, x0: np.array,
    x1: np.array, dt: float, vehicle_dynamics: VehicleDynamics, ftol: float = 1e-8, e: np.array) -> float`

    Feasibility criteria can be changed by passing a function with the signature `fun(x: np.array,
    x_sim: np.array, vehicle_dynamics: VehicleDynamics, e: np.array = np.array([2e-2, 2e-2, 3e-2]),
    d: int = 4) -> bool`

    :param trajectory: trajectory
    :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
    :param dt: delta time
    :param objective: callable `fun(u, x0, x1, dt, vehicle_dynamics) -> float`, objective function to be
        minimized in order to find a valid input for state transition
    :param criteria: callable `fun(x1, x_sim, vehicle_dynamics) -> bool`, feasibility criteria to be
        checked between the real next state and the simulated next state
    :param ftol: ftol passed to the minimizer function
    :param e: error margin passed to the feasibility criteria function
    :param d: decimal points where the difference values are rounded up to in order to avoid floating
        point errors set it based on the error margin, i.e e=0.02, d=4
    :param maxiter: maxiter passed to the minimizer function
    :param disp: disp passed to the minimizer function
    :return: feasibility ratio, {feasible frame}/{total frame}
    """
    trajectory_type = TrajectoryType.get_trajectory_type(trajectory, vehicle_dynamics.vehicle_model)
    if trajectory_type in [TrajectoryType.Input, TrajectoryType.PMInput]:
        raise FeasibilityException('Invalid trajectory type!')

    try:
        feasible_count = 0
        for x0, x1 in zip(trajectory.state_list[:-1], trajectory.state_list[1:]):
            feasible, reconstructed_input = state_transition_feasibility(x0, x1, vehicle_dynamics, dt, objective,
                                                                         criteria, ftol, e, d, maxiter, disp)
            if feasible:
                feasible_count += 1
        ref_feasible_total = len(trajectory.state_list[:-1])
        feasiblity_cost = feasible_count/ref_feasible_total
        return feasiblity_cost

    except StateTransitionException as ex:
        msg = f'An error occurred during feasibility checking!\n' \
              f'Vehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}, ' \
              f'e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}'
        raise TrajectoryFeasibilityException(msg) from ex

    except Exception as ex:  # catch any other exception (in order to debug if there is an unexpected error)
        msg = f'An exception occurred during trajectory feasibility checking!\n' \
              f'Vehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}, ' \
              f'e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}'
        raise Exception(msg) from ex


def input_vector_feasibility_cost(initial_state: State, input_vector: Trajectory,
                             vehicle_dynamics: VehicleDynamics, dt: float) -> float:
    """
    Checks whether the given input vector (as Trajectory object) is feasible according to the input and
    state constraints.

    The input bounds and friction circle constraint of corresponding vehicle model is being used as
    criteria of validity. During the process of feasibility checking, the trajectory will be simulated
    for the given initial state and input vector. If there is an infeasible input, all the trajectory
    states simulated up to that input will be returned instead.

    For example, if we have initial state wth time step 0, and valid input vector that contains 20 inputs,
    the trajectory will be completely simulated and returned together with the feasibility result. If we
    have an input vector that is valid up to 5th input, then the trajectory will be simulated up to 5th
    time step, but 6th time step will not be simulated since the input is not feasible.

    :param initial_state: initial state which the input vector will be applied
    :param input_vector: input vector s Trajectory object
    :param vehicle_dynamics: the vehicle dynamics model to be used for input constraint checks
    :param dt: delta time
    :return: feasibility ratio, {feasible frame}/{total frame}
    """
    trajectory_type = TrajectoryType.get_trajectory_type(input_vector, vehicle_dynamics.vehicle_model)
    if not (trajectory_type in [TrajectoryType.Input, TrajectoryType.PMInput]):
        raise FeasibilityException('Invalid trajectory type!')

    try:
        feasible_count = 0
        states = [vehicle_dynamics.convert_initial_state(initial_state)]
        for inp in input_vector.state_list:
            within_bounds = vehicle_dynamics.input_within_bounds(inp)
            violates_friction = vehicle_dynamics.violates_friction_circle(states[-1], inp)

            if within_bounds and not violates_friction:
                feasible_count += 1

            next_state = vehicle_dynamics.simulate_next_state(states[-1], inp, dt)
            states.append(next_state)

        ref_feasible_total = len(input_vector.state_list)
        feasiblity_cost = feasible_count/ref_feasible_total
        return feasiblity_cost

    except VehicleDynamicsException as ex:
        msg = f'An error occurred during input vector feasibility checking!\n' \
              f'Vehicle: {type(vehicle_dynamics)}\ndt: {dt}\n' \
              f'Initial State: {initial_state}'
        raise InputVectorFeasibilityException(msg) from ex

    except Exception as ex:
        msg = f'An exception occurred during input vector feasibility checking!\n' \
              f'Vehicle: {type(vehicle_dynamics)}'
        raise Exception(msg) from ex
