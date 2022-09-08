from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import List, Union, Tuple

import numpy as np
import math
from commonroad.common.solution import VehicleType, VehicleModel
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.trajectory import State, Trajectory
from scipy.integrate import odeint
from scipy.optimize import Bounds
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3
from vehiclemodels.parameters_sedan import parameters_sedan
from vehiclemodels.parameters_semi_trailer import parameters_semi_trailer
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_kst import vehicle_dynamics_kst
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_sedan import vehicle_dynamics_sedan
from vehiclemodels.vehicle_dynamics_semi_trailer import vehicle_dynamics_semi_trailer
from vehiclemodels.vehicle_parameters import VehicleParameters


class VehicleDynamicsException(Exception):
    pass


class FrictionCircleException(VehicleDynamicsException):
    pass


class InputBoundsException(VehicleDynamicsException):
    pass


class StateException(VehicleDynamicsException):
    pass


class InputException(VehicleDynamicsException):
    pass


@unique
class VehicleParameterMapping(Enum):
    """
    Mapping for VehicleType name to VehicleParameters
    """
    FORD_ESCORT = parameters_vehicle1()
    BMW_320i = parameters_vehicle2()
    VW_VANAGON = parameters_vehicle3()
    SEDAN = parameters_sedan()
    SEMI_TRAILER = parameters_semi_trailer()

    @classmethod
    def from_vehicle_type(cls, vehicle_type: VehicleType) -> VehicleParameters:
        return cls[vehicle_type.name].value


class VehicleDynamics(ABC):
    """
    VehicleDynamics abstract class that encapsulates the common methods of all VehicleDynamics classes.

    List of currently implemented vehicle models
     - Point-Mass Model (PM)
     - Kinematic Single-Track Model (KS)
     - Single-Track Model (ST)
     - Multi-Body Model (MB)
     - Single-Track Trailer Model (KST)
     - Sedan Model (SEDAN)
     - Semi-Trailer Model (SEMI_TRAILER)

    New types of VehicleDynamics can be defined by extending this class. If there isn't any mismatch with the state
    values, the new VehicleDynamics class can be used directly with the feasibility checkers as well.

    For detailed documentation of the Vehicle Models, please check the `Vehicle Model Documentation
    <https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/blob/master/vehicleModels_commonroad.pdf>`_
    """

    def __init__(self, vehicle_model: VehicleModel, vehicle_type: VehicleType):
        """
        Creates a VehicleDynamics model for the given VehicleType.

        :param vehicle_type: VehicleType
        """
        self.vehicle_model = vehicle_model
        self.vehicle_type = vehicle_type
        self.parameters = VehicleParameterMapping[self.vehicle_type.name].value
        self.shape = Rectangle(length=self.parameters.l,
                               width=self.parameters.w)
        if vehicle_type == VehicleType.SEMI_TRAILER:
            self.shape_trailer = Rectangle(length=self.parameters.trailer.l,
                               width=self.parameters.trailer.w)
        else:
            self.shape_trailer = None

    @classmethod
    def PM(cls, vehicle_type: VehicleType) -> 'PointMassDynamics':
        """
        Creates a PointMassDynamics model.

        :param vehicle_type: VehicleType, i.e. VehileType.FORD_ESCORT
        :return: PointMassDynamics instance with the given vehicle type.
        """
        return PointMassDynamics(vehicle_type)

    @classmethod
    def KS(cls, vehicle_type: VehicleType) -> 'KinematicSingleTrackDynamics':
        """
        Creates a KinematicSingleTrackDynamics model.

        :param vehicle_type: VehicleType, i.e. VehileType.FORD_ESCORT
        :return: KinematicSingleTrackDynamics instance with the given vehicle type.
        """
        return KinematicSingleTrackDynamics(vehicle_type)
    
    @classmethod
    def ST(cls, vehicle_type: VehicleType) -> 'SingleTrackDynamics':
        """
        Creates a SingleTrackDynamics VehicleDynamics model.

        :param vehicle_type: VehicleType, i.e. VehileType.FORD_ESCORT
        :return: SingleTrackDynamics instance with the given vehicle type.
        """
        return SingleTrackDynamics(vehicle_type)

    @classmethod
    def SEDAN(cls, vehicle_type: VehicleType) -> 'SedanTrackDynamics':
        """
        Creates a KinematicSingleTrackDynamics model.

        :param vehicle_type: VehicleType, i.e. VehileType.FORD_ESCORT
        :return: KinematicSingleTrackDynamics instance with the given vehicle type.
        """
        return SedanTrackDynamics(vehicle_type)

    @classmethod
    def SEMI_TRAILER(cls, vehicle_type: VehicleType) -> 'SemiTrailerDynamics':
        """
        Creates a SingleTrackDynamics VehicleDynamics model.

        :param vehicle_type: VehicleType, i.e. VehileType.FORD_ESCORT
        :return: SingleTrackDynamics instance with the given vehicle type.
        """
        return SemiTrailerDynamics(vehicle_type)

    @classmethod
    def MB(cls, vehicle_type: VehicleType) -> 'MultiBodyDynamics':
        """
        Creates a MultiBodyDynamics VehicleDynamics model.

        :param vehicle_type: VehicleType, i.e. VehileType.FORD_ESCORT
        :return: MultiBodyDynamics instance with the given vehicle type.
        """
        return MultiBodyDynamics(vehicle_type)
    
    @classmethod
    def KST(cls, vehicle_type: VehicleType) -> 'KinematicTrailerkDynamics':
        """
        Creates a KinematicSingleTrackDynamics model.

        :param vehicle_type: VehicleType, i.e. VehileType.FORD_ESCORT
        :return: KinematicSingleTrackDynamics instance with the given vehicle type.
        """
        return KinematicTrailerDynamics(vehicle_type)

    @classmethod
    def from_model(cls, vehicle_model: VehicleModel, vehicle_type: VehicleType) -> 'VehicleDynamics':
        """
        Creates a VehicleDynamics model for the given vehicle model and type.

        :param vehicle_model: VehicleModel, i.e. VehicleModel.KS
        :param vehicle_type: VehicleType, i.e. VehileType.FORD_ESCORT
        :return: VehicleDynamics instance with the given vehicle type.
        """
        model_constructor = getattr(cls, vehicle_model.name)
        return model_constructor(vehicle_type)

    @abstractmethod
    def dynamics(self, t, x, u) -> List[float]:
        """
        Vehicle dynamics function that models the motion of the vehicle during forward simulation.

        :param t: time point which the differentiation is being calculated at.
        :param x: state values
        :param u: input values
        :return: next state values
        """
        pass

    @property
    def input_bounds(self) -> Bounds:
        """
        Returns the bounds on inputs (constraints).

        Bounds are
            - min steering velocity <= steering_angle_speed <= max steering velocity
            - -max longitudinal acc <= acceleration <= max longitudinal acc

        :return: Bounds
        """
        return Bounds([self.parameters.steering.v_min, -self.parameters.longitudinal.a_max],
                      [self.parameters.steering.v_max, self.parameters.longitudinal.a_max])

    def input_within_bounds(self, u: Union[State, np.array], throw: bool = False) -> bool:
        """
        Checks whether the given input is within input constraints of the vehicle dynamics model.

        :param u: input values as np.array or State - Contains 2 values
        :param throw: if set to false, will return bool instead of throwing exception (default=False)
        :return: True if within constraints
        """
        inputs = self.input_to_array(u)[0] if isinstance(u, State) else u
        in_bounds = all([self.input_bounds.lb[idx] <= round(inputs[idx], 4) <= self.input_bounds.ub[idx]
                         for idx in range(len(self.input_bounds.lb))])
        if not in_bounds and throw:
            raise InputBoundsException(f'Input is not within bounds!\nInput: {u}')
        return in_bounds

    def violates_friction_circle(self, x: Union[State, np.array], u: Union[State, np.array],
                                 throw: bool = False) -> bool:
        """
        Checks whether given input violates the friction circle constraint for the given state.

        :param x: current state
        :param u: the input which was used to simulate the next state
        :param throw: if set to false, will return bool instead of throwing exception (default=False)
        :return: True if the constraint was violated
        """
        x_vals = self.state_to_array(x)[0] if isinstance(x, State) else x
        u_vals = self.input_to_array(u)[0] if isinstance(u, State) else u
        x_dot = self.dynamics(0, x_vals, u_vals)

        vals = np.array([u_vals[1], x_vals[3] * x_dot[4]])
        vals_sq = np.power(vals, 2)
        vals_sum = np.sum(vals_sq)
        violates = vals_sum > self.parameters.longitudinal.a_max ** 2

        if throw and violates:
            msg = f'Input violates friction circle constraint!\n' \
                  f'Init state: {x}\n\n Input:{u}'
            raise FrictionCircleException(msg)

        return violates

    def forward_simulation(self, x: np.array, u: np.array, dt: float, throw: bool = True) -> Union[None, np.array]:
        """
        Simulates the next state using the given state and input values as numpy arrays.

        :param x: state values.
        :param u: input values
        :param dt: scenario delta time.
        :param throw: if set to false, will return None as next state instead of throwing exception (default=True)
        :return: simulated next state values, raises VehicleDynamicsException if invalid input.
        """
        within_bounds = self.input_within_bounds(u, throw)
        violates_friction_constraint = self.violates_friction_circle(
            x, u, throw)
        if not throw and (not within_bounds or violates_friction_constraint):
            return None

        x0, x1 = odeint(self.dynamics, x, [0.0, dt], args=(u,), tfirst=True)
        return x1

    def simulate_next_state(self, x: State, u: State, dt: float, throw: bool = True) -> Union[None, State]:
        """
        Simulates the next state using the given state and input values as State objects.

        :param x: current state
        :param u: inputs for simulating the next state
        :param dt: scenario delta time.
        :param throw: if set to false, will return None as next state instead of throwing exception (default=True)
        :return: simulated next state, raises VehicleDynamicsException if invalid input.
        """
        x_vals, x_ts = self.state_to_array(x)
        u_vals, u_ts = self.input_to_array(u)
        x1_vals = self.forward_simulation(x_vals, u_vals, dt, throw)
        if x1_vals is None:
            return None
        x1 = self.array_to_state(x1_vals, x_ts + 1)
        return x1

    def simulate_trajectory(self, initial_state: State, input_vector: Trajectory,
                            dt: float, throw: bool = True) -> Union[None, Trajectory]:
        """
        Creates the trajectory for the given input vector.

        :param initial_state: initial state of the planning problem
        :param input_vector: input vector as Trajectory object
        :param dt: scenario delta time
        :param throw: if set to false, will return None as trajectory instead of throwing exception (default=True)
        :return: simulated trajectory, raises VehicleDynamicsException if there is an invalid input.
        """
        converted_init_state = self.convert_initial_state(initial_state)
        state_list = [converted_init_state]
        for input in input_vector.state_list:
            simulated_state = self.simulate_next_state(
                state_list[-1], input, dt, throw)
            if not throw and not simulated_state:
                return None
            state_list.append(simulated_state)
        trajectory = Trajectory(
            initial_time_step=initial_state.time_step, state_list=state_list)
        return trajectory

    @abstractmethod
    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """Actual conversion of state to array happens here, each vehicle will implement its own converter."""
        pass

    def state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """
        Converts the given State to numpy array.

        :param state: State
        :return: state values as numpy array and time step of the state
        """
        try:
            array, time_step = self._state_to_array(
                state, steering_angle_default)
            return array, time_step
        except Exception as e:
            err = f'Not a valid state!\nState:{str(state)}'
            raise StateException(err) from e

    @abstractmethod
    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """Actual conversion of the array to state happens here, each vehicle will implement its own converter."""
        pass

    def array_to_state(self, x: np.array, time_step: int) -> State:
        """
        Converts the given numpy array of values to State.

        :param x: list of state values
        :param time_step: time step of the converted state
        :return: State
        """
        try:
            state = self._array_to_state(x, time_step)
            return state
        except Exception as e:
            raise e
            err = f'Not a valid state array!\nTime step: {time_step}, State array:{str(x)}'
            raise StateException(err) from e

    def convert_initial_state(self, initial_state, steering_angle_default=0.0) -> State:
        """
        Converts the given default initial state to VehicleModel's state by setting the state values accordingly.

        :param initial_state: default initial state
        :param steering_angle_default: default steering_angle value as it is not given in intiial state
        :return: converted initial state
        """
        return self.array_to_state(self.state_to_array(initial_state, steering_angle_default)[0],
                                   initial_state.time_step)

    def _input_to_array(self, input: State) -> Tuple[np.array, int]:
        """
        Actual conversion of input to array happens here. Vehicles can override this method to implement their own converter.
        """
        values = [
            input.steering_angle_speed,
            input.acceleration,
        ]
        time_step = input.time_step
        return np.array(values), time_step

    def input_to_array(self, input: State) -> Tuple[np.array, int]:
        """
        Converts the given input (as State object) to numpy array.

        :param input: input as State object
        :return: state values as numpy array and time step of the state, raises VehicleDynamicsException if invalid
            input
        """
        try:
            array, time_step = self._input_to_array(input)
            return array, time_step
        except Exception as e:
            raise InputException(f'Not a valid input!\n{str(input)}') from e

    def _array_to_input(self, u: np.array, time_step: int) -> State:
        """
        Actual conversion of input array to input happens here. Vehicles can override this method to implement their
        own converter.
        """
        values = {
            'steering_angle_speed': u[0],
            'acceleration': u[1],
        }
        return State(**values, time_step=time_step)

    def array_to_input(self, u: np.array, time_step: int) -> State:
        """
        Converts the given numpy array of values to input (as State object).

        :param u: input values
        :param time_step: time step of the converted input
        :return: input as state object, raises VehicleDynamicsException if invalid input
        """
        try:
            state = self._array_to_input(u, time_step)
            return state
        except Exception as e:
            raise InputException(
                f'Not a valid input array!\nArray:{str(u)} Time Step: {time_step}') from e

    @staticmethod
    def _convert_from_directional_velocity(velocity, orientation) -> Tuple[float, float]:
        """
        Converts the given velocity and orientation to velocity_x and velocity_y values.

        :param velocity: velocity
        :param orientation: orientation
        :return: velocity_x, velocity_y
        """
        velocity_x = math.cos(orientation) * velocity
        velocity_y = math.sin(orientation) * velocity
        return velocity_x, velocity_y


class PointMassDynamics(VehicleDynamics):

    def __init__(self, vehicle_type: VehicleType):
        super(PointMassDynamics, self).__init__(VehicleModel.PM, vehicle_type)

    def dynamics(self, t, x, u) -> List[float]:
        """
        Point Mass model dynamics function. Overrides the dynamics function of VehicleDynamics for PointMass model.

        :param t:
        :param x: state values, [position x, position y, velocity x, velocity y]
        :param u: input values, [acceleration x, acceleration y]

        :return:
        """
        return [
            x[2],
            x[3],
            u[0],
            u[1],
        ]

    @property
    def input_bounds(self) -> Bounds:
        """
        Overrides the bounds method of Vehicle Model in order to return bounds for the Point Mass inputs.

        Bounds are
            - -max longitudinal acc <= acceleration <= max longitudinal acc
            - -max longitudinal acc <= acceleration_y <= max longitudinal acc

        :return: Bounds
        """
        return Bounds([-self.parameters.longitudinal.a_max, -self.parameters.longitudinal.a_max],
                      [self.parameters.longitudinal.a_max, self.parameters.longitudinal.a_max])

    def violates_friction_circle(self, x: Union[State, np.array], u: Union[State, np.array],
                                 throw: bool = False) -> bool:
        """
        Overrides the friction circle constraint method of Vehicle Model in order calculate
        friction circle constraint for the Point Mass model.

        :param x: current state
        :param u: the input which was used to simulate the next state
        :param throw: if set to false, will return bool instead of throwing exception (default=False)
        :return: True if the constraint was violated
        """
        u_vals = self.input_to_array(u)[0] if isinstance(u, State) else u

        vals_sq = np.power(u_vals, 2)
        vals_sqrt = np.sqrt(np.sum(vals_sq))
        violates = vals_sqrt > self.parameters.longitudinal.a_max
        if throw and violates:
            msg = f'Input violates friction circle constraint!\n' \
                  f'Init state: {x}\n\n Input:{u}'
            raise FrictionCircleException(msg)

        return violates

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        if hasattr(state, 'velocity') and hasattr(state, 'orientation') and not hasattr(state,
                                                                                        'velocity_y'):  # If initial state
            velocity_x, velocity_y = self._convert_from_directional_velocity(
                state.velocity, state.orientation)
        else:
            velocity_x, velocity_y = state.velocity, state.velocity_y

        values = [
            state.position[0],
            state.position[1],
            velocity_x,
            velocity_y
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'velocity': x[2],
            'velocity_y': x[3],
            'orientation': math.atan2(x[3], x[2])
        }
        return State(**values, time_step=time_step)

    def _input_to_array(self, input: State) -> Tuple[np.array, int]:
        """ Overrides VehicleDynamics method. """
        values = [
            input.acceleration,
            input.acceleration_y,
        ]
        time_step = input.time_step
        return np.array(values), time_step

    def _array_to_input(self, u: np.array, time_step: int) -> State:
        """ Overrides VehicleDynamics method. """
        values = {
            'acceleration': u[0],
            'acceleration_y': u[1],
        }
        return State(**values, time_step=time_step)


class KinematicSingleTrackDynamics(VehicleDynamics):
    def __init__(self, vehicle_type: VehicleType):
        super(KinematicSingleTrackDynamics, self).__init__(
            VehicleModel.KS, vehicle_type)

    def dynamics(self, t, x, u) -> List[float]:
        return vehicle_dynamics_ks(x, u, self.parameters)

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        values = [
            state.position[0] - self.parameters.b *
            math.cos(state.orientation),
            state.position[1] - self.parameters.b *
            math.sin(state.orientation),
            # not defined in initial state
            getattr(state, 'steering_angle', steering_angle_default),
            state.velocity,
            state.orientation
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0] + self.parameters.b * math.cos(x[4]),
                                  x[1] + self.parameters.b * math.sin(x[4])]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
        }
        state = State(**values, time_step=time_step)
        return state


class SedanTrackDynamics(VehicleDynamics):
    def __init__(self, vehicle_type: VehicleType):
        super(SedanTrackDynamics, self).__init__(
            VehicleModel.KS, vehicle_type)

    def dynamics(self, t, x, u) -> List[float]:
        return vehicle_dynamics_sedan(x, u, self.parameters)

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        values = [
            state.position[0] - self.parameters.b *
            math.cos(state.orientation),
            state.position[1] - self.parameters.b *
            math.sin(state.orientation),
            # not defined in initial state
            getattr(state, 'steering_angle', steering_angle_default),
            state.velocity,
            state.orientation
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0] + self.parameters.b * math.cos(x[4]),
                                  x[1] + self.parameters.b * math.sin(x[4])]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
        }
        state = State(**values, time_step=time_step)
        return state


class SingleTrackDynamics(VehicleDynamics):
    def __init__(self, vehicle_type: VehicleType):
        super(SingleTrackDynamics, self).__init__(
            VehicleModel.ST, vehicle_type)

    def dynamics(self, t, x, u) -> List[float]:
        return vehicle_dynamics_st(x, u, self.parameters)

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        values = [
            state.position[0],
            state.position[1],
            # not defined in initial state
            getattr(state, 'steering_angle', steering_angle_default),
            state.velocity,
            state.orientation,
            state.yaw_rate,
            state.slip_angle
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
            'yaw_rate': x[5],
            'slip_angle': x[6],
        }
        return State(**values, time_step=time_step)


class MultiBodyDynamics(VehicleDynamics):
    def __init__(self, vehicle_type: VehicleType):
        super(MultiBodyDynamics, self).__init__(VehicleModel.MB, vehicle_type)

    def dynamics(self, t, x, u) -> List[float]:
        return vehicle_dynamics_mb(x, u, self.parameters)

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        if not len(state.attributes) == 29:  # if initial state
            velocity_x, velocity_y = self._convert_from_directional_velocity(
                state.velocity, state.orientation)
        else:
            velocity_x, velocity_y = state.velocity, state.velocity_y

        p = self.parameters
        g = 9.81  # [m/s^2]
        F0_z_f = p.m_s * g * p.b / (p.a + p.b) + p.m_uf * g
        F0_z_r = p.m_s * g * p.a / (p.a + p.b) + p.m_ur * g
        position_z_front = F0_z_f / 2 * p.K_zt
        position_z_rear = F0_z_r / 2 * p.K_zt
        def wheel_speed(velocity_x): return velocity_x / p.R_w

        def velocitz_y_front(
            velocity_y, yaw_rate): return velocity_y + p.a * yaw_rate
        def velocitz_y_rear(
            velocity_y, yaw_rate): return velocity_y - p.b * yaw_rate

        values = [
            # sprung mass states
            state.position[0],
            state.position[1],
            # not defined in initial state
            getattr(state, 'steering_angle', steering_angle_default),
            velocity_x,
            state.orientation,
            state.yaw_rate,
            getattr(state, 'roll_angle', 0.0),
            getattr(state, 'roll_rate', 0.0),
            getattr(state, 'pitch_angle', 0.0),
            getattr(state, 'pitch_rate', 0.0),
            getattr(state, 'velocity_y', velocity_y),
            getattr(state, 'position_z', 0.0),
            getattr(state, 'velocity_z', 0.0),

            # unsprung mass states (front)
            getattr(state, 'roll_angle_front', 0.0),
            getattr(state, 'roll_rate_front', 0.0),
            getattr(state, 'velocity_y_front', velocitz_y_front(
                velocity_y, state.yaw_rate)),
            getattr(state, 'position_z_front', position_z_front),
            # not defined in initial state
            getattr(state, 'velocity_z_front', 0.0),

            # unsprung mass states (rear)
            # not defined in initial state
            getattr(state, 'roll_angle_rear', 0.0),
            # not defined in initial state
            getattr(state, 'roll_rate_rear', 0.0),
            getattr(state, 'velocity_y_rear', velocitz_y_rear(
                velocity_y, state.yaw_rate)),
            getattr(state, 'position_z_rear', position_z_rear),
            # not defined in initial state
            getattr(state, 'velocity_z_rear', 0.0),

            # wheel states
            getattr(state, 'left_front_wheel_angular_speed',
                    wheel_speed(velocity_x)),
            getattr(state, 'right_front_wheel_angular_speed',
                    wheel_speed(velocity_x)),
            getattr(state, 'left_rear_wheel_angular_speed',
                    wheel_speed(velocity_x)),
            getattr(state, 'right_rear_wheel_angular_speed',
                    wheel_speed(velocity_x)),
            getattr(state, 'delta_y_f', 0.0),  # not defined in initial state
            getattr(state, 'delta_y_r', 0.0),  # not defined in initial state
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
            'yaw_rate': x[5],
            'roll_angle': x[6],
            'roll_rate': x[7],
            'pitch_angle': x[8],
            'pitch_rate': x[9],
            'velocity_y': x[10],
            'position_z': x[11],
            'velocity_z': x[12],
            'roll_angle_front': x[13],
            'roll_rate_front': x[14],
            'velocity_y_front': x[15],
            'position_z_front': x[16],
            'velocity_z_front': x[17],
            'roll_angle_rear': x[18],
            'roll_rate_rear': x[19],
            'velocity_y_rear': x[20],
            'position_z_rear': x[21],
            'velocity_z_rear': x[22],
            'left_front_wheel_angular_speed': x[23],
            'right_front_wheel_angular_speed': x[24],
            'left_rear_wheel_angular_speed': x[25],
            'right_rear_wheel_angular_speed': x[26],
            'delta_y_f': x[27],
            'delta_y_r': x[28],
        }
        return State(**values, time_step=time_step)


class KinematicTrailerDynamics(VehicleDynamics):
    def __init__(self, vehicle_type: VehicleType):
        super(KinematicTrailerDynamics, self).__init__(
            VehicleModel.KST, vehicle_type)

    def dynamics(self, t, x, u) -> List[float]:
        return vehicle_dynamics_kst(x, u, self.parameters)

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        values = [
            state.position[0],
            state.position[1],
            state.steering_angle,
            state.velocity,
            state.orientation,
            state.hitch_angle
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
            'hitch_angle': x[5]
        }
        return State(**values, time_step=time_step)


class SemiTrailerDynamics(VehicleDynamics):
    def __init__(self, vehicle_type: VehicleType):
        super(SemiTrailerDynamics, self).__init__(
            VehicleModel.SEMI_TRAILER, vehicle_type)

    def dynamics(self, t, x, u) -> List[float]:
        return vehicle_dynamics_semi_trailer(x, u, self.parameters)

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        values = [
            state.position[0],
            state.position[1],
            state.steering_angle,
            state.velocity,
            state.orientation,
            state.hitch_angle,
            state.position_trailer[0],
            state.position_trailer[1],
            state.yaw_angle_trailer
        ]
        time_step = state.time_step
        return np.array(values), time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
            'hitch_angle': x[5],
            'position_trailer': np.array([x[6], x[7]]),
            'yaw_angle_trailer': x[8]
        }
        return State(**values, time_step=time_step)


