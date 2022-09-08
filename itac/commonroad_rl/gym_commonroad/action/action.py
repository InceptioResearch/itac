"""
Module containing the action base class
"""
import gym
from typing import Union
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad_rl.gym_commonroad.action.vehicle import *


def _rotate_to_curvi(vector: np.ndarray, local_ccosy: CurvilinearCoordinateSystem, pos: np.ndarray) \
        -> np.ndarray:
    """
    Function to rotate a vector in the curvilinear system to its counterpart in the normal coordinate system

    :param vector: The vector in question
    :returns: The rotated vector
    """
    try:
        long, _ = local_ccosy.convert_to_curvilinear_coords(pos[0], pos[1])
    except ValueError:
        long = 0.

    tangent = local_ccosy.tangent(long)
    theta = np.math.atan2(tangent[1], tangent[0])
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    return np.matmul(rot_mat, vector)


class Action(ABC):
    """
    Description:
        Abstract base class of all action spaces
    """

    def __init__(self):
        """ Initialize empty object """
        super().__init__()
        self.vehicle = None

    @abstractmethod
    def step(self, action: Union[np.ndarray, int], local_ccosy: CurvilinearCoordinateSystem = None) -> None:
        """
        Function which acts on the current state and generates the new state
        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        """
        pass


class DiscreteAction(Action):
    """
    Description:
        Abstract base class of all discrete action spaces. Each high-level discrete
        action is converted to a low-level trajectory by a specified planner.
    """

    def __init__(self, vehicle_params_dict: dict, long_steps: int, lat_steps: int):
        """ Initialize empty object """
        super().__init__()

        assert VehicleModel(vehicle_params_dict["vehicle_model"]) == VehicleModel.PM, \
            'ERROR in ACTION INITIALIZATION: DiscreteAction only supports the PM vehicle_type no'

        assert long_steps % 2 != 0 and lat_steps % 2 != 0, \
            'ERROR in ACTION INITIALIZATION: The discrete steps for longitudinal and lateral action ' \
            'have to be odd numbers, so constant velocity without turning is an possible action'

        self.vehicle = ContinuousVehicle(vehicle_params_dict)
        self.local_ccosy = None

    def reset(self, initial_state: State, dt: float) -> None:
        """
        resets the vehicle
        :param initial_state: initial state
        :param dt: time step size of scenario
        """
        self.vehicle.reset(initial_state, dt)

    def step(self, action: Union[np.ndarray, int], local_ccosy: CurvilinearCoordinateSystem = None) -> None:
        """
        Function which acts on the current state and generates the new state

        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        """
        self.local_ccosy = local_ccosy
        state = self.get_new_state(action)
        self.vehicle.set_current_state(state)

    @abstractmethod
    def get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """function which return new states given the action and current state"""
        pass

    def _propagate(self, control_input: np.array):
        # Rotate the action according to the curvilinear coordinate system
        if self.local_ccosy is not None:
            control_input = _rotate_to_curvi(control_input, self.local_ccosy, self.vehicle.state.position)

        # get the next state from the PM model
        return self.vehicle.get_new_state(control_input, "acceleration")


class DiscretePMJerkAction(DiscreteAction):
    """
    Description:
        Discrete / High-level action class with point mass model and jerk control
    """

    def __init__(self, vehicle_params_dict: dict, long_steps: int, lat_steps: int):
        """
        Initialize object
        :param vehicle_params_dict: vehicle parameter dictionary
        :param long_steps: number of discrete longitudinal jerk steps
        :param lat_steps: number of discrete lateral jerk steps
        """
        super().__init__(vehicle_params_dict, long_steps, lat_steps)

        self.j_max = 10  # set the maximum jerk
        self.long_step_size = (self.j_max * 2) / (long_steps - 1)
        self.lat_step_size = (self.j_max * 2) / (lat_steps - 1)
        self.action_mapping_long = {}
        self.action_mapping_lat = {}

        for idx in range(long_steps):
            self.action_mapping_long[idx] = (self.j_max - (idx * self.long_step_size))

        for idx in range(lat_steps):
            self.action_mapping_lat[idx] = (self.j_max - (idx * self.lat_step_size))

    def get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """
        calculation of next state depending on the discrete action
        :param action: discrete action
        :return: next state
        """
        # map discrete action to jerk and calculate a
        # correct rescale in order to make 0 acceleration achievable again when sign of acc switches
        a_long = self.action_mapping_long[action[0]] * self.vehicle.dt + self.vehicle.state.acceleration
        if self.vehicle.state.acceleration != 0 and np.sign(a_long) != np.sign(self.vehicle.state.acceleration) and \
                (np.abs(a_long) % (self.long_step_size * self.vehicle.dt)) != 0:
            a_long = self.action_mapping_long[action[0]] * self.vehicle.dt + self.vehicle.state.acceleration - \
                     np.sign(a_long) * (np.abs(a_long) % (self.long_step_size * self.vehicle.dt))

        a_lat = self.action_mapping_lat[action[1]] * self.vehicle.dt + self.vehicle.state.acceleration_y
        if self.vehicle.state.acceleration_y != 0 and np.sign(a_lat) != np.sign(self.vehicle.state.acceleration_y) and \
                (np.abs(a_lat) % (self.lat_step_size * self.vehicle.dt)) != 0:
            a_lat = self.action_mapping_long[action[1]] * self.vehicle.dt + self.vehicle.state.acceleration_y - \
                    np.sign(a_lat) * (np.abs(a_lat) % (self.lat_step_size * self.vehicle.dt))

        control_input = np.array([a_long, a_lat])

        return self._propagate(control_input)


class DiscretePMAction(DiscreteAction):
    """
    Description:
        Discrete / High-level action class with point mass model
    """

    def __init__(self, vehicle_params_dict: dict, long_steps: int, lat_steps: int):
        """
        Initialize object
        :param vehicle_params_dict: vehicle parameter dictionary
        :param long_steps: number of discrete acceleration steps
        :param lat_steps: number of discrete turning steps
        """
        super().__init__(vehicle_params_dict, long_steps, lat_steps)

        a_max = self.vehicle.parameters.longitudinal.a_max
        a_long_steps = (a_max * 2) / (long_steps - 1)
        a_lat_steps = (a_max * 2) / (lat_steps - 1)

        self.action_mapping_long = {}
        self.action_mapping_lat = {}

        for idx in range(long_steps):
            self.action_mapping_long[idx] = (a_max - (idx * a_long_steps))

        for idx in range(lat_steps):
            self.action_mapping_lat[idx] = (a_max - (idx * a_lat_steps))

    def propogate_one_state(self, state: State, action: Union[np.ndarray, int]):
        """
        Used to generate a trajectory from a given action
        :param state:
        :param action:
        :return:
        """
        control_input = np.array([self.action_mapping_long[action[0]],
                                  self.action_mapping_lat[action[1]]])
        # Rotate the action according to the curvilinear coordinate system
        if self.local_ccosy is not None:
            control_input = _rotate_to_curvi(control_input, self.local_ccosy, state.position)

        return self.vehicle.propagate_one_time_step(state, control_input, "acceleration")

    def get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """
        calculation of next state depending on the discrete action
        :param action: discrete action
        :return: next state
        """
        return self.propogate_one_state(state=self.vehicle.state, action=action)


class ContinuousAction(Action):
    """
    Description:
        Module for continuous action space; actions correspond to vehicle control inputs
    """

    def __init__(self, params_dict: dict, action_dict: dict):
        """ Initialize object """
        super().__init__()
        # create vehicle object
        self.action_base = action_dict['action_base']
        self._continous_collision_check = action_dict.get("continuous_collision_checking", True)
        self.vehicle = ContinuousVehicle(params_dict, continuous_collision_checking=self._continous_collision_check)

    def _set_rescale_factors(self):

        a_max = self.vehicle.parameters.longitudinal.a_max
        # rescale factors for PM model
        if self.vehicle.vehicle_model == VehicleModel.PM:
            self._rescale_factor = np.array([a_max, a_max])
            self._rescale_bias = 0.0
        # rescale factors for KS model
        elif self.vehicle.vehicle_model == VehicleModel.KS:
            steering_v_max = self.vehicle.parameters.steering.v_max
            steering_v_min = self.vehicle.parameters.steering.v_min
            self._rescale_factor = np.array([(steering_v_max - steering_v_min) / 2., a_max])
            self._rescale_bias = np.array([(steering_v_max + steering_v_min) / 2., 0.])
        # rescale factors for MB model
        elif self.vehicle.vehicle_model == VehicleModel.MB:
            steering_v_max = self.vehicle.parameters.steering.v_max
            steering_v_min = self.vehicle.parameters.steering.v_min
            self._rescale_factor = np.array([(steering_v_max - steering_v_min) / 2., a_max])
            self._rescale_bias = np.array([(steering_v_max + steering_v_min) / 2., 0.])
        elif self.vehicle.vehicle_model == VehicleModel.KST:
            steering_v_max = self.vehicle.parameters.steering.v_max
            steering_v_min = self.vehicle.parameters.steering.v_min
            self._rescale_factor = np.array([(steering_v_max - steering_v_min) / 2., a_max])
            self._rescale_bias = np.array([(steering_v_max + steering_v_min) / 2., 0.])
        elif self.vehicle.vehicle_model == VehicleModel.SEDAN:
            steering_v_max = self.vehicle.parameters.steering.v_max
            steering_v_min = self.vehicle.parameters.steering.v_min
            self._rescale_factor = np.array([(steering_v_max - steering_v_min) / 2., a_max])
            self._rescale_bias = np.array([(steering_v_max + steering_v_min) / 2., 0.])
        elif self.vehicle.vehicle_model == VehicleModel.SEMI_TRAILER:
            steering_v_max = self.vehicle.parameters.steering.v_max
            steering_v_min = self.vehicle.parameters.steering.v_min
            self._rescale_factor = np.array([(steering_v_max - steering_v_min) / 2., a_max])
            self._rescale_bias = np.array([(steering_v_max + steering_v_min) / 2., 0.])

    def reset(self, initial_state: State, dt: float) -> None:
        self.vehicle.reset(initial_state, dt)
        self._set_rescale_factors()

    def step(self, action: Union[np.ndarray, int], local_ccosy: CurvilinearCoordinateSystem = None) -> None:
        """
        Function which acts on the current state and generates the new state

        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        :return: New state of ego vehicle
        """
        rescaled_action = self.rescale_action(action)
        new_state = self.vehicle.get_new_state(rescaled_action, self.action_base)
        self.vehicle.set_current_state(new_state)
        if self.vehicle.vehicle_model == VehicleModel.MB:
            self._set_rescale_factors()

    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range

        :param action: action from the CommonroadEnv.
        :return: rescaled action
        """
        assert hasattr(self, "_rescale_bias") and hasattr(self, "_rescale_factor"), \
            "<ContinuousAction/rescale_action>: rescale factors not set, please run action.reset() first"
        # if self.vehicle.vehicle_model == VehicleModel.YawRate:
        #     # update rescale factors
        #     self._set_rescale_factors()

        return self._rescale_factor * action + self._rescale_bias


def action_constructor(action_configs: dict, vehicle_params: dict) -> \
        Tuple[Action, Union[gym.spaces.Box, gym.spaces.MultiDiscrete]]:

    if action_configs['action_type'] == "continuous":
        action = ContinuousAction(vehicle_params, action_configs)
    elif action_configs['action_type'] == "discrete":
        if action_configs['action_base'] == "acceleration":
            action = DiscretePMAction
        elif action_configs['action_base'] == "jerk":
            action = DiscretePMJerkAction
        else:
            raise NotImplementedError(f"action_base {action_configs['action_base']} not supported. "
                                      f"Please choose acceleration or jerk")
        action = action(vehicle_params, action_configs['long_steps'], action_configs['lat_steps'])
    else:
        raise NotImplementedError(f"action_type {action_configs['action_type']} not supported. "
                                  f"Please choose continuous or discrete")

        # Action space remove
        # TODO initialize action space with class
    if action_configs['action_type'] == "continuous":
        action_high = np.array([1.0, 1.0])
        action_space = gym.spaces.Box(low=-action_high, high=action_high, dtype="float32")
    else:
        action_space = gym.spaces.MultiDiscrete([action_configs['long_steps'], action_configs['lat_steps']])

    return action, action_space