from os import stat
import warnings
from copy import deepcopy
from typing import List

import numpy as np
from commonroad.common.solution import VehicleModel, VehicleType
from commonroad.scenario.trajectory import Trajectory, State


class TruckMotionPrimitive:
    """
    Class for motion primitives.
    """

    class PrimitiveState:
        """
        Inner class to represent the initial and final states of a motion primitive.
        """

        def __init__(self, x, y, steering_angle, velocity, orientation, hitch_angle, xt, yt, yaw_angle_trailer, time_step=0):
            """
            Initialisation of a state of a motion primitive.

            :param x: position in x axis
            :param y: position in y axis
            :param steering_angle: steering angle
            :param velocity: velocity
            :param orientation: orientation
            :param hitch_angle: hitch_angle
            :param xt: trailer position in x axis
            :param yt: trailer position in y axis
            :param yaw_angle_trailer: yaw_angle_trailer
            :param time_step: time step of the state
            """
            self.x = x
            self.y = y
            self.steering_angle = steering_angle
            self.velocity = velocity
            self.orientation = orientation
            self.hitch_angle = hitch_angle
            self.xt = xt
            self.yt = yt
            self.yaw_angle_trailer = yaw_angle_trailer
            self.time_step = time_step

        def __str__(self) -> str:
            """
            Returns the information of PrimitiveState.
            """
            return "pos: ({}, {})m, vel: {} m/s, yaw_tractor: {} rad, steer: {} rad, hitch: {} rad, trailer pos: ({}, {})m, yaw_trailer: {} rad".format(round(self.x, 2),
                                                                                                                                                        round(
                                                                                                                                                            self.y, 2),
                                                                                                                                                        round(
                                                                                                                                                            self.velocity, 2),
                                                                                                                                                        round(
                                                                                                                                                            self.orientation, 2),
                                                                                                                                                        round(
                                                                                                                                                            self.steering_angle, 2),
                                                                                                                                                        round(
                                                                                                                                                            self.hitch_angle, 2),
                                                                                                                                                        round(
                                                                                                                                                            self.xt, 2),
                                                                                                                                                        round(
                                                                                                                                                            self.yt, 2),
                                                                                                                                                        round(self.yaw_angle_trailer, 2))

    def __init__(self, state_initial: PrimitiveState, state_final: PrimitiveState, trajectory: Trajectory,
                 length_time_step):
        """
        Initialisation of a motion primitive.

        :param state_initial: initial state of the primitive
        :param state_final: final state of the primitive
        :param trajectory: trajectory of the primitive, which is a list of states between the initial and final states
        :param length_time_step: length of the time step of trajectory
        """
        self.state_initial = state_initial
        self.state_final = state_final
        self.trajectory = trajectory
        self.length_time_step = length_time_step

        # default vehicle type and model
        self.id_type_vehicle = 2
        self.type_vehicle = VehicleType.SEMI_TRAILER
        self.model_vehicle = VehicleModel.SEMI_TRAILER

        self._id = int(0)
        self._id_is_set = False
        # a list to store connectable successive primitives from this primitive
        self.list_successors = []
        self.list_ids_successors = []

    def __str__(self):
        return "Primitive: \n\t {}\t=>\n\t{}".format(str(self.state_initial), str(self.state_final))

    @property
    def id(self) -> int:
        """
        ID getter function.
        """
        assert self._id_is_set, f"The primitive id is not set!"
        return self._id

    @id.setter
    def id(self, primitive_id) -> None:
        """
        ID setter function.
        """
        assert isinstance(
            primitive_id, int), f"<ID> Provided id is not an instance of int, id = {primitive_id}"

        if not self._id_is_set:
            self._id = primitive_id
            self._id_is_set = True
        else:
            warnings.warn("Primitive ID is already set!")

    def print_info(self) -> None:
        """
        Prints the states of the primitive.
        """
        kwarg = {'position': np.array([self.state_initial.x, self.state_initial.y]),
                 'velocity': self.state_initial.velocity,
                 'steering_angle': self.state_initial.steering_angle,
                 'orientation': self.state_initial.orientation,
                 'hitch_angle': self.state_initial.hitch_angle,
                 'position_trailer': np.array([self.state_initial.xt, self.state_initial.yt]),
                 'yaw_angle_trailer': self.state_initial.yaw_angle_trailer,
                 'time_step': self.state_initial.time_step}
        state_initial = State(**kwarg)

        # add final state into the trajectory
        kwarg = {'position': np.array([self.state_final.x, self.state_final.y]),
                 'velocity': self.state_final.velocity,
                 'steering_angle': self.state_final.steering_angle,
                 'orientation': self.state_final.orientation,
                 'hitch_angle': self.state_final.hitch_angle,
                 'position_trailer': np.array([self.state_final.xt, self.state_final.yt]),
                 'yaw_angle_trailer': self.state_final.yaw_angle_trailer,
                 'time_step': self.state_final.time_step}
        state_final = State(**kwarg)

        print(state_initial)
        for state in self.trajectory.state_list:
            print(state)
        print(state_final)

    def mirror(self) -> None:
        """
        Mirrors the current primitive with regard to the x-axis.
        """
        self.state_final.y = -self.state_final.y
        self.state_final.orientation = -self.state_final.orientation
        self.state_final.hitch_angle = -self.state_final.hitch_angle
        self.state_final.steering_angle = -self.state_final.steering_angle
        self.state_final.yt = -self.state_final.yt
        self.state_final.yaw_angle_trailer = -self.state_final.yaw_angle_trailer
        for state in self.trajectory.state_list:
            state.position[1] = -state.position[1]
            state.position_trailer[1] = -state.position_trailer[1]
            state.orientation = -state.orientation
            state.hitch_angle = -state.hitch_angle
            state.steering_angle = -state.steering_angle
            state.yaw_angle_trailer = -state.yaw_angle_trailer

    def is_connectable(self, other: 'TruckMotionPrimitive') -> bool:
        """
        Any primitive whose initial state's velocity and steering angle are equal to those of the current primitive is
        deemed connectable.

        :param other: the motion primitive to which the connectivity is examined
        """

        return abs(self.state_final.velocity - other.state_initial.velocity) < 0.01 and abs(
            self.state_final.steering_angle - other.state_initial.steering_angle) < 0.01

    def attach_trajectory_to_state(self, state: State) -> List[State]:
        """
        Attaches the trajectory to the given state, and returns the new list of states.

        :param state: the state to which the trajectory will be attached
        """
        # deep copy to prevent manipulation of original data
        trajectory = deepcopy(self.trajectory)

        # rotate the trajectory by the orientation of the given state
        trajectory.translate_rotate(np.zeros(2), state.orientation)
        # translate the trajectory by the position of the given state
        trajectory.translate_rotate(state.position, 0)

        # as the initial state of the trajectory is exactly the given state, we thus pop out the initial state
        trajectory.state_list.pop(0)

        # we modify the time steps of the trajectory
        time_step_state = int(state.time_step)
        for state in trajectory.state_list:
            state.time_step += time_step_state

        return trajectory.state_list


class TruckMotionPrimitiveParser:
    """
    Class for motion primitive parsers, which parse and create motion primitives from given XML nodes.
    """

    @classmethod
    def create_from_node(cls, node_trajectory) -> TruckMotionPrimitive:
        """
        Creates a motion primitive from the given XML node.

        :param node_trajectory: node containing information of a trajectory

        """
        # read from xml node and create start state
        node_initial = node_trajectory.find('Initial')
        x_initial = float(node_initial.find('x').text)
        y_initial = float(node_initial.find('y').text)
        steering_angle_initial = float(
            node_initial.find('steering_angle').text)
        velocity_initial = float(node_initial.find('velocity').text)
        orientation_initial = float(node_initial.find('orientation').text)
        hitch_angle_initial = float(node_initial.find('hitch_angle').text)
        xt_initial = float(node_initial.find('xt').text)
        yt_initial = float(node_initial.find('yt').text)
        yaw_angle_trailer_initial = float(
            node_initial.find('yaw_angle_trailer').text)
        time_step_initial = int(node_initial.find('time_step').text)

        state_initial = TruckMotionPrimitive.PrimitiveState(x_initial, y_initial, steering_angle_initial, velocity_initial,
                                                            orientation_initial, hitch_angle_initial, xt_initial,
                                                            yt_initial, yaw_angle_trailer_initial,
                                                            time_step_initial)

        # read from xml node and create final state
        node_final = node_trajectory.find('Final')
        x_final = float(node_final.find('x').text)
        y_final = float(node_final.find('y').text)
        steering_angle_final = float(node_final.find('steering_angle').text)
        velocity_final = float(node_final.find('velocity').text)
        orientation_final = float(node_final.find('orientation').text)
        hitch_angle_final = float(node_final.find('hitch_angle').text)
        xt_final = float(node_final.find('xt').text)
        yt_final = float(node_final.find('yt').text)
        yaw_angle_trailer_final = float(
            node_final.find('yaw_angle_trailer').text)
        time_step_final = int(node_final.find('time_step').text)

        state_final = TruckMotionPrimitive.PrimitiveState(x_final, y_final, steering_angle_final, velocity_final,
                                                          orientation_final, hitch_angle_final, xt_final,
                                                          yt_final, yaw_angle_trailer_final,
                                                          time_step_final)

        # create trajectory from path node and initial/final states
        node_path = node_trajectory.find('Path')
        duration = node_trajectory.find('Duration')
        length_time_step = float(duration.text) / (len(node_path) + 1)

        trajectory = cls.create_trajectory(
            state_initial, state_final, node_path)

        # create and return motion primitive
        return TruckMotionPrimitive(state_initial, state_final, trajectory, length_time_step)

    @classmethod
    def create_trajectory(cls, state_initial: TruckMotionPrimitive.PrimitiveState,
                          state_final: TruckMotionPrimitive.PrimitiveState,
                          node_path) -> Trajectory:
        """
        Creates trajectory state list from the path values described in the xml file.

        :param state_initial: initial state of the trajectory
        :param state_final: final state of the trajectory
        :param node_path: xml node of the path of the trajectory
        """

        assert node_path is not None, "Input path node is empty!"

        # insert the initial state
        list_vertices = [(state_initial.x, state_initial.y)]
        list_steering_angles = [state_initial.steering_angle]
        list_velocities = [state_initial.velocity]
        list_orientation = [state_initial.orientation]
        list_hitch_angles = [state_initial.hitch_angle]
        list_trailer_vertices = [(state_initial.xt, state_initial.yt)]
        list_yaw_angle_trailer = [state_initial.yaw_angle_trailer]
        list_time_steps = [int(state_initial.time_step)]

        # insert trajectory states
        list_states_trajectory = node_path.findall('State')
        for state in list_states_trajectory:
            x = float(state.find('x').text)
            y = float(state.find('y').text)
            steering_angle = float(state.find('steering_angle').text)
            velocity = float(state.find('velocity').text)
            orientation = float(state.find('orientation').text)
            hitch_angle = float(state.find('hitch_angle').text)
            xt = float(state.find('xt').text)
            yt = float(state.find('yt').text)
            yaw_angle_trailer = float(state.find('yaw_angle_trailer').text)
            time_step = int(state.find('time_step').text)

            list_vertices.append((x, y))
            list_steering_angles.append(steering_angle)
            list_velocities.append(velocity)
            list_orientation.append(orientation)
            list_hitch_angles.append(hitch_angle)
            list_trailer_vertices.append((xt, yt))
            list_yaw_angle_trailer.append(yaw_angle_trailer)
            list_time_steps.append(time_step)

        # insert the final state
        list_vertices.append((state_final.x, state_final.y))
        list_steering_angles.append(state_final.steering_angle)
        list_velocities.append(state_final.velocity)
        list_orientation.append(state_final.orientation)
        list_hitch_angles.append(state_final.hitch_angle)
        list_trailer_vertices.append((state_final.xt, state_final.yt))
        list_yaw_angle_trailer.append(state_final.yaw_angle_trailer)
        list_time_steps.append(int(state_final.time_step))

        assert len(list_vertices) == len(list_steering_angles) == len(list_velocities) == len(list_orientation) == len(
            list_time_steps) == len(list_hitch_angles) == len(list_trailer_vertices) == len(list_yaw_angle_trailer), "The sizes of state elements should be equal!"

        # creates the trajectory of the primitive
        list_states_trajectory = []
        for i in range(len(list_vertices)):
            kwarg = {'position': np.array([list_vertices[i][0], list_vertices[i][1]]),
                     'velocity': list_velocities[i],
                     'steering_angle': list_steering_angles[i],
                     'orientation': list_orientation[i],
                     'hitch_angle': list_hitch_angles[i],
                     'position_trailer': np.array([list_trailer_vertices[i][0], list_trailer_vertices[i][1]]),
                     'yaw_angle_trailer': list_yaw_angle_trailer[i],
                     'time_step': list_time_steps[i]}

            # append states
            list_states_trajectory.append(State(**kwarg))

        return Trajectory(initial_time_step=int(list_time_steps[0]), state_list=list_states_trajectory)
