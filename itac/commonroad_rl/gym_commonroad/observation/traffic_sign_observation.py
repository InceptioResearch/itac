from collections import defaultdict, OrderedDict
from typing import Union, Dict, List, Tuple

import gym
import numpy as np
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficSign, TrafficSignIDGermany
from commonroad_rl.gym_commonroad.action.vehicle import Vehicle
from commonroad_rl.gym_commonroad.observation.observation import Observation
from numpy import ndarray
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from shapely.geometry import Point, LineString


class TrafficSignObservation(Observation):
    def __init__(self, configs: Dict, configs_name: str = "traffic_sign_configs"):
        # Read configs
        configs = configs[configs_name]
        self.observe_stop_sign: bool = configs.get("observe_stop_sign")
        self.observe_yield_sign: bool = configs.get("observe_yield_sign")
        self.observe_priority_sign: bool = configs.get("observe_priority_sign")
        self.observe_right_of_way_sign: bool = configs.get("observe_right_of_way_sign")

        self.observation_dict = OrderedDict()
        self.observation_history_dict = defaultdict(list)
        self.traffic_sign_id_list = []
        self.traffic_sign_id_list_successor = []

    def build_observation_space(self) -> OrderedDict:
        observation_space_dict = OrderedDict()

        if self.observe_stop_sign:
            observation_space_dict["stop_sign"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
            observation_space_dict["stop_sign_distance_long"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        if self.observe_yield_sign:
            observation_space_dict["yield_sign"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
            observation_space_dict["yield_sign_distance_long"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        if self.observe_priority_sign:
            observation_space_dict["priority_sign"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
            observation_space_dict["priority_sign_distance_long"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                   dtype=np.float32)
        if self.observe_right_of_way_sign:
            observation_space_dict["right_of_way_sign"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
            observation_space_dict["right_of_way_sign_distance_long"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                       dtype=np.float32)
        return observation_space_dict

    def observe(self, scenario: Scenario, ego_vehicle: Vehicle, ego_vehicle_lanelet: Lanelet,
                local_ccosy: Union[None, CurvilinearCoordinateSystem] = None) -> Union[ndarray, Dict]:
        """
        Create traffic sign observation for given state in an environment.
        """
        self.traffic_sign_id_list, self.traffic_sign_id_list_successor = \
            self._get_traffic_signs_on_lanelet_and_successor(scenario, ego_vehicle_lanelet)

        if self.observe_stop_sign:
            self._observe_traffic_sign("stop_sign", TrafficSignIDGermany.STOP, ego_vehicle,
                                       ego_vehicle_lanelet, scenario, True)

        if self.observe_yield_sign:
            self._observe_traffic_sign("yield_sign", TrafficSignIDGermany.YIELD, ego_vehicle,
                                       ego_vehicle_lanelet, scenario, True)

        if self.observe_priority_sign:
            self._observe_traffic_sign("priority_sign", TrafficSignIDGermany.PRIORITY, ego_vehicle,
                                       ego_vehicle_lanelet, scenario)

        if self.observe_right_of_way_sign:
            self._observe_traffic_sign("right_of_way_sign", TrafficSignIDGermany.RIGHT_OF_WAY, ego_vehicle,
                                       ego_vehicle_lanelet, scenario)

        return self.observation_dict

    def _observe_traffic_sign(self, sign_entry: str, sign_type: TrafficSignIDGermany, ego_vehicle: Vehicle,
                              ego_vehicle_lanelet: Lanelet, scenario: Scenario, successor=False):
        """
        finds traffic sign in the current or successor lanelets and distance to the sign (-1 if not present)
        :param sign_entry: sign entry name in the observation dictionary
        :param sign_type: traffic sign id enum
        :param ego_vehicle: the ego vehicle
        :param ego_vehicle_lanelet: the lanelet of the ego vehicle
        :param scenario: scenario with traffic signs
        """
        sign_distance_entry = f"{sign_entry}_distance_long"

        self.observation_dict[sign_entry] = np.array([0])
        self.observation_dict[sign_distance_entry] = np.array([-1])

        search_list = self.traffic_sign_id_list_successor if successor else self.traffic_sign_id_list
        # add traffic signs on current lanelet and successor lanelets
        for sign_id in search_list:
            sign = scenario.lanelet_network.find_traffic_sign_by_id(sign_id)
            if sign.traffic_sign_elements[0].traffic_sign_element_id == sign_type:
                self.observation_dict[sign_entry] = np.array([1])

                # calculate distance from ego vehicle head to the forward boundary of current lanelet
                forward_distance = self.get_distance_to_traffic_sign_long(scenario, ego_vehicle,
                                                                          ego_vehicle_lanelet, sign, successor)
                self.observation_dict[sign_distance_entry] = np.array([forward_distance])
                return

    @staticmethod
    def get_distance_to_traffic_sign_long(scenario: Scenario, ego_vehicle: Vehicle,
                                          ego_vehicle_lanelet: Lanelet,
                                          sign: TrafficSign, successor=False) -> float:
        """
        get distance from the head of ego vehicle to forward boundary of current lanelet
        :param scenario: current scenario
        :param ego_vehicle: the ego vehicle
        :param ego_vehicle_lanelet: lanelet of ego vehicle
        :param sign: observed traffic sign on current or successor lanelet
        :param successor: whether including traffic sign on the successor lanelet, True given stop and yield sign

        :return: distance from ego vehicle to forward boundary, returns -1.0 if the sign could not be found
        """
        # Check for stop line in ego and successor lanelets
        stop_line_present = \
            ego_vehicle_lanelet.stop_line and ego_vehicle_lanelet.stop_line.traffic_sign_ref == sign.traffic_sign_id
        if stop_line_present:
            stop_line = ego_vehicle_lanelet.stop_line
            forward_boundary = LineString((stop_line.start, stop_line.end))
        elif successor:
            for suc_id in ego_vehicle_lanelet.successor:
                suc_lanelet = scenario.lanelet_network.find_lanelet_by_id(suc_id)
                stop_line_present = \
                    suc_lanelet.stop_line and suc_lanelet.stop_line.traffic_sign_ref == sign.traffic_sign_id
                if stop_line_present:
                    stop_line = suc_lanelet.stop_line
                    forward_boundary = LineString((stop_line.start, stop_line.end))
                    break

        if not stop_line_present:
            if sign.traffic_sign_id in ego_vehicle_lanelet.traffic_signs:
                forward_boundary = LineString(
                    (ego_vehicle_lanelet.right_vertices[-1], ego_vehicle_lanelet.left_vertices[-1]))
            elif successor:
                for suc_id in ego_vehicle_lanelet.successor:
                    suc_lanelet = scenario.lanelet_network.find_lanelet_by_id(suc_id)
                    if sign.traffic_sign_id in suc_lanelet.traffic_signs:
                        forward_boundary = LineString((suc_lanelet.right_vertices[-1], suc_lanelet.left_vertices[-1]))
                        break
                # Couldn't find traffic sign, something went wrong (possibly mismatched successor flags when searching
                # for the sign and calculating the distance)
                try:
                    forward_boundary
                except NameError:
                    return -1.0
            # Traffic sign can't be found with current settings
            else:
                return -1.0

        # Calculate the actual distance
        ego_vehicle_point = Point(ego_vehicle.state.position)
        nearest_point = forward_boundary.interpolate(forward_boundary.project(ego_vehicle_point))
        forward_distance = nearest_point.distance(ego_vehicle_point)

        return abs(forward_distance)

    def _get_traffic_signs_on_lanelet_and_successor(self, scenario: Scenario, ego_vehicle_lanelet: Lanelet) \
            -> Tuple[List[int], List[int]]:
        """
        get traffic signs which located on current and successor lanelets
        :param scenario: the current scenario
        :param ego_vehicle_lanelet: lanelet of ego vehicle

        :return: list of traffic signs on current lanelet and a list of traffic signs on current and successor lanelets
        """
        traffic_sign_list = []
        traffic_sign_list += list(ego_vehicle_lanelet.traffic_signs)
        traffic_sign_list_successor = list.copy(traffic_sign_list)
        for successor_id in ego_vehicle_lanelet.successor:
            successor_lanelet = scenario.lanelet_network.find_lanelet_by_id(successor_id)
            if len(successor_lanelet.traffic_signs) != 0:
                traffic_sign_list_successor += list(successor_lanelet.traffic_signs)
        return traffic_sign_list, traffic_sign_list_successor

    def draw(self):
        pass


if __name__ == "__main__":
    import yaml
    from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

    config_file = PATH_PARAMS["configs"]["commonroad-v1"]
    with open(config_file, "r") as config_file:
        config = yaml.safe_load(config_file)
    configs = config["env_configs"]
    traffic_sign_observation = TrafficSignObservation(configs)
    print(traffic_sign_observation)
