import copy
from collections import defaultdict

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle

from commonroad_rl.gym_commonroad.utils.scenario import get_road_edge


def get_all_connected_lanelets(scenario: Scenario) -> dict:
    """
    Create all possible lanes by merging predecessors and successors, then create a dict with its keys as lanelet id
    and values as connected lanelet ids.
    
    :return: dict
    """
    merged_lanelet_dict = defaultdict(set)
    for l in scenario.lanelet_network.lanelets:  # iterate in all lanelet in this scenario
        if not l.predecessor and not l.successor:  # the lanelet is a lane itself
            merged_lanelet_dict[l.lanelet_id].add(l.lanelet_id)
        elif not l.predecessor:
            max_lane_merge_range = 1000.0
            _, sub_lanelet_ids = Lanelet.all_lanelets_by_merging_successors_from_lanelet(l, scenario.lanelet_network,
                                                                                         max_lane_merge_range)
            for s in sub_lanelet_ids:
                for i in s:
                    merged_lanelet_dict[i].update(s)
    return merged_lanelet_dict


def generate_reset_config(scenario: Scenario, open_lane_ends) -> dict:
    """
    Generate a dict of reset configurations which contains obstacle lanelet ids, road edge, collision checker,
    lanelet boundary and lanelet connection dict.

    :param scenario: commonroad scenario
    :return:
    """
    (left_road_edge_lanelet_id, left_road_edge, right_road_edge_lanelet_id, right_road_edge) = get_road_edge(scenario)
    _, lanelet_boundary = create_road_boundary_obstacle(scenario, method="obb_rectangles", open_lane_ends=open_lane_ends)
    connected_lanelet_dict = get_all_connected_lanelets(scenario)

    meta_scenario = copy.deepcopy(scenario)
    meta_scenario.remove_obstacle(scenario.obstacles)
    reset_config = {"left_road_edge_lanelet_id_dict": left_road_edge_lanelet_id,
                    "left_road_edge_dict": left_road_edge,
                    "right_road_edge_lanelet_id_dict": right_road_edge_lanelet_id,
                    "right_road_edge_dict": right_road_edge,
                    "boundary_collision_object": lanelet_boundary,
                    "connected_lanelet_dict": connected_lanelet_dict,
                    "meta_scenario": meta_scenario}

    return reset_config
