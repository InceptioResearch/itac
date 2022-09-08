"""
Unit tests of the module gym_commonroad.observation
"""

import numpy as np
from commonroad.scenario.scenario import Scenario, ScenarioID
from commonroad.scenario.trajectory import State
from commonroad_dc import pycrcc
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object

from commonroad_rl.gym_commonroad.observation import Observation
from commonroad_rl.tests.common.marker import *

# ====================================================================================================================
#
#                                    ObservationCollector
#
# ====================================================================================================================

from typing import List, Tuple
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad_rl.gym_commonroad.observation import ObservationCollector
from commonroad_rl.gym_commonroad.utils.navigator import Navigator


@pytest.mark.parametrize(("orientation", "lanelet_vertices", "expected_id"), [
    (6.280, [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]],),  # orientation 0.0... uh oh!
             ([[-1, -1], [-1, 1]], [[0, -1], [0, 1]], [[1, -1], [1, 1]],),  # orientation 1.57 should not be closer!
             ], 1,), (3 * np.pi / 2, [([[-1, -1], [-1, 1]], [[0, -1], [0, 1]], [[1, -1], [1, 1]],),
                                      # orientation -1.57 should be closest
                                      ([[-1, -1], [-1, 1]], [[0, -1], [0, 1]], [[1, -1], [1, 1]],),
                                      # orientation 1.57 should not be closer!
                                      ], 1,),  # some trivial cases
    (0, [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]])], 1),
    (0, [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]],),  # orientation 0 a perfect match!
         ([[-1, -1], [-1, 1]], [[0, -1], [0, 1]], [[1, -1], [1, 1]],),  # orientation 1.57, way too far away!
         ], 1,), (0, [([[-1, 1], [1, 1]], [[-1, 0], [1, -0.01]], [[-1, -1], [1, -1]],),
                      # orientation 3.14... uh oh! => handled to -0.01
                      ([[-1, -1], [-1, 1]], [[0, -1], [0, 1]], [[1, -1], [1, 1]],),
                      # orientation 1.57 should not be closer!
                      ], 1,),
    (0, [([[-1, 1], [1, 1]], [[-1, 0], [1, -0.01]], [[-1, -1], [1, -1]],),  # orientation 0.01... uh oh!
         ([[-1, -1], [-1, 1]], [[0, -1], [0, 1]], [[1, -1], [1, 1]],),  # orientation 1.57 should not be closer!
         ], 1,),
    (1.570, [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]],),  # orientation 0 should not be closer
             ([[-1, -1], [-1, 1]], [[0, -1], [0, 1]], [[1, -1], [1, 1]],),  # orientation 1.57 should  be perfect
             ], 2,),
    (1.570, [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]],),  # orientation 0 should not be closer
             ([[-1, -1], [-1, 1]], [[0, -1], [0.01, 1]], [[1, -1], [1, 1]],),  # orientation 1.57 should  be perfect
             ], 2,),
    (1.570, [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]],),  # orientation 0 should not be closer
             ([[-1, -1], [-1, 1]], [[0, -1], [-0.01, 1]], [[1, -1], [1, 1]],),  # orientation 1.57 should  be perfect
             ], 2,), ], )
@unit_test
@functional
def test_get_lanelet_id_by_state_orientation(orientation: float, lanelet_vertices: List[Tuple[List[List[float]]]],
                                             expected_id: int, ):
    """
    Test that get lanelet id by state makes correct use of orientations
    """
    s = State(position=np.array([0, 0]), orientation=orientation)
    scenario = Scenario(0.1, ScenarioID("test_id"))
    for i, lv in enumerate(lanelet_vertices, start=1):
        l = Lanelet(np.array(lv[0]), np.array(lv[1]), np.array(lv[2]), i)
        scenario.add_objects(l)

    lanelet_polygons = [(l.lanelet_id, l.convert_to_polygon()) for l in scenario.lanelet_network.lanelets]
    lanelet_polygons_sg = pycrcc.ShapeGroup()
    for l_id, poly in lanelet_polygons:
        lanelet_polygons_sg.add_shape(create_collision_object(poly))
    lanelet_ids = Navigator.sorted_lanelet_ids(
        ObservationCollector._related_lanelets_by_state(s, lanelet_polygons, lanelet_polygons_sg),
        s.orientation, s.position, scenario)
    lanelet_id = -1 if len(lanelet_ids) == 0 else lanelet_ids[0]
    assert expected_id == lanelet_id


@pytest.mark.parametrize(("lanelet_vertices", "expected_ids"), [(
        [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]],),
         ([[-1, -1], [-1, 1]], [[0, -1], [0, 1]], [[1, -1], [1, 1]],), ], [1, 2],), (
        [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]],),
         ([[2, 2], [2, 4]], [[3, 2], [3, 4]], [[4, 2], [4, 4]],), ], [1],), (
        [([[2, 2], [2, 4]], [[3, 2], [3, 4]], [[4, 2], [4, 4]],),
         ([[2, 2], [2, 4]], [[3, 2], [3, 4]], [[4, 2], [4, 4]],), ], [],), (
        [([[2, 2], [2, 4]], [[3, 2], [3, 4]], [[4, 2], [4, 4]],),
         ([[2, 2], [2, 4]], [[3, 2], [3, 4]], [[4, 2], [4, 4]],),
         ([[2, 5], [2, 4]], [[3, 2], [3, 4]], [[4, 2], [4, 4]],),
         ([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]],), ], [4],), ])
@unit_test
@functional
def test_related_lanelets_by_state(lanelet_vertices: List[Tuple[List[List[float]]]], expected_ids: List[int], ):
    """
    Test that related lanelet id by state
    """
    s = State(position=np.array([0, 0]), orientation=0)
    scenario = Scenario(dt=0.1, scenario_id=ScenarioID("test_id"))
    for i, lv in enumerate(lanelet_vertices, start=1):
        l = Lanelet(np.array(lv[0]), np.array(lv[1]), np.array(lv[2]), i)
        scenario.add_objects(l)

    lanelet_polygons = [(l.lanelet_id, l.convert_to_polygon()) for l in scenario.lanelet_network.lanelets]
    lanelet_polygons_sg = pycrcc.ShapeGroup()
    for l_id, poly in lanelet_polygons:
        lanelet_polygons_sg.add_shape(create_collision_object(poly))

    res = ObservationCollector._related_lanelets_by_state(s, lanelet_polygons, lanelet_polygons_sg)
    assert isinstance(res, list)
    assert all([a == b for a, b in zip(res, expected_ids)])


# TODO: update expected value after ccosy update
@pytest.mark.parametrize(("lanelets",
                          "cartesian_coordinates", "expected_curvilinear_coordinates", "desired_merged_lanelet"),
                         [([Lanelet(lanelet_id=0, left_vertices=np.array([[-1, 1], [1, 1]]),
                                    center_vertices=np.array([[-1, 0], [1, 0]]),
                                    right_vertices=np.array([[-1, -1], [1, -1]]), adjacent_right_same_direction=1),
                            Lanelet(lanelet_id=1, left_vertices=np.array([[-1, -1], [1, -1]]),
                                    center_vertices=np.array([[-1, -2], [1, -2]]),
                                    right_vertices=np.array([[-1, -3], [1, -3]]), adjacent_left_same_direction=0)],
                           np.array([0, 0]), np.array([1, 2]), 1),
                          ([Lanelet(lanelet_id=0, left_vertices=np.array([[-1, 1], [1, 1]]),
                                    center_vertices=np.array([[-1, 0], [1, 0]]),
                                    right_vertices=np.array([[-1, -1], [1, -1]]),
                                    successor=[1]),
                            Lanelet(lanelet_id=1, left_vertices=np.array([[1, 1], [3, 1]]),
                                    center_vertices=np.array([[1, 0], [3, 0]]),
                                    right_vertices=np.array([[1, -1], [3, -1]]),
                                    predecessor=[0])],
                           np.array([0, 0]), np.array([1, 0]), 1),
                          ([Lanelet(lanelet_id=0, left_vertices=np.array([[-1, 1], [1, 1]]),
                                    center_vertices=np.array([[-1, 0], [1, 0]]),
                                    right_vertices=np.array([[-1, -1], [1, -1]]), successor=[1],
                                    adjacent_right_same_direction=2),
                            Lanelet(lanelet_id=1, left_vertices=np.array([[1, 1], [3, 1]]),
                                    center_vertices=np.array([[1, 0], [3, 0]]),
                                    right_vertices=np.array([[1, -1], [3, -1]]), predecessor=[0]),
                            Lanelet(lanelet_id=2, left_vertices=np.array([[-1, -1], [1, -1]]),
                                    center_vertices=np.array([[-1, -2], [1, -2]]),
                                    right_vertices=np.array([[-1, -3], [1, -3]]), adjacent_left_same_direction=0,
                                    adjacent_right_same_direction=3),
                            Lanelet(lanelet_id=3, left_vertices=np.array([[-1, -3], [1, -3]]),
                                    center_vertices=np.array([[-1, -4], [1, -4]]),
                                    right_vertices=np.array([[-1, -5], [1, -5]]), adjacent_left_same_direction=2)],
                           np.array([2, 4]), np.array([3, 4]), 1)])
@unit_test
@functional
def test_get_local_curvi_cosy(lanelets: List[Lanelet], cartesian_coordinates: np.ndarray,
                              expected_curvilinear_coordinates: np.ndarray, desired_merged_lanelet: int):
    """test ObservationCollector get_local_curvi_cosy function"""
    ego_vehicle_lanelet_id = 1
    max_lane_merge_range = 5000.

    lanelet_network = LaneletNetwork()
    for lanelet in lanelets:
        lanelet_network.add_lanelet(lanelet)

    ref_path_dict = dict()

    curvi_cosy, ref_merged_lanelet = ObservationCollector.get_local_curvi_cosy(lanelet_network, ego_vehicle_lanelet_id,
                                                                               ref_path_dict, max_lane_merge_range)
    calculated_curv_coords = curvi_cosy.convert_to_curvilinear_coords(*cartesian_coordinates)

    assert all(np.isclose(calculated_curv_coords, expected_curvilinear_coordinates, atol=1e-3))
    if not ref_merged_lanelet is None:
        assert np.isclose(ref_merged_lanelet.lanelet_id, desired_merged_lanelet)
    else:
        assert desired_merged_lanelet is None
