from commonroad.common.util import Interval
from numpy import array

from commonroad_rl.gym_commonroad.utils.scenario import *
from commonroad_rl.tests.common.evaluation import *
from commonroad_rl.tests.common.marker import *


# @pytest.mark.parametrize(
#     ("angle", "expected"),
#     [
#         (np.pi * 2, 0),
#         (-3 / 2 * np.pi, np.pi / 2),
#         (7 / 2 * np.pi, -np.pi / 2),
#         (3 / 2 * np.pi, -np.pi / 2),
#     ],
# )
# @unit_test
# @functional
# def test_shift_orientation(angle, expected):
#     """
#     Tests the shift_orientation method
#     """
#     shifted_angle = shift_orientation(angle)
#     assert np.isclose(shifted_angle, expected)


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        (0, [1, 0]),
        (np.pi, [-1, 0]),
        (np.pi / 2, [0, 1]),
        (3 * np.pi / 2, [0, -1]),
        (-np.pi / 2, [0, -1]),
        (-np.pi, [-1, 0]),
        (2 * np.pi, [1, 0]),
        (-2 * np.pi, [1, 0]),
    ],
)
@unit_test
@functional
def test_approx_orientation_vector(orientation, expected):
    """
    Tests that the orientation vector for an orientation is well formed and normed
    """
    vec = approx_orientation_vector(orientation)
    assert np.all(np.isclose(vec, expected))
    assert np.isclose(np.linalg.norm(vec), 1)


# @pytest.mark.parametrize(
#     ("v1", "v2", "expected"),
#     [
#         ([0, 1], [0, 1], 0),
#         ([1, 0], [0, 1], np.pi / 2),
#         ([-1, 0], [0, 1], np.pi / 2),
#         ([1, 0], [0, -1], np.pi / 2),
#         ([-1, 0], [1, 0], np.pi),
#         ([1, 0], [-1, 0], np.pi),
#         (
#             [[0, 1], [1, 0], [-1, 0], [1, 0], [-1, 0], [1, 0]],
#             [[0, 1], [0, 1], [0, 1], [0, -1], [1, 0], [-1, 0]],
#             [0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi, np.pi],
#         ),
#     ],
# )
# @unit_test
# @functional
# def test_abs_angle_diff(v1, v2, expected):
#     """
#     Tests that angle diff always returns the inner angle
#     """
#     diff = abs_angle_diff(np.array(v1), np.array(v2))
#     assert np.all(np.isclose(diff, expected))
#

@pytest.mark.parametrize(
    ("v1", "v2", "expected"),
    [
        ([0, 1], [0, 1], 0),
        ([1, 0], [0, 1], np.pi / 2),
        ([-1, 0], [0, 1], -np.pi / 2),
        ([1, 0], [0, -1], -np.pi / 2),
        ([-1, 0], [1, 0], np.pi),
        ([1, 0], [-1, 0], np.pi),
        (
            [[0, 1], [1, 0], [-1, 0], [1, 0], [-1, 0], [1, 0]],
            [[0, 1], [0, 1], [0, 1], [0, -1], [1, 0], [-1, 0]],
            [0, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi, np.pi],
        ),
    ],
)
@unit_test
@functional
def test_angle_diff(v1, v2, expected):
    """
    Tests that angle diff always returns the inner angle
    """
    diff = angle_difference(np.array(v1), np.array(v2))
    assert np.all(np.isclose(diff, expected))


# @pytest.mark.parametrize(
#     ("orientation", "lanelet_vertices", "expected_id"),
#     [
#         (
#             6.280,
#             [
#                 (
#                     [[-1, 1], [1, 1]],
#                     [[-1, 0], [1, 0]],
#                     [[-1, -1], [1, -1]],
#                 ),  # orientation 0.0... uh oh!
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [0, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation 1.57 should not be closer!
#             ],
#             1,
#         ),
#         (
#             3 * np.pi / 2,
#             [
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [0, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation -1.57 should be closest
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [0, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation 1.57 should not be closer!
#             ],
#             1,
#         ),
#         # some trivial cases
#         (0, [([[-1, 1], [1, 1]], [[-1, 0], [1, 0]], [[-1, -1], [1, -1]])], 1),
#         (
#             0,
#             [
#                 (
#                     [[-1, 1], [1, 1]],
#                     [[-1, 0], [1, 0]],
#                     [[-1, -1], [1, -1]],
#                 ),  # orientation 0 a perfect match!
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [0, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation 1.57, way too far away!
#             ],
#             1,
#         ),
#         (
#             0,
#             [
#                 (
#                     [[-1, 1], [1, 1]],
#                     [[-1, 0], [1, -0.01]],
#                     [[-1, -1], [1, -1]],
#                 ),  # orientation 3.14... uh oh! => handled to -0.01
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [0, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation 1.57 should not be closer!
#             ],
#             1,
#         ),
#         (
#             0,
#             [
#                 (
#                     [[-1, 1], [1, 1]],
#                     [[-1, 0], [1, -0.01]],
#                     [[-1, -1], [1, -1]],
#                 ),  # orientation 0.01... uh oh!
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [0, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation 1.57 should not be closer!
#             ],
#             1,
#         ),
#         (
#             1.570,
#             [
#                 (
#                     [[-1, 1], [1, 1]],
#                     [[-1, 0], [1, 0]],
#                     [[-1, -1], [1, -1]],
#                 ),  # orientation 0 should not be closer
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [0, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation 1.57 should  be perfect
#             ],
#             2,
#         ),
#         (
#             1.570,
#             [
#                 (
#                     [[-1, 1], [1, 1]],
#                     [[-1, 0], [1, 0]],
#                     [[-1, -1], [1, -1]],
#                 ),  # orientation 0 should not be closer
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [0.01, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation 1.57 should  be perfect
#             ],
#             2,
#         ),
#         (
#             1.570,
#             [
#                 (
#                     [[-1, 1], [1, 1]],
#                     [[-1, 0], [1, 0]],
#                     [[-1, -1], [1, -1]],
#                 ),  # orientation 0 should not be closer
#                 (
#                     [[-1, -1], [-1, 1]],
#                     [[0, -1], [-0.01, 1]],
#                     [[1, -1], [1, 1]],
#                 ),  # orientation 1.57 should  be perfect
#             ],
#             2,
#         ),
#     ],
# )
# @unit_test
# @functional
# def test_get_lanelet_id_by_state_orientation(
#     orientation: float,
#     lanelet_vertices: List[Tuple[List[List[float]]]],
#     expected_id: int,
# ):
#     """
#     #TODO remove, moved to ObservationCollector
#
#     Test that get lanelet id by state makes correct use of orientations
#     """
#     s = State(position=np.array([0, 0]), orientation=orientation)
#     scenario = Scenario(0.1, "test_id")
#     for i, lv in enumerate(lanelet_vertices, start=1):
#         l = Lanelet(np.array(lv[0]), np.array(lv[1]), np.array(lv[2]), i)
#         scenario.add_objects(l)
#
#     lanelet_polygons = [
#         (l.lanelet_id, l.convert_to_polygon())
#         for l in scenario.lanelet_network.lanelets
#     ]
#     lanelet_polygons_sg = pycrcc.ShapeGroup()
#     for l_id, poly in lanelet_polygons:
#         lanelet_polygons_sg.add_shape(create_collision_object(poly))
#     lanelet_ids = sorted_lanelet_ids(
#         related_lanelets_by_state_realtime(s, lanelet_polygons, lanelet_polygons_sg),
#         s.orientation,
#         s.position,
#         scenario,
#     )
#     lanelet_id = -1 if len(lanelet_ids) == 0 else lanelet_ids[0]
#     assert expected_id == lanelet_id


@pytest.mark.parametrize(
    ("arr",),
    [
        (array([[60.8363, -101.74465], [54.0463, -105.99465]]),),
        (array([[60.8363, -101.74465], [54.05512039386607, -105.98912913758015]]),),
        (
            array(
                [
                    [136.1013, -52.64465],
                    [136.9463, -53.99465],
                    [138.9713, -55.94465],
                    [141.0563, -57.19465],
                    [144.0213, -58.59465],
                    [147.1013, -59.64465],
                    [152.3213, -60.39465],
                    [157.7213, -59.79465],
                    [163.6113, -58.04465],
                ]
            ),
        ),
        (
            array(
                [
                    [135.6863, 7.45535],
                    [135.0613, -1.19465],
                    [133.7763, -13.59465],
                    [132.1663, -27.94465],
                    [131.1563, -36.29465],
                ]
            ),
        ),
    ],
)
@unit_test
@functional
def test_ccosy_contains_orig(arr):
    """
    #TODO remove, moved to SurroundingObservation

    Makes sure the original vertices are actually always included in the resulting polyline
    and the new polyline ends and start at the same position as the original one
    """
    from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
    from commonroad_dc.geometry.util import resample_polyline
    arr = resample_polyline(arr, step=2.)
    res = CurvilinearCoordinateSystem(arr)
    with does_not_raise():
        for x in arr:
            # try that no error is thrown
            res.convert_to_curvilinear_coords(x[0], x[1])


dummy_time_step = Interval(0.0, 0.0)


# @pytest.mark.parametrize(
#     ("ego_angle", "expected_output"),
#     [
#         (0, 0),
#         (1, 1),
#         (6, 6 - 2 * np.pi),
#         (2 * np.pi, 0),
#         (7, 7 - 2 * np.pi),
#         (-1, -1),
#         (-2 * np.pi, 0),
#     ],
# )
# @unit_test
# @functional
# def test_get_lane_relative_heading(ego_angle, expected_output):
#     # TODO remove, has been moved to EgoObservation
#     dummy_state = {
#         "velocity": 0.0,
#         "position": np.array([0.0, 0.0]),
#         "yaw_rate": 0.0,
#         "slip_angle": 0.0,
#         "time_step": 0.0,
#     }
#     ego_vehicle_state = State(**dummy_state, orientation=ego_angle)
#     ego_vehicle_lanelet = Lanelet(
#         array([[0.0, 1.0], [1.0, 1.0]]),
#         array([[0.0, 0.0], [1.0, 0.0]]),
#         array([[0.0, -1.0], [1.0, -1.0]]),
#         0,
#     )
#     relative_angle = get_lane_relative_heading(ego_vehicle_state, ego_vehicle_lanelet)
#     assert np.isclose(relative_angle, expected_output)


# @pytest.mark.parametrize(
#     ("state_list", "expected_output"),
#     [
#         ([], []),  # empty list
#         (
#             [State(orientation=0, velocity=0)],
#             [0],
#         ),  # list with one element, default steering angle
#         (
#             [State(orientation=1, velocity=1), State(orientation=1.5, velocity=2)],
#             [0.6732513654862752, 0.6732513654862752],
#         ),  # complex list witho appropriate steering angles
#         (
#             [State(orientation=1, velocity=1), State(orientation=2.0, velocity=2)],
#             [0.91, 0.91],
#         ),  # complex list with out-of-bounds steering angles
#         (
#             [
#                 State(orientation=1, velocity=1),
#                 State(orientation=1.5, steering_angle=0, velocity=2),
#                 State(orientation=2, velocity=2),
#             ],
#             [0.6732513654862752, 0, 0.5390728256069625],
#         ),  # complex list with one given steering angle
#     ],
# )
# @unit_test
# @functional
# def test_interpolate_steering_angles(state_list, expected_output):
#     vehicle_type = VehicleType.FORD_ESCORT
#     parameters = VehicleParameterMapping[vehicle_type.name].value
#     interpolated_state_list = interpolate_steering_angles(state_list, parameters, 1)
#     steering_angles = [state.steering_angle for state in interpolated_state_list]
#
#     assert (
#         np.all(np.isclose(steering_angles, expected_output))
#         or steering_angles == expected_output
#     )
