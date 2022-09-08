import numpy as np

# GPC
import Polygon as gpc
import Polygon.Utils

# commonroad
from commonroad.geometry.shape import Polygon, ShapeGroup, Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State

import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.boundary import triangle_builder, lanelet_bounds, scenario_bounds


def construct(scenario, build_order, boundary_margin=20):
    """Construct road bounding boxes, using
        :param scenario: CommonRoad scenario
        :param build_order: list of strings that describe which shapes to construct,
            see build(orders) for the possible options
        :param boundary_margin: the axis aligned rectangle, which surrounds all lanelets of the scenario,
            is enlarged at each corner by this value

    """

    def build(orders):
        if orders == []:
            return

        # helper functions to handle the two different types of road triangles
        def copy_into_road(shapegroup):
            """copy the elements of a shapegroup into the road shapegroup"""
            for object in shapegroup.unpack():
                road.add_shape(object)

        def build_simple_triangles():
            triangle_builder.build_simple_triangles(lanelet_network.lanelets, simple_triangles)
            copy_into_road(simple_triangles)

        def build_section_triangles():
            for lane_polyline in lanelet_bounds.lane_hull(lanelet_network):
                lane_polyline = lane_polyline.tolist()
                triangle_builder.triangulate(lanelet_bounds.polyline_edges(lane_polyline), [], None, section_triangles,
                                             {})
            copy_into_road(section_triangles)

        def help_triangulate(params):
            # hull: flag that indicates whether a hull is build around the road to bound the triangulation area
            # -True: hull is built from a minkowski sum
            # -False: the corners of the scenario are used instead
            hull = params.get('hull', True)
            offset = params.get('hull_offset', 2)
            bounds = [bound for bound in lanelet_bounds.pairwise_bounds(lanelet_network)]

            if hull:
                points = [point for lane_section in lanelet_bounds.lane_hull(lanelet_network) for point in lane_section]
                hull_points = simple_minkowski(points, offset, road)
                triangle_builder.triangulate(bounds, hull_points, road, triangulation, params)
            else:
                corner_vertices = [(corners[0], corners[2]), (corners[1], corners[2]),
                                   (corners[1], corners[3]), (corners[0], corners[3])]
                corner_edges = lanelet_bounds.polyline_edges(corner_vertices)
                triangle_builder.triangulate(bounds + corner_edges, [], road, triangulation, params)

        def build_critical_area(params):
            width = params.get('width', 2)
            triangle_builder.build_offset_section_triangles(lanelet_network, critical_area, width)

        params = {}
        options = {
            "simple_triangles": lambda: build_simple_triangles(),
            "section_triangles": lambda: build_section_triangles(),
            "triangulation": lambda: help_triangulate(params),
            "critical_area": lambda: build_critical_area(params),
        }
        for order in orders:
            params = {}
            if not type(order) == str:
                # the first element of order determines which option is taken, the following is a dictionary with
                # parameters passed to the functions

                params = order[1]
                order = order[0]
            if order in options:
                options.get(order)()

    def handle_return():
        dict = {
            "simple_triangles": simple_triangles,
            "section_triangles": section_triangles,
            "triangulation": triangulation,
            "critical_area": critical_area,
            "corners": corners,
        }

        return dict

    def simple_minkowski(points, radius, collision_mesh):
        """
        Basic algorithm to enlarge a polygon by a constant amount.
        Can be interpreted as a minkowski sum between the polygon defined by the points and a circle with the radius.
        Computes the normal for each point and offsets them by half the radius.
        :param points: List of points of the polygon.
        :param radius: Distance by which the polygon is enlarged. Each point is moved by half the radius to the outside.
        :param collision_mesh: Collision objects that define the inside area of the polygon.
        :return: point list with offset points
        """
        normals = lanelet_bounds.polyline_normals(points)
        offsets = [normal * radius / 2 for normal in normals]
        # The normals could point inside or outside the polygon, offset in both directions
        sum_points = [[p - offset, p + offset] for offset, p in zip(offsets, points)]

        # uncomment for visualization
        # for p, offsets in zip(points, sum_points):
        #    draw_object(rectangle_builder.get_rectangle(0.1, p, offsets[0]))
        #    draw_object(rectangle_builder.get_rectangle(0.05, p, offsets[1]))

        # Flatten
        sum_points = [point for tuple in sum_points for point in tuple]

        return [p for p in sum_points if not pycrcc.Point(*p).collide(collision_mesh)]

    lanelet_network = scenario.lanelet_network
    lanelets = lanelet_network.lanelets

    def init():
        return pycrcc.ShapeGroup()

    road = init()
    simple_triangles = init()
    section_triangles = init()
    triangulation = init()
    critical_area = init()

    # min and max coordinates of the scenario
    corners = scenario_bounds.calc_corners(lanelets, boundary_margin)

    build(build_order)

    return handle_return()


def construct_boundary_obstacle_triangulation(scenario: Scenario,
                                              build=['section_triangles', ('triangulation', {'max_area': '1'})],
                                              submethod_return='triangulation', boundary_margin=20):
    return construct(scenario, build, boundary_margin)[submethod_return]


def _convert_tristrips_to_triangles(tri_strips, bb, axis):
    return pycrcc.Util.convert_tristrips_to_triangles(tri_strips, bb, axis)


def construct_boundary_obstacle_aligned_triangulation(scenario: Scenario, axis=0, boundary_margin=20):
    """
    Creates the road boundary for the given scenario.
    Divides the polygons representing the road boundary into axis-aligned tiles and triangulates the tiles separately into the triangles having one side parallel to the other axis (in order to reduce the number of candidate triangles for collision checks).

    :param scenario: The input scenario to be triangulated
    :param axis: 0: Use vertical tiles, 1: Use horizontal tiles, 'auto': Use horizontal tiles if width<height.
    :param boundary_margin Size of margin for the calculation of scenario corners.
    :return: ShapeGroup with the constructed Collision Checker triangles.
    """

    lanelet_network = scenario.lanelet_network
    lanelets = lanelet_network.lanelets
    corners = scenario_bounds.calc_corners(lanelets, boundary_margin)

    corner_vertices = [(corners[0], corners[2]), (corners[1], corners[2]),
                       (corners[1], corners[3]), (corners[0], corners[3])]
    corner_edges = lanelet_bounds.polyline_edges(corner_vertices)

    polyline_corners = [np.asarray(el) for el in corner_vertices]
    gpc_corners = gpc.Polygon(polyline_corners)

    def whole_polygon_setup(polylines):
        polygon = gpc.Polygon()
        for polyline in polylines:
            polygon = polygon | gpc.Polygon(polyline)

        return polygon

    lane_rep2 = whole_polygon_setup(lanelet_bounds.lane_hull(lanelet_network))

    gpc_corners = gpc_corners - lane_rep2

    bb = gpc_corners.boundingBox()
    if (axis == 'auto'):

        width = bb[1] - bb[0]
        height = bb[3] - bb[2]
        if (width < height):
            axis = 1

    if (axis == 1):
        bb = gpc_corners.boundingBox()
        x_c = (bb[0] + bb[1]) / 2
        y_c = (bb[2] + bb[3]) / 2
        diff_c = y_c - x_c
        sum_c = x_c + y_c
        rot = np.reshape(np.asarray([0, 1, -1, 0]), (2, 2))
        transl = np.repeat(np.reshape(np.asarray([-diff_c, sum_c]), (2, 1)), 3, axis=1)
        gpc_corners.rotate(np.pi / 2)
    else:
        axis = 0

    gpc_corners_tiles = gpc.Utils.tileEqual(gpc_corners, 32)

    sg_tri_res = pycrcc.ShapeGroup()

    tri_strips = list()

    for poly_tile in gpc_corners_tiles:
        tri_strips.append(poly_tile.triStrip())

    sg_tri_res = _convert_tristrips_to_triangles(tri_strips, bb, axis)

    return sg_tri_res


def _GPCToCollisionPolygons(GPC_polygons, triangulate=True):
    sg_poly = pycrcc.ShapeGroup()
    for GPC_polygon in GPC_polygons:
        cont_solid = list()
        cont_holes = list()
        triangles = pycrcc.ShapeGroup()
        for ind, el in enumerate(GPC_polygon):
            if GPC_polygon.isSolid(ind):
                cont_solid.append(gpc.Polygon(el))
            else:
                cont_holes.append(gpc.Polygon(el))

        for ind, el in enumerate(cont_solid):

            outer_boundary_list = list(el[0])
            if (el.orientation()[0] == -1):
                outer_boundary_list.reverse()
            inner_boundaries_list = list()
            if triangulate:
                triangle_builder.triangulate(lanelet_bounds.polyline_edges(outer_boundary_list), [], None,
                                             triangles, {})

            for hole in cont_holes:
                if hole.overlaps(el):
                    list_hole = list(hole[0])
                    if (hole.orientation()[0] == 1):
                        list_hole.reverse()
                    inner_boundaries_list.append(list_hole)
            polygon = pycrcc.Polygon(outer_boundary_list, inner_boundaries_list, triangles.unpack())
            sg_poly.add_shape(polygon)

    return sg_poly


def construct_boundary_obstacle_obb_rectangles(scenario: Scenario, width=1e-5, open_lane_ends=True):
    """
    In contrast to triangulation, this method creates a set of oriented rectangles that separates the road and the road boundary.
    To compute the set, we perform a polygon union operation for all lanelets and then use oriented rectangles to represent each line segment of the resulant polygons with holes.
    To compute the set, we first compute the union of all lanelets in the road network and then extract the inner and outer contours of the resulting polygon.
    Afterwards, we create an oriented rectangle for each line segment of the inner and outer contours.
    The oriented rectangles symmetrically overapproximate each line segment.
    In this way, we can reduce the false positive rate at road forks and merges.

    :param scenario: The input scenario to be triangulated.
    :param width: Width of the generated rectangles. Default: 1e-5.
    :param open_lane_ends: Do not create the boundary for the beginning and the end of the lanelets that have no successor or predecessor.
    :return: ShapeGroup with the Collision Checker rectangle objects.
    """

    def remove_border_rect(sg_rect):
        midpoints = list()
        sg_spheres = pycrcc.ShapeGroup()
        lanelets = scenario.lanelet_network.lanelets
        for ind, lanelet in enumerate(lanelets):
            if len(lanelet.successor) == 0:
                pt1 = (lanelet.right_vertices[-1])
                pt2 = (lanelet.left_vertices[-1])
                midpoints.append((pt1 + pt2) / 2)

            if len(lanelet.predecessor) == 0:
                pt1 = (lanelet.right_vertices[0])
                pt2 = (lanelet.left_vertices[0])

                midpoints.append((pt1 + pt2) / 2)

        for el in midpoints:
            sg_spheres.add_shape(pycrcc.Circle(1e-4, el[0], el[1]))

        overlap_map = sg_rect.overlap_map(sg_spheres)

        sg_rect_new = pycrcc.ShapeGroup()
        shapes = sg_rect.unpack()

        for el in overlap_map.keys():
            if (len(overlap_map[el]) == 0):
                sg_rect_new.add_shape(shapes[el])
        return sg_rect_new

    def whole_polygon_setup(polylines):
        polygon = gpc.Polygon()
        for polyline in polylines:
            polygon = polygon | gpc.Polygon(polyline)

        return polygon

    def polygon_contours_build_rectangles(sg_poly, width):
        return pycrcc.Util.polygon_contours_build_rectangles(sg_poly, width)

    lanelet_network = scenario.lanelet_network
    whole_lane_polygon = whole_polygon_setup(lanelet_bounds.lane_hull(lanelet_network))
    whole_lane_polygon.simplify()

    fill_holes = False

    if fill_holes:
        whole_lane_polygon = gpc.Utils.fillHoles(whole_lane_polygon)

    sg_poly = _GPCToCollisionPolygons([whole_lane_polygon], triangulate=False)

    sg_rectangles = polygon_contours_build_rectangles(sg_poly, width)

    if open_lane_ends:
        return remove_border_rect(sg_rectangles)
    else:
        return sg_rectangles

def postprocess_create_static_obstacle_triangles(scenario: Scenario, shape_group: pycrcc.ShapeGroup):
    initial_state = State(position=np.array([0, 0]), orientation=0.0, time_step=0)
    road_boundary_shape_list = list()
    for r in shape_group.unpack():
        p = Polygon(np.array(r.vertices()))
        road_boundary_shape_list.append(p)
    road_boundary_obstacle = StaticObstacle(obstacle_id=scenario.generate_object_id(),
                                            obstacle_type=ObstacleType.ROAD_BOUNDARY,
                                            obstacle_shape=ShapeGroup(road_boundary_shape_list),
                                            initial_state=initial_state)
    return road_boundary_obstacle


def postprocess_create_static_obstacle_obb_rectangles(scenario: Scenario, shape_group: pycrcc.ShapeGroup):
    initial_state = State(position=np.array([0, 0]), orientation=0.0, time_step=0)
    road_boundary_shape_list = list()
    for r in shape_group.unpack():
        p = Rectangle(r.r_x() * 2, r.r_y() * 2, r.center(), r.orientation())
        road_boundary_shape_list.append(p)
    road_boundary_obstacle = StaticObstacle(obstacle_id=scenario.generate_object_id(),
                                            obstacle_type=ObstacleType.ROAD_BOUNDARY,
                                            obstacle_shape=ShapeGroup(road_boundary_shape_list),
                                            initial_state=initial_state)
    return road_boundary_obstacle


def construct_boundary_obstacle(scenario: Scenario, method, return_scenario_obstacle, kwargs):
    build_func_dict = {
        'triangulation': construct_boundary_obstacle_triangulation,
        'aligned_triangulation': construct_boundary_obstacle_aligned_triangulation,
        'obb_rectangles': construct_boundary_obstacle_obb_rectangles,
    }
    postprocess_obstacle_func_dict = {
        'triangulation': postprocess_create_static_obstacle_triangles,
        'aligned_triangulation': postprocess_create_static_obstacle_triangles,
        'obb_rectangles': postprocess_create_static_obstacle_obb_rectangles,
    }

    obstacle = build_func_dict[method](scenario, **kwargs)

    if return_scenario_obstacle:
        scenario_obstacle = postprocess_obstacle_func_dict[method](scenario, obstacle)
        return scenario_obstacle, obstacle
    return obstacle


def _lane_polygons_postprocess(lane_polygons, buf_width, triangulate):
    return pycrcc.Util.lane_polygons_postprocess(lane_polygons, buf_width, triangulate)


# triangulates lane polygons using GPC polygon strips

def _triangulate_polyline_gpc(polyline: list(), hole_vertices: list()) -> pycrcc.ShapeGroup:
    gpc_poly = gpc.Polygon(polyline)
    for hole in hole_vertices:
        gpc_poly.addContour(hole, 1)
    return _convert_tristrips_to_triangles([gpc_poly.triStrip()], bb=gpc_poly.boundingBox(), axis=0)


def _lane_polygons_triangulate(sg_lane_polygons: pycrcc.ShapeGroup):
    sg_lane_polygons_new = pycrcc.ShapeGroup()
    for poly in sg_lane_polygons.unpack():
        sg_triangles = _triangulate_polyline_gpc(poly.vertices(), poly.hole_vertices())
        sg_lane_polygons_new.add_shape(pycrcc.Polygon(poly.vertices(), poly.hole_vertices(), sg_triangles.unpack()))
    return sg_lane_polygons_new


def construct_road_boundary_lane_polygons(scenario: Scenario, resample=0, resample_tolerance_distance=2e-5, buffer=0,
                                          buf_width=5e-5, triangulate=True):
    """
    Creates lane polygons for the given scenario. Optionally uses Douglas-Peucker resampling and buffering of the polygons.

    :param scenario: The input scenario
    :param resample: Use Douglas-Peucker resampling. 0 - no resampling, 1 - enable resampling.
    :param resample_tolerance_distance - tolerance distance for the resampling (default: 2e-5).
    :param buffer: Use polygon buffering. 0 - no buffering, 1 - enable buffering. The Boost Geometry library, mitre joins and flat ends are used for the buffering.
    :param buf_width: Buffer width by which the resulting polygons should be enlarged (default: 5e-5).
    :param triangulate: True: triangles will be generated for the interior of each lane polygon using GPC Polygon strips, False: two triangles will be created for each lane polygon from its AABB bounding box.
    :return: ShapeGroup with the lane polygons.
    """

    lanelet_network = scenario.lanelet_network

    def lane_rep_setup(lanelet_network):
        lane_rep = []
        for lane_polyline in lanelet_bounds.lane_hull(lanelet_network):
            triangles = pycrcc.ShapeGroup()

            lane_polyline_list = lane_polyline.tolist()
            if resample == 1:
                lane_polyline_list = list(gpc.Utils.reducePointsDP(lane_polyline_list, resample_tolerance_distance))

            lane_polyline_list.reverse()

            polygon = pycrcc.Polygon(lane_polyline_list, list(), triangles.unpack())

            lane_rep.append(polygon)
        return lane_rep

    lane_rep = lane_rep_setup(lanelet_network)

    lane_polygons = pycrcc.ShapeGroup()

    for poly in lane_rep:
        lane_polygons.add_shape(poly)

    if buffer != 1:
        buf_width = 0

    lane_polygons = _lane_polygons_postprocess(lane_polygons, buf_width, False)

    if triangulate:
        lane_polygons = _lane_polygons_triangulate(lane_polygons)

    return lane_polygons

def construct_road_boundary_lanelet_polygons(scenario: Scenario, resample=0, resample_tolerance_distance=2e-5, buffer=0,
                                          buf_width=5e-5, triangulate=True):
    """
    Creates lane polygons for the given scenario. Optionally uses Douglas-Peucker resampling and buffering of the polygons.

    :param scenario: The input scenario
    :param resample: Use Douglas-Peucker resampling. 0 - no resampling, 1 - enable resampling.
    :param resample_tolerance_distance - tolerance distance for the resampling (default: 2e-5).
    :param buffer: Use polygon buffering. 0 - no buffering, 1 - enable buffering. The Boost Geometry library, mitre joins and flat ends are used for the buffering.
    :param buf_width: Buffer width by which the resulting polygons should be enlarged (default: 5e-5).
    :param triangulate: True: triangles will be generated for the interior of each lane polygon using GPC Polygon strips, False: two triangles will be created for each lane polygon from its AABB bounding box.
    :return: ShapeGroup with the lane polygons.
    """

    lanelet_network = scenario.lanelet_network

    def lanelet_rep_setup(lanelet_network):
        lanelet_rep = []
        for lanelet in lanelet_network.lanelets:
            lanelet_polyline = np.append(lanelet.right_vertices, np.flip(lanelet.left_vertices, axis=0), axis=0)
            triangles = pycrcc.ShapeGroup()

            lanelet_polyline_list = lanelet_polyline.tolist()
            if resample == 1:
                lanelet_polyline_list = list(gpc.Utils.reducePointsDP(lanelet_polyline_list, resample_tolerance_distance))

            polygon = pycrcc.Polygon(lanelet_polyline_list, list(), triangles.unpack())
            lanelet_rep.append(polygon)


        return lanelet_rep

    lanelet_rep = lanelet_rep_setup(lanelet_network)

    lanelet_polygons = pycrcc.ShapeGroup()

    for poly in lanelet_rep:
        lanelet_polygons.add_shape(poly)

    if buffer != 1:
        buf_width = 0

    lanelet_polygons = _lane_polygons_postprocess(lanelet_polygons, buf_width, False)

    if triangulate:
        lanelet_polygons = _lane_polygons_triangulate(lanelet_polygons)

    return lanelet_polygons


def construct_road_boundary_whole_polygon(scenario: Scenario, triangulate=True):
    """
    Creates large polygon(s), possibly with holes, representing the lanelet network for the given scenario.

    :param: scenario: The input scenario
    :param: triangulate: True: triangles will be generated for the interior of each resulting polygon using GPC Polygon strips, False: two triangles will be created for each resulting polygon from its AABB bounding box.
    :return: ShapeGroup with the resulting polygon(s), possibly with holes.
    """

    lanelet_network = scenario.lanelet_network

    def whole_polygon_setup(polylines):
        polygon = gpc.Polygon()
        for polyline in polylines:
            polygon = polygon | gpc.Polygon(polyline)

        return polygon

    lane_rep2 = whole_polygon_setup(lanelet_bounds.lane_hull(lanelet_network))

    lane_polygons = pycrcc.ShapeGroup()

    lane_rep2.simplify()

    lane_polygons = _GPCToCollisionPolygons([lane_rep2], triangulate=False)

    if triangulate == False:
        lane_polygons = _lane_polygons_postprocess(lane_polygons, 0, False)
    else:
        lane_polygons = _lane_polygons_triangulate(lane_polygons)

    return lane_polygons


def construct_road_boundary_whole_polygon_tiled(scenario: Scenario, max_cell_width, max_cell_height, eps=0.1,
                                                triangulate=True):
    """
    Creates large polygon(s), possibly with holes, representing the scenario road network. After that, tiles the polygon(s) into uniform rectangular grid cells.
    For the creation of polygon tiles, the uniform grid cells used are enlarged by epsilon to avoid any gaps between the generated polygon tiles.

    :param: scenario: The input scenario
    :param: max_cell_width Maximal grid cell width.
    :param: max_cell_height Maximal grid cell height.
    :param triangulate: True: triangles will be generated for the interior of each resulting polygon using GPC Polygon strips, False: two triangles will be created for each resulting polygon from its AABB bounding box.
    :return: ShapeGroup with the resulting polygon(s), possibly with holes.
    """

    def whole_polygon_setup(polylines):
        polygon = gpc.Polygon()
        for polyline in polylines:
            polygon = polygon | gpc.Polygon(polyline)

        return polygon

    def cut_into_tiles(polygon, max_cell_width, max_cell_height):
        bb = polygon.boundingBox()
        ncells_x = int((bb[1] - bb[0]) / (max_cell_width)) + 1
        ncells_y = int((bb[3] - bb[2]) / (max_cell_height)) + 1

        cell_separators_x = np.linspace(bb[0], bb[1], ncells_x + 1)
        cell_separators_y = np.linspace(bb[2], bb[3], ncells_y + 1)

        corners = np.zeros(4)
        corners[2] = bb[2]
        corners[3] = bb[3]

        x_sep = cell_separators_x[:-1]
        y_sep = cell_separators_y[:-1]

        ret_polys = list()

        for ind, x in enumerate(x_sep):
            for ind2, y in enumerate(y_sep):
                corners[0] = x - eps
                corners[1] = cell_separators_x[ind + 1] + eps
                corners[2] = y - eps
                corners[3] = cell_separators_y[ind2 + 1] + eps
                corner_vertices = [(corners[0], corners[2]), (corners[1], corners[2]),
                                   (corners[1], corners[3]), (corners[0], corners[3])]

                corner_edges = lanelet_bounds.polyline_edges(corner_vertices)

                polyline_corners = [np.asarray(el) for el in corner_vertices]
                gpc_corners = gpc.Polygon(polyline_corners)

                cell_poly = gpc_corners & polygon

                cell_poly.simplify()

                ret_polys.append(cell_poly)

        return ret_polys

    lanelet_network = scenario.lanelet_network
    lane_rep2 = whole_polygon_setup(lanelet_bounds.lane_hull(lanelet_network))
    lane_rep2.simplify()
    gpc_polys = cut_into_tiles(lane_rep2, max_cell_width, max_cell_height)
    sg_res = _GPCToCollisionPolygons(gpc_polys, triangulate=False)

    if triangulate == False:
        sg_res = _lane_polygons_postprocess(sg_res, 0, False)
    else:
        sg_res = _lane_polygons_triangulate(sg_res)

    return sg_res


def construct_road_polygons(scenario: Scenario, method, kwargs):
    build_func_dict = {
        'lane_polygons': construct_road_boundary_lane_polygons,
        'lanelet_polygons': construct_road_boundary_lanelet_polygons,
        'whole_polygon': construct_road_boundary_whole_polygon,
        'whole_polygon_tiled': construct_road_boundary_whole_polygon_tiled,
    }

    road_polygons = build_func_dict[method](scenario, **kwargs)

    return road_polygons
