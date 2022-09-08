# commonroad
from commonroad.scenario.scenario import Scenario

# commonroad-boundary
from commonroad_dc.boundary import construction


def create_road_boundary_obstacle(scenario: Scenario, method='obb_rectangles', return_scenario_obstacle=True, **kwargs):
    """
    Creates the road boundary for the given scenario.

    :param scenario: the input scenario to be triangulated
    :param method: road boundary creation method: triangulation - Delaunay triangulation, aligned_triangulation - axis-aligned triangles using GPC polygon strips, obb_rectangles - OBB rectangles on the road border (default: obb_rectangles)
    :param return_scenario_obstacle: additionally create a commonroad StaticObstacle for the road boundary
    :param kwargs: settings of the method
    :return: [optional: StaticObstacle representing the road boundary,] ShapeGroup with the collision checker objects

    Available methods:

    -triangulation

    Delaunay triangulation

    -aligned_triangulation

    Divides the polygons representing the road boundary into axis-aligned tiles and triangulates the tiles separately into the triangles having one side parallel to the other axis (in order to reduce the number of candidate triangles for collision checks).

    Settings:

    axis - 0: Use vertical tiles, 1: Use horizontal tiles, 'auto': Use horizontal tiles if width<height.

    boundary_margin - Size of margin for the calculation of scenario corners.

    -obb_rectangles

    In contrast to triangulation, this method creates a set of oriented rectangles that separates the road and the road boundary.
    To compute the set, we first compute the union of all lanelets in the road network and then extract the inner and outer contours of the resulting polygon.
    Afterwards, we create an oriented rectangle for each line segment of the inner and outer contours.
    The oriented rectangles symmetrically overapproximate each line segment.
    In this way, we can reduce the false positive rate at road forks and merges.

    Settings:

    width: Width of the generated rectangles. Default: 1e-5.

    Example:

    create_road_boundary_obstacle(scenario, method = 'aligned_triangulation', axis='auto')

    """

    boundary = construction.construct_boundary_obstacle(scenario, method, return_scenario_obstacle, kwargs)
    return boundary


def create_road_polygons(scenario: Scenario, method='lane_polygons', **kwargs):
    """
    Creates a ShapeGroup with collision checker polygons representing the road.

    :param scenario: the input scenario for the creation of road polygons
    :param method: road polygons creation method: lane_polygons - lane polygons, lanelet_polygons - lanelet polygons, whole_polygon - whole polygon, whole_polygon_tiled - whole polygon subdivided into tiles (default: lane_polygons)
    :param kwargs: settings of the method
    :return: ShapeGroup with the collision checker polygons

    Available methods:

    -lane_polygons

    Creates lane polygons for the given scenario. Optionally uses Douglas-Peucker resampling and buffering of the polygons.

    Settings:

    resample - use Douglas-Peucker resampling. 0 - no resampling, 1 - enable resampling.

    resample_tolerance_distance - tolerance distance for the resampling (default: 2e-5).

    buffer - use polygon buffering. 0 - no buffering, 1 - enable buffering. The Boost Geometry library, mitre joins and flat ends are used for the buffering.

    buf_width - buffer width by which the resulting polygons should be enlarged (default: 5e-5).

    triangulate - True: triangles will be generated for the interior of each lane polygon using GPC Polygon strips, False: two triangles will be created for each lane polygon from its AABB bounding box.

    -lanelet_polygons

    Creates lanelet polygons for the given scenario. Optionally uses Douglas-Peucker resampling and buffering of the polygons.

    Settings:

    resample - use Douglas-Peucker resampling. 0 - no resampling, 1 - enable resampling.

    resample_tolerance_distance - tolerance distance for the resampling (default: 2e-5).

    buffer - use polygon buffering. 0 - no buffering, 1 - enable buffering. The Boost Geometry library, mitre joins and flat ends are used for the buffering.

    buf_width - buffer width by which the resulting polygons should be enlarged (default: 5e-5).

    triangulate - True: triangles will be generated for the interior of each lanelet polygon using GPC Polygon strips, False: two triangles will be created for each lanelet polygon from its AABB bounding box.

    -whole_polygon

    Creates large polygon(s), possibly with holes, representing the lanelet network for the given scenario.

    Settings:

    triangulate - True: triangles will be generated for the interior of each resulting polygon using GPC Polygon strips, False: two triangles will be created for each resulting polygon from its AABB bounding box.

    -whole_polygon_tiled

    Creates large polygon(s), possibly with holes, representing the scenario road network. After that, tiles the polygon(s) into uniform rectangular grid cells.
    For the creation of polygon tiles, the uniform grid cells used are enlarged by epsilon to avoid any gaps between the generated polygon tiles.

    Settings:

    max_cell_width - maximal grid cell width.

    max_cell_height - maximal grid cell height.

    triangulate - True: triangles will be generated for the interior of each resulting polygon using GPC Polygon strips, False: two triangles will be created for each resulting polygon from its AABB bounding box.

    Example:

    create_road_polygons(scenario, method = 'lane_polygons', resample=1, buffer=1, triangulate=True)

    """

    boundary = construction.construct_road_polygons(scenario, method, kwargs)
    return boundary
