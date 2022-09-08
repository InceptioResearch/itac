import commonroad_dc.pycrcc as pycrcc


def trajectories_collision_static_obstacles_grid(trajectories: list, static_obstacles: pycrcc.ShapeGroup, num_cells=32,
                                                 auto_orientation=True, optimize_triangles=True,
                                                 enable_verification=False, return_debug_info=False):
    """
    Checks the given trajectories for a collision with any collision checker objects of the given ShapeGroup using Uniform Grid broadphase.

    :param trajectories: list() of TimeVariantCollisionObstacle - trajectories to check.
    :param static_obstacles: The ShapeGroup with obstacles.
    :param num_cells: Number of cells of the uniform grid broadphase accelerator structure.
    :param auto_orientation: Enable automatic choice of uniform grid orientation.
    :param optimize_triangles Use optimized 2D SAT solver for triangles instead of FCL solver.
    :param enable_verification Debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).
    :return: First timestep of collision or -1 is no collision was detected - for each trajectory. After that, debug information (optional).

    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.
    """
    result = pycrcc.Util.trajectories_collision_static_obstacles_grid(trajectories, static_obstacles,
                                                                      enable_verification, auto_orientation, num_cells,
                                                                      optimize_triangles)
    if return_debug_info:
        return result[0:-2], result[-2:]
    else:
        return result[0:-2]


def trajectories_collision_static_obstacles_fcl(trajectories: list, static_obstacles: pycrcc.ShapeGroup,
                                                broadphase_type=0, num_cells=32, num_buckets=1000,
                                                enable_verification=False, return_debug_info=False):
    """
    Checks the given trajectories for a collision with any collision checker objects of the given ShapeGroup using an FCL broadphase accelerator structure.

    :param trajectories: list() of TimeVariantCollisionObstacle - trajectories to check.
    :param static_obstacles: The ShapeGroup with obstacles.
    :param broadphase_type: Type of FCL broadphase to be used. 0: Binary Tree, 1: SpatialHashing 2: IntervalTree 3: Simple SAP 4: SAP
    :param num_cells: Determines cell size for the SpatialHashing broadphase accelerator structure. Cell size = length of obstacles bounding box along x axis divided by num_cells.
    :param num_buckets:  Number of hash buckets for the SpatialHashing broadphase accelerator structure.
    :param enable_verification: Debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).
    :return: First timestep of collision or -1 is no collision was detected - for each trajectory. After that, debug information (optional).

    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.
    """

    result = pycrcc.Util.trajectories_collision_static_obstacles_fcl(trajectories, static_obstacles,
                                                                     enable_verification, broadphase_type, num_cells,
                                                                     num_buckets)
    if return_debug_info:
        return result[0:-2], result[-2:]
    else:
        return result[0:-2]


def trajectories_collision_static_obstacles_box2d(trajectories: list, static_obstacles: pycrcc.ShapeGroup,
                                                  enable_verification=False, return_debug_info=False):
    """
    Checks the given trajectories for a collision with any collision checker objects of the given ShapeGroup using the Box2D Dynamic AABB Tree broadphase.

    :param trajectories: list() of TimeVariantCollisionObstacle - trajectories to check.
    :param static_obstacles: The ShapeGroup with obstacles.
    :param enable_verification: Debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).
    :return: First timestep of collision or -1 is no collision was detected - for each trajectory. After that, debug information (optional).

    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.
    """
    result = pycrcc.Util.trajectories_collision_static_obstacles_box2d(trajectories, static_obstacles,
                                                                       enable_verification)
    if return_debug_info:
        return result[0:-2], result[-2:]
    else:
        return result[0:-2]


def trajectories_enclosure_polygons_static_grid(trajectories: list, sg_polygons: pycrcc.ShapeGroup, num_cells=32,
                                                enable_verification=False, return_debug_info=False):
    """
    Checks the given trajectories for enclosure within the union of the lane polygons contained in road_polygons.
    Uses iterative difference operations and uniform grid broadphase for finding the candidate polygons for subtraction.
    It is necessary to buffer the lane polygons in the preprocessing step before calling this function such that there is no gap between neighboring lanelets.

    :param trajectories: list() of TimeVariantCollisionObstacle - trajectories to check.
    :param sg_polygons: The ShapeGroup with lane polygons of type pycrcc.Polygon for subtraction.
    :param num_cells: Parameter of the Uniform Grid broadphase (number of horizontal grid cells) used to find the candidate Polygons for subtraction. This is not the same as the grid cells used to cut the polygon in the IV 2020 publication.
    :param enable_verification: Debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).
    :return: First timestep of collision or -1 is no collision was detected - for each trajectory. After that, debug information (optional).

    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.
    """
    result = pycrcc.Util.trajectories_enclosure_polygons_static_grid(trajectories, sg_polygons, enable_verification,
                                                                     num_cells)
    if return_debug_info:
        return result[0:-2], result[-2:]
    else:
        return result[0:-2]


def trajectories_collision_dynamic_obstacles_fcl(trajectories: list, dynamic_obstacles: list, broadphase_type=0,
                                                 num_cells=32, num_buckets=1000, enable_verification=False,
                                                 return_debug_info=False):
    """
    Checks the given trajectories for collision with dynamic obstacles using an FCL broadphase accelerator structure.

    :param trajectories: list() of TimeVariantCollisionObstacle - trajectories to check.
    :param dynamic_obstacles: list() of TimeVariantCollisionObstacle - dynamic obstacles for collision check.
    :param broadphase_type: Type of FCL broadphase to be used. 0: Binary Tree, 1: SpatialHashing 2: IntervalTree 3: Simple SAP 4: SAP
    :param num_cells: Determines cell size for the SpatialHashing broadphase accelerator structure. Cell size = length of obstacles bounding box along x axis divided by num_cells.
    :param num_buckets:  Number of hash buckets for the SpatialHashing broadphase accelerator structure.
    :param enable_verification: Debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).
    :return: First timestep of collision or -1 is no collision was detected - for each trajectory. After that, debug information (optional).
    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.
    """

    result = pycrcc.Util.trajectories_collision_dynamic_obstacles_fcl(trajectories, dynamic_obstacles,
                                                                      enable_verification, broadphase_type, num_cells,
                                                                      num_buckets)
    if return_debug_info:
        return result[0:-2], result[-2:]
    else:
        return result[0:-2]


def trajectories_collision_dynamic_obstacles_box2d(trajectories: list, dynamic_obstacles: list,
                                                   enable_verification=False, return_debug_info=False):
    """
    Checks the given trajectories for collision with dynamic obstacles.

    :param trajectories: list() of TimeVariantCollisionObstacle - trajectories to check.
    :param dynamic_obstacles: list() of TimeVariantCollisionObstacle - dynamic obstacles for collision check.
    :param enable_verification: Debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).
    :return: First timestep of collision or -1 is no collision was detected - for each trajectory. After that, debug information (optional).
    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.
    """
    result = pycrcc.Util.trajectories_collision_dynamic_obstacles_box2d(trajectories, dynamic_obstacles,
                                                                        enable_verification)
    if return_debug_info:
        return result[0:-2], result[-2:]
    else:
        return result[0:-2]


def trajectories_collision_dynamic_obstacles_grid(trajectories, dynamic_obstacles, num_cells=32,
                                                  enable_verification=False, return_debug_info=False):
    """
    Checks the given trajectories for collision with dynamic obstacles using Uniform Grid broadphase.

    :param trajectories: list() of TimeVariantCollisionObstacle - trajectories to check.
    :param dynamic_obstacles: list() of TimeVariantCollisionObstacle - dynamic obstacles for collision check.
    :param num_cells: Number of cells of the uniform grid broadphase accelerator structure.
    :param enable_verification Debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).
    :return: First timestep of collision or -1 is no collision was detected - for each trajectory. After that, debug information (optional).
    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.

    """

    result = pycrcc.Util.trajectories_collision_dynamic_obstacles_grid(trajectories, dynamic_obstacles,
                                                                       enable_verification, num_cells)
    if return_debug_info:
        return result[0:-2], result[-2:]
    else:
        return result[0:-2]
