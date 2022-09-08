from .trajectory_queries_specialized import *

import commonroad_dc.pycrcc as pycrcc

from commonroad_dc.pycrcc.Util import \
    trajectory_collision_static_obstacles as pycrcc_trajectory_collision_static_obstacles
from commonroad_dc.pycrcc.Util import \
    trajectory_enclosure_polygons_static as pycrcc_trajectory_enclosure_polygons_static
from commonroad_dc.pycrcc.Util import obb_enclosure_polygons_static as pycrcc_obb_enclosure_polygons_static


class OBBSumException(Exception):
    pass


trajectories_collision_static_obstacles_function_dict = \
    {
        'grid': trajectories_collision_static_obstacles_grid,
        'fcl': trajectories_collision_static_obstacles_fcl,
        'box2d': trajectories_collision_static_obstacles_box2d
    }

trajectories_enclosure_polygons_function_dict = \
    {
        'grid': trajectories_enclosure_polygons_static_grid
    }

trajectories_collision_dynamic_obstacles_function_dict = \
    {
        'grid': trajectories_collision_dynamic_obstacles_grid,
        'fcl': trajectories_collision_dynamic_obstacles_fcl,
        'box2d': trajectories_collision_dynamic_obstacles_box2d
    }


# Trajectory queries for a list of trajectories (batch computation):

def trajectories_collision_static_obstacles(trajectories: list, static_obstacles: pycrcc.ShapeGroup, method='fcl',
                                            **kwargs):
    """
    For each trajectory in the list, returns its time of collision with static obstacles. If the trajectory is not colliding with the static obstacles, the returned timestep of collision is -1.

    :param trajectories: list of input trajectories (each trajectory in the list must be a pycrcc.TimeVariantCollisionObject)
    :param static_obstacles: ShapeGroup with static obstacles (pycrcc.ShapeGroup)
    :param method: broadphase method for the filtering of collision candidates
    :param kwargs: settings of the method

    Available methods:

    -grid

    Use Uniform Grid broadphase algorithm for candidate pair filtering.

    Settings:

    num_cells - number of cells of the uniform grid broadphase accelerator structure.

    auto_orientation - enable automatic choice of uniform grid orientation.

    optimize_triangles - use optimized 2D SAT solver for triangles instead of FCL solver.

    enable_verification - debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).

    -fcl

    Use a Flexible Collision Library-supported broadphase algorithm for candidate pair filtering.

    Settings:

    broadphase_type- type of FCL broadphase to be used. 0: Binary Tree, 1: SpatialHashing 2: IntervalTree 3: Simple SAP 4: SAP

    num_cells - determines cell size for the SpatialHashing broadphase accelerator structure. Cell size = length of obstacles bounding box along x axis divided by num_cells.

    num_buckets - number of hash buckets for the SpatialHashing broadphase accelerator structure.

    enable_verification - debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).

    -box2d

    Use 2D AABB Tree broadphase broadphase algorithm from the library Box2D for candidate pair filtering.

    Settings:

    enable_verification - debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).

    Additional settings parameters for all broadphase methods:

    return_debug_info - True: additionally return debug information.

    :return: First timestep of collision or -1 is no collision was detected - for each trajectory (as a list of integers). After that, debug information (optional).

    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.

    """

    return trajectories_collision_static_obstacles_function_dict[method](trajectories=trajectories,
                                                                         static_obstacles=static_obstacles, **kwargs)


def trajectories_enclosure_polygons_static(trajectories: list, sg_polygons: pycrcc.ShapeGroup, method='grid', **kwargs):
    """
    Checks the given trajectories for enclosure within the union of the lane polygons contained in sg_polygons.
    Uses iterative difference operations and Uniform Grid broadphase for finding the candidate polygons for subtraction.
    It is necessary to buffer the lane polygons in the preprocessing step before calling this function such that there is no small gap between neighboring lanelets due to numerics.

    :param trajectories: list of input trajectories (each trajectory in the list must be a pycrcc.TimeVariantCollisionObject)
    :param sg_polygons: ShapeGroup with lane polygons of type pycrcc.Polygon for subtraction
    :param method: broadphase method for the finding of candidate polygons
    :param kwargs: settings of the method

    Available methods:

    -grid

    Settings:

    num_cells - parameter of the Uniform Grid broadphase (number of vertical grid cells) used to find the candidate Polygons for subtraction. These are not the same as the grid cells into which the lanelet polygons can be cut using boundary.create_road_polygons(scenario, method='whole_polygon_tiled').

    enable_verification - debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).

    return_debug_info - True: additionally return debug information.

    :return: First timestep of incomplete enclosure or -1 if always completely enclosed - for each trajectory. After that, debug information (optional).

    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.
    """

    return trajectories_enclosure_polygons_function_dict[method](trajectories=trajectories, sg_polygons=sg_polygons,
                                                                 **kwargs)


def trajectories_collision_dynamic_obstacles(trajectories: list, dynamic_obstacles: list, method='fcl', **kwargs):
    """
    For each trajectory in the list, returns its time of collision with dynamic obstacles. If the trajectory is not colliding with the dynamic obstacles, the returned timestep of collision is -1.

    :param trajectories: list of input trajectories (each must be a pycrcc.TimeVariantCollisionObject)
    :param dynamic_obstacles: list of dynamic obstacles (each must be a pycrcc.TimeVariantCollisionObject)
    :param method: broadphase method for the filtering of collision candidates
    :param kwargs: settings of the method

    Available methods:

    -grid

    Use Uniform Grid broadphase algorithm for candidate pair filtering.

    Settings:

    num_cells - number of cells of the uniform grid broadphase accelerator structure.

    enable_verification - debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).

    -fcl

    Use a Flexible Collision Library-supported broadphase algorithm for candidate pair filtering.

    Settings:

    broadphase_type- type of FCL broadphase to be used. 0: Binary Tree, 1: SpatialHashing 2: IntervalTree 3: Simple SAP 4: SAP

    num_cells - determines cell size for the SpatialHashing broadphase accelerator structure. Cell size = length of obstacles bounding box along x axis divided by num_cells.

    num_buckets - number of hash buckets for the SpatialHashing broadphase accelerator structure.

    enable_verification - Debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).

    -box2d

    Use 2D AABB Tree broadphase broadphase algorithm from the library Box2D for candidate pair filtering.

    Settings:

    enable_verification - debug check if any colliding obstacles were missed by the broadphase (as compared to the bruteforce checker).

    Additional settings parameters for all broadphase methods:

    return_debug_info - True: additionally return debug information.

    :return: First timestep of collision or -1 is no collision was detected - for each trajectory (as a list of integers). After that, debug information (optional).

    Please refer to the tutorial example commonroad_road_compliance_checking for usage example.

    """

    return trajectories_collision_dynamic_obstacles_function_dict[method](trajectories=trajectories,
                                                                          dynamic_obstacles=dynamic_obstacles, **kwargs)


def trajectory_preprocess_obb_sum(trajectory: pycrcc.TimeVariantCollisionObject):
    """
    Preprocess a trajectory using OBB sum hull (for continuous collision detection). The input trajectory must consist of one oriented rectangle per time step.
    The occupancies of the OBB boxes for two subsequent states are overapproximated with a tightly fitting OBB box.

    :param trajectory: trajectory for preprocessing (pycrcc.TimeVariantCollisionObject)

    :return: postprocessed trajectory (pycrcc.TimeVariantCollisionObject). Its length is one step smaller compared to the input trajectory.

    """
    if trajectory.time_end_idx() - trajectory.time_start_idx() <= 0:
        raise OBBSumException("Invalid input for trajectory_preprocess_obb_sum: Input trajectory must consists of at least two time steps")

    return pycrcc.Util.trajectory_preprocess_obb_sum(trajectory)

def filter_trajectories_polygon_enclosure_first_timestep(trajectories, road_polygons):
    """
    Selects trajectories that have the vehicle fully within the union of the road polygons at their first time step.

    :param trajectories: input list of trajectories for filtering (list of pycrcc.TimeVariantCollisionObject). Only trajectories containing OBB and AABB boxes are supported.

    :param road_polygons: pycrcc.ShapeGroup containing road polygons.

    :return: List of trajectories fully within the road at their first time step.

    """
    return pycrcc.Util.filter_trajectories_polygon_enclosure_first_timestep(trajectories,road_polygons)


# Trajectory queries for 1 trajectory (in numpy format):

# Please note that when the functions below were imported directly from pycrcc.Util and not from this module,
# mean computation time was about 5% lower, and std was also lower when they were used for multiple trajectories in a loop
# (to detect collision between the first step occupancy and static obstacles).
# The C++ functions are wrapped as python functions below in case of possible future API changes.

def trajectory_collision_static_obstacles(sg_obstacles: pycrcc.ShapeGroup, half_box_length, half_box_width,
                                          trajectory: list):
    """
     Returns the time of collision between the input trajectory and the static obstacles. If the trajectory is not colliding with the static obstacles, the returned timestep of collision is -1.
     The function uses numpy arrays to represent the input trajectory. The trajectory is represented by a moving oriented bounding box.

     :param sg_obstacles: ShapeGroup with obstacles
     :param half_box_length: the half-length (r_x) of the trajectory oriented bounding box
     :param half_box_width: the half-width (r_y) of the trajectory oriented bounding box
     :param trajectory: list of numpy arrays each having 3 elements: x coordinate, y coordinate, OBB box orientation angle in radians
     :return: First timestep of collision or -1 is no collision was detected.

    """
    return pycrcc_trajectory_collision_static_obstacles(sg_obstacles, half_box_length, half_box_width, trajectory)


def trajectory_enclosure_polygons_static(sg_polygons: pycrcc.ShapeGroup, half_box_length, half_box_width,
                                         trajectory: list):
    """
    Checks if the given trajectory is enclosed within the lanelet polygons contained in the given ShapeGroup.
    The function uses numpy arrays to represent the input trajectory. The trajectory is represented by a moving oriented bounding box.
    The function can be slower than the functions checking a batch of trajectories.
    But in some cases it can be faster to use this function, for example, when one needs to check for the collision between the first time step of the input trajectory and the static obstacles.
    In this situation, the creation of the numpy array was considerably faster than the extraction of the occupancy for the first time step.

    :param sg_polygons: ShapeGroup with road polygons
    :param half_box_length: the half-length (r_x) of the trajectory oriented bounding box
    :param half_box_width: the half-width (r_y) of the trajectory oriented bounding box
    :param trajectory: list of numpy arrays each having 3 elements: x coordinate, y coordinate, OBB box orientation angle in radians
    :return: First timestep of incomplete enclosure or -1 if the trajectory is always completely enclosed within the polygons union.
    """

    return pycrcc_trajectory_enclosure_polygons_static(sg_polygons, half_box_length, half_box_width, trajectory)


def obb_enclosure_polygons_static(sg_polygons: pycrcc.ShapeGroup, obb: pycrcc.RectOBB):
    """
    Checks if the given obb box is enclosed within the lanelet polygons contained in the given ShapeGroup.

    :param sg_polygons: ShapeGroup with road polygons
    :param obb: the input obb box for enclosure checking
    :return: First timestep of incomplete enclosure or -1 if the trajectory is always completely enclosed within the polygons union.
    """
    return pycrcc_obb_enclosure_polygons_static(sg_polygons, obb)
