"""
Lanelet-based Curvilinear Coordinate system
"""
import warnings
from enum import Enum
from typing import Tuple

import numpy as np
import commonroad_dc.pycrccosy as pycrccosy
from commonroad.scenario.lanelet import Lanelet

from commonroad_dc.geometry.util import compute_curvature_from_polyline, \
    chaikins_corner_cutting, resample_polyline_with_length_check

__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "0.1.0"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Integration"


class LongitudinalRelativePointPosition(Enum):
    BeforeDomain = -1
    InDomain = 0
    AfterDomain = 1


class LaneletCoordinateSystem(pycrccosy.CurvilinearCoordinateSystem):
    """
    Class to transform between curvilinear and cartesian coordinates over a given lanelet
    """

    def __init__(self, lanelet: Lanelet):
        """
        Create new object
        :param lanelet: The lanelet, which will be used as curvilinear coordinate system
        """
        self.lanelet = lanelet
        lanelet_polyline = self._create_resampled_polyline_for_ccosy(lanelet)
        super().__init__(lanelet_polyline, 25.0, 0.1)

        # TODO: This is workaround for the problem
        #  that the first and last vertices are missed in Ccosy
        eps = 0.0001
        curvi_coords_of_projection_domain = np.array(self.curvilinear_projection_domain())

        self.longitudinal_min, self.normal_min = np.min(curvi_coords_of_projection_domain,
                                                        axis=0) + eps
        self.longitudinal_max, self.normal_max = np.max(curvi_coords_of_projection_domain,
                                                        axis=0) - eps
        self.normal_center = (self.normal_min + self.normal_max) / 2
        bounding_points = np.array(
            [self.convert_to_cartesian_coords(self.longitudinal_min, self.normal_center),
             self.convert_to_cartesian_coords(self.longitudinal_max, self.normal_center)])
        self.bounding_points = np.array([bounding_point for bounding_point in bounding_points])

        self.vertex_curvi_coords = [self.get_extrapolated_curvi_coords(vertex) for vertex in
                                      self.lanelet.center_vertices]
        self.vertex_long_distances = [vertex_curvi_coord[0][0] for vertex_curvi_coord in
                                      self.vertex_curvi_coords]
        self.vertex_tangents = [self.tangent(vertex_long_distance) for vertex_long_distance in
                                self.vertex_long_distances]

    def get_extrapolated_curvi_coords(self, cartesian_position: np.ndarray) \
            -> Tuple[np.ndarray, LongitudinalRelativePointPosition]:
        """
        Returns the projected curvilinear coordinates.
        If the given position is outside of the projection domain, the domain will be extrapolated
        :param cartesian_position: The cartesian coordinates of the position
        :return: The curvilinear coordinates of the position
        """
        try:
            long_lat_distance = self.convert_to_curvilinear_coords(cartesian_position[0],
                                                                   cartesian_position[1])
            rel_pos_to_domain = LongitudinalRelativePointPosition.InDomain
        except ValueError:
            long_lat_distance, rel_pos_to_domain = self._project_out_of_domain_to_lanelet(
                cartesian_position)

        return np.array(long_lat_distance), rel_pos_to_domain

    def get_extrapolated_cartesian_coords(self, curvi_position: np.ndarray) \
            -> Tuple[np.ndarray, LongitudinalRelativePointPosition]:
        """
        Returns the position defined as curvilinear coordinates as cartesian coordinates
        If the given position is outside of the projection domain, the domain will be extrapolated
        :param curvi_position: The curvilinear coordinates of the position
        :return: The cartesian coordinates of the position
        """

        long_dist, lat_dist = curvi_position

        if long_dist < self.longitudinal_min:
            # Nearer to the first bounding point
            rel_pos_to_domain = LongitudinalRelativePointPosition.BeforeDomain
            position = self.convert_to_cartesian_coords(self.longitudinal_min, 0)
            position = position + self.tangent(self.longitudinal_min) * (
                    long_dist - self.longitudinal_min)
            position = position + self.normal(self.longitudinal_min) * lat_dist
        elif long_dist > self.longitudinal_max:
            # Nearer to the last bounding point
            rel_pos_to_domain = LongitudinalRelativePointPosition.AfterDomain
            position = self.convert_to_cartesian_coords(self.longitudinal_max, 0)
            position = position + self.tangent(self.longitudinal_max) * (
                    long_dist - self.longitudinal_max)
            position = position + self.normal(self.longitudinal_max) * lat_dist
        else:
            # Inside of the domain
            rel_pos_to_domain = LongitudinalRelativePointPosition.InDomain
            try:
                position = self.convert_to_cartesian_coords(long_dist, lat_dist)
            except ValueError:
                position = self.convert_to_cartesian_coords(long_dist, 0)
                position = position + self.normal(long_dist) * lat_dist

        return position, rel_pos_to_domain

    @staticmethod
    def _create_resampled_polyline_for_ccosy(lanelet: Lanelet) -> np.ndarray:
        """
        Prepare polyline for ccosy
        :param lanelet: The lanelet, which center vertices will be used as reference path
        :return: Refined polyline for the ccosy
        """
        polyline = lanelet.center_vertices
        for _ in range(20):
            polyline = chaikins_corner_cutting(polyline)
            polyline = resample_polyline_with_length_check(polyline, length_to_check=10.0)

            abs_curvature = abs(compute_curvature_from_polyline(polyline))
            max_curvature = max(abs_curvature)

            if max_curvature < 0.1:
                break

        return polyline

    def _project_out_of_domain_to_lanelet(self, cartesian_position: np.ndarray) \
            -> Tuple[np.ndarray, LongitudinalRelativePointPosition]:
        """
        Projects out of domain point orthogonally to the extrapolated curvilinear coordinate system
        :param cartesian_position: The cartesian position to project
        :return: Tuple of the curvilinear coordinates and the relative longitudinal position
        """
        # TODO: The longitudinal_min and similar variables are workaround for the problem
        #  that the first and last vertices are missed

        rel_positions_to_vertices = cartesian_position - self.lanelet.center_vertices
        rel_positions_to_bounding_points = cartesian_position - self.bounding_points
        distances = np.linalg.norm(rel_positions_to_vertices, axis=1)

        nearest_vertex_idx = np.argmin(distances)
        if isinstance(nearest_vertex_idx, np.ndarray):
            nearest_vertex_idx = nearest_vertex_idx[0]

        projected_long_distance_to_front = np.dot(self.tangent(self.longitudinal_min),
                                                  rel_positions_to_vertices[0])

        projected_long_distance_to_back = np.dot(self.tangent(self.longitudinal_max),
                                                 rel_positions_to_vertices[-1])

        if nearest_vertex_idx == 0 and projected_long_distance_to_front <= 0:
            # Nearer to the first bounding point
            rel_pos_to_domain = LongitudinalRelativePointPosition.BeforeDomain
            long_dist = self.longitudinal_min + np.dot(self.tangent(self.longitudinal_min),
                                                       rel_positions_to_bounding_points[0])
            lat_dist = self.normal_center + np.dot(self.normal(self.longitudinal_min),
                                                   rel_positions_to_bounding_points[0])
        elif nearest_vertex_idx == len(distances) - 1 and projected_long_distance_to_back >= 0:
            # Nearer to the last bounding point
            rel_pos_to_domain = LongitudinalRelativePointPosition.AfterDomain
            long_dist = self.longitudinal_max + np.dot(self.tangent(self.longitudinal_max),
                                                       rel_positions_to_bounding_points[1])
            lat_dist = self.normal_center + np.dot(self.normal(self.longitudinal_max),
                                                   rel_positions_to_bounding_points[1])
        else:
            # In the domain longitudinally but outside laterally
            warnings.warn(f"The to-be-projected point is out-of-domain laterally, "
                          f"the projection may be inaccurate")
            rel_pos_to_domain = LongitudinalRelativePointPosition.InDomain

            # TODO: Using nearest vertex for now
            #       Will be integrated in cpp later and handled correctly there

            nearest_vertex = self.lanelet.center_vertices[nearest_vertex_idx]
            rel_position = cartesian_position - nearest_vertex
            (long_dist_vertex, lat_dist_vertex), _ = self.get_extrapolated_curvi_coords(nearest_vertex)

            long_dist = long_dist_vertex + np.dot(self.tangent(long_dist_vertex), rel_position)
            lat_dist = lat_dist_vertex + np.dot(self.normal(self.longitudinal_max), rel_position)

        return np.array([long_dist, lat_dist]), rel_pos_to_domain

    def tangent(self, long_dist):
        """
        Returns the tangent at the given longitudinal distance
        :param long_dist: The longitudinal distance, where the tangent is queried
        :return: The tangent vector at the given longitudinal distance
        """
        try:
            return super().tangent(long_dist)
        except ValueError:
            if long_dist < self.longitudinal_min:
                # Nearer to the first bounding point
                return self.tangent(self.longitudinal_min)
            elif long_dist > self.longitudinal_max:
                # Nearer to the last bounding point
                return self.tangent(self.longitudinal_max)
            else:
                raise RuntimeError

    def normal(self, long_dist):
        """
        Returns the normal at the given longitudinal distance
        :param long_dist: The longitudinal distance, where the tangent is queried
        :return: The normal vector at the given longitudinal distance
        """
        try:
            return super().normal(long_dist)
        except ValueError:
            if long_dist < self.longitudinal_min:
                # Nearer to the first bounding point
                return self.normal(self.longitudinal_min)
            elif long_dist > self.longitudinal_max:
                # Nearer to the last bounding point
                return self.normal(self.longitudinal_max)
            else:
                raise RuntimeError

    def orientation_at_distance(self, longitudinal_distance: float):
        """
        Finds the lanelet orientation in the given distance

        :param longitudinal_distance: Distance where the lanelet's orientation should be calculated
        :return: An orientation in interval [-pi,pi]
        """
        tangent = self.tangent(longitudinal_distance)
        return np.arctan2(tangent[1], tangent[0])

    def orientation_at_position(self, cartesian_position: np.ndarray):
        """
        Finds the lanelet orientation in a given position

        :param cartesian_position: Position where the lanelet's orientation should be calculated
        :return: An orientation in interval [-pi,pi]
        """
        long_lat_distance, _ = self.get_extrapolated_curvi_coords(cartesian_position)
        return self.orientation_at_distance(long_lat_distance[0])
