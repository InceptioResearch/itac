import warnings
import abc
from copy import deepcopy

import numpy as np
from typing import List, Union, Optional, Tuple

import shapely.geometry
import shapely.affinity

from commonroad.geometry.transform import translate_rotate, rotate_translate
from commonroad.common.validity import is_valid_polyline, is_real_number, is_real_number_vector, is_valid_orientation
from commonroad.common.util import make_valid_orientation

__author__ = "Stefanie Manzinger"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "2022.1"
__maintainer__ = "Stefanie Manzinger"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"

from commonroad.visualization.drawable import IDrawable
from commonroad.visualization.param_server import ParamServer
from commonroad.visualization.renderer import IRenderer


class Shape(IDrawable, metaclass=abc.ABCMeta):
    """ Abstract class for CommonRoad shapes."""

    @abc.abstractmethod
    def translate_rotate(self, translation: np.ndarray, angle: float) -> 'Shape':
        """ First translates and then rotates a shape around the origin."""
        pass

    @abc.abstractmethod
    def rotate_translate_local(self, translation: np.ndarray, angle: float) -> 'Shape':
        """ First rotates a shape around the center and the translates it."""
        pass

    @abc.abstractmethod
    def contains_point(self, point: np.ndarray) -> bool:
        """ Checks whether point is contained in shape."""
        pass


class Rectangle(Shape):
    """ The class Rectangle can be used to model occupied regions or rectangular obstacles, e.g., a vehicle. The
    rectangle is specified by the length in longitudinal direction, the width in lateral direction, the orientation,
    and its geometric center. If we model the shape of an obstacle, the orientation and geometric center can be
    omitted; therefore, we set the orientation, and the x- and y-coordinate of the geometric center to zero."""

    def __init__(self, length: float, width: float, center: np.ndarray = None,
                 orientation: float = 0.0):
        """

        :param length: length of the rectangle in longitudinal direction
        :param width: width of the rectangle in lateral direction
        :param center: geometric center [x, y] of the rectangle in [m]. If the rectangle is used to model the shape
        of an obstacle, the geometric center can be omitted and is set to [0.0, 0.0].
        :param orientation: orientation of the rectangle in [rad]. If the rectangle is used to model the shape of an
        obstacle, the orientation can be omitted and is set to zero.
        """
        self.length: float = length
        self.width: float = width
        self.center: np.ndarray = center if center is not None else np.array([0.0, 0.0])
        self.orientation: float = orientation

        self._vertices: np.ndarray = None
        self.__shapely_polygon: shapely.geometry.Polygon = None

    @property
    def _shapely_polygon(self) -> shapely.geometry.Polygon:
        if self.__shapely_polygon is None:
            self.__shapely_polygon: shapely.geometry.Polygon = shapely.geometry.Polygon(self.vertices)

        return self.__shapely_polygon

    @_shapely_polygon.setter
    def _shapely_polygon(self, _shapely_polygon):
        self.__shapely_polygon = _shapely_polygon

    @property
    def length(self) -> float:
        """ Length of the rectangle in longitudinal direction."""
        return self._length

    @length.setter
    def length(self, length: float):
        if not hasattr(self, '_length'):
            assert is_real_number(length), '<Rectangle/length>: argument "length" is not valid. length = {}'.format(
                    length)
            self._length = length
        else:
            warnings.warn('<Rectangle/length>: length of rectangle is immutable.')

    @property
    def width(self) -> float:
        """ Width of the rectangle in lateral direction."""
        return self._width

    @width.setter
    def width(self, width: float):
        if not hasattr(self, '_width'):
            assert is_real_number(width), '<Rectangle/width>: argument "width" is not valid. width = {}'.format(width)
            self._width = width
        else:
            warnings.warn('<Rectangle/width>: width of rectangle is immutable.')

    @property
    def center(self) -> np.ndarray:
        """ Geometric center of the rectangle [x, y]. If the rectangle is used to describe the shape of an obstacle,
        we set the center to the coordinates [0.0, 0.0]."""
        return self._center

    @center.setter
    def center(self, center: np.ndarray):
        if not hasattr(self, '_center'):
            assert is_real_number_vector(center, 2), '<Rectangle/center>: argument "center" is not a vector ' \
                                                     'of real numbers of length 2. center = {}'.format(center)
            self._center = center
        else:
            warnings.warn('<Rectangle/center>: center of rectangle is immutable.')

    @property
    def orientation(self) -> float:
        """ Orientation of the rectangle. If the rectangle is used to describe the shape of an obstacle,
        we set the orientation to 0.0."""
        return self._orientation

    @orientation.setter
    def orientation(self, orientation: float):
        if not hasattr(self, '_orientation'):
            assert is_valid_orientation(orientation), '<Rectangle/orientation>: argument "orientation" is not valid. ' \
                                                      'orientation = {}'.format(orientation)
            self._orientation = orientation
        else:
            warnings.warn('<Rectangle/orientation>: orientation of rectangle is immutable.')

    @property
    def vertices(self) -> np.ndarray:
        """ Vertices of the rectangle: [[x_0, y_0], [x_1, y_1], ...]. The vertices are sorted clockwise and the
            first and last point are the same.
        """
        if self._vertices is None:
            self._vertices = self._compute_vertices()
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray):
        warnings.warn('<Rectangle/vertices>: vertices of rectangle are immutable.')

    @property
    def shapely_object(self) -> shapely.geometry.Polygon:
        return self._shapely_polygon

    def translate_rotate(self, translation: np.ndarray, angle: float) -> 'Rectangle':
        """ A new rectangle is created by first translating and then rotating the rectangle around the origin.

            :param translation: translation vector [x_off, y_off] in x- and y-direction
            :param angle: rotation angle in radian (counter-clockwise)
            :return: transformed rectangle
        """
        assert is_real_number_vector(translation, 2), '<Rectangle/translate_rotate>: argument "translation" is ' \
                                                      'not a vector of real numbers of length 2. translation = ' \
                                                      '{}'.format(translation)
        assert is_valid_orientation(angle), '<Rectangle/translate_rotate>: argument "orientation" is not valid.' \
                                            'orientation = {}'.format(angle)
        new_center = translate_rotate(self._center.reshape([1, -1]), translation, angle)[0]
        new_orientation = make_valid_orientation(self._orientation + angle)
        return Rectangle(self._length, self._width, new_center, new_orientation)

    def rotate_translate_local(self, translation: np.ndarray, angle: float) -> 'Rectangle':
        """ A new rectangle is created by first rotating the rectangle around its center and then translating it.

            :param translation: translation vector [x_off, y_off] in x- and y-direction
            :param angle: rotation angle in radian (counter-clockwise)
            :return: transformed rectangle
        """
        new_center = self._center + translation
        new_orientation = make_valid_orientation(self._orientation + angle)
        return Rectangle(self._length, self._width, new_center, new_orientation)

    def contains_point(self, point: np.ndarray) -> bool:
        """ Checks if a point is contained in a rectangle.

            :param point: 2D point as array [x, y]
            :return: true if the rectangle’s interior or boundary intersects with the given point, otherwise false
        """
        assert is_real_number_vector(point, 2), '<Rectangle/contains_point>: argument "point" is ' \
                                                'not a vector of real numbers of length 2. point = {}'.format(point)
        return self._shapely_polygon.intersects(shapely.geometry.Point(point))

    def _compute_vertices(self) -> np.ndarray:
        """ Computes the vertices of the rectangle."""
        vertices = np.array([[- 0.5 * self._length, - 0.5 * self._width], [- 0.5 * self._length, + 0.5 * self._width],
                             [+ 0.5 * self._length, + 0.5 * self._width], [+ 0.5 * self._length, - 0.5 * self._width],
                             [- 0.5 * self._length, - 0.5 * self._width]])
        return rotate_translate(vertices, self._center, self._orientation)

    def __str__(self):
        output = "Rectangle: \n"
        output += '\t width: {} \n'.format(self._width)
        output += '\t length: {} \n'.format(self._length)
        output += '\t center: {} \n'.format(self._center)
        output += '\t orientation: {} \n'.format(self._orientation)
        return output

    def draw(self, renderer: IRenderer, draw_params: Union[ParamServer, dict, None] = None,
             call_stack: Optional[Tuple[str, ...]] = tuple()):
        renderer.draw_rectangle(self.vertices, draw_params, call_stack)


class Circle(Shape):
    """ The class Circle can be used to model occupied regions or circular obstacles, e.g., a pedestrian.
    A circle is defined by its radius and its geometric center. If we model the shape of an obstacle,
    the geometric center can be omitted and is set to [0.0, 0.0]."""

    def __init__(self, radius: float, center: np.ndarray = None):
        """
        :param radius: radius of the circle in [m]
        :param center: geometric center [x, y] of the circle in [m]. If we model the shape of an obstacle,
        the geometric center can be omitted and is set to [0.0, 0.0].
        """
        self.radius: float = radius
        self.center: np.ndarray = center if center is not None else np.array([0.0, 0.0])
        self._shapely_circle: shapely.geometry = \
            shapely.geometry.Point(self.center[0], self.center[1]).buffer(radius / 2)

    @property
    def radius(self) -> float:
        """ The radius of the circle."""
        return self._radius

    @radius.setter
    def radius(self, radius: float):
        if not hasattr(self, '_radius'):
            assert is_real_number(radius), '<Rectangle/radius>: argument "radius" is not a real number. ' \
                                           'radius = {}'.format(radius)
            self._radius = radius
        else:
            warnings.warn('<Rectangle/radius>: radius of circle is immutable.')

    @property
    def center(self) -> np.ndarray:
        """ Geometric center [x, y] of the circle. If the circle is used to describe the shape of an obstacle,
        we set the center to the coordinates [0.0, 0.0]."""
        return self._center

    @center.setter
    def center(self, center: np.ndarray):
        if not hasattr(self, '_center'):
            assert is_real_number_vector(center, 2), '<Circle/center>: argument "center" is not a vector ' \
                                                     'of real numbers of length 2. center = {}'.format(center)
            self._center = center
        else:
            warnings.warn('<Circle/center>: center of circle is immutable.')

    @property
    def shapely_object(self) -> shapely.geometry.Polygon:
        return self._shapely_circle

    def translate_rotate(self, translation: np.ndarray, angle: float) -> 'Circle':
        """ A new circle is created by first translating and then rotating the current circle around the origin.

            :param translation: translation vector [x_off, y_off] in x- and y-direction
            :param angle: rotation angle in radian (counter-clockwise)
            :return: transformed circle
        """
        assert is_real_number_vector(translation, 2), '<Circle/translate_rotate>: argument "translation" is ' \
                                                      'not a vector of real numbers of length 2. translation = ' \
                                                      '{}'.format(translation)
        new_center = translate_rotate(np.array([self._center]), translation, angle)[0]
        return Circle(self._radius, new_center)

    def rotate_translate_local(self, translation: np.ndarray, angle: float) -> 'Circle':
        """ A new circle is created by translating the center.

            :param translation: translation vector [x_off, y_off] in x- and y-direction
            :param angle: rotation angle in radian (counter-clockwise)
            :return: transformed circle
        """
        assert is_real_number_vector(translation, 2), '<Circle/rotate_translate_local>: argument "translation" is ' \
                                                      'not a vector of real numbers of length 2. translation = ' \
                                                      '{}'.format(translation)
        new_center = self._center + translation
        return Circle(self._radius, new_center)

    def contains_point(self, point: np.ndarray):
        """ Checks if a point is contained in a circle.

            :param point: 2D point [x, y]
            :return: true if the circles’s interior or boundary intersects with the given point, otherwise false
        """
        assert is_real_number_vector(point, 2), '<Circle/contains_point>: argument ' \
                                                '"point" is ' \
                                                'not a vector of real numbers of ' \
                                                'length 2. point = {}'.format(point)
        return np.greater_equal(self._radius, np.linalg.norm(point - self._center))

    def __str__(self):
        output = "Circle: \n"
        output += '\t radius: {} \n'.format(self._radius)
        output += '\t center: {} \n'.format(self._center)
        return output

    def draw(self, renderer: IRenderer, draw_params: Union[ParamServer, dict, None] = None,
             call_stack: Optional[Tuple[str, ...]] = tuple()):
        renderer.draw_ellipse(self.center, self.radius, self.radius, draw_params, call_stack)


class Polygon(Shape):
    """ The class Polygon can be used to model occupied regions or obstacles. A polygon is defined by an array of
    ordered points (clockwise or counterclockwise)."""

    def __init__(self, vertices: np.ndarray):
        """
        :param vertices: array of ordered vertices of the polygon [[x_0, y_0], [x_1, y_1], ...]
        """
        self.vertices: np.ndarray = vertices
        self._min: np.ndarray = np.min(vertices, axis=0)
        self._max: np.ndarray = np.max(vertices, axis=0)
        self._shapely_polygon: shapely.geometry.Polygon = shapely.geometry.Polygon(self._vertices)
        # ensure that vertices are sorted clockwise and the first and last point are the same
        self._vertices = np.array(shapely.geometry.polygon.orient(self._shapely_polygon, sign=-1.0).exterior.coords)

    def __eq__(self, other):
        if not isinstance(other, Polygon):
            return False

        thresh = 1e-10

        return np.allclose(self._vertices, other.vertices, rtol=thresh, atol=thresh) \
            and self._shapely_polygon == other.shapely_object

    @property
    def vertices(self) -> np.ndarray:
        """ Vertices of the polygon [[x_0, y_0], [x_1, y_1], ...]. The vertices are sorted clockwise and the
            first and last point are the same.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray):
        if not hasattr(self, '_vertices'):
            assert is_valid_polyline(vertices), '<Polygon/vertices>: argument "vertices" is not valid. vertices = ' \
                                                '{}'.format(vertices)
            self._vertices = vertices
            self._min = np.min(vertices, axis=0)
            self._max = np.max(vertices, axis=0)
        else:
            warnings.warn('<Polygon/vertices>: vertices of polygon are immutable.')

    @property
    def center(self) -> np.ndarray:
        """ Computes the geometric center of the polygon."""
        return np.array(self._shapely_polygon.centroid)

    @property
    def shapely_object(self) -> shapely.geometry.Polygon:
        return self._shapely_polygon

    def translate_rotate(self, translation: np.ndarray, angle: float) -> 'Polygon':
        """ A new polygon is created by first translating and then rotating the current polygon.

            :param translation: translation vector [x_off, y_off] in x- and y-direction
            :param angle: rotation angle in radian (counter-clockwise)
            :return: transformed polygon
        """
        assert is_real_number_vector(translation, 2), '<Polygon/translate_rotate>: argument "translation" is ' \
                                                      'not a vector of real numbers of length 2. translation = ' \
                                                      '{}'.format(translation)
        assert is_valid_orientation(angle), '<Polygon/translate_rotate>: argument "orientation" is not valid.' \
                                            'orientation = {}'.format(angle)
        return Polygon(translate_rotate(self._vertices, translation, angle))

    def rotate_translate_local(self, translation: np.ndarray, angle: float) -> 'Polygon':
        """ A new polygon is created by first rotating the polygon around its center and then translating it.

            :param translation: translation vector [x_off, y_off] in x- and y-direction
            :param angle: rotation angle in radian (counter-clockwise)
            :return: transformed polygon
        """
        assert is_real_number_vector(translation, 2), '<Polygon/rotate_translate_local>: argument "translation" is ' \
                                                      'not a vector of real numbers of length 2. translation = ' \
                                                      '{}'.format(translation)
        assert is_valid_orientation(angle), '<Polygon/rotate_translate_local>: argument "orientation" is not valid.' \
                                            'orientation = {}'.format(angle)
        rotated_shapely_polygon = shapely.affinity.rotate(self._shapely_polygon, angle, origin='centroid',
                                                          use_radians=True)
        new_vertices = np.array(rotated_shapely_polygon.exterior.coords) + translation
        return Polygon(new_vertices)

    def contains_point(self, point: np.ndarray) -> bool:
        """ Checks if a point is contained in the polygon.

            :param point: 2D point
            :return: true if the polygons’s interior or boundary intersects with the given point, otherwise false
        """
        assert is_real_number_vector(point, 2), '<Polygon/contains_point>: argument ' \
                                                '"point" is ' \
                                                'not a vector of real numbers of ' \
                                                'length 2. point = {}'.format(point)

        def in_axis_aligned_bounding_box(point: np.ndarray) -> bool:
            """
            fast check if a point is inside the axis aligned bounding box of a lanelet
            """
            return all(np.less_equal(self._min, point)) and all(np.less_equal(point, self._max))

        return in_axis_aligned_bounding_box(point) and self._shapely_polygon.intersects(shapely.geometry.Point(point))

    def __str__(self):
        output = "Polygon: \n"
        output += '\t vertices: {} \n'.format(self._vertices.tolist())
        output += '\t center: {} \n'.format(self.center)
        return output

    def draw(self, renderer: IRenderer, draw_params: Union[ParamServer, dict, None] = None,
             call_stack: Optional[Tuple[str, ...]] = tuple()):
        renderer.draw_polygon(self.vertices, draw_params, call_stack)


class ShapeGroup(Shape):
    """ The class ShapeGroup represents a collection of primitive shapes, e.g., rectangles and polygons,
    which can be used to model occupied regions."""

    def __init__(self, shapes: List[Shape]):
        """
        :param shapes: list of shapes
        """
        self.shapes = shapes

    @property
    def shapes(self) -> List[Shape]:
        """ Collection of shapes."""
        return self._shapes

    @shapes.setter
    def shapes(self, shapes: List[Shape]):
        if not hasattr(self, '_shapes'):
            assert isinstance(shapes, list) and all(isinstance(elem, Shape) for elem in
                                                    shapes), '<ShapeGroup/shapes>: argument "shapes" is not a valid ' \
                                                             'list of shapes. shapes = {}'.format(shapes)
            self._shapes = shapes
        else:
            warnings.warn('<ShapeGroup/shapes>: shapes of shape group are immutable.')

    def translate_rotate(self, translation: np.ndarray, angle: float) -> 'ShapeGroup':
        """ A new shape group is created by first translating and then rotating all shapes around the origin.

            :param translation: translation vector [x_off, y_off] in x- and y-direction
            :param angle: rotation angle in radian (counter-clockwise)
            :return: transformed shape group
        """
        assert is_real_number_vector(translation, 2), '<ShapeGroup/translate_rotate>: argument "translation" is ' \
                                                      'not a vector of real numbers of length 2. translation = ' \
                                                      '{}'.format(translation)
        assert is_valid_orientation(angle), '<ShapeGroup/translate_rotate>: argument "orientation" is not valid.' \
                                            'orientation = {}'.format(angle)

        new_shapes = list()
        for s in self._shapes:
            new_shapes.append(s.translate_rotate(translation, angle))
        return ShapeGroup(new_shapes)

    def rotate_translate_local(self, translation: np.ndarray, angle: float) -> 'ShapeGroup':
        """ A new shape group is created by first rotating each shape around its center and then translating it.

            :param translation: translation vector [x_off, y_off] in x- and y-direction
            :param angle: rotation angle in radian (counter-clockwise)
            :return: transformed shape group
        """
        assert is_real_number_vector(translation, 2), '<ShapeGroup/rotate_translate_local>: argument "translation" ' \
                                                      'is not a vector of real numbers of length 2. translation = ' \
                                                      '{}'.format(translation)
        assert is_valid_orientation(angle), '<ShapeGroup/rotate_translate_local>: argument "orientation" is not ' \
                                            'valid. orientation = {}'.format(angle)

        new_shapes = list()
        for s in self._shapes:
            new_shapes.append(s.rotate_translate_local(translation, angle))
        return ShapeGroup(new_shapes)

    def contains_point(self, point: np.array):
        """ Checks if a point is contained in any shape of the shape group.

            :param point: 2D point [x, y]
            :return: true if the interior or boundary of any shape intersects with the given point, otherwise false
        """
        assert is_real_number_vector(point, 2), '<ShapeGroup/contains_point>: argument "point" is not a vector ' \
                                                'of real numbers of length 2. point = {}'.format(point)
        for s in self._shapes:
            if s.contains_point(point):
                return True
        return False

    def __str__(self):
        output = 'ShapeGroup: \n'
        output += '\t number of shapes: {} \n'.format(len(self._shapes))
        return output

    def draw(self, renderer: IRenderer, draw_params: Union[ParamServer, dict, None] = None,
             call_stack: Optional[Tuple[str, ...]] = tuple()):
        for s in self._shapes:
            s.draw(renderer, draw_params, call_stack)


def occupancy_shape_from_state(shape, state):
    if state.is_uncertain_position or state.is_uncertain_orientation:
        # From M. Althoff and J. M. Dolan, “Online Verification of Automated Road Vehicles Using Reachability Analysis,”
        # IEEE Transactions on Robotics, vol. 30, no. 4, pp. 903–918, Aug. 2014, doi: 10.1109/TRO.2014.2312453.
        # Section IV.C
        if isinstance(shape, Rectangle) or isinstance(shape, Polygon):
            min_x, min_y, max_x, max_y = shape.shapely_object.bounds
            l_v = np.abs(max_x - min_x)
            w_v = np.abs(max_y - min_y)
        elif isinstance(shape, Circle):
            w_v = 2.0 * shape.radius
            l_v = 2.0 * shape.radius
        else:
            raise ValueError

        if state.is_uncertain_orientation:
            # Using middle of orientation interval as reference orientation
            psi_d = state.orientation.start + 0.5 * state.orientation.length
            delta_psi = 0.5 * state.orientation.length
        else:
            psi_d = state.orientation
            delta_psi = 0.0

        if state.is_uncertain_position:
            center = state.position.center
            if isinstance(state.position, Rectangle) or isinstance(state.position, Polygon):
                # Rotate shape to obtain bounding box with edges parallel to the
                # frame of the reference trajectory
                rot_shape = state.position.rotate_translate_local(np.array([0, 0]), -psi_d)
                min_x, min_y, max_x, max_y = rot_shape.shapely_object.bounds
                l_s = np.abs(max_x - min_x)
                w_s = np.abs(max_y - min_y)
            elif isinstance(state.position, Circle):
                l_s = 2.0 * state.position.radius
                w_s = l_s
            else:
                raise ValueError
        else:
            l_s = 0.0
            w_s = 0.0
            center = state.position

        # Maximum enlargement at these angles
        delta_psi_l = min(delta_psi, np.arctan(w_v / l_v))
        delta_psi_w = min(delta_psi, np.arctan(l_v / w_v))
        l_psi = np.abs((1.0 - np.cos(delta_psi_l)) * l_v - np.sin(delta_psi_l) * w_v)
        w_psi = np.abs((1.0 - np.cos(delta_psi_w)) * w_v - np.sin(delta_psi_w) * l_v)
        l_enclosing = l_s + l_v + l_psi
        w_enclosing = w_s + w_v + w_psi
        occupied_region = Rectangle(l_enclosing, w_enclosing, center, psi_d)
    else:
        occupied_region = shape.rotate_translate_local(state.position, state.orientation)
    return occupied_region
