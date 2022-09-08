import commonroad.geometry.shape
import numpy as np
import shapely


def minkowski_sum_circle(shape: commonroad.geometry.shape.Shape,
                         radius: float, resolution: int) -> commonroad.geometry.shape.Shape:
    return minkowski_sum_circle_func_dict[type(shape)](shape, radius, resolution)


def minkowski_sum_circle_shapely_polygon(polygon: shapely.geometry.Polygon,
                                         radius: float, resolution: int) -> np.ndarray:
    """
    Computes the minkowski sum of a provided polygon and a circle with
    parametrized radius

    :param polygon: The polygon as a numpy array with columns as x and y
    coordinates
    :param radius: The radius of the circle
    :return: The minkowski sum of the provided polygon and the circle with the
    parametrized radius
    """
    assert isinstance(polygon, shapely.geometry.Polygon), \
        '<> Provided polygon is not an instance of shapely.geometry.Polygon,' \
        ' polygon = {}'.format(polygon)
    assert radius > 0, '<> Provided radius must be positive. radius = {}'. \
        format(radius)
    # dilation with round cap style (style = 1)
    dilation = polygon.buffer(radius, cap_style=1, resolution=resolution)
    if isinstance(dilation, shapely.geometry.MultiPolygon):
        polys = [polygon for polygon in dilation]
        for p in polys:
            p_vertices = list()
            vertices = p.exterior.coords
            for v in vertices:
                p_vertices.append(np.array([v[0], v[1]]))
    # convert to array
    else:
        p_vertices = list()
        vertices = dilation.exterior.coords
        for v in vertices:
            p_vertices.append(np.array([v[0], v[1]]))
    return np.array(p_vertices)


def minkowski_sum_circle_polygon(polygon: commonroad.geometry.shape.Polygon,
                                 radius: float, resolution: int) \
        -> commonroad.geometry.shape.Polygon:
    if np.isclose(radius, 0.0):
        return polygon
    else:
        return commonroad.geometry.shape.Polygon(
            minkowski_sum_circle_shapely_polygon(polygon._shapely_polygon, radius, resolution))


def minkowski_sum_circle_circle(circle: commonroad.geometry.shape.Circle,
                                radius: float, resolution: int) \
        -> commonroad.geometry.shape.Circle:
    return commonroad.geometry.shape.Circle(
        circle.radius + radius, circle.center)


def minkowski_sum_circle_rectangle(
        rectangle: commonroad.geometry.shape.Rectangle, radius: float, resolution: int) \
        -> commonroad.geometry.shape.Polygon:
    return commonroad.geometry.shape.Polygon(
        minkowski_sum_circle_shapely_polygon(rectangle._shapely_polygon, radius, resolution))


def minkowski_sum_circle_shape_group(
        shape_group: commonroad.geometry.shape.ShapeGroup, radius: float, resolution: int) \
        -> commonroad.geometry.shape.ShapeGroup:
    new_shapes = list()
    for s in shape_group.shapes:
        new_shapes.append(minkowski_sum_circle(s, radius, resolution))
    return commonroad.geometry.shape.ShapeGroup(new_shapes)


minkowski_sum_circle_func_dict = {
    commonroad.geometry.shape.ShapeGroup: minkowski_sum_circle_shape_group,
    commonroad.geometry.shape.Polygon: minkowski_sum_circle_polygon,
    commonroad.geometry.shape.Circle: minkowski_sum_circle_circle,
    commonroad.geometry.shape.Rectangle: minkowski_sum_circle_rectangle,
}
