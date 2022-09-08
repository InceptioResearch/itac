import numpy as np

from commonroad_dc.boundary import rectangle_builder


def lateral_bounds(lanelet_network):
    """generator function to find the leftmost and rightmost boundaries of each lane section
        the vertices and the boundaries are ordered in the same direction
    """
    lanelets = lanelet_network.lanelets
    lanelet_id_list = list(map(lambda l: l.lanelet_id, lanelets))
    while lanelet_id_list != []:
        # search for the leftmost and rightmost lanelets adjacent to the current one
        id = lanelet_id_list.pop()
        right = left = lanelet_network.find_lanelet_by_id(id)
        right_direction = left_direction = True

        # expand left while there is a lanelet on the left, looking from the direction of the first lanelet
        while (left_direction and left.adj_left != None) or (not left_direction and left.adj_right != None):
            if (left_direction):
                next_id = left.adj_left
                left_direction = (left_direction == left.adj_left_same_direction)  # flip of direction
            else:
                next_id = left.adj_right
                left_direction = (left_direction == left.adj_right_same_direction)

            left = lanelet_network.find_lanelet_by_id(next_id)
            lanelet_id_list.remove(next_id)
        if (left_direction):
            left_vertices = left.left_vertices
        else:
            # vertices are in a coloumn vector, flipud to reverse the order of the vertices
            left_vertices = np.flipud(left.right_vertices)

        # expand right
        while (right_direction and right.adj_right != None) or (not right_direction and right.adj_left != None):
            if (right_direction):
                next_id = right.adj_right
                right_direction = (right_direction == right.adj_right_same_direction)
            else:
                next_id = right.adj_left
                right_direction = (right_direction == right.adj_left_same_direction)
            right = lanelet_network.find_lanelet_by_id(next_id)
            lanelet_id_list.remove(next_id)
        if (right_direction):
            right_vertices = right.right_vertices
        else:
            right_vertices = np.flipud(right.left_vertices)

        yield left_vertices, right_vertices


def longitudinal_bounds(lanelets):
    """ generator function to find the first and last boundaries of successive lanelets
    """
    for l in lanelets:
        left = l.left_vertices
        right = l.right_vertices
        if l.predecessor == []:
            yield left[0], right[0]
        if l.successor == []:
            yield left[-1], right[-1]


def lane_sections(lanelet_network):
    """ generator function that generates the boundaries of each lane section

        creates the left and right bounds like longitudinal_bounds(), but also generates the polylines at the
        start and end of each lane section. These can be used for delaunay triangulation of a lane section, which
        leads to a perfect collision representation of a lane.

        yields the boundary of each lane section as four polylines, namely:
        start_vertices, left_vertices, end_vertices, right_vertices
    """
    lanelets = lanelet_network.lanelets
    lanelet_id_list = list(map(lambda l: l.lanelet_id, lanelets))

    while lanelet_id_list != []:
        # search for the leftmost and rightmost lanelets adjacent to the current one
        id = lanelet_id_list.pop()
        current = lanelet_network.find_lanelet_by_id(id)
        direction = True

        start_vertices = []
        end_vertices = []

        def add_initial_lateral_vertices(current, direction):
            if direction:
                start_vertices.append(current.right_vertices[0])
                end_vertices.insert(0, current.right_vertices[-1])
            else:
                start_vertices.append(current.left_vertices[-1])
                end_vertices.insert(0, current.left_vertices[0])

        def add_lateral_vertices(current, direction):
            if direction:
                start_vertices.append(current.left_vertices[0])
                end_vertices.insert(0, current.left_vertices[-1])
            else:
                start_vertices.append(current.right_vertices[-1])
                end_vertices.insert(0, current.right_vertices[0])

        # ____________________ go to the rightmost lanelet _______________________
        while (direction and current.adj_right != None) or (not direction and current.adj_left != None):
            if (direction):
                next_id = current.adj_right
                direction = (direction == current.adj_right_same_direction)
            else:
                next_id = current.adj_left
                direction = (direction == current.adj_left_same_direction)
            current = lanelet_network.find_lanelet_by_id(next_id)

        if direction:
            right_vertices = np.flipud(current.right_vertices)
        else:
            right_vertices = current.left_vertices

        add_initial_lateral_vertices(current, direction)
        add_lateral_vertices(current, direction)

        # ___________________ expand left ____________________________
        # expand left while there is a lanelet on the left, looking from the direction of the first lanelet
        while (direction and current.adj_left != None) or (not direction and current.adj_right != None):
            if direction:
                next_id = current.adj_left
                direction = (direction == current.adj_left_same_direction)  # flip of direction
            else:
                next_id = current.adj_right
                direction = (direction == current.adj_right_same_direction)

            current = lanelet_network.find_lanelet_by_id(next_id)
            if next_id in lanelet_id_list:
                lanelet_id_list.remove(next_id)
            add_lateral_vertices(current, direction)

        if direction:
            left_vertices = current.left_vertices
        else:
            # vertices are in a coloumn vector, flipud to reverse the order of the vertices
            left_vertices = np.flipud(current.right_vertices)

        start_vertices = np.array(start_vertices)
        end_vertices = np.array(end_vertices)
        yield start_vertices, left_vertices, end_vertices, right_vertices


def lane_hull(lanelet_network):
    """Yields the single polyline which describes the boundary of each lane section"""
    for start_vertices, left_vertices, end_vertices, right_vertices in lane_sections(lanelet_network):
        # the corner vertices are included twice
        yield np.concatenate((start_vertices, left_vertices[1:-1], end_vertices, right_vertices[1:-1]))


def offset_bounds(lanelet_network, offset):
    """generator function that generates the lateral and longitudinal boundaries
        and moves them to the outside by the offset value
    """

    def elongate_line(v, w):
        """move both points which form a line, so that the line is elongated by offset*2"""
        if np.linalg.norm(w - v) == 0:
            return v, w
        tangent = offset * (w - v) / np.linalg.norm(w - v)
        return v - tangent, w + tangent

    def elongate_boundary(bound):
        """insert points at the beginning and end, so that the boundary is elongated"""
        bound = np.insert(bound, 0, elongate_line(*bound[:2])[0], axis=0)
        # had problems with np.append, so use of insert at len - 1
        return np.insert(bound, len(bound), elongate_line(*bound[-2:])[1], axis=0)

    # TODO: reduce, special case of elongate_line
    def offset_point(v, w):
        """offset point v away from point w"""
        tangent = offset * (w - v) / np.linalg.norm(w - v)
        return v - tangent

    def offset_bound_rel_other(bound, other):
        """offset a boundary relative to another, which is opposite to it"""

        def offset_point_rel_other(v):
            """offset point away from closest point on the opposite boundary"""
            index = np.argmin([np.linalg.norm(w - v) for w in other])
            return offset_point(v, other[index])

        return np.apply_along_axis(lambda v: offset_point_rel_other(v), 1, bound)

    def longitudinal_bounds_offset(lanelets):
        """rewritten longitudinal_bounds function
            required because we need to know for each bound which lanelet it corresponds to"""
        for l in lanelets:
            left = l.left_vertices
            right = l.right_vertices
            if l.predecessor == []:
                v, w = elongate_line(left[0], right[0])
                v = offset_point(v, left[1])
                w = offset_point(w, right[1])
                yield np.array((v, w))
            if l.successor == []:
                v, w = elongate_line(left[-1], right[-1])
                v = offset_point(v, left[-2])
                w = offset_point(w, right[-2])
                yield np.array((v, w))
        return None

    for left_vertices, right_vertices in lateral_bounds(lanelet_network):
        left_vertices = offset_bound_rel_other(left_vertices, right_vertices)
        left_vertices = elongate_boundary(left_vertices)

        right_vertices = offset_bound_rel_other(right_vertices, left_vertices)
        right_vertices = elongate_boundary(right_vertices)
        yield left_vertices
        yield right_vertices

    for bound in longitudinal_bounds_offset(lanelet_network.lanelets):
        yield bound


def offset_bounds_lateral(lanelet_network, offset):
    """Same as offset_bounds, but returns only lateral bounds, in pairs (left,right)"""

    def elongate_line(v, w):
        """move both points which form a line, so that the line is elongated by offset*2"""
        if np.linalg.norm(w - v) == 0:
            return v, w
        tangent = offset * (w - v) / np.linalg.norm(w - v)
        return v - tangent, w + tangent

    def elongate_boundary(bound):
        """insert points at the beginning and end, so that the boundary is elongated"""
        bound = np.insert(bound, 0, elongate_line(*bound[:2])[0], axis=0)
        # had problems with np.append, so use of insert at len - 1
        return np.insert(bound, len(bound), elongate_line(*bound[-2:])[1], axis=0)

    # TODO: reduce, special case of elongate_line
    def offset_point(v, w):
        """offset point v away from point w"""
        tangent = offset * (w - v) / np.linalg.norm(w - v)
        return v - tangent

    def offset_bound_rel_other(bound, other):
        """offset a boundary relative to another, which is opposite to it"""

        def offset_point_rel_other(v):
            """offset point away from closest point on the opposite boundary"""
            index = np.argmin([np.linalg.norm(w - v) for w in other])
            return offset_point(v, other[index])

        return np.apply_along_axis(lambda v: offset_point_rel_other(v), 1, bound)

    for left_vertices, right_vertices in lateral_bounds(lanelet_network):
        left_vertices = offset_bound_rel_other(left_vertices, right_vertices)
        left_vertices = elongate_boundary(left_vertices)

        right_vertices = offset_bound_rel_other(right_vertices, left_vertices)
        right_vertices = elongate_boundary(right_vertices)
        yield left_vertices, right_vertices


def outer_vertices(lanelet_network, quads):
    """Generates all bounds that collide with the quads and are therefore considered to be outside
        Return: outer vertices in pairs
    """
    RECTANGLE_WIDTH = 0.2

    def iterate_vertices(vertices):
        v = vertices[0]
        for w in vertices[1:]:
            rect = rectangle_builder.get_rectangle(RECTANGLE_WIDTH, v, w)
            if rect.collide(quads):
                yield v, w
            v = w

    for leftvertices, rightvertices in lateral_bounds(lanelet_network):
        yield from iterate_vertices(leftvertices)
        yield from iterate_vertices(rightvertices)
    for v1, v2 in longitudinal_bounds(lanelet_network.lanelets):
        yield from iterate_vertices([v1, v2])


def pairwise_bounds(lanelet_network):
    """Yields the longitudinal and lateral bounds of the lanelets in the network in pairs"""
    for left_vertices, right_vertices in lateral_bounds(lanelet_network):
        v1 = left_vertices[0]
        for v2 in left_vertices[1:]:
            yield v1, v2
            v1 = v2

        v1 = right_vertices[0]
        for v2 in right_vertices[1:]:
            yield v1, v2
            v1 = v2

    for bound in longitudinal_bounds(lanelet_network.lanelets):
        yield bound


def pairwise_offset_bounds(lanelet_network, offset):
    """Yields the offset longitudinal and lateral bounds of the lanelets in the network in pairs"""
    for bound in offset_bounds(lanelet_network, offset):
        v1 = bound[0]
        for v2 in bound[1:]:
            yield v1, v2
            v1 = v2


def pairs(l, cyclic=True):
    """
    Creates pairs of adjacent elements in l
    :param l: list
    :param cyclic: last and first element make a pair
    :return: iterator of pairs
    """
    partner = l[1:]
    if cyclic:
        partner.append(l[0])
    return zip(l, partner)


def polyline_edges(polyline, cyclic=True):
    """Returns the edges between each two points of a polyline.

        cyclic: flags if there exists an edge between the last and first point
    """
    return list(pairs(polyline, cyclic))


def polyline_normals(polyline, cyclic=True):
    """Returns the normals of each points of a cyclic polyline.
    The sign of the normals can be seen as arbitrary,
    i.e. they do not necessarily point to the outwards of the polygon that the line describes.

    The normal for a point p at the index i s defined as following:
        normal = normal(p[i-1] - p[i]) + normal(p[i+1] - p[i])
    with all the involved normals being normalized.
    """
    edges = polyline_edges(polyline, cyclic)

    def normalized(a):
        if np.linalg.norm(a) == 0:
            return a
        return a / np.linalg.norm(a)

    def segment_normal(point_a, point_b):
        # normal that is orthogonal on the line between point_a and point_b (in 2D)
        segment = point_a - point_b
        return normalized(np.array([-segment[1], segment[0]]))

    edge_normals = [segment_normal(point_a, point_b) for point_a, point_b in edges]
    # Compute normals of all points between two edges
    point_normals = [normalized(normal_a + normal_b) for normal_a, normal_b in pairs(edge_normals, cyclic)]

    if cyclic:
        # Normal for first point is at the end of the list, swap
        point_normals = [point_normals[-1]] + point_normals[:-1]
    else:
        # The start and end points take the normal of their single edge
        point_normals.insert(0, edge_normals[0])
        point_normals.append(edge_normals[-1])

    assert len(polyline) == len(point_normals), "{} <> {}".format(len(polyline), len(point_normals))
    return point_normals
