import numpy as np
import commonroad_dc.pycrcc as pycrcc
import sys

try:
    import triangle
except:
    pass

from commonroad_dc.boundary import lanelet_bounds


def triangle_zig_zag(left, right, triangles):
    """create triangles between two polylines that have the same length
    """
    counter = 1
    previous = left[0]
    for vertex in left[1:]:
        triangles.add_shape(pycrcc.Triangle(*previous, *right[counter - 1], *right[counter]))
        triangles.add_shape(pycrcc.Triangle(*vertex, *previous, *right[counter]))
        previous = vertex
        counter += 1


def triangle_fan(point, line, triangles):
    """create triangles between a point and a polyline, so that the triangles form a fan
    """
    previous = line[0]
    for vertex in line[1:]:
        triangles.add_shape(pycrcc.Triangle(*point, *previous, *vertex))
        previous = vertex


def build_simple_triangles(lanelets, triangles):
    """create triangles for each lanelet individually
    """
    for l in lanelets:
        triangle_zig_zag(l.left_vertices, l.right_vertices, triangles)


def build_offset_section_triangles(lanelet_network, triangles, offset):
    """create triangles that cover the enlarged lanelets
    """
    for left_vertices, right_vertices in lanelet_bounds.offset_bounds_lateral(lanelet_network, offset):
        minlength = min(len(left_vertices), len(right_vertices))
        triangle_zig_zag(left_vertices[0:minlength], right_vertices[0:minlength], triangles)
        if (len(left_vertices) < len(right_vertices)):
            triangle_fan(left_vertices[minlength - 1], right_vertices[minlength - 1:], triangles)
        elif (len(left_vertices) > len(right_vertices)):
            triangle_fan(right_vertices[minlength - 1], left_vertices[minlength - 1:], triangles)


def triangulate(bounds, vertices, input_triangles, output_triangles, params):
    """Fill the scenario with triangles using the Triangle Library by Jonathan Richard Shewchuk:
        https://www.cs.cmu.edu/~quake/triangle.html

        To use it, Triangle is called from the wrapper library "triangle"
        Step 1: Write the vertices and edges of the lane boundaries
        Step 2: Call Triangle
        Step 3: Read the triangles, construct them as collision objects, remove triangles that are in the road

    """
    if 'triangle' not in sys.modules:
        raise Exception(
            'This operation requires a non-free third-party python package triangle to be installed. It can be installed using pip (pip install triangle). Please refer to its license agreement for more details.')

    # steiner_ratio: determines the maximum amount of steiner points based on the number of vertices of the input
    # eg: steiner_ratio = 0.1, 100 vertices => 0.1*100 = 10 steiner points allowed at most
    steiner_ratio = params.get('steiner_ratio', 0.1)
    steiner_max = (len(bounds) + len(vertices)) * steiner_ratio

    # Triangle call string, default options used:
    # - q quality mesh generation with minimal 20 degrees for each triangle
    # - S no additional steiner points
    # - c the points are enclosed by their convex hull, by default the option is only enabled if single vertices without
    #     edges are given
    # for further options see
    # http://dzhelil.info/triangle/index.html#api and https://www.cs.cmu.edu/~quake/triangle.switch.html
    # eg. call_options: 'S1000q25'
    defaul_calloptions = 'qS' + str(steiner_max)

    # Build triangles in the convex hull defined by the vertices, if vertices without edges are given as input
    if vertices:
        defaul_calloptions += 'c'
    call_options = 'p' + params.get('call_options', defaul_calloptions)

    # Parameter that determines if the hole areas are written to file
    # Can be problematic when using section triangle representation
    input_holes = params.get('input_holes', False)

    # radius of the circles that are used for filtering triangles in the road
    # value of 0 leads to a point
    # a bigger radius can be used if some triangles between lanelets are not filtered
    # a small radius can be used if small triangles near the road are filtered unwarrented
    filter_radius = params.get('filter_radius', 0)

    # Make sets for nodes, segments, holes
    # The sets eliminate all duplicate elements, so each node/segment/hole is written to file only once.
    nodeset = set()
    segset = set()
    holeset = set()

    def addnode(vertex):
        # The required format for nodes is:  x, y
        # The index is added to each entry after all nodes have been added.
        nodeset.add((*vertex,))

    # Input all individual vertices, that don't have an edge
    for v in vertices:
        addnode(v)

    def addseg(v1, v2):
        # The required format for segments is: v1_index, v2_index
        # The frozenset makes sure that there are no duplicate edges with different order of vertices, ie. (v,w) and (w,v).
        t = tuple([tuple([*v1]), tuple([*v2])])  # wrapping vertices for frozenset
        segset.add(frozenset(t))

    def addedge(v1, v2):
        addnode(v1)
        addnode(v2)
        addseg(v1, v2)

    # Input all edges from the lanelet boundaries
    for bound in bounds:
        addedge(*bound)

    def addhole(v1, v2, v3):
        # The required format for holes is: x, y
        # A hole marks an area where no triangles should be generated.
        # In this case, each input triangles defines a hole area, so that the road is not triangulated.
        center = (v1 + v2 + v3) / 3
        holeset.add((*center,))

    # Input holes from input_triangles
    if input_holes:
        for t in input_triangles.unpack():
            vertices = t.vertices()
            vertices = [np.array(v) for v in vertices]
            addhole(*vertices)

    # Convert each set so that they apply to the required format
    def map_segment(entry, vertices):
        v1, v2 = entry
        v1_index = [i for i, entry in enumerate(vertices) if entry == v1][0]
        v2_index = [i for i, entry in enumerate(vertices) if entry == v2][0]
        return (v1_index, v2_index)

    vertices = [node for node in nodeset]
    # the if condition eliminates edges that have two identical vertices as endpoints
    segset = set([map_segment(entry, vertices) for entry in segset if len(entry) == 2])
    vertices = np.array(vertices)
    segments = np.array([seg for seg in segset])
    holes = np.array([hole for hole in holeset])

    if (len(holes)):
        output = triangle.triangulate({'vertices': vertices, 'segments': segments, 'holes': holes}, call_options)
    else:
        output = triangle.triangulate({'vertices': vertices, 'segments': segments}, call_options)

    vertices = output.get('vertices')
    tris = output.get('triangles')

    for t in tris:
        points = [vertices[index] for index in t]
        a, b, c = (points[0], points[1], points[2])
        collision_triangle = pycrcc.Triangle(*a, *b, *c)

        # remove problem spots:
        middle = (a + b + c) / 3
        if filter_radius == 0:
            middle_point = pycrcc.Point(*middle)
        else:
            middle_point = pycrcc.Circle(filter_radius, *middle)

        # uncomment for drawing middle points
        # draw_object(middle_point, draw_params={'collision': {'circle': {'facecolor': '#1f78b4', 'zorder': 30, 'edgecolor' :'#000000'}}})
        if not input_triangles or not middle_point.collide(input_triangles):
            output_triangles.add_shape(collision_triangle)
