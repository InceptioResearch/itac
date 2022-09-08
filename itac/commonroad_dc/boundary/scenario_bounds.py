def calc_corners(lanelets, boundary_margin=20):
    """calculate the corners of the scenario from the lanelets
        corners: outermost coordinates in x and y direction:
        corners = [xmin, xmax, ymin, ymax]
    """
    corners = [float('inf'), float('-inf'), float('inf'), float('-inf')]

    for l in lanelets:
        for vertex in l.left_vertices:
            if vertex[0] < corners[0]:
                corners[0] = vertex[0]
            if vertex[0] > corners[1]:
                corners[1] = vertex[0]
            if vertex[1] < corners[2]:
                corners[2] = vertex[1]
            if vertex[1] > corners[3]:
                corners[3] = vertex[1]
        for vertex in l.right_vertices:
            if vertex[0] < corners[0]:
                corners[0] = vertex[0]
            if vertex[0] > corners[1]:
                corners[1] = vertex[0]
            if vertex[1] < corners[2]:
                corners[2] = vertex[1]
            if vertex[1] > corners[3]:
                corners[3] = vertex[1]

    corners[0] -= boundary_margin
    corners[2] -= boundary_margin
    corners[1] += boundary_margin
    corners[3] += boundary_margin
    return corners


def calc_boundary_box(corners):
    """calculates the coordinates needed to build an axis aligned rectangle which spans the corners
        coordinates: width/2, height/2, center x, center y
    """
    halfwidth = (corners[1] - corners[0]) / 2
    halfheight = (corners[3] - corners[2]) / 2
    return [halfwidth, halfheight, halfwidth + corners[0], halfheight + corners[2]]
