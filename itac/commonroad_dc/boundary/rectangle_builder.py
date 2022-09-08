import numpy as np
import commonroad_dc.pycrcc as pycrcc


def get_rectangle(width, v1, v2):
    """builds a rectangle object which has the line v1,v2 as middle line"""
    v = v2 - v1
    r_x = np.linalg.norm(v) / 2
    r_y = width / 2
    orientation = np.arctan2(v[1], v[0])
    center = v1 + (v) / 2
    return pycrcc.RectOBB(r_x, r_y, orientation, *center)
