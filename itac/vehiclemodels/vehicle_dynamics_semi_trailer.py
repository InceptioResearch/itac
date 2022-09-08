import math

from vehiclemodels.utils.steering_constraints import steering_constraints
from vehiclemodels.utils.acceleration_constraints import acceleration_constraints


def vehicle_dynamics_semi_trailer(x, u_init, p):
    """
    vehicle_dynamics_semi_trailer - kinematic single-track with one on-axle trailer vehicle dynamics
    reference point: rear axle

    Syntax:
        f = vehicle_dynamics_semi_trailer(x,u,p)

    Inputs:
        :param x: vehicle state vector
        :param u_init: vehicle input vector
        :param p: vehicle parameter vector

    Outputs:
        :return f: right-hand side of differential equations
    """

    # ------------- BEGIN CODE --------------

    # create equivalent kinematic single-track parameters
    l_wb = p.a + p.b  # wheel base
    l_wbt = p.trailer.l_wb  # wheel base trailer

    # states
    # x1 = x-position in a global coordinate system, tractor
    # x2 = y-position in a global coordinate system, tractor
    # x3 = steering angle of front wheels
    # x4 = velocity in longitudinal direction
    # x5 = yaw angle, tractor
    # x6 = hitch angle
    # x7 = x-position in a global coordinate system, trailer
    # x8 = y-position in a global coordinate system, trailer
    # x9 = yaw angle, trailer

    # inputs
    # u1 = steering angle velocity of front wheels
    # u2 = longitudinal acceleration

    u = []
    # consider steering constraints
    u.append(steering_constraints(
        x[2], u_init[0],
        p.steering))  # different name uInit/u due to side effects of u
    # consider acceleration constraints
    u.append(acceleration_constraints(
        x[3], u_init[1],
        p.longitudinal))  # different name uInit/u due to side effects of u

    # hitch angle constraints
    if -math.pi / 2 <= x[5] <= math.pi / 2:
        d_alpha = -x[3] * (math.sin(x[5]) / l_wbt + math.tan(x[2]) / l_wb)
    else:
        d_alpha = 0
        x[5] = -math.pi / 2 if x[5] < -math.pi / 2 else math.pi / 2

    # system dynamics
    f = [
        x[3] * math.cos(x[4]),
        x[3] * math.sin(x[4]),
        u[0],
        u[1],
        x[3] / l_wb * math.tan(x[2]),
        d_alpha,
        x[3] * math.cos(x[4]) + l_wbt * math.sin(x[8]) * (x[3] / l_wb * math.tan(x[2]) + d_alpha),
        x[3] * math.sin(x[4]) - l_wbt * math.cos(x[8]) * (x[3] / l_wb * math.tan(x[2]) + d_alpha),
        x[3] / l_wb * math.tan(x[2]) + d_alpha
    ]

    return f

    # ------------- END OF CODE --------------
