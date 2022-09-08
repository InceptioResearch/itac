import math

# from vehiclemodels.steeringConstraints import steering_constraints
# from vehiclemodels.utils.accelerationConstraints import acceleration_constraints
from vehiclemodels.utils.acceleration_constraints import acceleration_constraints
from vehiclemodels.utils.steering_constraints import steering_constraints


def vehicle_dynamics_sedan(x, u_init, p):
    """
    vehicleDynamics_ks - kinematic single-track vehicle dynamics
    reference point: rear axle

    Syntax:
        f = vehicleDynamics_ks(x,u,p)

    Inputs:
        :param x: vehicle state vector
        :param u_init: vehicle input vector
        :param p: vehicle parameter vector

    Outputs:
        :return f: right-hand side of differential equations

    """

    #------------- BEGIN CODE --------------

    #create equivalent kinematic single-track parameters
    l = p.a + p.b

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle

    #u1 = steering angle velocity of front wheels
    #u2 = longitudinal acceleration

    #consider steering constraints
    u = []
    u.append(steering_constraints(
        x[2], u_init[0],
        p.steering))  # different name u_init/u due to side effects of u
    #consider acceleration constraints
    u.append(acceleration_constraints(
        x[3], u_init[1],
        p.longitudinal))  # different name u_init/u due to side effects of u

    #system dynamics
    f = [
        x[3] * math.cos(x[4]), x[3] * math.sin(x[4]), u[0], u[1],
        x[3] / l * math.tan(x[2])
    ]

    return f

    #------------- END OF CODE --------------
