import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
from itac import *

    
class ConstraintsPointMass:
    x_s_min = 0  # feasible min longitudinal displacement
    x_s_max = 150  # feasible max longitudinal displacement
    x_l_min = -7 # feasible min lateral displacement
    x_l_max = 7 # feasible max lateral displacement
    
    v_s_min = 0  # feasible min longitudinal velocity
    v_s_max = 40 # feasible max longitudinal velocity
    v_l_min = -1.5 # feasible min lateral velocity
    v_l_max = 1.5 # feasible max lateral velocity
    
    a_s_min = -6 # feasible min longitudinal acceleration
    a_s_max =  6 # feasible max longitudinal acceleration
    a_l_min =  -1.0 # feasible min lateral acceleration
    a_l_max =  1.0 # feasible max lateral acceleration

def plot_state_vector(x : cp.Variable, u : cp.Variable, c : ConstraintsPointMass):
    N = x.shape[1]-1

    # Plot (x_t)_1.
    plt.subplot(6,1,1)
    x1 = (x.value)[0,:].flatten()
    plt.plot(np.array(range(N+1)),x1,'g')
    plt.yticks(np.linspace(c.x_s_min, c.x_s_max, 3))
    plt.ylim([c.x_s_min, c.x_s_max])
    plt.ylabel(r"$x_s$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_2.
    plt.subplot(6,1,2)
    x2 = (x.value)[1,:].flatten()
    plt.plot(np.array(range(N+1)),x2,'g')
    plt.yticks(np.linspace(c.x_l_min, c.x_l_max,3))
    plt.ylim([c.x_l_min, c.x_l_max+2])
    plt.ylabel(r"$x_l$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_3.
    plt.subplot(6,1,3)
    x2 = (x.value)[2,:].flatten()
    plt.plot(np.array(range(N+1)),x2,'g')
    plt.yticks(np.linspace(c.v_s_min,c.v_s_max,3))
    plt.ylim([c.v_s_min,c.v_s_max+2])
    plt.ylabel(r"$v_s$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_4.
    plt.subplot(6,1,4)
    x2 = (x.value)[3,:].flatten()
    plt.plot(np.array(range(N+1)), x2,'g')
    plt.yticks(np.linspace(c.v_l_min,c.v_l_max,3))
    plt.ylim([c.v_l_min,c.v_l_max+1])
    plt.ylabel(r"$v_l$", fontsize=16)
    plt.xticks(np.arange(0,N+1,5))
    plt.tight_layout()

    # Plot (u_t)_1.
    plt.subplot(6,1,5)
    u1 = (u.value)[0,:].flatten()
    plt.plot(np.array(range(N)), u1,'g')
    plt.yticks(np.linspace(c.a_s_min,c.a_s_max,3))
    plt.ylim([c.a_s_min,c.a_s_max+2])
    plt.ylabel(r"$a_s$", fontsize=16)
    plt.xticks(np.arange(0,N+1,5))
    plt.tight_layout()

    # Plot (u_t)_2.
    plt.subplot(6,1,6)
    u2 = (u.value)[1,:].flatten()
    plt.plot(np.array(range(N)), u2,'g')
    plt.yticks(np.linspace(c.a_l_min,c.a_l_max,3))
    plt.ylim([c.a_l_min,c.a_l_max+2])
    plt.ylabel(r"$a_l$", fontsize=16)
    plt.xticks(np.arange(0,N+1,5))
    plt.xlabel(r"$k$", fontsize=16)
    plt.tight_layout()

    plt.show()
    
    
    
vehicle4 = parameters_semi_trailer.parameters_semi_trailer()
class ConstraintsKST:
    x_s_min = 0  # feasible min longitudinal displacement
    x_s_max = 100  # feasible max longitudinal displacement
    x_l_min = -5 # feasible min lateral displacement
    x_l_max = 5 # feasible max lateral displacement
    
    v_s_min = vehicle4.longitudinal.v_min  # feasible min longitudinal velocity
    v_s_max = vehicle4.longitudinal.v_max # feasible max longitudinal velocity

    a_s_min = -vehicle4.longitudinal.a_max # feasible min longitudinal acceleration
    a_s_max =  vehicle4.longitudinal.a_max # feasible max longitudinal acceleration

    steer_min = vehicle4.steering.min  # minimum steering angle [rad]
    steer_max = vehicle4.steering.max  # maximum steering angle [rad]
    steer_v_min = vehicle4.steering.v_min  # minimum steering velocity [rad/s]
    steer_v_max = vehicle4.steering.v_max  # maximum steering velocity [rad/s]

    yaw_min = -np.pi / 2 # minimum tractor orientation in frenet coordinate sytem [rad]
    yaw_max = np.pi / 2 # maximum tractor orientation in frenet coordinate sytem [rad]

    hitch_angle_min = -np.pi / 2 # minimum angle between tractor and trailer [rad]
    hitch_angle_max = np.pi / 2 # maximum angle between tractor and trailer [rad]


class EgoParametersKST:
    tractor_l = vehicle4.l
    tractor_w = vehicle4.w
    # tractor axis distance
    l_wb = vehicle4.a + vehicle4.b

    # trailer parameters
    trailer_l = vehicle4.trailer.l   # trailer length
    trailer_w = vehicle4.trailer.w   # trailer width
    l_hitch = vehicle4.trailer.l_hitch   # hitch length
    l_total = vehicle4.trailer.l_total   # total system length
    l_wbt = vehicle4.trailer.l_wb   # trailer wheelbase


def plot_state_vector_kst(x : cp.Variable, u : cp.Variable, c : ConstraintsKST):
    N = x.shape[1]-1

    # Plot (x_t)_1.
    plt.subplot(5,1,1)
    x1 = (x.value)[0,:].flatten()
    plt.plot(np.array(range(N+1)),x1,'g')
    plt.yticks(np.linspace(c.x_l_min, c.x_l_max, 3))
    plt.ylim([c.x_l_min, c.x_l_max])
    plt.ylabel(r"$l$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_2.
    plt.subplot(5,1,2)
    x2 = (x.value)[1,:].flatten()
    plt.plot(np.array(range(N+1)),x2,'g')
    # plt.yticks(np.linspace(c.yaw_min, c.yaw_max,3))
    # plt.ylim([c.yaw_min, c.yaw_max+2])
    plt.ylabel(r"$\Psi$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_3.
    plt.subplot(5,1,3)
    x2 = (x.value)[2,:].flatten()
    plt.plot(np.array(range(N+1)),x2,'g')
    # plt.yticks(np.linspace(c.steer_min,c.steer_max,3))
    # plt.ylim([c.steer_min,c.steer_max+2])
    plt.ylabel(r"$\delta$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_4.
    plt.subplot(5,1,4)
    x2 = (x.value)[3,:].flatten()
    plt.plot(np.array(range(N+1)), x2,'g')
    # plt.yticks(np.linspace(c.hitch_angle_min,c.hitch_angle_max,3))
    # plt.ylim([c.hitch_angle_min,c.hitch_angle_max+1])
    plt.ylabel(r"$\alpha$", fontsize=16)
    plt.xticks(np.arange(0,N+1,5))
    plt.tight_layout()

    # Plot (u_t)_1.
    plt.subplot(5,1,5)
    u1 = (u.value)[0,:].flatten()
    plt.plot(np.array(range(N)), u1,'g')
    # plt.yticks(np.linspace(c.steer_v_min,c.steer_v_max,3))
    # plt.ylim([c.steer_v_min,c.steer_v_max+2])
    plt.ylabel(r"$v_{\delta}$", fontsize=16)
    plt.xticks(np.arange(0,N+1,5))
    plt.tight_layout()

    plt.show()


def plot_state_vector_lon_kst(x : cp.Variable, u : cp.Variable, c : ConstraintsKST):
    N = x.shape[1]-1

    # Plot (x_t)_1.
    plt.subplot(3,1,1)
    x1 = (x.value)[0,:].flatten()
    plt.plot(np.array(range(N+1)),x1,'g')
    plt.yticks(np.linspace(c.x_s_min, c.x_s_max, 3))
    plt.ylim([c.x_s_min, c.x_s_max])
    plt.ylabel(r"$s$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_2.
    plt.subplot(3,1,2)
    x2 = (x.value)[1,:].flatten()
    plt.plot(np.array(range(N+1)),x2,'g')
    plt.yticks(np.linspace(c.v_s_min, c.v_s_max,3))
    # plt.ylim([c.v_s_min, c.v_s_max+2])
    plt.ylabel(r"$v$", fontsize=16)
    plt.xticks([])

    # Plot (u_t)_1.
    plt.subplot(3,1,3)
    u1 = (u.value)[0,:].flatten()
    plt.plot(np.array(range(N)), u1,'g')
    # plt.yticks(np.linspace(c.steer_v_min,c.steer_v_max,3))
    # plt.ylim([c.steer_v_min,c.steer_v_max+2])
    plt.ylabel(r"$a$", fontsize=16)
    plt.xticks(np.arange(0,N+1,5))
    plt.tight_layout()

    plt.show()