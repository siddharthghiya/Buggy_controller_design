'''
This is a realization of the controller from Vehicle Dynamics and Control by Rajesh Rajamani.
Yaohui Guo
'''
from BuggySimulator import *
import numpy as np
import scipy
import scipy.signal as signal
import control
from scipy.ndimage import gaussian_filter1d
from util import *
from simple_pid import PID

def dlqr(A,B,Q,R):
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    eigVals, eigVecs = scipy.linalg.eig(A - B*K)
 
    return K, X, eigVals

def controller(traj, vehicle, tangent, curvature, itr):

	#defining the constants
    m = 2000
    l_r = 1.7
    l_f = 1.1
    c_alpha = 15000
    i_z = 3344
    mu = 0.01
    dt = 0.05

    #paramters for F PID
    xd_k_u = 72000
    xd_t_u = 0.1
    xd_t_i = xd_t_u / 2
    xd_t_d = xd_t_u / 8
    xd_k_p = 0.8 * xd_k_u
    xd_k_i = 0
    xd_k_d = xd_k_p * xd_t_d

    #paramters for deltad PID
    # delta_k_u = 1
    # delta_t_u = 0
    # delta_t_i = delta_t_u / 2
    # delta_t_d = delta_t_u / 8
    delta_k_p = 1
    delta_k_i = 0.8
    delta_k_d = 0.1

    #creating the current error matrix
    currentstate = vehicle.state

    xd = currentstate.xd
    yd = currentstate.yd
    phid = currentstate.phid
    delta = currentstate.delta
    X = currentstate.X
    Y = currentstate.Y
    phi = currentstate.phi

    #finding the closest index
    _, close_idx = closest_node(X, Y, traj)

    #changing speed according to curvature in future
    if not ((curvature[close_idx-10:close_idx+100] > 0.01).any()):
        xd_des = 2
        la = 60
    elif (curvature[close_idx-100:close_idx+100] > 0.4).any():
        xd_des = 2
        la = 60
    else:
        xd_des = 2
        la = 60
        
    #deciding the desired index
    if close_idx < (traj.shape[0]-1) - la: 
        des_idx = close_idx + la
    else :
        des_idx = close_idx + 50 

	#defining the LQR constants
    Q_1 = np.identity(4)
    R_1 = 1
    
    #defining the continous linear system
    A_1 = np.array([[0,1,0,0],
        [0,(-4 * c_alpha)/(m * xd),(4 * c_alpha)/m, (2 * c_alpha * (l_r - l_f))/(m * xd)],
        [0,0,0,1],
        [0, 2 * c_alpha *(l_r - l_f)/(i_z * xd),-2 * c_alpha *(l_r - l_f)/i_z ,-2 * c_alpha * (l_f**2 + l_r**2)/(i_z * xd)]])

    B_1 = np.array([[0],
        [2*c_alpha/m],
        [0],
        [2*l_f*c_alpha/i_z]])

    C_1 = np.zeros((4,1))

    D_1 = 0

    sys = scipy.signal.cont2discrete((A_1,B_1,C_1,D_1),dt)

    #getting the feedback matrix by LQR
    K_1,_,_ = dlqr(sys[0],sys[1],Q_1,R_1)

    #defning the error terms
    error = np.zeros((4,1))
    error[0] = -(X - traj[des_idx,0])*np.sin(tangent[des_idx]) + (Y - traj[des_idx,1])*np.cos(tangent[des_idx])
    error[2] = wrap2pi(phi - tangent[des_idx])
    error[1] = yd + xd*error[2]
    error[3] = phid - xd * curvature[des_idx]

    #calculating the delta_dot
    delta_des = np.asscalar(-K_1 @ error)
    #delta_dot_des = (delta_new_scalar - delta)/dt

    #calculaing error terms for deltad pid control
    if itr == 0:
        global error_delta_po
        global error_delta_i
        error_delta_po = 0
        error_delta_i = 0
    error_delta_p = delta_des - delta
    error_delta_i += error_delta_p * dt
    error_delta_d = (error_delta_p - error_delta_po) / dt
    error_delta_po = error_delta_p
    deltad_des = delta_k_p * error_delta_p + delta_k_d * error_delta_d + delta_k_i * error_delta_i

    #calculating F by bang-bang control
    # if xd < xd_des:
    #     F = 5000
    # else:
    #     F = -5000

    #calculaing error terms for F pid control
    if itr == 0:
        global errorxd_po
        global errorxd_i
        errorxd_po = 0
        errorxd_i = 0
    errorxd_p = xd_des - xd
    errorxd_i += errorxd_p * dt
    errorxd_d = (errorxd_p - errorxd_po) / dt
    errorxd_po = errorxd_p
    F_des = xd_k_p * errorxd_p + xd_k_d * errorxd_d + xd_k_i * errorxd_i

    print('itr',itr,'close_idx',close_idx,'xd_des',xd_des,'xd',xd,'F',F_des)

    command = vehicle.command(F_=F_des,deltad_=deltad_des)

    return command