import numpy as np
import matplotlib.pyplot as plt
from util import *
from scipy.ndimage.filters import gaussian_filter1d
import scipy.signal as signal
import scipy


def dlqr(A,B,Q,R):
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    eigVals, eigVecs = scipy.linalg.eig(A - B*K)
 
    return K, X, eigVals

my_data = np.genfromtxt('buggyTrace.csv', delimiter=',')


traj = get_trajectory('buggyTrace.csv')

m = 2000
l_r = 1.7
l_f = 1.1
c_alpha = 15000
i_z = 3344
mu = 0.01
dt = 0.05
xd = 0.3

A_1 = np.array([[0,1,0,0],
        [0,(-4 * c_alpha)/(m * xd),(4 * c_alpha)/m, (2 * c_alpha * (l_r - l_f))/(m * xd)],
        [0,0,0,1],
        [0, 2 * c_alpha *(l_r - l_f)/(i_z * xd),-2 * c_alpha *(l_r - l_f)/i_z ,-2 * c_alpha * (l_f**2 + l_r**2)/(i_z*xd)]])

B_1 = np.array([[0],
        [2*c_alpha/m],
        [0],
        [2*l_f*c_alpha/i_z]])

C_1 = np.zeros((4,1))

D_1 = 0

Q_1 = np.zeros((4,4))
np.fill_diagonal(Q_1,1)

R_1 = 1

sys = signal.cont2discrete((A_1,B_1,C_1,D_1),dt)

K_1,_,_ = dlqr(sys[0],sys[1],Q_1,R_1)

dx = np.gradient(traj[:,0])
dy = np.gradient(traj[:,1])
dxd = np.gradient(dx)
dyd = np.gradient(dy)

tangent = dy/dx

xp = gaussian_filter1d(traj[:,0], sigma = 10, order =1)
yp = gaussian_filter1d(traj[:,1], sigma = 10, order =1)
xpp = gaussian_filter1d(traj[:,0], sigma = 10, order =2)
ypp = gaussian_filter1d(traj[:,1], sigma = 10, order =2)
curvature = np.abs(xp*ypp - yp*xpp)/(xp**2 + yp**2)**1.5

if (curvature[:2000] > 0.4).any():
    print('ah shit')

B = not (curvature[:2500] > 0.4).any()
print(B)