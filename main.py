from BuggySimulator import *
import numpy as np
from controller import *
from util import *
import math
from scipy.ndimage.filters import gaussian_filter1d
from Evaluation import *


# get the trajectory
traj = get_trajectory('buggyTrace.csv')
# initial the Buggy
vehicle = initail(traj,0)
n = 5000
X = []
Y = []
delta = []
xd = []
yd = []
phi = []
phid = []
deltad = []
F = []
minDist =[]
cur_state = np.zeros(7)
'''
your code starts here
'''
# preprocess the trajectory
passMiddlePoint = False
nearGoal = False

#defining the radius at every trajectory point
dx = np.gradient(traj[:,0])
dy = np.gradient(traj[:,1])
tangent = np.arctan2(dy,dx)

xp = gaussian_filter1d(traj[:,0], sigma = 10, order =1)
yp = gaussian_filter1d(traj[:,1], sigma = 10, order =1)
xpp = gaussian_filter1d(traj[:,0], sigma = 10, order =2)
ypp = gaussian_filter1d(traj[:,1], sigma = 10, order =2)
curvature = np.abs(xp*ypp - yp*xpp)/(xp**2 + yp**2)**1.5

for i in range(n):
    command = controller(traj, vehicle, tangent, curvature, i)
    vehicle.update(command = command)

    # termination check
    disError,nearIdx = closest_node(vehicle.state.X, vehicle.state.Y, traj)
    stepToMiddle = nearIdx - len(traj)/2.0
    if abs(stepToMiddle) < 100.0:
        passMiddlePoint = True
        print('middle point passed')
    nearGoal = nearIdx >= len(traj)-50
    if nearGoal and passMiddlePoint:
        print('destination reached!')
        break
    # record states
    X.append(vehicle.state.X)
    Y.append(vehicle.state.Y)
    delta.append(vehicle.state.delta)
    xd.append(vehicle.state.xd)
    yd.append(vehicle.state.yd)
    phid.append(vehicle.state.phid)
    phi.append(vehicle.state.phi)
    deltad.append(command.deltad)
    F.append(command.F)
    minDist.append(disError)

    cur_state = np.array(save_state(vehicle.state))

    if i == 0:
        state_saved = cur_state.reshape((1, 7))
    else:
        state_saved = np.concatenate((state_saved, cur_state.reshape((1, 7))), axis=0)

np.save('24-677_Project_2_BuggyStates_Fateh.npy', state_saved)
showResult(traj,X,Y,delta,xd,yd,F,phi,phid,minDist)

taskNum = 3
evaluation(minDist, traj, X, Y, taskNum)