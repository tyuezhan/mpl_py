#!/usr/bin/env python

import rospy
from planning_ros_msgs.msg import Primitive, PrimitiveArray, Trajectory
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import numpy as np
from mpl.primitive import Primitive as Primitive3D
from mpl.trajectory import Trajectory as Trajectory3D

def to_primitive_ros_msg(pr):
    # U is u_v, u_w
    # P_zero is x, y, z, theta_0
    pr_u = pr.u()
    p_zero = pr.p_zero()
    msg = Primitive()
    msg.cx = [pr_u[0], p_zero[0]] # u_v, x_0
    msg.cy = [pr_u[0], p_zero[1]] # u_v, y_0
    msg.cz = [0, p_zero[2]] # 0, z_0
    msg.cyaw = [pr_u[1], p_zero[3]] #u_w, theta_0

    msg.t = pr.t
    msg.control_car = True
    return msg

def to_primitive_array_ros_msg(prs, z=0):
    print("to_primitive_array_ros_msg")
    print(len(prs))
    print(prs)
    msg = PrimitiveArray()
    for pr in prs:
        msg.primitives.append(to_primitive_ros_msg(pr))
    return msg

def to_trajectory_ros_msg(traj, z=0):
    msg = Trajectory()
    for seg in traj.segs:
        msg.primitives.append(to_primitive_ros_msg(seg))

    # if traj.lambda().exist():
    #     l = traj.lambda()
    #     msg.lambda = [LambdaSegment(dT=seg.dT, ti=seg.ti, tf=seg.tf, ca=seg.a.tolist()) for seg in l.segs]
    return msg

def to_primitive3d(pr):
    if pr.control_car:
        return Primitive3D(pr.cx[1], pr.cy[1], pr.cz[1], pr.cyaw[1], pr.cx[0], pr.cyaw[0], pr.t)

    else:
        print("Unsupported primitive type")
        return None

def to_trajectory3d(traj_msg):
    traj = Trajectory3D()
    traj.taus.append(0)
    for it in traj_msg.primitives:
        traj.segs.append(to_primitive3d(it))
        traj.taus.append(traj.taus[-1] + it.t)

    # if traj_msg.lambda:
    #     l = Lambda()
    #     for i, seg in enumerate(traj_msg.lambda):
    #         l.segs.append(LambdaSeg(np.array(seg.ca), seg.ti, seg.tf, seg.dT))
    #         traj.total_t_ += seg.dT
    #     traj.lambda_ = l
    #     traj.Ts = [traj.lambda_.getT(tau) for tau in traj.taus]
    # else:
    #     traj.total_t_ = traj.taus[-1]

    return traj

def vec_to_cloud(pts):
    cloud = PointCloud()
    cloud.points = [Point32(x=pt[0], y=pt[1], z=pt[2]) for pt in pts]
    return cloud