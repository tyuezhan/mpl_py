#!/usr/bin/env python

from mpl.planner import Planner
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud
from planning_ros_msgs.msg import PrimitiveArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mpl.waypoint import Waypoint
from mpl.map_util import MapUtil
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray
from utils.primitive_ros_utils import *
import torch

import numpy as np
import rospy
# Create some random points within a 2D space range [x_min, y_min] to [x_max, y_max]
x_min, x_max = -10, 10
y_min, y_max = -10, 10

class PlannerTest:
    def __init__(self):
        rospy.init_node("mpl_planner_test")

        self.cloud_pub_ = rospy.Publisher("mpl/cloud", PointCloud, queue_size=1)
        self.prs_pub_ = rospy.Publisher("mpl/primitives", PrimitiveArray, queue_size=1)
        self.map_pub_ = rospy.Publisher("mpl/map", MarkerArray, queue_size=1)
        self.plan_sub_ = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.plan_cb, queue_size=1)
        self.odom_sub_ = rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=1)
        self.fake_odom_pub_ = rospy.Publisher("/fake_odom", Odometry, queue_size=1)

        self.odom_init_ = False
        self.odom_pos_ = np.zeros(3)
        self.odom_vel_ = np.zeros(3)
        self.odom_yaw_ = 0
        self.start_ = Waypoint()
        self.goal_ = Waypoint()
        
        #TODO: make param
        self.x_min = -1
        self.x_max = 10
        self.y_min = -1
        self.y_max = 10
        self.v_max = 0.5
        self.a_max = 0.5
        self.yaw_max = 0.628
        self.dt = 1.0
        self.goal_tolerance_ = 0.5
        self.yaw_tolerance_ = 3.14
        self.robot_radius_ = 0.3
        self.num = 1

        # Compute U
        # du = 0.8 * self.v_max / (2*self.num)
        du_yaw = self.yaw_max / self.num
        self.U = []
        for dv in np.linspace(0.2 * self.v_max, self.v_max, 3*self.num):
            for dw in np.linspace(-self.yaw_max, self.yaw_max, 3*self.num):
                self.U.append(np.array([dv, dw]))
        print("Control:", self.U)

        # set planner
        self.planner = Planner()
        self.map_util = MapUtil(x_min, x_max, y_min, y_max, self.robot_radius_)
        self.planner.setMapUtil(self.map_util)
        self.planner.setU(self.U)
        self.planner.setDt(self.dt)
        self.planner.setEpsilon(self.dt)
        self.planner.setVmax(self.v_max)
        self.planner.setAmax(self.a_max)
        self.planner.setYawmax(self.yaw_max)
        self.planner.setTol(self.goal_tolerance_, -1, -1)
        self.planner.setTolYaw(self.yaw_tolerance_)
        self.planner.setPlanTmax(1.0)

    def set_map(self):
        # Now we need to come up with a few points in the space with size
        points = np.zeros((5, 3))
        points[0] = np.array([2, 2, 1])
        points[1] = np.array([0, 3, 0.5])
        points[2] = np.array([4, 0, 1])
        points[3] = np.array([8, 1, 1.5])
        points[4] = np.array([1, 9, 1.0])
        gaussians = {}
        gaussians['means3D'] = torch.from_numpy(points)
        gaussians['radius'] = torch.tensor(np.array([0.1, 0.1, 0.2, 0.1, 0.2]))
        self.map = gaussians
        
        self.map_util.set_gaussians(gaussians)

    def pubish_map(self):
        marker_array = MarkerArray()
        marker_array.markers = []
        map_pts = self.map['means3D'].numpy()
        pts_radius = self.map['radius'].numpy()
        for i, point in enumerate(map_pts):
            print(point)
            marker = self.create_marker(point, 3*pts_radius[i], i)
            marker_array.markers.append(marker)
        self.map_pub_.publish(marker_array)

    def create_marker(self, point, size, marker_id):
        marker = Marker()
        marker.header.frame_id = "world"  # Set the frame ID to your fixed frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "points"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        return marker


    def odom_cb(self, msg):
        self.odom_pos_ = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.odom_vel_ = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.odom_yaw_ = R.from_quat(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        ).as_euler("xyz")[2]
        self.odom_init_ = True
        

    def plan_cb(self, msg):
        # self.start_.pos = np.array([self.odom_pos_[0], self.odom_pos_[1], self.odom_pos_[2]])
        # self.start_.yaw = self.odom_yaw_
        self.start_.pos = np.array([0, 0, 0])
        self.start_.yaw = 0

        self.goal_.pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.goal_.yaw = 0

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "world"
        odom_msg.pose.pose.position.x = self.start_.pos[0]
        odom_msg.pose.pose.position.y = self.start_.pos[1]
        odom_msg.pose.pose.position.z = self.start_.pos[2]
        odom_msg.pose.pose.orientation.x = 0
        odom_msg.pose.pose.orientation.y = 0
        odom_msg.pose.pose.orientation.z = 0
        odom_msg.pose.pose.orientation.w = 1
        self.fake_odom_pub_.publish(odom_msg)
        self.set_map()
        self.pubish_map()
        self.plan_traj(self.start_, self.goal_)
        

    def plan_traj(self, start, goal):
        rospy.loginfo("Called plan traj!")

        t0 = rospy.Time.now()
        valid = self.planner.plan(start, goal)
        if not valid:
            if self.planner.initialized():
                rospy.logwarn("Failed! Takes {} sec for planning, expand {} nodes".format((rospy.Time.now() - t0).to_sec(),
                                len(self.planner.getCloseSet())))
            else:
                rospy.logwarn("Failed! Takes {} sec for planning".format((rospy.Time.now() - t0).to_sec()))

        else:
            rospy.loginfo("Succeed! Takes {} sec for planning, expand {} nodes".format(
                            (rospy.Time.now() - t0).to_sec(),
                            len(self.planner.getCloseSet())))
            if len(self.planner.getCloseSet()) == 0:
                rospy.logwarn("Reach goal. No traj!")
                return 0

            traj = self.planner.getTraj()
            plan_stime_ = rospy.Time.now()

            # Publish trajectory
            header = Header()
            header.frame_id = "world"
            header.stamp = t0
            print(type(traj))
            # Publish trajectory as primitives
            prs_msg = to_primitive_array_ros_msg(traj.get_primitives())
            prs_msg.header = header
            self.prs_pub_.publish(prs_msg)

            print(
                "Refined traj -- J(VEL): %f, J(ACC): %f, J(JRK): %f, J(SNP): %f, "
                "total time: %f\n" %
                (traj.J('VEL'), traj.J('ACC'), traj.J('JRK'), traj.J('SNP'),
                traj.get_total_time())
            )

        # No matter succeed or not. Publish expanded nodes
        header = Header()
        header.frame_id = "world"
        header.stamp = t0
        ps = vec_to_cloud(self.planner.getExpandedNodes())
        ps.header = header
        self.cloud_pub_.publish(ps)

        if not valid:
            return -1
        else:
            return 0
        
if __name__ == "__main__":
    planner_test = PlannerTest()
    rospy.spin()