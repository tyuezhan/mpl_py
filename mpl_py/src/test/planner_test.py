from mpl.planner import Planner
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud
from planning_ros_msgs.msg import PrimitiveArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mpl.waypoint import Waypoint
from scipy.spatial.transform import Rotation as R

import numpy as np
import rospy
# Create some random points within a 2D space range [x_min, y_min] to [x_max, y_max]
x_min, x_max = -10, 10
y_min, y_max = -10, 10

class PlannerTest:
    def __init__(self):
        self.planner = Planner()
        self.cloud_pub_ = rospy.Publisher("mpl/cloud", PointCloud, queue_size=1)
        self.prs_pub_ = rospy.Publisher("mpl/primitives", PrimitiveArray, queue_size=1)
        self.plan_sub_ = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.plan_cb, queue_size=1)
        self.odom_sub_ = rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=1)
        self.odom_init_ = False
        self.odom_pos_ = np.zeros(3)
        self.odom_vel_ = np.zeros(3)
        self.odom_yaw_ = 0
        self.start_ = Waypoint()
        self.goal_ = Waypoint()

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
        self.start_.pos = np.array([self.odom_pos_[0], self.odom_pos_[1], self.odom_pos_[2]])
        self.start_.yaw = self.odom_yaw_
        self.goal_.pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.goal_.yaw = R.from_quat(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        ).as_euler("xyz")[2]
        self.plan_traj(self.start_, self.goal_)
        

    def plan_traj(self, start, goal):
        rospy.loginfo("Called plan traj!")

        t0 = rospy.Time.now()
        valid = self.planner.plan(start, goal)
        if not valid:
            if self.planner.initialized():
                rospy.logwarn("Failed! Takes %f sec for planning, expand [%zu] nodes",
                                (rospy.Time.now() - t0).to_sec(),
                                len(self.planner.getCloseSet()))
            else:
                rospy.logwarn("Failed! Takes %f sec for planning", (rospy.Time.now() - t0).to_sec())

        else:
            rospy.loginfo("Succeed! Takes %f sec for planning, expand [%zu] nodes",
                            (rospy.Time.now() - t0).to_sec(),
                            len(self.planner.getCloseSet()))
            if len(self.planner.getCloseSet()) == 0:
                rospy.logwarn("Reach goal. No traj!")
                return 0

            traj = self.planner.getTraj()
            plan_stime_ = rospy.Time.now()

            # Publish trajectory
            header = Header()
            header.frame_id = "world"
            header.stamp = t0

            # Publish trajectory as primitives
            prs_msg = self.toPrimitiveArrayROSMsg(traj.getPrimitives())
            prs_msg.header = header
            self.prs_pub_.publish(prs_msg)

            print(
                "Refined traj -- J(VEL): %f, J(ACC): %f, J(JRK): %f, J(SNP): %f, "
                "J(YAW): %f, total time: %f\n" %
                (traj.J('VEL'), traj.J('ACC'), traj.J('JRK'), traj.J('SNP'),
                    traj.Jyaw(), traj.getTotalTime())
            )

        # No matter succeed or not. Publish expanded nodes
        header = Header()
        header.frame_id = "world"
        header.stamp = t0
        ps = self.vec_to_cloud(self.planner.getExpandedNodes())
        ps.header = header
        self.cloud_pub_.publish(ps)

        if not valid:
            return -1
        else:
            return 0