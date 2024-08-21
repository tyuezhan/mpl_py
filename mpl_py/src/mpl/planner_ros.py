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
from mpl_utils.primitive_ros_utils import *
import torch

from sensor_msgs.msg import PointField, PointCloud2
from sensor_msgs import point_cloud2
from jackal_tracker_msgs.msg import JackalMPTrackerGoal
from jackal_tracker_msgs.msg import JackalMPTrackerAction
import tf2_ros
import numpy as np
import rospy
from actionlib import SimpleActionClient
from planning_ros_msgs.msg import Trajectory

class LocalPlanner:

    def __init__(self):
        # rospy.init_node("mpl_planner_test")
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
        self.robot_radius_ = 0.1
        self.num = 1
        self.map_set_ = False
        self.odom_init_ = False
        self.debug = True
        self.pc_fields_ = self.make_fields()

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
        self.map_util = MapUtil(self.x_min, self.x_max, self.y_min, self.y_max, self.robot_radius_)
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

        self.cloud_pub_ = rospy.Publisher("mpl/cloud", PointCloud, queue_size=1)
        self.prs_pub_ = rospy.Publisher("mpl/primitives", PrimitiveArray, queue_size=1)
        self.map_pub_ = rospy.Publisher("mpl/map", PointCloud2, queue_size=1)
        self.plan_sub_ = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.plan_cb, queue_size=1)
        self.odom_sub_ = rospy.Subscriber("/ground_truth/husky/odom", Odometry, self.odom_cb, queue_size=1)
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Tracker client
        self.tracker_client = SimpleActionClient('/jackal_tracker/mp_tracker_server', JackalMPTrackerAction)
        self.tracker_client.wait_for_server(rospy.Duration(2.0))


    def set_map(self, means, radius_log):
        # It should contain at least means3D and radius
        # lookup the TF between map frame and the world frame
        try:
            # Lookup the static transform
            source_frame = 'world'
            target_frame = 'map'
            transform = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            # Print out the transform details
            rospy.loginfo(f"Transform from {source_frame} to {target_frame}:")
            rospy.loginfo(f"Translation: {transform.transform.translation.x}, {transform.transform.translation.y}, {transform.transform.translation.z}")
            rospy.loginfo(f"Rotation: {transform.transform.rotation.x}, {transform.transform.rotation.y}, {transform.transform.rotation.z}, {transform.transform.rotation.w}")
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform lookup failed: {e}")
        except tf2_ros.ConnectivityException as e:
            rospy.logerr(f"Transform connectivity issue: {e}")
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"Transform extrapolation issue: {e}")
        
        self.map = {}
        # Apply tf to means
        # check means device
        print(means.device)
        # use torch to apply the transform
        # create H from the transform
        map2world = np.eye(4)
        map2world[:3, :3] = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]).as_matrix()
        map2world[:3, 3] = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ]
        map2world = torch.tensor(map2world, dtype=torch.float32).to(means.device)
        # Make means homogeneous
        means = torch.cat([means, torch.ones(means.shape[0], 1, device=means.device)], dim=1)
        means_w = torch.matmul(map2world, means.t()).t()[:, :3]

        self.map["means3D"] = means_w
        self.map["radius"] = torch.exp(radius_log)
        self.map_set_ = True
        self.map_util.set_gaussians(self.map)

        if self.debug:
            self.publish_map()


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
        self.start_.pos = self.odom_pos_
        self.start_.yaw = self.odom_yaw_

        self.goal_.pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.goal_.yaw = 0

        self.plan_traj(self.start_, self.goal_)


    def plan_traj(self, start, goal):
        if not self.odom_init_:
            rospy.logwarn("No odometry!")
            return
        if not self.map_set_:
            rospy.logwarn("No map!")
            return
        rospy.loginfo("Called plan traj!")

        t0 = rospy.Time.now()
        valid = self.planner.plan(start, goal)
        if not valid:
            if self.planner.initialized():
                rospy.logerr("Failed! Takes {} sec for planning, expand {} nodes".format((rospy.Time.now() - t0).to_sec(),
                                len(self.planner.getCloseSet())))
            else:
                rospy.logerr("Failed! Takes {} sec for planning".format((rospy.Time.now() - t0).to_sec()))
            # cancel goal
            self.tracker_client.cancel_goal()
            rospy.loginfo("Cancel goal!")
            # Publish empty trajectory
            prs_msg = PrimitiveArray()
            prs_msg.header.stamp = t0
            prs_msg.header.frame_id = "world"
            self.prs_pub_.publish(prs_msg)

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
            header.stamp = rospy.Time.now()
            print(type(traj))
            # Publish trajectory as primitives
            prs_msg = to_primitive_array_ros_msg(traj.get_primitives())
            prs_msg.header = header
            self.prs_pub_.publish(prs_msg)

            # Publish goal
            traj_msg = to_trajectory_ros_msg(traj)
            goal = JackalMPTrackerGoal()
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "world"
            goal.trajectory = traj_msg
            self.tracker_client.send_goal(goal)
            rospy.loginfo("Send new goal to tracker!")
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


    def make_fields(self):
        fields = []
        field = PointField()
        field.name = 'x'
        field.count = 1
        field.offset = 0
        field.datatype = PointField.FLOAT32
        fields.append(field)

        field = PointField()
        field.name = 'y'
        field.count = 1
        field.offset = 4
        field.datatype = PointField.FLOAT32
        fields.append(field)

        field = PointField()
        field.name = 'z'
        field.count = 1
        field.offset = 8
        field.datatype = PointField.FLOAT32
        fields.append(field)
        return fields

    def publish_map(self):
        if not self.map_set_:
            rospy.logwarn("No map!")
            return
        
        s_time = rospy.Time.now()
        # Publish the GS points as PointCloud2 
        original_gs_points = self.map['means3D'].detach().cpu().numpy()
        gs_points = original_gs_points[0:original_gs_points.shape[0]:1]
        point_data = gs_points.tolist()

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        cloud = point_cloud2.create_cloud(header, self.pc_fields_, point_data)
        self.map_pub_.publish(cloud)
        e_time = rospy.Time.now()
        rospy.loginfo(f"[MPL planner] Published Map! Time: {(e_time-s_time).to_sec()} s")


    def test_plan(self):
        self.start_.pos = self.odom_pos_
        self.start_.yaw = self.odom_yaw_

        self.goal_.pos = np.array([2, 2, 0])
        self.goal_.yaw = 0

        self.plan_traj(self.start_, self.goal_)