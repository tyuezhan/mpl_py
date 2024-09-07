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
from mpl.primitive_dict import PrimitiveDict

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

    def __init__(self, eval_traj_func):
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
        self.v_max = rospy.get_param("~v_max", 0.5)
        self.a_max = rospy.get_param("~a_max", 0.5)
        self.yaw_max = rospy.get_param("~yaw_max", np.pi/10)
        self.dt = rospy.get_param("~dt", 1.0)
        self.goal_tolerance_ = rospy.get_param("~goal_tolerance", 0.8)
        self.yaw_tolerance_ = rospy.get_param("~yaw_tolerance", 1.0)
        self.robot_size = rospy.get_param("~robot_size", 0.6)
        self.robot_radius_ = self.robot_size / 2.0
        self.num = rospy.get_param("~num_discretization", 5)
        self.plan_max_time = rospy.get_param("~plan_t_max", 0.3)
        self.collision_tol = rospy.get_param("~collision_tol", 3)
        self.pub_collision_map_ = rospy.get_param("~pub_collision_map", False)
        self.along_path_ = rospy.get_param("~crop_goal_along_path", False)
        self.sim_ = rospy.get_param("~gs_sim", False)
        if self.sim_:  # for sim
            self.odom_frame_id = rospy.get_param("~odom_frame_id", "odom")
            self.world_frame_id = rospy.get_param("~world_frame_id", "world")
        else:
            self.odom_frame_id = rospy.get_param("~robot_odom_frame_id", "odom")
            self.world_frame_id = rospy.get_param("~robot_world_frame_id", "world")
        self.map_set_ = False
        self.odom_init_ = False
        self.debug = False
        self.map2world = None
        self.world2map = None
        self.pc_fields_ = self.make_fields()
        self.debug_pc_fields_ = self.make_debug_fields()
        self.prev_traj_ = None
        self.last_plan_success_ = False
        self.reversing_ = False
        self.horizon = rospy.get_param("~planning_horizon", 3)
        self.curr_path = None
        self.cropped_path = None

        # Compute U
        # du = 0.8 * self.v_max / (2*self.num)
        du_yaw = self.yaw_max / self.num
        self.U = []
        for dv in np.linspace(0.2 * self.v_max, self.v_max, 3*self.num):
            for dw in np.linspace(-self.yaw_max, self.yaw_max, 5*self.num):
                self.U.append((dv, dw))
        print("Control:", self.U)

        # set planner
        self.planner = Planner(eval_traj_func)
        self.primitive_dict = PrimitiveDict(5, self.U)
        self.primitive_dict.precompute()
        self.map_util = MapUtil(self.x_min, self.x_max, self.y_min, self.y_max, self.robot_radius_, self.collision_tol)
        self.planner.setMapUtil(self.map_util, self.primitive_dict)
        self.planner.setU(self.U)
        self.planner.setDt(self.dt)
        self.planner.setEpsilon(self.dt)
        self.planner.setVmax(self.v_max)
        self.planner.setAmax(self.a_max)
        self.planner.setYawmax(self.yaw_max)
        self.planner.setTol(self.goal_tolerance_, -1, -1)
        self.planner.setTolYaw(self.yaw_tolerance_)
        self.planner.setPlanTmax(self.plan_max_time)

        self.odom_topic = rospy.get_param("~odom_topic", "/ground_truth/husky/odom")
        self.cloud_pub_ = rospy.Publisher("mpl/cloud", PointCloud, queue_size=1)
        self.prs_pub_ = rospy.Publisher("mpl/primitives", PrimitiveArray, queue_size=1)
        self.map_pub_ = rospy.Publisher("mpl/map", PointCloud2, queue_size=1)
        self.collision_pub_ = rospy.Publisher("mpl/collision", PointCloud2, queue_size=1)
        self.plan_sub_ = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.plan_cb, queue_size=1)
        # self.odom_sub_ = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.goal_pub_ = rospy.Publisher("mpl/goal", PoseStamped, queue_size=1)
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Tracker client
        self.tracker_client = SimpleActionClient('/jackal_tracker/mp_tracker_server', JackalMPTrackerAction)
        self.tracker_client.wait_for_server(rospy.Duration(2.0))

        # Print params from config
        rospy.loginfo(f"v_max: {self.v_max}")
        rospy.loginfo(f"a_max: {self.a_max}")
        rospy.loginfo(f"yaw_max: {self.yaw_max}")
        rospy.loginfo(f"dt: {self.dt}")
        rospy.loginfo(f"goal_tolerance: {self.goal_tolerance_}")
        rospy.loginfo(f"yaw_tolerance: {self.yaw_tolerance_}")
        rospy.loginfo(f"collision_tol: {self.collision_tol}")
        rospy.loginfo(f"robot_radius: {self.robot_radius_}")
        rospy.loginfo(f"num_discretization: {self.num}")
        rospy.loginfo(f"plan_t_max: {self.plan_max_time}")
        rospy.loginfo(f"odom_topic: {self.odom_topic}")
        rospy.loginfo(f"planning_horizon: {self.horizon}")
        rospy.loginfo(f"pub_collision_map: {self.pub_collision_map_}")
        rospy.loginfo(f"crop_goal_along_path: {self.along_path_}")

        rospy.loginfo("Local planner initialized!")


    def set_map(self, means, radius_log, ground_labels):
        if self.map2world is None:
            # It should contain at least means3D and radius
            # lookup the TF between map frame and the world frame
            try:
                # Lookup the static transform
                source_frame = 'world'
                target_frame = 'gs_map'
                transform = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
                # Print out the transform details
                # rospy.loginfo(f"Transform from {source_frame} to {target_frame}:")
                # rospy.loginfo(f"Translation: {transform.transform.translation.x}, {transform.transform.translation.y}, {transform.transform.translation.z}")
                # rospy.loginfo(f"Rotation: {transform.transform.rotation.x}, {transform.transform.rotation.y}, {transform.transform.rotation.z}, {transform.transform.rotation.w}")
            except tf2_ros.LookupException as e:
                rospy.logerr(f"Transform lookup failed: {e}")
                return
            except tf2_ros.ConnectivityException as e:
                rospy.logerr(f"Transform connectivity issue: {e}")
                return
            except tf2_ros.ExtrapolationException as e:
                rospy.logerr(f"Transform extrapolation issue: {e}")
                return
            
            self.map = {}
            # Apply tf to means
            # check means device
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
            self.map2world = torch.tensor(map2world, dtype=torch.float32).to(means.device)
        # Make means homogeneous
        means = torch.cat([means, torch.ones(means.shape[0], 1, device=means.device)], dim=1)
        means_w = torch.matmul(self.map2world, means.t()).t()[:, :3]

        z_mask = (ground_labels == 0) # keep points that are not ground
        print('Num gaussians before mask: ', means_w.shape[0])
        means_w = means_w[z_mask]
        radius_log = radius_log[z_mask]
        print('Num gaussians after mask: ', means_w[0])

        self.map["means3D"] = means_w
        self.map["radius"] = torch.exp(radius_log)
        # self.map["ground_labels"] = ground_labels
        self.map_set_ = True
        self.map_util.set_gaussians(self.map)
        # means_clone = torch.clone(means).to(means.device)
        # means_clone = torch.cat([means_clone, torch.ones(means_clone.shape[0], 1, device=means_clone.device)], dim=1)
        # means_w = torch.matmul(self.map2world, means_clone.t()).t()[:, :3]

        # radius_log_clone = torch.clone(radius_log).to(means.device)
        # self.map["means3D"] = means_w
        # self.map["radius"] = torch.exp(radius_log_clone)
        # self.map_set_ = True
        # self.map_util.set_gaussians(self.map)
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

        # self.plan_traj(self.start_, self.goal_, params, intrinsics)


    def plan_traj(self, start, goal, params, intrinsics, img_w, img_h):
        # Lookup transform
        try:
            # Lookup the static transform
            source_frame = self.world_frame_id
            target_frame = self.odom_frame_id
            transform = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            # Print out the transform details
            # rospy.loginfo(f"Transform from {source_frame} to {target_frame}:")
            # rospy.loginfo(f"Translation: {transform.transform.translation.x}, {transform.transform.translation.y}, {transform.transform.translation.z}")
            # rospy.loginfo(f"Rotation: {transform.transform.rotation.x}, {transform.transform.rotation.y}, {transform.transform.rotation.z}, {transform.transform.rotation.w}")
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform lookup failed: {e}")
            return -1
        except tf2_ros.ConnectivityException as e:
            rospy.logerr(f"Transform connectivity issue: {e}")
            return -1
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"Transform extrapolation issue: {e}")
            return -1
        self.odom_pos_ = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        self.odom_yaw_ = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]).as_euler("xyz")[2]
        rospy.loginfo(f"Odom pos: {self.odom_pos_}, Odom yaw: {self.odom_yaw_}")

        # if not self.odom_init_:
        #     rospy.logwarn("No odometry!")
        #     return
        if not self.map_set_:
            rospy.logwarn("No map!")
            return -1
        if not self.find_world2map():
            rospy.logwarn("No world2map!")
            return -1
        
        rospy.loginfo("Called plan traj!")

        t0 = rospy.Time.now()
        status = self.planner.plan(start, goal, params, intrinsics, img_w, img_h)
        
        # For debug collision points.
        if self.pub_collision_map_:
            self.publish_collision_pts(start.pos)

        if status <= 0:
            if self.planner.initialized():
                rospy.logerr("[planner ros] Failed! Takes {} sec for planning, expand {} nodes".format((rospy.Time.now() - t0).to_sec(),
                                len(self.planner.getCloseSet())))
            else:
                rospy.logerr("[planner ros] Failed! Takes {} sec for planning".format((rospy.Time.now() - t0).to_sec()))
            # cancel goal
            # check if the tracker is active
            if self.tracker_client.get_state() == 1 and self.reversing_ == False:
                self.tracker_client.cancel_goal()
                rospy.loginfo("[planner ros] Cancel goal!")
            # Publish empty trajectory
            prs_msg = PrimitiveArray()
            prs_msg.header.stamp = t0
            prs_msg.header.frame_id = "world"
            self.prs_pub_.publish(prs_msg)
            self.last_plan_success_ = False
            if self.prev_traj_ is not None:
                # reverse
                goal = JackalMPTrackerGoal()
                goal.header.stamp = rospy.Time.now()
                goal.header.frame_id = "world"
                traj_msg = to_trajectory_ros_msg(self.prev_traj_)
                goal.trajectory = traj_msg
                goal.reverse = True
                self.tracker_client.send_goal(goal)
                rospy.loginfo("[planner ros] Reverse previous traj. Send goal to tracker!")
                self.prev_traj_ = None
                self.reversing_ = True
                
        elif status == 1:
            rospy.loginfo("[planner ros] Succeed! Takes {} sec for planning, expand {} nodes".format(
                            (rospy.Time.now() - t0).to_sec(),
                            len(self.planner.getCloseSet())))
            if len(self.planner.getCloseSet()) == 0:
                rospy.logwarn("Reach goal. No traj!")
                return 0

            traj = self.planner.getTraj()
            self.prev_traj_ = traj
            self.last_plan_success_ = True
            self.reversing_ = False
            plan_stime_ = rospy.Time.now()

            # Publish trajectory
            header = Header()
            header.frame_id = "world"
            header.stamp = rospy.Time.now()
            # Publish trajectory as primitives
            prs_msg = to_primitive_array_ros_msg(traj.get_primitives())
            prs_msg.header = header
            self.prs_pub_.publish(prs_msg)

            #DEBUG:
            # for pr in traj.get_primitives():
            #     print("control:", pr.u())

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
        else:
            rospy.logerr("[planner ros] Unknown status!")
            return -1


        # No matter succeed or not. Publish expanded nodes
        header = Header()
        header.frame_id = "world"
        header.stamp = t0
        ps = vec_to_cloud(self.planner.getExpandedNodes())
        ps.header = header
        self.cloud_pub_.publish(ps)

        if status <= 0:
            return -1
        else:
            return 0


    def find_world2map(self):
        if self.world2map is not None:
            return True
        # It should contain at least means3D and radius
        # lookup the TF between map frame and the world frame
        try:
            # Lookup the static transform
            source_frame = 'gs_map'
            target_frame = 'world'
            transform = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            # Print out the transform details
            rospy.loginfo(f"Transform from {source_frame} to {target_frame}:")
            rospy.loginfo(f"Translation: {transform.transform.translation.x}, {transform.transform.translation.y}, {transform.transform.translation.z}")
            rospy.loginfo(f"Rotation: {transform.transform.rotation.x}, {transform.transform.rotation.y}, {transform.transform.rotation.z}, {transform.transform.rotation.w}")
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform lookup failed: {e}")
            return False
        except tf2_ros.ConnectivityException as e:
            rospy.logerr(f"Transform connectivity issue: {e}")
            return False
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"Transform extrapolation issue: {e}")
            return False
        # create H from the transform
        self.world2map = np.eye(4)
        self.world2map[:3, :3] = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]).as_matrix()
        self.world2map[:3, 3] = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ]
        self.planner.setWorld2Map(self.world2map)
        return True
        

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

    def make_debug_fields(self):
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

        field = PointField()
        field.name = 'rgb'
        field.count = 1
        field.offset = 12
        field.datatype = PointField.UINT32
        fields.append(field)

        field = PointField()
        field.name = 'size'
        field.count = 1
        field.offset = 16
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


    def plan_to_ftr(self, params, intrinsics, path_to_ftr, img_w, img_h):
        # set start pos as 1.0 second on the previous traj.
        # if self.last_plan_success_:
        #     # Take previous traj
        #     wp = self.prev_traj_.evaluate(0.3)
        #     # if goal is not far from current odom, use it
        #     if np.linalg.norm(wp.pos[:2] - self.odom_pos_[:2]) < 0.6:
        #         self.start_.pos = wp.pos
        #         self.start_.yaw = wp.yaw
        #     else:
        #         self.start_.pos = self.odom_pos_
        #         self.start_.yaw = self.odom_yaw_
        # else:
        #     self.start_.pos = self.odom_pos_
        #     self.start_.yaw = self.odom_yaw_
        self.start_.pos = self.odom_pos_
        self.start_.yaw = self.odom_yaw_

        # call get_local_goal function to get local goal
        goal_pos, goal_yaw = self.get_local_goal(self.start_.pos, path_to_ftr, horizon=self.horizon, along_path=self.along_path_)
        self.goal_.pos[:2] = goal_pos
        self.goal_.pos[2] = self.odom_pos_[2]
        self.goal_.yaw = goal_yaw
        self.plan_traj(self.start_, self.goal_, params, intrinsics, img_w, img_h)

        # Compose goal msg
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "world"
        goal_msg.pose.position.x = self.goal_.pos[0]
        goal_msg.pose.position.y = self.goal_.pos[1]
        goal_msg.pose.position.z = self.goal_.pos[2]    
        quat = R.from_euler('z', goal_yaw).as_quat()
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]
        self.goal_pub_.publish(goal_msg)


    def get_local_goal(self, s_pos, path, horizon=3, along_path=False):
        # Path is a Nx2 array
        # horizon is the distance to look ahead
        # iterate through the path and find waypoints that are outside the horizon
        # return the first waypoint that is outside the horizon
        
        # first identify which node we are closest to
        if self.curr_path is not None:
            # check if path changed
            if np.array_equal(path, self.curr_path):
                # path is the same, just used the cropped path
                if self.cropped_path is not None:
                    closest_idx = np.argmin(np.linalg.norm(self.cropped_path - s_pos[:2], axis=1))
                    self.cropped_path = self.cropped_path[closest_idx:]
                else:
                    # path is not cropped yet, crop it and save as cropped path
                    closest_idx = np.argmin(np.linalg.norm(path - s_pos[:2], axis=1))
                    self.cropped_path = path[closest_idx:]
            else:
                # path changed, update the current path
                self.curr_path = path
                closest_idx = np.argmin(np.linalg.norm(path - s_pos[:2], axis=1))
                self.cropped_path = path[closest_idx:]
        else:
            self.curr_path = path
            closest_idx = np.argmin(np.linalg.norm(path - s_pos[:2], axis=1))
            self.cropped_path = path[closest_idx:]
            
        if self.cropped_path.shape[0] == 0:
            return s_pos[:2], 0
        elif self.cropped_path.shape[0] == 1:
            return self.cropped_path[0], np.arctan2(self.cropped_path[0][1] - s_pos[1], self.cropped_path[0][0] - s_pos[0])
        else:
            if along_path:
                dist = 0
                for i in range(self.cropped_path.shape[0]):
                    if i == 0:
                        dist = np.linalg.norm(s_pos[:2] - self.cropped_path[i])
                    else:
                        dist += np.linalg.norm(self.cropped_path[i-1] - self.cropped_path[i])
                    if dist > horizon:
                        return self.cropped_path[i], np.arctan2(self.cropped_path[i][1] - s_pos[1], self.cropped_path[i][0] - s_pos[0])
                return self.cropped_path[-1], np.arctan2(self.cropped_path[-1][1] - s_pos[1], self.cropped_path[-1][0] - s_pos[0])
            else:
                for i in range(self.cropped_path.shape[0]):
                    dist = np.linalg.norm(s_pos[:2] - self.cropped_path[i])
                    if dist > horizon:
                        return self.cropped_path[i], np.arctan2(self.cropped_path[i][1] - s_pos[1], self.cropped_path[i][0] - s_pos[0])
                return self.cropped_path[-1], np.arctan2(self.cropped_path[-1][1] - s_pos[1], self.cropped_path[-1][0] - s_pos[0])


    def publish_collision_pts(self, pos):
        s_time = rospy.Time.now()
        # Publish the GS points as PointCloud2
        pos = torch.tensor(pos).reshape(1, 3).to(self.map['means3D'].device)
        # original_gs_points = original_gs_points[gs_ground_labels == 0]
        # print("shape of latest_means3D: ", self.latest_means3D.shape)

        # Check for collision 
        ret, gaussians = self.map_util.collision_testing_debug(pos)
        gaussians = gaussians.squeeze(0)
        collision_pts = gaussians[ret[:, :, 1] == 1].detach().cpu().numpy()
        # convert to ros point cloud
        # color points as red
        gs_colors = np.zeros((collision_pts.shape[0], 3))
        gs_colors[:, 2] = 1
        gs_sizes = np.ones(collision_pts.shape[0]) * 0.1
        # create the point cloud    
        # create the point cloud
        rgb_data = np.array([
                (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
                for b, g, r in gs_colors], dtype=np.uint32)                

        point_data = [[None] *5] * len(collision_pts)
        for i in range(len(collision_pts)):
            point_data[i] = [collision_pts[i][0], collision_pts[i][1], collision_pts[i][2], int(rgb_data[i]), gs_sizes[i]]
        # point_data = np.concatenate((gs_points, rgb_data[:, np.newaxis], gs_sizes), axis=1).tolist()
        # change rgb_data in point_data to int
        # for i in range(len(point_data)):
        #     point_data[i][3] = int(point_data[i][3])
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        # print(point_data[0])
        cloud = point_cloud2.create_cloud(header, self.debug_pc_fields_, point_data)
        self.collision_pub_.publish(cloud)
        e_time = rospy.Time.now()
        rospy.loginfo(f"Published Collision Map! Time: {(e_time-s_time).to_sec()} s")