#ifndef _MP_TRACKER_SERVER_H_
#define _MP_TRACKER_SERVER_H_

#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/Twist.h>
#include <jackal_tracker_msgs/JackalMPTrackerAction.h>
#include <nav_msgs/Odometry.h>
#include <planning_ros_msgs/Trajectory.h>
#include <planning_ros_utils/primitive_ros_utils.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include "tf2_ros/transform_listener.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <list>


#include <Eigen/Geometry>

namespace mp_tracker {

class PIDController {
 public:
  PIDController(double kp, double ki, double kd, double max_i, double max_output)
      : kp_(kp), ki_(ki), kd_(kd), max_i_(max_i), max_output_(max_output) {
    error_prev_ = 0.0;
    error_integral_ = 0.0;
    ROS_INFO("Init PID Controller with kp: %f, ki: %f, kd: %f, max_i: %f, max_output: %f",
              kp_, ki_, kd_, max_i_, max_output_);
  }

  double compute(double error, double dt, int error_sign) {
    error_integral_ += error * dt;
    if (error_integral_ > max_i_) {
      error_integral_ = max_i_;
    } else if (error_integral_ < -max_i_) {
      error_integral_ = -max_i_;
    }
    // ROS_INFO("error: %f, error_integral: %f, error_prev: %f", error, error_integral_, error_prev_);
    if (dt < 1e-6) {
      ROS_WARN("dt is too small, p only");
      return kp_ * error;
    }
    double diff_term = (error - error_prev_) / dt;
    // ROS_INFO("diff_term: %f", diff_term);
    // ROS_INFO("dt: %f", dt);

    double output = kp_ * error + ki_ * error_integral_ + kd_ * diff_term;
    // ROS_INFO("output before cutoff: %f", output);
    if (output > max_output_) {
      output = max_output_;
    } else if (output < -max_output_) {
      output = -max_output_;
    }
    // ROS_INFO("max_output: %f", max_output_);
    error_prev_ = error;
    // ROS_INFO("output: %f", output);
    if (error_sign == 1) {
      return output;
    } else if (error_sign == -1) {
      return -output;
    } else {
      return 0;
    }
    return output;
  }

  void resetError() {
    error_prev_ = 0.0;
    error_integral_ = 0.0;
  }

 private:
  double kp_, ki_, kd_;
  double max_i_;
  double max_output_;
  double error_prev_;
  double error_integral_;
};

class MPTrackerServer {
 public:
  MPTrackerServer(ros::NodeHandle& nh_, ros::NodeHandle& pnh_);
  ~MPTrackerServer();

  /* set traj that is embedded in the action goal*/
  /* if goal canceled or preempted, traj_set = false*/
  void trackerGoalCB();
  void preemptCb_();
  /* set odom, if traj is also set, call update function */
  void odomCB(const nav_msgs::Odometry::ConstPtr& msg);

  /* compute control command (Twist) with traj and odom */
  void update();

  /* reset parameters */
  void resetParam();

  /* convert the trajectory message to Trajectory3D */
  void toTrajectory3D(const planning_ros_msgs::Trajectory& traj_msg);
  double shortestAngularDistance(double from, double to);
  double angleToNextPoint(const Eigen::Vector3d& curr, const Eigen::Vector3d& next);
  bool lookupOdomTransform();

 private:
  typedef actionlib::SimpleActionServer<jackal_tracker_msgs::JackalMPTrackerAction>
      ServerType;

  // Nodehandles
  ros::NodeHandle nh, pnh;
  ros::Subscriber odom_sub_;
  ros::Publisher cmd_vel_pub_;

  // Strings
  std::string mp_tracker_server_topic_;

  // action Server and param: mp tracker server
  std::unique_ptr<ServerType> mp_tracker_server_ptr_;

  std::unique_ptr<Trajectory3D> traj_;

  ros::Time odom_time_;
  Eigen::Vector3d odom_pos_;
  Eigen::Vector3d odom_vel_;
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;
  bool odom_set_;
  bool traj_set_;
  bool traj_finished_;
  bool active_;
  ros::Time traj_start_, t_prev_;
  Eigen::Vector3d current_pos_;
  Eigen::Quaterniond current_orient_;
  double current_yaw_;
  double v_kp_, v_ki_, v_kd_, v_max_i_, v_max_;
  double w_kp_, w_ki_, w_kd_, w_max_i_, w_max_;
  double distance_traveled_;
  double current_traj_length_;
  double traj_total_time_;
  // TF param
  tf2_ros::Buffer tfBuffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_ptr_;
  bool use_sim_;
  std::string sensor_frame_;
  std::string body_frame_;
  Eigen::Matrix4d b2s_;
  bool sensor_tf_init_;
  bool use_cmd_queue_;
  std::list<geometry_msgs::Twist> cmd_queue_;
  int cmd_queue_size_;
  bool reset_cmd_queue_;


  // double error_prev_;
  // double error_integral_;
  // Use command struct from mpl to query trajectory
  Command3D last_traj_cmd_;
  // Reverse traj
  bool reverse_traj_;
  std::unique_ptr<PIDController> linear_controller_;
  std::unique_ptr<PIDController> angular_controller_;
};

}  // namespace mp_tracker

#endif