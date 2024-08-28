#include "jackal_mp_tracker/mp_tracker_server.h"

using namespace mp_tracker;

MPTrackerServer::MPTrackerServer(ros::NodeHandle& nh_, ros::NodeHandle& pnh_)
    : nh(nh_), pnh(pnh_) {
  // Topics
  pnh_.param<std::string>("tracker/mp_tracker_server",
                        mp_tracker_server_topic_,
                        "/jackal_tracker/mp_tracker_server");
  pnh_.param<double>("tracker/vel/kp", v_kp_, 1.0);
  pnh_.param<double>("tracker/vel/ki", v_ki_, 1.0);
  pnh_.param<double>("tracker/vel/kd", v_kd_, 1.0);
  pnh_.param<double>("tracker/vel/max_i", v_max_i_, 1.0);
  pnh_.param<double>("tracker/vel/max_v", v_max_, 1.0);
  pnh_.param<double>("tracker/yaw/kp", w_kp_, 1.0);
  pnh_.param<double>("tracker/yaw/ki", w_ki_, 1.0);
  pnh_.param<double>("tracker/yaw/kd", w_kd_, 1.0);
  pnh_.param<double>("tracker/yaw/max_i", w_max_i_, 1.0);
  pnh_.param<double>("tracker/yaw/max_w", w_max_, 1.0);
  pnh_.param("use_sim", use_sim_, false);
  pnh_.param("sensor_frame", sensor_frame_, std::string("odom"));
  pnh_.param("body_frame", body_frame_, std::string("base_link"));

  // Controller
  linear_controller_.reset(new PIDController(v_kp_, v_ki_, v_kd_, v_max_i_, v_max_));
  angular_controller_.reset(new PIDController(w_kp_, w_ki_, w_kd_, w_max_i_, w_max_));

  // Action server
  mp_tracker_server_ptr_.reset(new ServerType(nh_, mp_tracker_server_topic_, false));
  mp_tracker_server_ptr_->registerGoalCallback(
      boost::bind(&MPTrackerServer::trackerGoalCB, this));
  mp_tracker_server_ptr_->registerPreemptCallback(
      boost::bind(&MPTrackerServer::preemptCb_, this));
  mp_tracker_server_ptr_->start();

  // Subscriber
  odom_sub_ = nh.subscribe("/odometry/filtered", 1, &MPTrackerServer::odomCB, this);

  // Publisher
  cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/tracker/cmd_vel", 1);
  odom_set_ = false;
  traj_set_ = false;
  traj_finished_ = true;
  active_ = false;

  // Find the static tf between lidar sensor to base_link, especially when lidar odom give odom in sensor_frame
  // tf listener
  tf_listener_ptr_.reset(new tf2_ros::TransformListener(tfBuffer_));
  sensor_tf_init_ = false;
}

MPTrackerServer::~MPTrackerServer(){};


bool MPTrackerServer::lookupOdomTransform() {
  // get current transform between sensor frame to base link
  geometry_msgs::TransformStamped transformStamped;
  try {
    transformStamped = tfBuffer_.lookupTransform(sensor_frame_, body_frame_, ros::Time(0));
  }
  catch (tf2::TransformException &ex) {
    ROS_ERROR("Couldn't get current body frame to sensor frame transform");
    ROS_WARN("%s", ex.what());
    return false;
  }
  Eigen::Vector3d trans(transformStamped.transform.translation.x,
                              transformStamped.transform.translation.y,
                              transformStamped.transform.translation.z);

  // Extract the rotation (quaternion)
  Eigen::Quaterniond rot(transformStamped.transform.rotation.w,
                              transformStamped.transform.rotation.x,
                              transformStamped.transform.rotation.y,
                              transformStamped.transform.rotation.z);

  // Create the transformation matrix
  b2s_ = Eigen::Matrix4d::Identity();
  b2s_.block<3, 3>(0, 0) = rot.toRotationMatrix();
  b2s_.block<3, 1>(0, 3) = trans;
  ROS_INFO("Body frame to sensor frame transform set");
  ROS_INFO("Transform:");
  std::cout << b2s_ << std::endl;
  sensor_tf_init_ = true;
  return true;
}

void MPTrackerServer::odomCB(const nav_msgs::Odometry::ConstPtr& msg) {
  if (!sensor_tf_init_) {
    if (use_sim_) {
      b2s_ = Eigen::Matrix4d::Identity();
      sensor_tf_init_ = true;
    } else {
      if (!lookupOdomTransform()) return;
    }
  }
  if (use_sim_) {
    odom_time_ = msg->header.stamp;
    odom_pos_(0) = msg->pose.pose.position.x;
    odom_pos_(1) = msg->pose.pose.position.y;
    odom_pos_(2) = msg->pose.pose.position.z;

    odom_vel_(0) = msg->twist.twist.linear.x;
    odom_vel_(1) = msg->twist.twist.linear.y;
    odom_vel_(2) = msg->twist.twist.linear.z;

    odom_orient_.w() = msg->pose.pose.orientation.w;
    odom_orient_.x() = msg->pose.pose.orientation.x;
    odom_orient_.y() = msg->pose.pose.orientation.y;
    odom_orient_.z() = msg->pose.pose.orientation.z;
    double yaw, _pitch, _roll;
    tf2::Matrix3x3(tf2::Quaternion(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                                  msg->pose.pose.orientation.z, msg->pose.pose.orientation.w)).getEulerYPR(yaw, _pitch, _roll);
    odom_yaw_ = yaw;
    odom_set_ = true;
  } else {
    odom_time_ = msg->header.stamp;
    Eigen::Quaterniond sensor_q = Eigen::Quaterniond(msg->pose.pose.orientation.w,
                                                 msg->pose.pose.orientation.x,
                                                 msg->pose.pose.orientation.y,
                                                 msg->pose.pose.orientation.z);
    Eigen::Matrix3d sensor_r_m = sensor_q.toRotationMatrix();
    Eigen::Matrix4d s2w = Eigen::Matrix4d::Identity();
    s2w.block<3, 3>(0, 0) = sensor_r_m;
    s2w(0, 3) = msg->pose.pose.position.x;
    s2w(1, 3) = msg->pose.pose.position.y;
    s2w(2, 3) = msg->pose.pose.position.z;
    s2w(3, 3) = 1.0;
    // std::cout<< "Before transform, s2w:\n " << s2w << std::endl;
    Eigen::Matrix4d body_pose = s2w * b2s_;
    odom_pos_ = body_pose.block<3, 1>(0, 3);
    odom_orient_ = Eigen::Quaterniond(body_pose.block<3, 3>(0, 0));
    double yaw, _pitch, _roll;
    tf2::Matrix3x3(tf2::Quaternion(odom_orient_.x(), odom_orient_.y(), odom_orient_.z(), odom_orient_.w())).getEulerYPR(yaw, _pitch, _roll);
    odom_yaw_ = yaw;
    // std::cout<< "After transform, body_pose:\n " << body_pose << std::endl;
    // Apply to linear velocity
    // std::cout<< "Before transform, sensor_vel:\n " << msg->twist.twist.linear << std::endl;
    Eigen::Vector3d sensor_vel(msg->twist.twist.linear.x,
                               msg->twist.twist.linear.y,
                               msg->twist.twist.linear.z);
    Eigen::Vector3d body_vel = sensor_r_m * sensor_vel;
    odom_vel_ = body_vel;
    // std::cout<< "After transform, body_vel:\n " << body_vel << std::endl;
    odom_set_ = true;
  }

  // When there is odom, and there is a traj, we call update function to compute new
  // control (Twist)
  update();
}

void MPTrackerServer::resetParam() {
  traj_set_ = false;
  traj_finished_ = true;
  current_traj_length_ = 0.0;
  linear_controller_->resetError();
  angular_controller_->resetError();
  // error_prev_ = 0.0;
  // error_integral_ = 0.0;
}

void MPTrackerServer::trackerGoalCB() {
  ROS_INFO("[MPTrackerServer]: Received MP tracker goal.");
  if (!odom_set_) {
    ROS_ERROR_THROTTLE(1.0, "[MPTrackerServer] No odom, cannot accept new goal.");
  }
  // If there is another goal active, cancel it
  if (mp_tracker_server_ptr_->isActive()) {
    ROS_WARN("[MPTrackerServer] Received a new goal, canceling the previous one");
    mp_tracker_server_ptr_->setAborted();
  }
  // Pointer to msg
  const auto goal = mp_tracker_server_ptr_->acceptNewGoal();

  if (mp_tracker_server_ptr_->isPreemptRequested()) {
    ROS_INFO("[MPTrackerServer] goal preempted.");
    mp_tracker_server_ptr_->setPreempted();
    resetParam();
    return;
  }

  // Save new trajectory into traj_
  toTrajectory3D(goal->trajectory);
  traj_start_ = ros::Time::now();
  t_prev_ = traj_start_;
  traj_total_time_ = traj_->getTotalTime();
  bool ret = traj_->evaluate(traj_total_time_, last_traj_cmd_);
  if (!ret) ROS_ERROR("evaluate traj failed");
  ROS_INFO(
      "[MPTrackerServer]: new traj total time %2.2f last pos on traj: (%2.2f, %2.2f, "
      "%2.2f)",
      traj_total_time_,
      last_traj_cmd_.pos(0),
      last_traj_cmd_.pos(1),
      last_traj_cmd_.pos(2));
  traj_set_ = true;
  traj_finished_ = false;
  current_traj_length_ = 0.0;
  linear_controller_->resetError();
  angular_controller_->resetError();

  // Set reverse
  reverse_traj_ = goal->reverse;
}

void MPTrackerServer::preemptCb_() {
  if (mp_tracker_server_ptr_->isActive()) {
    ROS_INFO("[MPTrackerServer] goal aborted.");
    mp_tracker_server_ptr_->setAborted();
  } else {
    ROS_INFO("[MPTrackerServer] goal preempted.");
    mp_tracker_server_ptr_->setPreempted();
  }
  resetParam();
}

// Calculate the shortest angular distance between two angles in radians
double MPTrackerServer::shortestAngularDistance(double from, double to) {
  double result = to - from;
  while (result > M_PI) result -= 2.0 * M_PI;
  while (result < -M_PI) result += 2.0 * M_PI;
  return result;
}

double MPTrackerServer::angleToNextPoint(const Eigen::Vector3d& curr,
                                         const Eigen::Vector3d& next) {
  return atan2(next(1) - curr(1), next(0) - curr(0));
}

void MPTrackerServer::toTrajectory3D(const planning_ros_msgs::Trajectory& traj_msg) {
  traj_.reset(new Trajectory3D());

  traj_->taus.push_back(0);
  // If not reverse:
  if (!reverse_traj_) {
    for (const auto& it : traj_msg.primitives) {
      // ROS_INFO("primitive type: %d", it.control_car);
      // auto seg = toPrimitive3D(it);
      // Vec2f U = seg.pr_car().coeff();
      // ROS_ERROR("seg_control: %f, %f", U(0), U(1));
      Waypoint3D seg_end = seg.evaluate(1.0);
      traj_->segs.push_back(toPrimitive3D(it));
      traj_->taus.push_back(traj_->taus.back() + it.t);
    }

    if (!traj_msg.lambda.empty()) {
      Lambda l;
      for (int i = 0; i < (int)traj_msg.lambda.size(); i++) {
        LambdaSeg seg;
        seg.a(0) = traj_msg.lambda[i].ca[0];
        seg.a(1) = traj_msg.lambda[i].ca[1];
        seg.a(2) = traj_msg.lambda[i].ca[2];
        seg.a(3) = traj_msg.lambda[i].ca[3];
        seg.ti = traj_msg.lambda[i].ti;
        seg.tf = traj_msg.lambda[i].tf;
        seg.dT = traj_msg.lambda[i].dT;
        l.segs.push_back(seg);
        traj_->total_t_ += seg.dT;
      }
      traj_->lambda_ = l;
      std::vector<decimal_t> ts;
      for (const auto& tau : traj_->taus) ts.push_back(traj_->lambda_.getT(tau));
      traj_->Ts = ts;
    } else
      traj_->total_t_ = traj_->taus.back();
  } else {
    // If reverse:
    int seg_id = 0;
    int closest_seg_id = 0;
    double closest_dist = std::numeric_limits<double>::max();
    // 1. Identify which is the closest segment end in the previous traj
    for (const auto& it : traj_msg.primitives) {
      ROS_INFO("primitive type: %d", it.control_car);
      // In reverse case, check the closest point of current odom to the segment
      Eigen::Vector3d p(current_pos_(0), current_pos_(1), current_pos_(2));
      auto seg = toPrimitive3D(it);
      Waypoint3D seg_end = seg.evaluate(1.0);
      Eigen::Vector3d seg_end_p(seg_end.pos(0), seg_end.pos(1), seg_end.pos(2));
      // Keep distance to seg_end
      double tmp_dist = (seg_end_p - p).norm();
      if (tmp_dist < closest_dist) {
        closest_dist = tmp_dist;
        closest_seg_id = seg_id;
      }
      seg_id ++;
    }
    ROS_ERROR("closest_seg_id: %d", closest_seg_id);
    // Add the segments up to the closest seg
    seg_id = 0;
    for (const auto& it : traj_msg.primitives) {
      if (seg_id > closest_seg_id) break;
      traj_->segs.push_back(toPrimitive3D(it));
      traj_->taus.push_back(traj_->taus.back() + it.t);
      seg_id ++;
    }
    // Add time up to the closest seg
    if (!traj_msg.lambda.empty()) {
      Lambda l;
      for (int i = 0; i <= closest_seg_id; i++) {
        LambdaSeg seg;
        seg.a(0) = traj_msg.lambda[i].ca[0];
        seg.a(1) = traj_msg.lambda[i].ca[1];
        seg.a(2) = traj_msg.lambda[i].ca[2];
        seg.a(3) = traj_msg.lambda[i].ca[3];
        seg.ti = traj_msg.lambda[i].ti;
        seg.tf = traj_msg.lambda[i].tf;
        seg.dT = traj_msg.lambda[i].dT;
        l.segs.push_back(seg);
        traj_->total_t_ += seg.dT;
      }
      traj_->lambda_ = l;
      std::vector<decimal_t> ts;
      for (const auto& tau : traj_->taus) ts.push_back(traj_->lambda_.getT(tau));
      traj_->Ts = ts;
    } else
      traj_->total_t_ = traj_->taus.back();
      ROS_ERROR("total_t_: %f", traj_->total_t_);
  }

}

void MPTrackerServer::update() {
  if (!traj_set_) {
    ROS_ERROR_THROTTLE(1.0, "[MPTrackerServer] No traj, cannot update.");
    return;
  }
  if (!odom_set_) {
    ROS_ERROR_THROTTLE(1.0, "[MPTrackerServer] No odom, cannot update.");
    return;
  }

  if (!mp_tracker_server_ptr_->isActive()) {
    // return Twist msg with 0
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = 0;
    cmd_vel.linear.y = 0;
    cmd_vel.linear.z = 0;
    cmd_vel.angular.x = 0;
    cmd_vel.angular.y = 0;
    cmd_vel.angular.z = 0;
    cmd_vel_pub_.publish(cmd_vel);
    ROS_INFO("[MPTrackerServer] Tracker is not active, returning 0 cmd_vel.");
    return;
  }

  const ros::Time t_now = ros::Time::now();

  // Record distance between last position and current.
  const double ds = Eigen::Vector2d(current_pos_(0) - odom_pos_(0),
                                    current_pos_(1) - odom_pos_(1)).norm();

  current_pos_ = odom_pos_;
  current_orient_ = odom_orient_;
  current_yaw_ = odom_yaw_;

  current_traj_length_ += ds;

  geometry_msgs::Twist cmd_vel;

  if (traj_finished_) {
    ROS_WARN("[MPTrackerServer] Trajectory finished.");
    ROS_WARN("[MPTrackerServer] Trajectory finished.");
    // TODO: Debug when we will enter this condition
    cmd_vel.linear.x = 0;
    cmd_vel.angular.z = 0;

    ROS_INFO("cmd_vel.linear.x: %f, cmd_vel.angular.z: %f", cmd_vel.linear.x, cmd_vel.angular.z);
    jackal_tracker_msgs::JackalMPTrackerResult result;
    result.total_time = traj_total_time_;
    result.total_distance_travelled = current_traj_length_;
    mp_tracker_server_ptr_->setSucceeded(result);
    cmd_vel_pub_.publish(cmd_vel);
    t_prev_ = t_now;
    return;
  }

  double traj_time = (t_now - traj_start_).toSec();
  ROS_INFO("Traj time: %f. Total time: %f", traj_time, traj_total_time_);

  if (traj_time >= traj_total_time_)  // Reached goal
  {
    ROS_WARN("[MPTrackerServer] Trajectory finished.");
    Eigen::Vector3d x;
    cmd_vel.linear.x = 0;
    cmd_vel.angular.z = 0;
    cmd_vel_pub_.publish(cmd_vel);
    ROS_INFO("cmd_vel.linear.x: %f, cmd_vel.angular.z: %f", cmd_vel.linear.x, cmd_vel.angular.z);

    traj_finished_ = true;
    current_traj_length_ = 0.0;
    jackal_tracker_msgs::JackalMPTrackerResult result;
    result.total_time = traj_total_time_;
    result.total_distance_travelled = current_traj_length_;
    mp_tracker_server_ptr_->setSucceeded(result);
  } else if (traj_time >= 0) {
    Command3D next_cmd;
    // Handle reverse cases
    if (reverse_traj_) {
      traj_time = traj_total_time_ - traj_time;
    }
    traj_->evaluate(traj_time, next_cmd);
    Eigen::Vector3d x;
    x(0) = next_cmd.pos(0);
    x(1) = next_cmd.pos(1);
    x(2) = next_cmd.pos(2);
    // linear velocity magnitude
    double traj_vel = sqrt(next_cmd.vel(0)*next_cmd.vel(0) + next_cmd.vel(1)*next_cmd.vel(1));
    double yaw_vel = next_cmd.yaw_dot;
    ROS_INFO("Traj vel: %f, Yaw vel: %f", traj_vel, yaw_vel);

    // Find the desired position in body frame
    // Take odom and construct H
    Eigen::Vector3d p(current_pos_(0), current_pos_(1), current_pos_(2));
    Eigen::Matrix3d R = current_orient_.toRotationMatrix();
    Eigen::Matrix4d H_b2w = Eigen::Matrix4d::Identity();
    H_b2w.block<3, 3>(0, 0) = R;
    H_b2w.block<3, 1>(0, 3) = p;
    Eigen::Vector4d x_world(x(0), x(1), x(2), 1);
    Eigen::Vector4d x_body = H_b2w.inverse() * x_world;
    int pos_error_sign = 1;
    // if x(0) is negative, it means the point is behind the robot, error sign is -1.
    if (x_body(0) < 0) {
      pos_error_sign = -1;
    } else if (x_body(0) == 0) {
      pos_error_sign = 0;
    }

    double dx = x(0) - current_pos_(0);
    double dy = x(1) - current_pos_(1);
    double yaw_des = next_cmd.yaw;
    // double yaw_des = angleToNextPoint(current_pos_, x);
    // ROS_INFO("current yaw: %f, des yaw: %f", current_yaw_, yaw_des);
    double error_yaw = shortestAngularDistance(current_yaw_, yaw_des);
    // ROS_WARN("Error yaw: %f", error_yaw);

    // double error_v = sqrt(dx * dx + dy * dy);
    // Find angle between heading and the error vector
    // double error_angle = atan2(dy, dx);
    // double angle_diff = shortestAngularDistance(current_yaw_, error_angle);
    // if (angle_diff > M_PI / 2 || angle_diff < -M_PI / 2) {
    //   error_v = -error_v;
    // }
    double error_pos = sqrt(dx * dx + dy * dy);

    // ROS_WARN("Before adding controller output: v: %f, w: %f", traj_vel, yaw_vel);
    if (reverse_traj_) {
      // 1. the original traj_vel needs to be flipped (we go reverse direction)
      // 2. position error sign needs to be flipped
      cmd_vel.linear.x = -traj_vel + linear_controller_->compute(error_pos, (t_now - t_prev_).toSec(), -pos_error_sign);
      cmd_vel.angular.z = -yaw_vel + angular_controller_->compute(error_yaw, (t_now - t_prev_).toSec(), -1);
    } else {
      cmd_vel.linear.x = traj_vel + linear_controller_->compute(error_pos, (t_now - t_prev_).toSec(), pos_error_sign);
      cmd_vel.angular.z = yaw_vel + angular_controller_->compute(error_yaw, (t_now - t_prev_).toSec(), 1);
    }

    // cmd_vel.linear.x = kv_ * sqrt(dx * dx + dy * dy);
    // cmd_vel.angular.z = kw_ * dtheta;
    cmd_vel_pub_.publish(cmd_vel);
    ROS_INFO("cmd_vel.linear.x: %f, cmd_vel.angular.z: %f", cmd_vel.linear.x, cmd_vel.angular.z);

  }
  t_prev_ = t_now;
  return;
}