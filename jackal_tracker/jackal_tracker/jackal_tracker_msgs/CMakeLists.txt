cmake_minimum_required(VERSION 3.10)
project(jackal_tracker_msgs)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation
  actionlib_msgs
  planning_ros_msgs
  planning_ros_utils
)



add_action_files(
  FILES
  JackalMPTracker.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  planning_ros_msgs
  actionlib_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ocp_msgs
  CATKIN_DEPENDS 
  nav_msgs 
  roscpp  
  sensor_msgs 
  std_msgs 
  message_runtime
  geometry_msgs 
  actionlib_msgs
  planning_ros_msgs
  planning_ros_utils
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)


