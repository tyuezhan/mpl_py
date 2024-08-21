cmake_minimum_required(VERSION 3.10)
project(jackal_mp_tracker)

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    rospy
    roscpp
    std_msgs
    nav_msgs
    geometry_msgs
    sensor_msgs
    message_generation
    planning_ros_utils
    motion_primitive_library
    catkin_simple
    tf2
)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${MOTION_PRIMITIVE_LIBRARY_INCLUDE_DIRS}
  ${DECOMP_UTIL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  include
)


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    rospy
    roscpp
    std_msgs
    nav_msgs
    geometry_msgs
    sensor_msgs
    message_runtime
    planning_ros_utils
    tf2
)

add_library(mp_tracker_server src/mp_tracker_server.cpp)

add_executable(mp_tracker_server_node src/mp_tracker_server_node.cpp)

target_link_libraries(mp_tracker_server_node
  mp_tracker_server
  ${catkin_LIBRARIES}
  ${MOTION_PRIMITIVE_LIBRARY_LIBRARIES}
)

add_dependencies(mp_tracker_server_node
  ${catkin_EXPORTED_TARGETS}
)
