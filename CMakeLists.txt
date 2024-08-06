cmake_minimum_required(VERSION 3.10)
project(mpl_py)

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    std_msgs
    nav_msgs
    geometry_msgs
    message_generation
    catkin_simple
    tf2
)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${MOTION_PRIMITIVE_LIBRARY_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  include
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    std_msgs
    nav_msgs
    geometry_msgs
    message_runtime
    tf2
)

# Python package installation
catkin_python_setup()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)