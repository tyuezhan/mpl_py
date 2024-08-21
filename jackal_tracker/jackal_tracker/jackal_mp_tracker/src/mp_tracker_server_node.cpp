#include "jackal_mp_tracker/mp_tracker_server.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "mp_tracker_server_node");
  ros::NodeHandle nh, pnh("~");
  mp_tracker::MPTrackerServer mp_tracker_server_node(nh, pnh);

  while (ros::ok()) {
    ros::spin();
  }
}