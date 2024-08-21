# active_dsg_launch

## To build the docker file
Replace the ssh key with your ssh key which has access to MIT edu repos.
```
docker build -t jackal_docker --build-arg ssh_prv_key="$(cat ~/.ssh/id_yz)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_yz.pub)" -f jackal_dsg.Dockerfile --squash .
```

## Jackal Experiment

1. ```roscore```

2. ```roslaunch active_dsg_launch jackal_autonomy.launch```. This should launch Jackal Hardware, Ouster Lidar, RealSense D455 and Faster LIO. Feel free to create launch file that is suitable for your hardware.

# Launch ARL sim from active dsg launch
1. Mount this repo to the arl sim. Modify the run.sh for arl sim by adding this:
```
--volume {path_to_ws}/src/active_dsg_launch:/arl-unity-ros/ws/src/active_dsg_launch \
```
2. Enter ARL docker
3. Build active_dsg_launch in the docker
```
cd ws && catkin init && catkin build
```
4. source it
```
source devel/setup.bash
```
5. copy Uhumans2 sim binary into docker
```
docker cp arl-unity-robotics-sim/. admiring_gould::/arl-unity-ros/install/lib/arl_unity_simulator/
```
6. launch sim
```
roslaunch active_dsg_launch arl_sim.launch
```

## Launching

To deal with the fact that the Regions part of Hydra runs separately from ROS
and increase the flexibility when launching, I recommend using `tmuxp` and the
launch configuration in `tmux/hydra_launch.yaml`. You will have to pip install
tmuxp, but then you can run `tmux p load hydra_launch.yaml`, and it will load
hydra and the frontier-based planner in a tmux session. Because of how
`master.launch` is set up, it's really easy to choose which nodes should run in
their own separate tmux panes.

Currently there are two external environment variables that must be set:
1. ACTIVE_DSG_WS -- the path to your catkin workspace

2. HYDRA_REGIONS_VENV_PATH -- the path to the virtual environment you set up
for the Automatic-Abstractions code.


# Receiving a scene graph

There's an example of receiving a scene graph in `include/active_dsg_planner/dsg_receiver_example.py`.
You can see this example in action by first building this package (that takes care of the python install),
and then launching `master.launch`. If Hydra and the simulator are running, you should see it print
out the number of places and frontiers.

```
catkin build active_dsg_launch
roslaunch active_dsg_launch master.launch
```

If you haven't built the python bindings for `spark_dsg` yet, you need to run
`pip install --user .` inside of `spark_dsg`.

## Receiving in python and using pytorch

We probably don't need this functionality, but if we ever need to receive a
scene graph in python that we are planning on doing something with in Pytorch,
there is a slightly different receiver that we should use. Catkin doesn't play
well with virtual environments, and pytorch basically requires them. There is a
slightly different api that can be used to receive scene graphs *without*
needing anything ros-related on the receiving side.

# Frontier Planner

I implemented a preliminary version of a planner for going to a goal point that
may require traversing unknown space. Planning logic is in
`include/active_dsg_planner/frontier_planner.py`, and the ROS wrapper is in
`include/active_dsg_planner/frontier_planner_ros.py`. The planner relies on a
version of the scene graph that's easier to work with (implemented in
`pydsg.py`) than what's provided by `spark_dsg`. However, the code quality is
not great and I hope to clean it up a bit.

You can see the planner in action by launching the node (after the simulator and hydra are running)

```
roslaunch arl_unity_ros_ground simulator_with_husky.launch                        # Terminal 1
roslaunch active_dsg_launch arl_sim_husky.launch start_rviz:=false                # Terminal 2
roslaunch active_dsg_launch master.launch launch_frontier_planner:=true launch_rviz:=true # Terminal 3
```

Make sure to `catkin build` because there's a new service definition.

In rviz, click on the `Publish Point` widget. Now if you click on the rviz
grid, the planner will try planning a path from the robot's current position to
the clicked point. NOTE: you probably need to turn off the DSG visualization
before trying to publish a point. For me, rviz segfaults if you try to publish
a point while the DSG marker messages are displayed (an instance of this bug
https://github.com/ros-visualization/rviz/issues/1082)
