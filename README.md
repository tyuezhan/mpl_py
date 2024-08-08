# mpl_py
Motion Primitive Library in Python

Currently only contains implementation of the unicycle model for car with control input (linear velocity, angular velocity)

# dependency
```
pip install HeapDict
```
or install it from source: https://github.com/DanielStutzbach/heapdict

Go to workspace source folder `catkin_ws/src`
```
wstool init & wstool merge mpl_py/planner.rosinstall && wstool update
```
