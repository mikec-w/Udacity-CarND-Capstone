# Udacity Self Driving Car Nano Degree
# Capstone Project

## Introduction

This project is the final Capstone project in the Udacity Self Driving Car Nano Degree. It involves using ROS nodes to implement a full self driving vehicle. 

This project includes two elements, a simulated section and a real car (Carla).

While this is intended as a Team Project, personal constraints meant that the work was performed alone.

## ROS Nodes

Various ROS nodes had to be made to interoperate in order to drive the car. This included:

* Waypoint Load Node - to load the path
* Waypoint Follower Node - to follow the path
ROS
* DBW Node - to drive the car (throttle, brake, steer) 
* Simulator Bridge - for testing
* Traffic Light detection - to stop at red lights

More details can be found in the project documentation.

## Traffic Light Detection

This node formed the majority of the project and using a Convolution Neural Network it was possible for this node to use the camera feed to determine if a traffic light was Red, Yellow, Green or unknown. This was then used to influence the waypoint follower and DBW node in order to stop the car appropriately.

The network was trained using two sets of data, one for the simulator and a seperate one based on real work video footage. The result being two models. In order to swap between the two, it is necessary to alter line 46 in tl_detector.py.

`# Set if using real car`

`self.is_site = True`


## Implementation of ROS 

The implementation was intially performed using the Udacity provided VM before moving the final result to the provided Workspace.

## Results 

The end result was the ROS control system managed to happily drive the simulator around the test track, obeying the lights as expected and without issue.

Furthermore, it demonstrated a high level of accuracy when replaying the rosbag test data provided. 
