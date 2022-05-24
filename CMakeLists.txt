cmake_minimum_required(VERSION 2.8.3)
project(bite_timing_robot)

## Compile as C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  std_msgs
  sensor_msgs
  image_transport
  message_generation
  message_filters
)

## Declare ROS messages
add_message_files(
  FILES
  Pixel.msg
  BodyPart.msg
  Person.msg
  Frame.msg
  AudioData.msg
  GazeData.msg
  Orientation.msg
)
add_service_files(
  FILES
  CheckBiteTiming.srv
  MouthOpen.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

# catkin specific configuration
catkin_package(
  CATKIN_DEPENDS
  std_msgs
  geometry_msgs
  message_runtime
  sensor_msgs
  message_filters
)
