#!/usr/bin/env python

# import modules
import sys
import cv2
import rospy
import argparse
import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from ros_openpose.msg import Frame, Person, BodyPart, Pixel
from sensor_msgs.msg import Image, CameraInfo
import os

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData

import collections
# Import Openpose (Ubuntu)
rospy.init_node('bite_trigger')



class BiteTrigger:
    def __init__(self, frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic):

        self.pub = rospy.Publisher(pub_topic, Frame, queue_size=10)

        self.frame_id = frame_id
        self.no_depth = no_depth

        self.bridge = CvBridge()

        self.frame = None

        # Populate necessary K matrix values for 3D pose computation.
        # cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
        # self.fx = cam_info.K[0]
        # self.fy = cam_info.K[4]
        # self.cx = cam_info.K[2]
        # self.cy = cam_info.K[5]
        # TODO: FIX THIS ON REAL ROBOT
        self.fx = 616.755
        self.fy = 616.829
        self.cx = 313.394
        self.cy = 251.594

        # Obtain depth topic encoding
        # print('Waiting for depth topic...', depth_topic)
        # encoding = rospy.wait_for_message(depth_topic, CompressedImage).format
        # self.mm_to_m = 0.001 if "16UC1" in encoding else 1.

        print('/camera1/' + color_topic)

        # image_sub = message_filters.Subscriber('/camera1/' + color_topic, CompressedImage)
        # depth_sub = message_filters.Subscriber(depth_topic, CompressedImage)
        # openpose_sub = message_filters.Subscriber("/camera1/openpose", Frame)
        
        subs = [message_filters.Subscriber('/camera1/' + color_topic, CompressedImage), message_filters.Subscriber("/camera1/openpose", Frame),
                message_filters.Subscriber('/camera2/' + color_topic, CompressedImage), message_filters.Subscriber("/camera2/openpose", Frame),
                message_filters.Subscriber('/camera3/' + color_topic, CompressedImage), message_filters.Subscriber("/camera3/openpose", Frame)]

        self.ts = message_filters.ApproximateTimeSynchronizer(subs, 5, .01)
        self.ts.registerCallback(self.callback)
        
        # subscribe to audio topic and register callback
        audio_sub = rospy.Subscriber('/audio', AudioData, self.audio_callback)

        self.speech_prefetch_buffer = b""
        self.speech_audio_buffer = self.speech_prefetch_buffer

        self.data_buffer = collections.deque(maxlen=180) # 180 is 6 seconds

        self.last_bite_time = rospy.Time.now()

        self.trigger_pub = rospy.Publisher('/biteTiming/trigger', String, queue_size=10)



    def audio_callback(self, msg):
        if len(self.speech_audio_buffer) == 0:
            self.speech_audio_buffer = self.speech_prefetch_buffer
        self.speech_audio_buffer += msg.data


    def process_audio(self):
        if len(self.speech_audio_buffer) == 0:
            return
        audio_data = self.speech_audio_buffer
        self.speech_audio_buffer =  b""
        # process the audio here!

        is_talking = False
        return is_talking

    # def callback(self, ros_image, openpose_frame):
    def callback(self, ros_image1, openpose_frame1, ros_image2, openpose_frame2, ros_image3, openpose_frame3):
        trigger = 0

        timestamp = rospy.Time.now()

        # Process audio
        is_talking = self.process_audio()

        # Buffer is_talking, and other input data!
        curr_data = (ros_image1, openpose_frame1, ros_image2, openpose_frame2, ros_image3, openpose_frame3, is_talking)
        self.data_buffer.append(curr_data)

        # We should only run if we have a full buffer, for the model's sake
        if len(self.data_buffer) < 180:
            return

        # Call model with what we have in our buffer
        print('Time since last bite', timestamp.to_sec() - self.last_bite_time.to_sec())

        if timestamp.to_sec() - self.last_bite_time.to_sec() >= 45:
            trigger = 1

        
        if trigger == 1:
            self.last_bite_time = timestamp
            self.trigger_pub.publish(String(""))

            # Clear the buffer because the model will be turned off during feeding stage
            self.data_buffer.clear()
            

        




def main():


    frame_id = 'camera_color_optical_frame'
    no_depth = False
    pub_topic = 'openpose_pose'
    color_topic = "color/image_raw/compressed/time_synced"
    depth_topic = "aligned_depth_to_color/image_raw/compressed"
    cam_info_topic = "/camera1/color/camera_info"

    try:
        # Flags, refer to include/openpose/flags.hpp for more parameters
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Start ros wrapper
        print('creating node')
        rop = BiteTrigger(frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic)
        rospy.spin()

    except Exception as e:
        rospy.logerr(e)
        sys.exit(-1)


if __name__ == "__main__":
    main()
