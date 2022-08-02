#!/usr/bin/env python

# import modules
import sys
import cv2
import rospy
import argparse
import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from bite_timing_robot.msg import Frame, Person, BodyPart, Pixel, AudioData, GazeData, Orientation
from bite_timing_robot.srv import CheckBiteTiming, CheckBiteTimingResponse, Toggle, ToggleResponse
from sensor_msgs.msg import Image, CameraInfo
import os

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Int32

import collections

import concurrent.futures

from rt_gene_processor import RTGene

rospy.init_node('rt_gene_synchronous_ayase')

import faulthandler; faulthandler.enable()

class RTGeneSyncr:
    def __init__(self, frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic):

        self.pub = rospy.Publisher(pub_topic, Frame, queue_size=10)

        self.frame_id = frame_id
        self.no_depth = no_depth

        self.bridge = CvBridge()

        self.frame = None

        self.rt_gene = RTGene(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage))
        self.rt_genes = [self.rt_gene, self.rt_gene, self.rt_gene]
        # self.rt_genes = [RTGene(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage)) for _ in range(3)] # 16-17 fps for all 3 at the same time

        self.last_seq_2 = 0
        self.dropped_msgs_2 = 0
        # subscribe to audio topic and register callback

        self.color_topic = color_topic

        # test pub publishes a string
        # self.test_pub2 = rospy.Publisher('/camera1/gaze', String, queue_size=10000)
        self.last_process_time = 0

        self.subs1 = None
        self.subs2 = None
        self.subs3 = None
        self.toggle_service = rospy.Service('rt_gene/toggle_on', Toggle, self.toggle_callback)

        self.is_on = False

    def create_data_subs(self):
        print('Creating data subs')
        queue_size = 180*180
        buff_size = 65536*180

        gaze_num_threads = 1
        
        # self.pubs = [rospy.Publisher("/camera1/gaze", GazeData, queue_size=10),
        #              rospy.Publisher("/camera2/gaze", GazeData, queue_size=10),
        #              rospy.Publisher("/camera3/gaze", GazeData, queue_size=10)]

        self.pubs = [None,
                     rospy.Publisher("/camera2/gaze", GazeData, queue_size=10),
                     rospy.Publisher("/camera3/gaze", GazeData, queue_size=10)]

        # # hardcoding because some is running on ayase        
        # # self.pubs = [rospy.Publisher("/camera1/gaze", GazeData, queue_size=10),
        # #              rospy.Publisher("/camera2/gaze", GazeData, queue_size=10),
        #             #  rospy.Publisher("/camera3/gaze", GazeData, queue_size=10)]
        # self.pubs = [rospy.Publisher("/camera1/gaze", GazeData, queue_size=10),
        #              rospy.Publisher("/camera2/gaze", GazeData, queue_size=10),
        #              None]

        # self.subs1 = [message_filters.Subscriber("/camera1/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]

        # self.data_sub1 = message_filters.ApproximateTimeSynchronizer(self.subs1, queue_size, .1)
        # for i in range(gaze_num_threads):
        #     self.data_sub1.registerCallback(lambda img, i=i: self.gaze_callback(img, 0, i, gaze_num_threads))

        self.subs2 = [message_filters.Subscriber("/camera2/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.data_sub2 = message_filters.ApproximateTimeSynchronizer(self.subs2, queue_size, .1)
        for i in range(gaze_num_threads):
            self.data_sub2.registerCallback(lambda img, i=i: self.gaze_callback(img, 1, i, gaze_num_threads))

        self.subs3 = [message_filters.Subscriber("/camera3/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.data_sub3 = message_filters.ApproximateTimeSynchronizer(self.subs3, queue_size, .1)
        for i in range(gaze_num_threads):
            self.data_sub3.registerCallback(lambda img, i=i: self.gaze3_callback(img, 2, i, gaze_num_threads))


        # self.subs1 = [message_filters.Subscriber("/camera1/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size),
        #                 message_filters.Subscriber("/camera2/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size),
        #                 message_filters.Subscriber("/camera3/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]

        # self.data_sub1 = message_filters.ApproximateTimeSynchronizer(self.subs1, queue_size, .1)
        # self.data_sub1.registerCallback(lambda img1,img2,img3, i=0: self.all_gaze_callback(img1,img2,img3, 0, i, gaze_num_threads))
        # print('Data subs created')


    def delete_data_subs(self):
        print('deleting data subs')
        for sub in self.subs1:
            sub.sub.unregister()
            del sub

        if self.subs2 is not None:
            for sub in self.subs2:
                sub.sub.unregister()
                del sub
            for sub in self.subs3:
                sub.sub.unregister()
                del sub

        self.subs1 = None
        self.subs2 = None
        self.subs3 = None


    def toggle_callback(self, msg):
        toggle_on = msg.toggle_on

        if toggle_on and not self.is_on:
            self.create_data_subs()
        elif not toggle_on:
            if self.subs1 is not None:
                self.delete_data_subs()

        out = ToggleResponse(True)
        return out




    ########################################################################################################################
    ### gaze data callbacks
    ########################################################################################################################


    def gaze3_callback(self, img, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            # block for a bit
            return # drop every other frame!

        if not(img.header.seq % num_threads == thread_idx):
            return
        # check only on one thread
        if num_callback == 0:
            if self.last_seq_2 + 2 != img.header.seq:
                self.dropped_msgs_2 += 1
                print("rt_gene sync. gaze Processing Dropped msg: ",self.dropped_msgs_2, " seq: ", img.header.seq, " last seq: ", self.last_seq_2)
                print(num_callback, thread_idx, num_threads)
                print(img.header.seq, img.header.seq % num_threads)
            self.last_seq_2 = img.header.seq


        recieved_time = rospy.Time.now()
        # frame = self.face_openposes[num_callback].processOpenPose(img, depth)
        # attach all relevant features here
  
        gaze, headpose = self.rt_genes[num_callback].process(img)


        msg = GazeData()
        msg.header = img.header
        msg.header.frame_id = str(img.header.seq)
        if gaze is not None:
            msg.gaze.phi = gaze[0]
            msg.gaze.theta = gaze[1]
            msg.headpose.phi = headpose[0]
            msg.headpose.theta = headpose[1]
        else:
            msg.gaze.phi = -1000
            msg.gaze.theta = -1000
            msg.headpose.phi = -1000
            msg.headpose.theta = -1000
        self.pubs[2].publish(msg)

        # look at callback 0 to verify speed
        if num_callback == 0:
            finish = rospy.Time.now().to_sec()
            # print('gaze processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish



    
    # def gaze_callback(self, img,img2,img3, num_callback, thread_idx, num_threads):
    def gaze_callback(self, img, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            # block for a bit
            return # drop every other frame!

        if not(img.header.seq % num_threads == thread_idx):
            return
        # check only on one thread
        if num_callback == 0:
            if self.last_seq_2 + 2 != img.header.seq:
                self.dropped_msgs_2 += 1
                print("rt_gene sync. gaze Processing Dropped msg: ",self.dropped_msgs_2, " seq: ", img.header.seq, " last seq: ", self.last_seq_2)
                print(num_callback, thread_idx, num_threads)
                print(img.header.seq, img.header.seq % num_threads)
            self.last_seq_2 = img.header.seq


        recieved_time = rospy.Time.now()
        # frame = self.face_openposes[num_callback].processOpenPose(img, depth)
        # attach all relevant features here
  
        gaze, headpose = self.rt_genes[num_callback].process(img)


        msg = GazeData()
        msg.header = img.header
        msg.header.frame_id = str(msg.header.seq)
        if gaze is not None:
            msg.gaze.phi = gaze[0]
            msg.gaze.theta = gaze[1]
            msg.headpose.phi = headpose[0]
            msg.headpose.theta = headpose[1]
        else:
            msg.gaze.phi = -1000
            msg.gaze.theta = -1000
            msg.headpose.phi = -1000
            msg.headpose.theta = -1000
        self.pubs[1].publish(msg)

        # look at callback 0 to verify speed
        if num_callback == 0:
            finish = rospy.Time.now().to_sec()
            # print('gaze processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish

    def worker(self, img, num_callback):
            gaze, headpose = self.rt_genes[num_callback].process(img)

            msg = GazeData()
            msg.header = img.header
            msg.header.frame_id = str(img.header.seq)
            if gaze is not None:
                msg.gaze.phi = gaze[0]
                msg.gaze.theta = gaze[1]
                msg.headpose.phi = headpose[0]
                msg.headpose.theta = headpose[1]
            else:
                msg.gaze.phi = -1000
                msg.gaze.theta = -1000
                msg.headpose.phi = -1000
                msg.headpose.theta = -1000
            self.pubs[num_callback].publish(msg)
    
    def all_gaze_callback(self, img,img2,img3, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            # block for a bit
            return # drop every other frame!

        if not(img.header.seq % num_threads == thread_idx):
            return
        # check only on one thread
        if num_callback == 0:
            if self.last_seq_2 + 2 != img.header.seq:
                self.dropped_msgs_2 += 1
                print("rt_gene sync. gaze Processing Dropped msg: ",self.dropped_msgs_2, " seq: ", img.header.seq, " last seq: ", self.last_seq_2)
                print(num_callback, thread_idx, num_threads)
                print(img.header.seq, img.header.seq % num_threads)
            self.last_seq_2 = img.header.seq


        # frame = self.face_openposes[num_callback].processOpenPose(img, depth)
        # attach all relevant features here
        # inputs = [img, img2, img3]
        inputs = [img2]

        # with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        #     future_to_input = {executor.submit(self.worker, inp, i): inp for i, inp in enumerate(inputs)}
  
        # gaze, headpose = self.rt_genes[num_callback].process(img)
        gaze2, headpose2 = self.rt_genes[num_callback].process(img2)
        # gaze3, headpose3 = self.rt_genes[num_callback].process(img3)
        # gazes = [gaze, gaze2, gaze3]
        # headposese = [headpose, headpose2, headpose3]
        # imgs = [img, img2, img3]
        # gazes = [gaze, gaze2]
        gazes = [gaze2]
        # headposese = [headpose, headpose2]
        headposese = [headpose2]
        imgs = [img2]
        # imgs = [img, img2]
        headpose = headpose2
        gaze = gaze2


        # for i, (gaze, headpose) in enumerate(zip(gazes, headposese)):

            # so much hardcoding because of real-time issues
            # if i == 2 or i == 0:
                # continue

        msg = GazeData()
        msg.header = img
        msg.header.frame_id = str(imgs[i].header.seq)
        if gaze is not None:
            msg.gaze.phi = gaze[0]
            msg.gaze.theta = gaze[1]
            msg.headpose.phi = headpose[0]
            msg.headpose.theta = headpose[1]
        else:
            msg.gaze.phi = -1000
            msg.gaze.theta = -1000
            msg.headpose.phi = -1000
            msg.headpose.theta = -1000
        self.pubs[1].publish(msg)

        # look at callback 0 to verify speed
        if num_callback == 0:
            finish = rospy.Time.now().to_sec()
            # print('gaze processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish

        


def main():


    frame_id = 'camera_color_optical_frame'
    no_depth = False
    pub_topic = 'openpose_pose'
    color_topic = "color/image_raw/compressed/"
    depth_topic = "aligned_depth_to_color/image_raw/compressed"
    cam_info_topic = "/camera1/color/camera_info"

    try:
        # Flags, refer to include/openpose/flags.hpp for more parameters
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Start ros wrapper
        print('creating node')
        rtgene = RTGeneSyncr(frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic)
        # rtgene.create_data_subs()
        rospy.spin()

    except Exception as e:
        rospy.logerr(e)
        sys.exit(-1)


if __name__ == "__main__":
    main()
