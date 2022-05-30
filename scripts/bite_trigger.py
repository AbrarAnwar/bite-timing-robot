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
from bite_timing_robot.srv import CheckBiteTiming, CheckBiteTimingResponse, MouthOpen, MouthOpenResponse
from sensor_msgs.msg import Image, CameraInfo
import os

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Int32
# from audio_common_msgs.msg import AudioData

import collections

from sklearn.neighbors import NearestNeighbors

import webrtcvad
import io
from scipy.io import wavfile
from scipy.io.wavfile import write

from rt_gene_processor import RTGene
from openpose_processor import OpenPose

import copy

from ml_models import SocialDiningModel

import pandas as pd

import threading
from multiprocessing.pool import ThreadPool
import concurrent.futures
PERSON1 = -180 # +/- 180
PERSON2 = 60
PERSON3 = -60


from threading import Thread, Lock

rospy.init_node('bite_trigger')

class BiteTrigger:
    def __init__(self, frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic):

        self.pub = rospy.Publisher(pub_topic, Frame, queue_size=10)

        self.frame_id = frame_id
        self.no_depth = no_depth

        self.bridge = CvBridge()

        self.frame = None
        self.mutex = Lock()


        # self.openposes = []
        # for i in range(3):
        #     self.openposes.append(OpenPose(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage), self.frame_id, face=False))
        
        # self.face_openposes = []
        # for i in range(3):
        #     self.face_openposes.append(OpenPose(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage), self.frame_id, face=True))


        self.openpose = OpenPose(rospy.wait_for_message("/camera1/" + depth_topic, CompressedImage), self.frame_id)

        # self.openposes = [self.openpose, self.openpose, self.openpose]

        # NOTE: UNCOMMENT THE ONE BELOW IF I DON'T WANT TO USE BODY DETECTION (like for target participant!)
        # self.face_openpose = OpenPose(rospy.wait_for_message("/camera1/" + depth_topic, CompressedImage), self.frame_id, face=True, body=False)
        # self.face_openposes = [self.face_openpose, self.face_openpose, self.face_openpose]
        self.openposes = [self.openpose, self.openpose, self.openpose] # 

        # self.rt_gene = RTGene(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage))
        # self.rt_genes = [self.rt_gene, self.rt_gene, self.rt_gene]
        # self.rt_genes = [RTGene(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage)) for _ in range(3)] # 16-17 fps for all 3 at the same time

        self.last_seq_1 = 0
        self.dropped_msgs_1 = 0
        self.last_seq_2 = 0
        self.dropped_msgs_2 = 0
        # subscribe to audio topic and register callback

        self.data_buffers = [collections.deque(maxlen=180) for _ in range(3)] # 90 is 6 seconds at 15fps

        self.face_buffers = [collections.deque(maxlen=180) for _ in range(3)] # 90 is 6 seconds

        self.gaze_buffers = [collections.deque(maxlen=180*2) for _ in range(3)] # 90 is 6 seconds


        self.audio_buffer = collections.deque(maxlen=180*4)
        self.direction_buffer = collections.deque(maxlen=180*4)


        self.color_topic = color_topic
        self.depth_topic = depth_topic

        self.last_bite_time = rospy.Time.now()


        # self.check_sub = rospy.Subscriber('/biteTiming/shouldFeed', String, self.check_callback)
        # rospy.Timer(rospy.Duration(3), self.check_callback)
        print('registering service')
        self.check_service = rospy.Service("/check_bite_timing", CheckBiteTiming, self.check_callback)

        self.mouth_open_service = rospy.Service("/mouth_open", MouthOpen, self.mouth_open_callback)


        self.feeding_in_progress = True
        self.vad = webrtcvad.Vad(3)
        # test pub publishes a string
        self.test_pub1 = rospy.Publisher('/camera1/openpose', String, queue_size=10000)
        # self.test_pub2 = rospy.Publisher('/camera1/gaze', String, queue_size=10000)

        self.last_process_time = 0

        # a lock to prevent race conditions

        self.model = SocialDiningModel('Linear SVM')

        # import multiprocessing
        # manager = multiprocessing.Manager()
        self.pool = ThreadPool(processes=3)

    def create_data_subs(self):
        print('Creating data subs')
        queue_size = 180*180
        buff_size = 65536*180
        num_threads = 1

        gaze_num_threads = 1
        

        # self.subs1 = [message_filters.Subscriber("/camera1/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size), message_filters.Subscriber('/camera1/' + self.depth_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.subs1 = [message_filters.Subscriber("/camera1/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]

        self.data_sub1 = message_filters.ApproximateTimeSynchronizer(self.subs1, queue_size, .1)
        for i in range(num_threads):
            self.data_sub1.registerCallback(lambda img, i=i: self.data_callback(img, 0, i, num_threads))
        # for i in range(gaze_num_threads):
        #     self.data_sub1.registerCallback(lambda img, i=i: self.gaze_callback(img, 0, i, gaze_num_threads))

        # self.subs2 = [message_filters.Subscriber("/camera2/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size), message_filters.Subscriber('/camera2/' + self.depth_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.subs2 = [message_filters.Subscriber("/camera2/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.data_sub2 = message_filters.ApproximateTimeSynchronizer(self.subs2, queue_size, .1)
        for i in range(num_threads):
            self.data_sub2.registerCallback(lambda img, i=i: self.data_callback(img, 1, i, num_threads))
        # for i in range(gaze_num_threads):
        #     self.data_sub2.registerCallback(lambda img, i=i: self.gaze_callback(img, 1, i, gaze_num_threads))

        # self.subs3 = [message_filters.Subscriber("/camera3/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size), message_filters.Subscriber('/camera3/' + self.depth_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.subs3 = [message_filters.Subscriber("/camera3/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]

        self.data_sub3 = message_filters.ApproximateTimeSynchronizer(self.subs3, queue_size, .1)
        for i in range(num_threads):
            self.data_sub3.registerCallback(lambda img, i=i: self.data_callback(img, 2, i, num_threads))
        # for i in range(gaze_num_threads):
        #     self.data_sub3.registerCallback(lambda img, i=i: self.gaze_callback(img, 2, i, gaze_num_threads))

        for i in range(gaze_num_threads):
            self.gaze_sub1 = rospy.Subscriber('/camera1/gaze', GazeData, lambda img, i=i: self.gaze_callback(img, 0, i, gaze_num_threads) , queue_size=queue_size, buff_size=buff_size)
            self.gaze_sub2 = rospy.Subscriber('/camera2/gaze', GazeData, lambda img, i=i: self.gaze_callback(img, 1, i, gaze_num_threads) , queue_size=queue_size, buff_size=buff_size)
            self.gaze_sub3 = rospy.Subscriber('/camera3/gaze', GazeData, lambda img, i=i: self.gaze_callback(img, 2, i, gaze_num_threads) , queue_size=queue_size, buff_size=buff_size)


        # self.subs1 = [message_filters.Subscriber("/camera1/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size),
        #                 message_filters.Subscriber("/camera2/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size),
        #                 message_filters.Subscriber("/camera3/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]


        # self.data_sub1 = message_filters.ApproximateTimeSynchronizer(self.subs1, queue_size, .1)
        # self.data_sub1.registerCallback(lambda img1,img2,img3, i=0: self.data_callback(img1,img2,img3, 0, i, gaze_num_threads))


        self.audio_sub = rospy.Subscriber('/audio', AudioData, self.audio_callback)
        self.direction_sub = rospy.Subscriber('/sound_direction', Int32, self.direction_callback)

        print('Data subs created')


    def delete_data_subs(self):
            for sub in self.subs1:
                sub.sub.unregister()
                del sub
            for sub in self.subs2:
                sub.sub.unregister()
                del sub
            for sub in self.subs3:
                sub.sub.unregister()
                del sub

            self.gaze_sub1.unregister()
            self.gaze_sub2.unregister()
            self.gaze_sub3.unregister()

            self.audio_sub.unregister()
            self.direction_sub.unregister()

            del self.gaze_sub1
            del self.gaze_sub2
            del self.gaze_sub3

            del self.audio_sub
            del self.direction_sub


            # Clear the buffer because the model will be turned off during feeding stage
            for buffer in self.data_buffers:
                buffer.clear()
            for buffer in self.face_buffers:
                buffer.clear()
            for buffer in self.gaze_buffers:
                buffer.clear()

            self.audio_buffer.clear()
            self.direction_buffer.clear()

            self.last_seq_1 = 0
            self.last_seq_2 = 0

    ########################################################################################################################
    ### Openpose and image callbacks
    ########################################################################################################################
    def worker(self, img, num_callback):
        frame = self.openposes[num_callback].processOpenPose(img)
        return num_callback, frame

    def all_data_callback(self, img,img2,img3, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            return # drop every other frame!

        # if not(img.header.seq % num_threads == thread_idx):
        #     return

        if num_callback == 0:
            if self.last_seq_1+ 2 != img.header.seq:
                self.dropped_msgs_1 += 1
                print("Openpose Processing Dropped msg: ", self.dropped_msgs_1, " seq: ", img.header.seq, " last seq: ", self.last_seq_1)
            self.last_seq_1 = img.header.seq


        recieved_time = rospy.Time.now()

        inputs = [img, img2, img3]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_input = {executor.submit(self.worker, inp, i): inp for i, inp in enumerate(inputs)}

            for future in concurrent.futures.as_completed(future_to_input):
                i, frame = future.result()
                all_data = {'header':img.header, 'openpose':frame}
                with self.mutex:
                    # condition 1. there is no face data, then use previous frame data
                    # print(len(self.data_buffers[num_callback]), 'callback: ', num_callback) 
                    # print(frame)
                    if len(frame.persons) == 0:
                        if len(self.data_buffers[i]) > 0:
                            all_data = self.data_buffers[i][-1]['data']
                            all_data['header'] = img.header
                        else:
                            print("1. data callback is invalid", i)
                            continue
                    # condition 2. there is a person, but there is no face data
                    elif len(frame.persons) > 0 and len(frame.persons[0].face) == 0:
                        if len(self.data_buffers[i]) > 0:
                            prev_frame = self.data_buffers[i][-1]['data']['openpose']
                            frame.persons[0].face = prev_frame.persons[0].face
                            all_data['openpose'] = frame
                        else:
                            print("2. data callback is invalid", i)
                            continue
                    # condition 3. there is a person and no body
                    elif len(frame.persons) > 0 and len(frame.persons[0].bodyParts) == 0:
                        # check if we are supposed to have body data, let's then look at previous values
                        if self.openposes[i].body == True:
                            if len(self.data_buffers[i]) > 0:
                                prev_frame = self.data_buffers[i][-1]['data']['openpose']
                                frame.persons[0].bodyParts = prev_frame.persons[0].bodyParts
                                all_data['openpose'] = frame
                            else:
                                print("3. data callback is invalid", i)
                                continue
                            



                    # after reselecting the frame, we can compute the order again
                    face_kp, body_kp = self.indexOpenposeFrame(frame)
                    all_data['face_kp'] = face_kp
                    all_data['body_kp'] = body_kp

                    # normal condition. we have a face and a body
                    self.data_buffers[i].append({'time':recieved_time, 'data':all_data})



    def data_callback(self, img, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            return # drop every other frame!

        # if not(img.header.seq % num_threads == thread_idx):
        #     return

        if num_callback == 0:
            if self.last_seq_1+ 2 != img.header.seq:
                self.dropped_msgs_1 += 1
                print("Openpose Processing Dropped msg: ", self.dropped_msgs_1, " seq: ", img.header.seq, " last seq: ", self.last_seq_1)
            self.last_seq_1 = img.header.seq


        recieved_time = rospy.Time.now()

        
        i = num_callback
        frame = self.openposes[i].processOpenPose(img)
        
        all_data = {'header':img.header, 'openpose':frame}
        with self.mutex:
            # condition 1. there is no face data, then use previous frame data
            # print(len(self.data_buffers[num_callback]), 'callback: ', num_callback) 
            # print(frame)
            if len(frame.persons) == 0:
                if len(self.data_buffers[i]) > 0:
                    all_data = self.data_buffers[i][-1]['data']
                    all_data['header'] = img.header
                else:
                    print("1. data callback is invalid", i)
                    return
            # condition 2. there is a person, but there is no face data
            elif len(frame.persons) > 0 and len(frame.persons[0].face) == 0:
                if len(self.data_buffers[i]) > 0:
                    prev_frame = self.data_buffers[i][-1]['data']['openpose']
                    frame.persons[0].face = prev_frame.persons[0].face
                    all_data['openpose'] = frame
                else:
                    print("2. data callback is invalid", i)
                    return
            # condition 3. there is a person and no body
            elif len(frame.persons) > 0 and len(frame.persons[0].bodyParts) == 0:
                # check if we are supposed to have body data, let's then look at previous values
                if self.openposes[i].body == True:
                    if len(self.data_buffers[i]) > 0:
                        prev_frame = self.data_buffers[i][-1]['data']['openpose']
                        frame.persons[0].bodyParts = prev_frame.persons[0].bodyParts
                        all_data['openpose'] = frame
                    else:
                        print("3. data callback is invalid", i)
                        return
                    

            # after reselecting the frame, we can compute the order again
            face_kp, body_kp = self.indexOpenposeFrame(frame)
            all_data['face_kp'] = face_kp
            all_data['body_kp'] = body_kp

            # normal condition. we have a face and a body
            self.data_buffers[i].append({'time':recieved_time, 'data':all_data})



        if num_callback == 0:
            # for debugging

            self.test_pub1.publish("bleh")        
            finish = rospy.Time.now().to_sec()
            # print('pose processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish

    def indexOpenposeFrame(self, frame):
        # We assume 1 person per frame
        face_kp = []
        body_kp = []
        if len(frame.persons) > 0:        
            for i in range(len(frame.persons[0].face)):
                face = frame.persons[0].face[i]
                face_kp.append([face.point.x, face.point.y])

            for i in range(len(frame.persons[0].bodyParts)):
                body = frame.persons[0].bodyParts[i]
                body_kp.append([body.point.x, body.point.y])

        return np.array(face_kp), np.array(body_kp)
            
        


    ########################################################################################################################
    ### face data callbacks
    ########################################################################################################################
    def face_callback(self, img, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            return # drop every other frame!

        # if not(img.header.seq % num_threads == thread_idx):
        #     return
        # check only on one thread
        if num_callback == 0:

            if self.last_seq_2 + 2 != img.header.seq:
                self.dropped_msgs_2 += 1
                print("Face Processing Dropped msg: ",self.dropped_msgs_2, " seq: ", img.header.seq, " last seq: ", self.last_seq_2)
                print(num_callback, thread_idx, num_threads)
                print(img.header.seq, img.header.seq % num_threads)
            self.last_seq_2 = img.header.seq


        recieved_time = rospy.Time.now()
        frame = self.face_openposes[num_callback].processOpenPose(img)
        # attach all relevant features here

        all_data = {'openpose':frame}
        # add to buffer
        with self.mutex:
            self.face_buffers[num_callback].append({'time':recieved_time, 'data':all_data})

        # look at callback 0 to verify speed
        if num_callback == 0:
            # self.test_pub.publish("bleh")        
            finish = rospy.Time.now().to_sec()
            # print('face processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish

    ########################################################################################################################
    ### gaze data callbacks
    ########################################################################################################################
    
    def gaze_callback(self, msg, num_callback, thread_idx, num_threads):
        if msg.header.seq % 2 == 0:
            return # drop every other frame!

        # if not(msg.header.seq % num_threads == thread_idx):
        #     return
        # check only on one thread
        if num_callback == 0:
            if self.last_seq_2 + 2 != msg.header.seq:
                self.dropped_msgs_2 += 1
                print("bite_trigger. gaze Processing Dropped msg: ",self.dropped_msgs_2, " seq: ", msg.header.seq, " last seq: ", self.last_seq_2)
                print(num_callback, thread_idx, num_threads)
                print(msg.header.seq, msg.header.seq % num_threads)
            self.last_seq_2 = msg.header.seq


        recieved_time = rospy.Time.now()

        all_data = {'header':msg.header, 'gaze':msg.gaze, 'headpose':msg.headpose}
        # add to buffer
        with self.mutex:
            if msg.gaze.phi == 0 and msg.gaze.theta == 0:
                # ensure we have a valid gaze
                if len(self.gaze_buffers[num_callback]) > 0:
                    prev_data = self.gaze_buffers[num_callback][-1]['data']
                    all_data['gaze'] = prev_data['gaze']
                    all_data['headpose'] = prev_data['headpose']
                else:
                    print("WARNING: gaze data is invalid", num_callback)
                    return
                # all_data = self.gaze_buffers[num_callback][-1]['data']
                # all_data['header'] = msg.header
            self.gaze_buffers[num_callback].append({'time':recieved_time, 'data':all_data})

        # look at callback 0 to verify speed
        if num_callback == 0:
            finish = rospy.Time.now().to_sec()
            # print('gaze processing time: \t', msg.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish

        

    ########################################################################################################################
    ### Audio processing Callbacks
    ########################################################################################################################

    def convert_bytearray_to_wav_ndarray(self, input_bytearray, sampling_rate=16000):
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, sampling_rate, np.frombuffer(input_bytearray, dtype=np.int16))
        output_wav = byte_io.read()
        samplerate, output = wavfile.read(io.BytesIO(output_wav))
        return output, samplerate

    def get_binary_speaking(self, x):
        wav, sr = self.convert_bytearray_to_wav_ndarray(x)
        is_speeches = []
        for window in np.lib.stride_tricks.sliding_window_view(wav, 640)[::80]:
            is_speech = self.vad.is_speech(window, 16000)
            is_speeches.append(is_speech)
        return np.array(is_speeches).mean() > .7

    def audio_callback(self, msg):
        recieved_time = rospy.Time.now()
        audio = msg.data

        # process frame here
        is_talking = self.get_binary_speaking(msg.data)

        # attach all relevant features here
        all_data = {'audio': audio, 'is_talking': is_talking}
        # add to buffer
        with self.mutex:
            self.audio_buffer.append({'time':recieved_time, 'data':all_data})

    def direction_callback(self, msg):
        recieved_time = rospy.Time.now()

        direction = msg.data
        with self.mutex:
            self.direction_buffer.append({'time':recieved_time, 'data':direction})


    def who_is_talking(self, is_talking, directions):
        # nearest neighbor audio directions, but first sin/cos them
        directions = np.array(directions)
        sin_directions = np.sin(np.radians(directions))

        person_directions = np.array([PERSON1, PERSON2, PERSON3])
        person_sin_dirs = np.sin(np.radians(person_directions))


        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(person_sin_dirs).reshape(-1,1))
        dists, idxs = nbrs.kneighbors(np.array(sin_directions).reshape(-1,1))
        person_annotations = idxs[:, 0] + 1

        # now, combine person annotations and is_talking
        is_talking = np.array(is_talking) # binary annotations
        who_is_talking = person_annotations * is_talking
        print(who_is_talking)

        return who_is_talking


    ########################################################################################################################
    ### ROS Service management
    ########################################################################################################################

    def interpolate_gaze_headpose(self, input_data):
        # df = pd.read_csv("/Users/tongwu/Downloads/rt_gene_feats.csv", usecols=['name', 'gaze'])
        # turn inputs into df

        filtered = []
        for i in range(len(input_data)):
            gaze = [input_data[i]['gaze'].phi, input_data[i]['gaze'].theta]
            headpose = [input_data[i]['headpose'].phi, input_data[i]['headpose'].theta]

            if gaze == [0,0]:
                print("NONE HAPPENED")
                continue
            
            filtered.append({'gaze_phi':gaze[0], 'gaze_theta':gaze[1], 'headpose_phi':headpose[0], 'headpose_theta':headpose[1], 'name':i})
            

        df = pd.DataFrame(filtered)

        l = []
        for i in range(1, len(df) - 1):
            pre = df.iloc[i-1]['name']
            cur = df.iloc[i]['name']
            if int(pre) != int(cur) - 1:
                diff = int(cur) - int(pre)
                for j in range(diff):
                    should = int(pre) + j
                    l.append(should)
            else:
                l.append(df.iloc[i-1]['name'])


        new_col = pd.DataFrame({"new_name": l})
        df = df.set_index("name")
        new_col = new_col.set_index("new_name")
        df_new = new_col.join(df)
        df_new = df_new.interpolate() # NOTE: EXPENSIVE OPERATION!


        # size should be 90, so we fill in df_new to be the last valid value!
        end = []
        last_valid = {'gaze_phi':df_new.iloc[-1]['gaze_phi'], 'gaze_theta':df_new.iloc[-1]['gaze_theta'], 'headpose_phi':df_new.iloc[-1]['headpose_phi'], 'headpose_theta':df_new.iloc[-1]['headpose_theta']}
        for i in range(len(l), 90):
            last_valid.update({'new_name':i})
            end.append(last_valid)
            # df_new =df_new.append({'gaze_phi':df_new.iloc[-1]['gaze_phi'], 'gaze_theta':df_new.iloc[-1]['gaze_theta'], 'headpose_phi':df_new.iloc[-1]['headpose_phi'], 'headpose_theta':df_new.iloc[-1]['headpose_theta'], 'new_name':i}, ignore_index=True)
        df_new = df_new.append(end, ignore_index=True)

        # do the same thing from 0 to min value
        first_valid = {'gaze_phi':df_new.iloc[0]['gaze_phi'], 'gaze_theta':df_new.iloc[0]['gaze_theta'], 'headpose_phi':df_new.iloc[0]['headpose_phi'], 'headpose_theta':df_new.iloc[0]['headpose_theta']}
        begin = []
        for i in range(0, int(l[0])):
            first_valid.update({'new_name':i})
            begin.append(first_valid)
        df_new = pd.concat([pd.DataFrame(begin), df_new])

        print(df_new)
        return df_new


    def align_data(self):
        # we have 3 data inputs, direction, audio, and video
        # they are inside data_buffer, audio_buffer, and direction_buffer

        # save bufferes into local variables 
        # and lock the buffers
        with self.mutex:
            data_buffers = copy.deepcopy(self.data_buffers)
            gaze_buffers = copy.deepcopy(self.gaze_buffers)
            audio_buffer = copy.deepcopy(self.audio_buffer)
            direction_buffer = copy.deepcopy(self.direction_buffer)

        
        # these are a deque of size 180 of format (time, (data1, data2, ...))
        video_times = []
        video_seqs = []
        for item in data_buffers[0]: # using only buffer 1
            video_times.append(item['time'].to_sec())
            video_seqs.append(item['data']['header'].stamp)
        

        ### Ensure that the rostopics are publishing information at the same rate!
        video_seqs = np.array(video_seqs)

        gaze_times = []
        gaze_seqs = []
        for item in gaze_buffers[0]:
            gaze_seqs.append(item['data']['header'].stamp)
            gaze_times.append(item['time'].to_sec())

        gaze_seqs = np.array(gaze_seqs)

        max_gaze_seq = np.max(gaze_seqs)
        max_video_seq = np.max(video_seqs)

        if max_gaze_seq > max_video_seq:
            print("Case 1")
            # then look for where max_video_seq occurs in gaze_seqs
            gaze_idxs = np.where(gaze_seqs <= max_video_seq)[0]
            video_idxs = np.arange(len(video_seqs))
        elif max_video_seq > max_gaze_seq:
            print("Case 2")
            # then look for where max_gaze_seq occurs in video_seqs
            video_idxs = np.where(video_seqs <= max_gaze_seq)[0]
            # thus gaze idxs are everything else
            gaze_idxs = np.arange(len(gaze_seqs))
        else:
            print("Case 3")
            # then they are equal
            video_idxs = np.arange(len(video_seqs))
            gaze_idxs = np.arange(len(gaze_seqs))

        # now we take the last 90 frames of each
        video_idxs = video_idxs[-90:]
        gaze_idxs = gaze_idxs[-90:]

        print(gaze_idxs)
        print(video_idxs)

        # this is in case it's not publishing at the same rate (one is faster than the other)
        if len(video_idxs) < 90 or len(gaze_idxs) < 90:
            print("Not enough data to align")
            return

        # these should now be the same length!

        # we now make sure all buffers use the same indices when aligning and calling the model

        video_times = np.array(video_times)[video_idxs]

        audio_times = []
        audio_data = []
        for item in audio_buffer:
            audio_times.append(item['time'].to_sec())
            audio_data.append(item['data'])



        direction_times = []
        direction_data = []
        for item in direction_buffer:
            direction_times.append(item['time'].to_sec())
            direction_data.append(item['data'])
        # NOTE THIS COULD BE EMPTY!!!



            
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(direction_times).reshape(-1,1))
        dists, idxs = nbrs.kneighbors(np.array(video_times).reshape(-1,1))
        dir_mapping = idxs[:, 0]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(audio_times).reshape(-1,1))
        dists, idxs = nbrs.kneighbors(np.array(video_times).reshape(-1,1))
        audio_mapping = idxs[:, 0]


        # get who is speaking annotations
        direction_data = np.array(direction_data)[dir_mapping]
        audio_data = np.array(audio_data)[audio_mapping]
        is_talking = [item['is_talking'] for item in audio_data]

        who_is_talking = self.who_is_talking(is_talking, direction_data)

        # need to return the following:
        # aligned pose frames 1,2,3, gazes 1,2,3
        # who's talking 1,2,3
        # time, count (given by callback messages)
        out_feats = {}



        # interpolate missing data
        # print(pd.DataFrame([item['data'] for item in gaze_buffers[0]])['gaze'] )
        # print(np.array(gaze_buffers[0])[gaze_idxs]['data'])




        # GAZE FEATURES. 90*(2+2)
        print(len(gaze_buffers[0]), len(gaze_buffers[1]), len(gaze_buffers[2]))
        print(len(gaze_idxs))

        gaze_1_data = np.array([item['data'] for item in gaze_buffers[0]])[gaze_idxs]
        gazes1 = np.array([ [item['gaze'].phi, item['gaze'].theta] for item in gaze_1_data])
        headpose1 = np.array([[item['gaze'].phi, item['gaze'].theta] for item in gaze_1_data])
        out_feats['gaze1'] = gazes1
        out_feats['headpose1'] = headpose1
        # out_feats['gaze_1'] = self.interpolate_gaze_headpose(gaze_1_data)

        gaze_2_data = np.array([item['data'] for item in gaze_buffers[1]])[gaze_idxs]
        gazes2 = np.array([ [item['gaze'].phi, item['gaze'].theta] for item in gaze_2_data])
        headpose2 = np.array([[item['gaze'].phi, item['gaze'].theta] for item in gaze_2_data])
        out_feats['gaze2'] = gazes2
        out_feats['headpose2'] = headpose2

        gaze_3_data = np.array([item['data'] for item in gaze_buffers[2]])[gaze_idxs]
        gazes3 = np.array([ [item['gaze'].phi, item['gaze'].theta] for item in gaze_3_data])
        headpose3 = np.array([[item['gaze'].phi, item['gaze'].theta] for item in gaze_3_data])
        out_feats['gaze3'] = gazes3
        out_feats['headpose3'] = headpose3

        # OPENPOSE FEATURES
        op_1_data = np.array([item['data'] for item in data_buffers[0]])[video_idxs]
        face1 = np.array([item['face_kp'] for item in op_1_data])
        body1 = np.array([item['body_kp'] for item in op_1_data])

        op_2_data = np.array([item['data'] for item in data_buffers[1]])[video_idxs]
        face2 = np.array([item['face_kp'] for item in op_2_data])
        body2 = np.array([item['body_kp'] for item in op_2_data])


        op_3_data = np.array([item['data'] for item in data_buffers[2]])[video_idxs]
        face3 = np.array([item['face_kp'] for item in op_3_data])
        body3 = np.array([item['body_kp'] for item in op_3_data])

        # delete the right lower body parts
        if self.openposes[0].body:
            body1 = np.delete(body1, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 1)
        body2 = np.delete(body2, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 1)
        body3 = np.delete(body3, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 1)

        if not self.openposes[0].body:
            body1 = np.zeros_like(body2)

        out_feats['face1'] = face1
        out_feats['body1'] = body1

        out_feats['face2'] = face2
        out_feats['body2'] = body2

        out_feats['face3'] = face3
        out_feats['body3'] = body3

        # TALKING FEATURES
        out_feats['person1_is_talking'] = (who_is_talking == 1)
        out_feats['person2_is_talking'] = (who_is_talking == 2)
        out_feats['person3_is_talking'] = (who_is_talking == 3)


        
        t = rospy.Time.now().to_sec() 
        print('Time since last video time preprocessing', (t - video_times[0])-6)


        # 46980 is the input size
        # each person is of vector 90*(14*2 + 70*2 + 2 + 2 + 1 + 1)
        # this will be 90 of the following. pose, face, gaze, head, audio, time
        # out_feats['person1'] = 

        return out_feats
    
    def check_callback(self, msg):
        start_time = rospy.Time.now().to_sec()
        print("Service call recieved")
        num_bites = msg.num_bites
        time_since_last_bite = msg.time_since_last_bite
        time_since_start = msg.time_since_start
        
        # this is called every 3 seconds.
        if self.feeding_in_progress:
            # if feeding is in progress, we check a special rosparam that indicates "idle"
            is_percieving = rospy.get_param("/social_dining_study/timingPerceptionOn")
            if not is_percieving:
                print("We are not percieving right now. Service was likely called at the wrong time.")
                return CheckBiteTimingResponse(False)
            else:
                # let's start taking data again
                self.feeding_in_progress = False
                self.create_data_subs()


        # print('in check callback with size', len(self.data_buffer1), len(self.data_buffer2), len(self.data_buffer3))
        trigger = 0

        for buffer in self.data_buffers:
            if len(buffer) < 95: # a little higher than 6 seconds to deal with latency alignment issues
                print("Buffer is not full with 6 seconds of information yet...", len(buffer))
                return CheckBiteTimingResponse(False)

        # if len(self.face_buffer1) < 180 or len(self.face_buffer2) < 180 or len(self.face_buffer3) < 180:
            # print("Buffer is not full with 6 seconds of information yet...")
            # return CheckBiteTimingResponse(False)

            
        aligned_data = self.align_data()

        aligned_data['time_since_start'] = time_since_start
        aligned_data['time_since_last_bite'] = time_since_last_bite
        aligned_data['num_bites'] = num_bites

        for k, item in aligned_data.items():
            if type(item) == np.ndarray:
                print(k, item.shape)

        # call model
        person1 = np.concatenate([aligned_data['face1'][:,:,0],aligned_data['face1'][:,:,1], aligned_data['body1'][:,:,0], aligned_data['body1'][:,:,1], aligned_data['gaze1'], aligned_data['headpose1'], aligned_data['person1_is_talking'].reshape(-1,1), np.array(time_since_last_bite).repeat(90).reshape(-1,1) ], axis=1).flatten()
        person2 = np.concatenate([aligned_data['face2'][:,:,0],aligned_data['face2'][:,:,1], aligned_data['body2'][:,:,0], aligned_data['body2'][:,:,1], aligned_data['gaze2'], aligned_data['headpose2'], aligned_data['person2_is_talking'].reshape(-1,1), np.array(time_since_last_bite).repeat(90).reshape(-1,1) ], axis=1).flatten()
        person3 = np.concatenate([aligned_data['face3'][:,:,0],aligned_data['face3'][:,:,1], aligned_data['body3'][:,:,0], aligned_data['body3'][:,:,1], aligned_data['gaze3'], aligned_data['headpose3'], aligned_data['person3_is_talking'].reshape(-1,1), np.array(time_since_last_bite).repeat(90).reshape(-1,1) ], axis=1).flatten()

        # person2 = np.concatenate([aligned_data['face2'], aligned_data['body2'], aligned_data['gaze2'], aligned_data['headpose2'], aligned_data['person2_is_talking'].reshape(1,-1), np.array(time_since_last_bite).repeat(90).reshape(1,-1) ], axis=1).flatten()
        # person3 = np.concatenate([aligned_data['face3'], aligned_data['body3'], aligned_data['gaze3'], aligned_data['headpose3'], aligned_data['person3_is_talking'].reshape(1,-1), np.array(time_since_last_bite).repeat(90).reshape(1,-1) ], axis=1).flatten()

        print('person1', person1.shape)
        print('person2', person2.shape)
        print('person3', person3.shape)

        combined = np.concatenate([person1, person2, person3])
        print('combined', combined.shape)

        out = self.model.predict(combined.reshape(1, -1))


        print('out', out)

        t = rospy.Time.now().to_sec() 
        print('Processing time in callback', t - start_time) # takes about 2.5 seconds
        trigger = 1

        if trigger == 1:
            self.last_bite_time = rospy.Time.now()

            self.feeding_in_progress = True
            rospy.set_param("/social_dining_study/timingPerceptionOn", False)

            # delete all the subs. let us guarantee a call to this after feeding is done to reinitialize
            self.delete_data_subs()

            return CheckBiteTimingResponse(True)
        return CheckBiteTimingResponse(False)

    ########################################################################################################################
    ### Mouth open service callback
    ########################################################################################################################

    def mouth_open_callback(self, msg):
        # check if mouth is open using the front camera
        print("Mouth open service called")

        # wait for message in camera3
        img = rospy.wait_for_message("/camera1/" + self.color_topic, CompressedImage)

        # pass into openpose
        frame = self.openposes[0].processOpenPose(img)
        print(frame)

        out = MouthOpenResponse(False)

        mouth_open = False
        mouth_points = []
        if len(frame.persons) > 0:
            if len(frame.persons[0].face) > 0:
                mouth_points.append(frame.persons[0].face[50].point)
                mouth_points.append(frame.persons[0].face[51].point)
                mouth_points.append(frame.persons[0].face[52].point)
                mouth_points.append(frame.persons[0].face[56].point)
                mouth_points.append(frame.persons[0].face[57].point)
                mouth_points.append(frame.persons[0].face[58].point)
                mouth_points.append(frame.persons[0].face[61].point)
                mouth_points.append(frame.persons[0].face[62].point)
                mouth_points.append(frame.persons[0].face[63].point)
                mouth_points.append(frame.persons[0].face[65].point)
                mouth_points.append(frame.persons[0].face[66].point)
                mouth_points.append(frame.persons[0].face[67].point)

        if len(mouth_points) > 0:

            lipDist = np.sqrt(pow((mouth_points[10].x - mouth_points[7].x), 2) + \
                                        pow((mouth_points[10].y - mouth_points[7].y), 2))

            lipThickness = np.sqrt(pow((mouth_points[1].x - mouth_points[7].x), 2) + \
                            pow((mouth_points[1].y - mouth_points[7].y), 2)) / 2.0 + \
                np.sqrt(pow((mouth_points[4].x - mouth_points[10].x), 2) + \
                    pow((mouth_points[4].y - mouth_points[10].y), 2)) / 2

            if (lipDist >= 1.0 * lipThickness):
                mouthOpen = True
            else:
                mouthOpen = False
        
        else:
            mouthOpen = False

        return mouthOpen
        
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
        rop = BiteTrigger(frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic)
        rospy.spin()

    except Exception as e:
        rospy.logerr(e)
        sys.exit(-1)


if __name__ == "__main__":
    main()
