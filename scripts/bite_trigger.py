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

        # self.openposes = []
        # for i in range(3):
        #     self.openposes.append(OpenPose(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage), self.frame_id, face=False))
        
        # self.face_openposes = []
        # for i in range(3):
        #     self.face_openposes.append(OpenPose(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage), self.frame_id, face=True))


        self.openpose = OpenPose(rospy.wait_for_message("/camera1/" + depth_topic, CompressedImage), self.frame_id)
        # self.openposes = [self.openpose, self.openpose, self.openpose]

        # NOTE: UNCOMMENT THE ONE BELOW IF I DON'T WANT TO USE BODY DETECTION (like for target participant!)
        self.face_openpose = OpenPose(rospy.wait_for_message("/camera1/" + depth_topic, CompressedImage), self.frame_id, face=True, body=False)
        # self.face_openposes = [self.face_openpose, self.face_openpose, self.face_openpose]
        self.openposes = [self.face_openpose, self.openpose, self.openpose] # 

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

        self.gaze_buffers = [collections.deque(maxlen=180) for _ in range(3)] # 90 is 6 seconds


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
        # self.test_pub = rospy.Publisher('/biteTiming/test', String, queue_size=10)
        self.test_pub1 = rospy.Publisher('/camera1/openpose', String, queue_size=10000)
        # self.test_pub2 = rospy.Publisher('/camera1/gaze', String, queue_size=10000)

        self.last_process_time = 0

        # a lock to prevent race conditions

        self.mutex = Lock()

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
        frame = self.openposes[num_callback].processOpenPose(img)
        all_data = {'header':img.header, 'image':img, 'openpose':frame}
        with self.mutex:
            self.data_buffers[num_callback].append({'time':recieved_time, 'data':all_data})

        if num_callback == 0:
            # for debugging

            self.test_pub1.publish("bleh")        
            finish = rospy.Time.now().to_sec()
            # print('pose processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish
        


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
        # frame = self.face_openposes[num_callback].processOpenPose(img, depth)
        # attach all relevant features here


        all_data = {'header':msg.header, 'gaze':msg.gaze, 'headpose':msg.headpose}
        # add to buffer
        with self.mutex:
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
            # then look for where max_video_seq occurs in gaze_seqs
            gaze_idxs = np.where(gaze_seqs == max_video_seq)[0]
            video_idxs = np.arange(len(video_seqs))
        elif max_video_seq > max_gaze_seq:
            # then look for where max_gaze_seq occurs in video_seqs
            video_idxs = np.where(video_seqs < max_gaze_seq)[0]
            # thus gaze idxs are everything else
            gaze_idxs = np.arange(len(gaze_seqs))
        else:
            # then they are equal
            video_idxs = np.arange(len(video_seqs))
            gaze_idxs = np.arange(len(gaze_seqs))

        # now we take the last 90 frames of each
        video_idxs = video_idxs[-90:]
        gaze_idxs = gaze_idxs[-90:]

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
        print(len(is_talking), len(direction_data))
        print(len(dir_mapping))
        print(len(audio_mapping))

        who_is_talking = self.who_is_talking(is_talking, direction_data)

        print(direction_data)
        print(is_talking)
        print(who_is_talking)

        t = rospy.Time.now().to_sec() 
        print((t - video_times[0])-6)



        # convert rospy times to regular floats

    
    def check_callback(self, msg):
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
            if len(buffer) < 100: # a little higher than 6 seconds to deal with latency alignment issues
                print("Buffer is not full with 6 seconds of information yet...", len(buffer))
                return CheckBiteTimingResponse(False)

        # if len(self.face_buffer1) < 180 or len(self.face_buffer2) < 180 or len(self.face_buffer3) < 180:
            # print("Buffer is not full with 6 seconds of information yet...")
            # return CheckBiteTimingResponse(False)

            
        aligned_data = self.align_data()

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
