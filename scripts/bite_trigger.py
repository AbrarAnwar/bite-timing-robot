#!/usr/bin/env python

# import modules
import sys
import cv2
import rospy
import argparse
import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from bite_timing_robot.msg import Frame, Person, BodyPart, Pixel, AudioData
from sensor_msgs.msg import Image, CameraInfo
import os

# import multiprocessing
from pathos.pools import ProcessPool

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Int32
# from audio_common_msgs.msg import AudioData

import collections

from sklearn.neighbors import NearestNeighbors


# Openpose stuff here
py_openpose_path = os.path.expanduser('/home/abrar/openpose/build/python')

try:
    # If you run `make install` (default path is `/usr/local/python` for Ubuntu)
    sys.path.append(py_openpose_path)
    from openpose import pyopenpose as op
except ImportError as e:
    rospy.logerr('OpenPose library could not be found. '
                'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

OPENPOSE1POINT7_OR_HIGHER = 'VectorDatum' in op.__dict__

class OpenPose:
    
    def __init__(self, depth_msg_example, frame_id):

        # Custom Params
        params = dict()
        # Can manually set params like this as well
        params["model_folder"] = "/home/abrar/openpose/models"
        params['number_people_max'] = 1
        params['tracking'] = 1
        params['render_pose'] = 0
        params['display'] = 0

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        self.op_wrapper = op_wrapper
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

        self.frame_id = frame_id
        self.no_depth = False

        self.bridge = CvBridge()

        self.frame = None

        encoding = depth_msg_example.format
        self.mm_to_m = 0.001 if "16UC1" in encoding else 1.

        # Function wrappers for OpenPose version discrepancies
        if OPENPOSE1POINT7_OR_HIGHER:
            self.emplaceAndPop = lambda datum: self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
            self.detect = lambda kp: kp is not None
        else:
            self.emplaceAndPop = lambda datum: self.op_wrapper.emplaceAndPop([datum])
            self.detect = lambda kp: kp.shape != ()

    def compute_3D_vectorized(self, kp, depth):
        # Create views (no copies made, so this remains efficient)
        U = kp[:, :, 0]
        V = kp[:, :, 1]

        # Extract the appropriate depth readings
        num_persons, body_part_count = U.shape
        XYZ = np.zeros((num_persons, body_part_count, 3), dtype=np.float32)
        for i in range(num_persons):
            for j in range(body_part_count):
                u, v = int(U[i, j]), int(V[i, j])
                if v < depth.shape[0] and u < depth.shape[1]:
                    XYZ[i, j, 2] = depth[v, u]

        XYZ[:, :, 2] *= self.mm_to_m  # convert to meters

        # Compute 3D coordinates in vectorized way
        Z = XYZ[:, :, 2]
        XYZ[:, :, 0] = (Z / self.fx) * (U - self.cx)
        XYZ[:, :, 1] = (Z / self.fy) * (V - self.cy)
        return XYZ

    def processOpenPose(self, ros_image, ros_depth):
        # Construct a frame with current time !before! pushing to OpenPose
        fr = Frame()
        fr.header.frame_id = self.frame_id

        # Convert images to cv2 matrices
        image = depth = None
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image)
            depth = self.bridge.compressed_imgmsg_to_cv2(ros_depth, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        # Push data to OpenPose and block while waiting for results
        datum = op.Datum()
        datum.cvInputData = image
        self.emplaceAndPop(datum)

        pose_kp = datum.poseKeypoints
        lhand_kp = datum.handKeypoints[0]
        rhand_kp = datum.handKeypoints[1]

        # Set number of people detected
        if self.detect(pose_kp):
            num_persons = pose_kp.shape[0]
            body_part_count = pose_kp.shape[1]
        else:
            num_persons = 0
            body_part_count = 0

        # Check to see if hands were detected
        lhand_detected = False
        rhand_detected = False
        hand_part_count = 0

        if self.detect(lhand_kp):
            lhand_detected = True
            hand_part_count = lhand_kp.shape[1]

        if self.detect(rhand_kp):
            rhand_detected = True
            hand_part_count = rhand_kp.shape[1]

        # Handle body points
        fr.persons = [Person() for _ in range(num_persons)]
        if num_persons != 0:
            # Perform vectorized 3D computation for body keypoints
            b_XYZ = self.compute_3D_vectorized(pose_kp, depth)

            # Perform the vectorized operation for left hand
            if lhand_detected:
                lh_XYZ = self.compute_3D_vectorized(lhand_kp, depth)

            # Do same for right hand
            if rhand_detected:
                rh_XYZ = self.compute_3D_vectorized(rhand_kp, depth)

            for person in range(num_persons):
                fr.persons[person].bodyParts = [BodyPart() for _ in range(body_part_count)]
                fr.persons[person].leftHandParts = [BodyPart() for _ in range(hand_part_count)]
                fr.persons[person].rightHandParts = [BodyPart() for _ in range(hand_part_count)]

                detected_hands = []
                if lhand_detected:
                    detected_hands.append((lhand_kp, fr.persons[person].leftHandParts, lh_XYZ))
                if rhand_detected:
                    detected_hands.append((rhand_kp, fr.persons[person].rightHandParts, rh_XYZ))

                # Process the body
                for bp in range(body_part_count):
                    u, v, s = pose_kp[person, bp]
                    x, y, z = b_XYZ[person, bp]
                    arr = fr.persons[person].bodyParts[bp]
                    arr.pixel.x = u
                    arr.pixel.y = v
                    arr.score = s
                    arr.point.x = x
                    arr.point.y = y
                    arr.point.z = z

                # Process left and right hands
                for kp, harr, h_XYZ in detected_hands:
                    for hp in range(hand_part_count):
                        u, v, s = kp[person, hp]
                        x, y, z = h_XYZ[person, hp]
                        arr = harr[hp]
                        arr.pixel.x = u
                        arr.pixel.y = v
                        arr.score = s
                        arr.point.x = x
                        arr.point.y = y
                        arr.point.z = z
        
        return fr


# Import Openpose (Ubuntu)
rospy.init_node('bite_trigger')

class BiteTrigger:
    def __init__(self, frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic):

        self.pub = rospy.Publisher(pub_topic, Frame, queue_size=10)

        self.frame_id = frame_id
        self.no_depth = no_depth

        self.bridge = CvBridge()

        self.frame = None

        print('creating audio sub')
        # subscribe to audio topic and register callback
        audio_sub = rospy.Subscriber('/audio', AudioData, self.audio_callback)

        self.data_buffer1 = collections.deque(maxlen=180) # 180 is 6 seconds
        self.data_buffer2 = collections.deque(maxlen=180) # 180 is 6 seconds
        self.data_buffer3 = collections.deque(maxlen=180) # 180 is 6 seconds

        self.audio_buffer = collections.deque(maxlen=180*4)
        self.direction_buffer = collections.deque(maxlen=180*4)


        data_subs1 = [message_filters.Subscriber("/camera1/" + color_topic, CompressedImage), message_filters.Subscriber('/camera1/' + depth_topic, CompressedImage)]
        self.data_sub1 = message_filters.ApproximateTimeSynchronizer(data_subs1, 5, .01)
        self.data_sub1.registerCallback(self.data1_callback)

        data_subs2 = [message_filters.Subscriber("/camera2/" + color_topic, CompressedImage), message_filters.Subscriber('/camera2/' + depth_topic, CompressedImage)]
        self.data_sub2 = message_filters.ApproximateTimeSynchronizer(data_subs2, 5, .01)
        self.data_sub2.registerCallback(self.data2_callback)

        data_subs3 = [message_filters.Subscriber("/camera3/" + color_topic, CompressedImage), message_filters.Subscriber('/camera3/' + depth_topic, CompressedImage)]
        self.data_sub3 = message_filters.ApproximateTimeSynchronizer(data_subs3, 5, .01)
        self.data_sub3.registerCallback(self.data3_callback)


        print('creating diretion sub')
        direction_sub = rospy.Subscriber('/sound_direction', Int32, self.direction_callback)



        self.last_bite_time = rospy.Time.now()


        self.check_sub = rospy.Subscriber('/biteTiming/shouldFeed', String, self.check_callback)

        self.trigger_pub = rospy.Publisher('/biteTiming/trigger', String, queue_size=10)

        self.openpose1 = OpenPose(rospy.wait_for_message("/camera1/" + depth_topic, CompressedImage), self.frame_id)
        self.openpose2 = OpenPose(rospy.wait_for_message("/camera1/" + depth_topic, CompressedImage), self.frame_id)


    def data1_callback(self, img, depth):
        recieved_time = rospy.Time.now()
        # attach all relevant features here
        all_data = {'image':img, 'depth':depth}
        # add to buffer
        
        self.data_buffer1.append({'time':recieved_time, 'data':all_data})

    def data2_callback(self, img, depth):
        recieved_time = rospy.Time.now()
        # attach all relevant features here
        all_data = {'image':img, 'depth':depth}
        # add to buffer
        self.data_buffer2.append({'time':recieved_time, 'data':all_data})


    def data3_callback(self, img, depth):
        recieved_time = rospy.Time.now()
        # attach all relevant features here
        all_data = {'image':img, 'depth':depth}
        # add to buffer
        self.data_buffer3.append({'time':recieved_time, 'data':all_data})

    def audio_callback(self, msg):
        recieved_time = rospy.Time.now()
        audio = msg.data

        # TODO: process frame here
        is_talking = 1


        # attach all relevant features here
        all_data = {'audio': audio, 'is_talking': is_talking}
        # add to buffer
        self.audio_buffer.append({'time':recieved_time, 'data':all_data})

    def direction_callback(self, msg):
        recieved_time = rospy.Time.now()

        direction = msg.data
        self.direction_buffer.append({'time':recieved_time, 'data':direction})

    def align_data(self):
        # we have 3 data inputs, direction, audio, and video
        # they are inside data_buffer, audio_buffer, and direction_buffer
        
        # these are a deque of size 180 of format (time, (data1, data2, ...))
        video_times = []
        video_data = []
        for item in self.data_buffer1: # using only buffer 1
            video_times.append(item['time'].to_sec())
            video_data.append(item['data'])

        audio_times = []
        audio_data = []
        for item in self.audio_buffer:
            audio_times.append(item['time'].to_sec())
            audio_data.append(item['data'])


        direction_times = []
        direction_data = []
        for item in self.direction_buffer:
            direction_times.append(item['time'].to_sec())
            direction_data.append(item['data'])

            
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(direction_times).reshape(-1,1))
        dists, idxs = nbrs.kneighbors(np.array(video_times).reshape(-1,1))
        dir_mapping = idxs[:, 0]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(audio_times).reshape(-1,1))
        dists, idxs = nbrs.kneighbors(np.array(video_times).reshape(-1,1))
        audio_mapping = idxs[:, 0]

        print(len(dir_mapping))
        print(len(audio_mapping))

        print(audio_mapping)

        prev = rospy.Time.now()


        for i in range(180):
            fr = self.openpose.processOpenPose(self.data_buffer1[i]['data']['image'],self.data_buffer1[i]['data']['depth'])
            fr = self.openpose.processOpenPose(self.data_buffer2[i]['data']['image'],self.data_buffer2[i]['data']['depth'])
            fr = self.openpose.processOpenPose(self.data_buffer3[i]['data']['image'],self.data_buffer3[i]['data']['depth'])

        # pool = multiprocessing.Pool(processes=4)
        # pool = ProcessPool(nodes=4)

        # inputs = [ (self.data_buffer1[i]['data']['image'],self.data_buffer1[i]['data']['depth']) for i in range(180)]
        # pool.map(self.openpose.processOpenPose, inputs)


        now = rospy.Time.now()
        print((now-prev).to_sec())


        # these are now mapped at 30fps. We should now compute openpose on all of these in a batch, so it's faster!




        # convert rospy times to regular floats
        exit()

    
    def check_callback(self, msg):
        print('in check callback with size', len(self.data_buffer1), len(self.data_buffer2), len(self.data_buffer3))
        trigger = 0

        if len(self.data_buffer1) < 180:
            return
        if len(self.data_buffer2) < 180:
            return
        if len(self.data_buffer3) < 180:
            return
            
        aligned_data = self.align_data()

        # TODO: Add check if we should run the model

        # Call model with what we have in our buffer
        trigger = evenly_spaced_trigger(timestamp)

        # Later, introduce switching behaviors depending on number of bites

        if trigger == 1:
            self.last_bite_time = timestamp
            self.trigger_pub.publish(String(""))

            # Clear the buffer because the model will be turned off during feeding stage
            self.data_buffer1.clear()
            self.data_buffer2.clear()
            self.data_buffer3.clear()

            self.audio_buffer.clear()
            self.direction_buffer.clear()


    # def callback(self, ros_image, openpose_frame):
    def callback(self, ros_image1, openpose_frame1, ros_image2, openpose_frame2, ros_image3, openpose_frame3):
        trigger = 0

        timestamp = ros_image1.header.stamp
        print(1/(rospy.Time.now() - ros_image1.header.stamp).to_sec(), 1/(rospy.Time.now() - ros_image2.header.stamp).to_sec(),1/(rospy.Time.now() - ros_image3.header.stamp).to_sec())

        # Buffer is_talking, and other input data!
        curr_data = (ros_image1, openpose_frame1, ros_image2, openpose_frame2, ros_image3, openpose_frame3)
        self.data_buffer.append((timestamp, curr_data))

        # We should only run if we have a full buffer, for the model's sake
        if len(self.data_buffer) < 180:
            return

        # TODO: Add check if we should run the model

        aligned_data = self.align_data()
        

        # Call model with what we have in our buffer
        trigger = evenly_spaced_trigger(timestamp)

        # Later, introduce switching behaviors depending on number of bites


        
        if trigger == 1:
            self.last_bite_time = timestamp
            self.trigger_pub.publish(String(""))

            # Clear the buffer because the model will be turned off during feeding stage
            self.data_buffer.clear()
            self.audio_buffer.clear()
            self.direction_buffer.clear()

    """
        Must output a 1 or a 0 for whether it should trigger or not
    """
    def evenly_spaced_trigger(current_time):
        print('Time since last bite', timestamp.to_sec() - self.last_bite_time.to_sec())

        if timestamp.to_sec() - self.last_bite_time.to_sec() >= 23.5:
            return 1
        return 0

    def paznet_trigger():
        # align the buffers using nearest neighbor, which produces input to the model



        # call the model given the input

        raise NotImplementedError

    def mouth_open_trigger():
        # this one will probably change a rosparam flag!
        raise NotImplementedError

        



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
