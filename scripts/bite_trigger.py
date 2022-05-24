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
from bite_timing_robot.srv import CheckBiteTiming, CheckBiteTimingResponse
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


PERSON1 = -180 # +/- 180
PERSON2 = 60
PERSON3 = -60


from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix


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
    # NOTE: We remove depth, but the option to keep it as False is not implemented
    def __init__(self, depth_msg_example, frame_id, face=False, body=1, no_depth=True):

        # Custom Params
        params = dict()
        # Can manually set params like this as well
        params["model_folder"] = "/home/abrar/openpose/models"
        params['number_people_max'] = 1
        params['tracking'] = 1
        params['render_pose'] = 1
        params['display'] = 0
        params['net_resolution'] = "-1x256"
        # params['disable_multi_thread'] = True

        if face:
            params['face'] = 1
            params['face_net_resolution'] = "320x320"
            params['face_detector'] = 1
            params['body'] = 0

        params['face'] = 1
        params['face_net_resolution'] = "320x320"
        params['face_detector'] = 1
        params['body'] = 1
        params['disable_multi_thread'] = True


        if not body:
            params['face_detector'] = 1
            params['body'] = 0


        # Starting OpenPose
        # op_wrapper = op.WrapperPython(op.ThreadManagerMode.Asynchronous) # performance decrease bc of cpu bottleneck
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        self.op_wrapper = op_wrapper

        self.no_depth = no_depth

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

        self.bridge = CvBridge()

        self.frame = None

        self.face = face
        encoding = depth_msg_example.format
        self.mm_to_m = 0.001 if "16UC1" in encoding else 1.

        # Function wrappers for OpenPose version discrepancies
        if OPENPOSE1POINT7_OR_HIGHER:
            self.emplaceAndPop = lambda datum: self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
            self.detect = lambda kp: kp.ndim > 1
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

        if self.no_depth:
            U = kp[:, :, 0]
            V = kp[:, :, 1]
            XYZ[:, :, 0] = U
            XYZ[:, :, 1] = V
            # leave Z as 0
            return XYZ


        for i in range(num_persons):
            for j in range(body_part_count):
                u, v = int(U[i, j]), int(V[i, j])
                if v < depth.shape[0] and u < depth.shape[1] and u > 0 and v > 0:
                    XYZ[i, j, 2] = depth[v, u]

        XYZ[:, :, 2] *= self.mm_to_m  # convert to meters

        # Compute 3D coordinates in vectorized way
        Z = XYZ[:, :, 2]
        XYZ[:, :, 0] = (Z / self.fx) * (U - self.cx)
        XYZ[:, :, 1] = (Z / self.fy) * (V - self.cy)
        return XYZ

    # def processOpenPose(self, ros_image, ros_depth):
    def processOpenPose(self, ros_image, ros_depth=None): # removing depth for the moment!
        # Construct a frame with current time !before! pushing to OpenPose
        fr = Frame()
        fr.header.frame_id = self.frame_id

        # Convert images to cv2 matrices
        image = depth = None
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image)
            if not self.no_depth:
                depth = self.bridge.compressed_imgmsg_to_cv2(ros_depth, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        # Push data to OpenPose and block while waiting for results
        datum = op.Datum()
        datum.cvInputData = image
        self.emplaceAndPop(datum)

        # pose_kp = datum.poseKeypoints
        # lhand_kp = datum.handKeypoints[0]
        # rhand_kp = datum.handKeypoints[1]
        pose_kp = datum.getPoseKeypoints()
        lhand_kp = datum.getLeftHandKeypoints()
        rhand_kp = datum.getRightHandKeypoints()
        face_kp = datum.getFaceKeypoints()

        # print(pose_kp)
        # Set number of people detected
        if self.detect(pose_kp):
            num_persons = pose_kp.shape[0]
            body_part_count = pose_kp.shape[1]
        else:
            num_persons = 0
            body_part_count = 0
        
        if self.detect(face_kp):
            num_faces = face_kp.shape[0]
            face_part_count = face_kp.shape[1]
        else:
            num_faces = 0
            face_part_count = 0

        # Check to see if hands were detected
        lhand_detected = False
        rhand_detected = False
        hand_part_count = 0

        if self.detect(lhand_kp):
            lhand_detected = True
            # print(lhand_kp)
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
                fr.persons[person].face = [BodyPart() for _ in range(0)] # to begin with. changed below

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

            if num_faces != 0:
                for person in range(num_persons):
                    f_XYZ = self.compute_3D_vectorized(face_kp, depth)
                    fr.persons[person].face = [BodyPart() for _ in range(face_part_count)]

                    # Process face
                    for bp in range(face_part_count):
                        u, v, s = face_kp[person, bp]

                        x, y, z = f_XYZ[person, bp]
                        arr = fr.persons[person].face[bp]
                        arr.pixel.x = u
                        arr.pixel.y = v
                        arr.score = s
                        arr.point.x = x
                        arr.point.y = y
                        arr.point.z = z
    
        return fr


class RTGene:
    """
    This class is used to perform the real-time processing of the data gaze data.
    """

    def __init__(self, img_msg_example, model_nets_path='~/rt_gene/rt_gene/model_nets'):
        """
        Initialize the class.
        """
        self.bridge = CvBridge()

        self.model_nets_path = os.path.expanduser(model_nets_path)
        tqdm.write('Loading networks')
        self.landmark_estimator = LandmarkMethodBase(device_id_facedetection='cuda:0',
                                                checkpoint_path_face=os.path.join(self.model_nets_path, "SFD/s3fd_facedetector.pth"),
                                                checkpoint_path_landmark=os.path.join(self.model_nets_path, "phase1_wpdc_vdc.pth.tar"),
                                                model_points_file=os.path.join(self.model_nets_path, "face_model_68.txt"))
        print('loaded landmark_estimator')

        from rt_gene.estimate_gaze_pytorch import GazeEstimator
        print('Loading model', os.path.join(self.model_nets_path, 'gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'))
        self.gaze_estimator = GazeEstimator("cuda:0", [os.path.join(self.model_nets_path, 'gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model')])

        print('loaded gaze_estimator')

        # get camera params
        image = None
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(img_msg_example)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        im_width, im_height = image.shape[1], image.shape[0]
        tqdm.write('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
        self.dist_coefficients, self.camera_matrix = np.zeros((1, 5)), np.array(
            [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

    def process(self, ros_image):
        """
        Process the data.
        """

        # Convert images to cv2 matrices
        image = None
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


        gaze, headpose = self.estimate_gaze(image, self.dist_coefficients, self.camera_matrix)
        
        return gaze, headpose


    def load_camera_calibration(self, calibration_file):
        import yaml
        with open(calibration_file, 'r') as f:
            cal = yaml.safe_load(f)

        dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
        camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

        return dist_coefficients, camera_matrix


    def extract_eye_image_patches(self, subjects):
        for subject in subjects:
            le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, self.landmark_estimator.eye_image_size)
            subject.left_eye_color = le_c
            subject.right_eye_color = re_c


    def estimate_gaze(self, color_img, dist_coefficients, camera_matrix):
        faceboxes = self.landmark_estimator.get_face_bb(color_img)
        if len(faceboxes) == 0:
            tqdm.write('Could not find faces in the image')
            return None, None

        subjects = self.landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
        self.extract_eye_image_patches(subjects)

        input_r_list = []
        input_l_list = []
        input_head_list = []
        valid_subject_list = []

        for idx, subject in enumerate(subjects):
            if subject.left_eye_color is None or subject.right_eye_color is None:
                tqdm.write('Failed to extract eye image patches')
                continue

            success, rotation_vector, _ = cv2.solvePnP(self.landmark_estimator.model_points,
                                                    subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                    cameraMatrix=camera_matrix,
                                                    distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_DLS)

            if not success:
                tqdm.write('Not able to extract head pose for subject {}'.format(idx))
                continue

            _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
            _m = np.zeros((4, 4))
            _m[:3, :3] = _rotation_matrix
            _m[3, 3] = 1
            # Go from camera space to ROS space
            _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]
            roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
            roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

            phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)

            # commented out to save rendering compute
            # face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            # head_pose_image = self.landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))



            input_r_list.append(self.gaze_estimator.input_from_image(subject.right_eye_color))
            input_l_list.append(self.gaze_estimator.input_from_image(subject.left_eye_color))
            input_head_list.append([theta_head, phi_head])
            valid_subject_list.append(idx)

        if len(valid_subject_list) == 0:
            return None, None

        gaze_est = self.gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                        inference_input_right_list=input_r_list,
                                                        inference_headpose_list=input_head_list)

        # comment out visualization
        # for subject_id, gaze, headpose in zip(valid_subject_list, gaze_est.tolist(), input_head_list):
        #     subject = subjects[subject_id]
        #     # Build visualizations
        #     r_gaze_img = self.gaze_estimator.visualize_eye_result(subject.right_eye_color, gaze)
        #     l_gaze_img = self.gaze_estimator.visualize_eye_result(subject.left_eye_color, gaze)
        #     s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

        gazes = []
        headposes = []
        for subject_id, gaze, headpose in zip(valid_subject_list, gaze_est.tolist(), input_head_list):
            gazes.append([gaze[1], gaze[0]])
            headposes.append([headpose[1], headpose[0]])

        if len(gazes) == 0:
            return None, None

        return gaze_est[0], headposes[0]





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
        self.openposes = [self.openpose, self.openpose, self.openpose]

        # NOTE: UNCOMMENT THE ONE BELOW IF I DON'T WANT TO USE BODY DETECTION (like for target participant!)
        self.face_openpose = OpenPose(rospy.wait_for_message("/camera1/" + depth_topic, CompressedImage), self.frame_id, face=True, body=False)
        # self.face_openposes = [self.face_openpose, self.face_openpose, self.face_openpose]
        self.openposes = [self.openpose, self.openpose, self.face_openpose] # 

        self.rt_gene = RTGene(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage))
        self.rt_genes = [self.rt_gene, self.rt_gene, self.rt_gene]
        # self.rt_genes = [RTGene(rospy.wait_for_message("/camera1/" + color_topic, CompressedImage)) for _ in range(3)] # 16-17 fps for all 3 at the same time

        self.last_seq_1 = 0
        self.dropped_msgs_1 = 0
        self.last_seq_2 = 0
        self.dropped_msgs_2 = 0
        # subscribe to audio topic and register callback

        self.data_buffers = [collections.deque(maxlen=180) for _ in range(3)] # 180 is 6 seconds

        self.face_buffers = [collections.deque(maxlen=180) for _ in range(3)] # 180 is 6 seconds

        self.gaze_buffers = [collections.deque(maxlen=180) for _ in range(3)] # 180 is 6 seconds


        self.audio_buffer = collections.deque(maxlen=180*4)
        self.direction_buffer = collections.deque(maxlen=180*4)


        self.color_topic = color_topic
        self.depth_topic = depth_topic

        self.last_bite_time = rospy.Time.now()


        # self.check_sub = rospy.Subscriber('/biteTiming/shouldFeed', String, self.check_callback)
        # rospy.Timer(rospy.Duration(3), self.check_callback)
        print('registering service')
        self.check_service = rospy.Service("/check_bite_timing", CheckBiteTiming, self.check_callback)


        self.feeding_in_progress = True
        self.vad = webrtcvad.Vad(3)
        # test pub publishes a string
        # self.test_pub = rospy.Publisher('/biteTiming/test', String, queue_size=10)
        self.test_pub1 = rospy.Publisher('/camera1/openpose', String, queue_size=10000)
        self.test_pub2 = rospy.Publisher('/camera1/gaze', String, queue_size=10000)

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
        for i in range(gaze_num_threads):
            self.data_sub1.registerCallback(lambda img, i=i: self.gaze_callback(img, 0, i, gaze_num_threads))

        # self.subs2 = [message_filters.Subscriber("/camera2/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size), message_filters.Subscriber('/camera2/' + self.depth_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.subs2 = [message_filters.Subscriber("/camera2/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.data_sub2 = message_filters.ApproximateTimeSynchronizer(self.subs2, queue_size, .1)
        for i in range(num_threads):
            self.data_sub2.registerCallback(lambda img, i=i: self.data_callback(img, 1, i, num_threads))
        for i in range(gaze_num_threads):
            self.data_sub2.registerCallback(lambda img, i=i: self.gaze_callback(img, 1, i, gaze_num_threads))

        # self.subs3 = [message_filters.Subscriber("/camera3/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size), message_filters.Subscriber('/camera3/' + self.depth_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]
        self.subs3 = [message_filters.Subscriber("/camera3/" + self.color_topic, CompressedImage, queue_size=queue_size, buff_size=buff_size)]

        self.data_sub3 = message_filters.ApproximateTimeSynchronizer(self.subs3, queue_size, .1)
        for i in range(num_threads):
            self.data_sub3.registerCallback(lambda img, i=i: self.data_callback(img, 2, i, num_threads))
        for i in range(gaze_num_threads):
            self.data_sub3.registerCallback(lambda img, i=i: self.gaze_callback(img, 2, i, gaze_num_threads))


        self.audio_sub = rospy.Subscriber('/audio', AudioData, self.audio_callback)
        self.direction_sub = rospy.Subscriber('/sound_direction', Int32, self.direction_callback)

        print('Data subs created')

        self.last_process_time = 0



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

            self.audio_sub.unregister()
            self.direction_sub.unregister()
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

    ########################################################################################################################
    ### Openpose and image callbacks
    ########################################################################################################################
    def data_callback(self, img, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            return # drop every other frame!

        if not(img.header.seq % num_threads == thread_idx):
            return

        if num_callback == 0:
            if self.last_seq_1+ 2 != img.header.seq:
                self.dropped_msgs_1 += 1
                print("Openpose Processing Dropped msg: ", self.dropped_msgs_1, " seq: ", img.header.seq, " last seq: ", self.last_seq_1)
            self.last_seq_1 = img.header.seq


        recieved_time = rospy.Time.now()
        frame = self.openposes[num_callback].processOpenPose(img)
        all_data = {'image':img, 'openpose':frame}
        self.data_buffers[num_callback].append({'time':recieved_time, 'data':all_data})

        # ideally gaze is computed in its own callback, but cpu restrictions make it difficult
        # gaze, headpose = self.rt_genes[num_callback].process(img)
        # all_data = {'gaze':gaze, 'headpose':headpose}
        # self.gaze_buffers[num_callback].append({'time':recieved_time, 'data':all_data})

        if num_callback == 0:
            # for debugging

            self.test_pub1.publish("bleh")        
            finish = rospy.Time.now().to_sec()
            print('pose processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish
        


    ########################################################################################################################
    ### face data callbacks
    ########################################################################################################################
    def face_callback(self, img, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            return # drop every other frame!

        if not(img.header.seq % num_threads == thread_idx):
            return
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
        self.face_buffers[num_callback].append({'time':recieved_time, 'data':all_data})

        # look at callback 0 to verify speed
        if num_callback == 0:
            # self.test_pub.publish("bleh")        
            finish = rospy.Time.now().to_sec()
            print('face processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
            self.last_process_time = finish

    ########################################################################################################################
    ### gaze data callbacks
    ########################################################################################################################
    
    def gaze_callback(self, img, num_callback, thread_idx, num_threads):
        if img.header.seq % 2 == 0:
            return # drop every other frame!

        if not(img.header.seq % num_threads == thread_idx):
            return
        # check only on one thread
        if num_callback == 0:

            if self.last_seq_2 + 2 != img.header.seq:
                self.dropped_msgs_2 += 1
                print("gaze Processing Dropped msg: ",self.dropped_msgs_2, " seq: ", img.header.seq, " last seq: ", self.last_seq_2)
                print(num_callback, thread_idx, num_threads)
                print(img.header.seq, img.header.seq % num_threads)
            self.last_seq_2 = img.header.seq


        recieved_time = rospy.Time.now()
        # frame = self.face_openposes[num_callback].processOpenPose(img, depth)
        # attach all relevant features here
        gaze, headpose = self.rt_genes[num_callback].process(img)

        all_data = {'gaze':gaze, 'headpose':headpose}
        # add to buffer
        self.gaze_buffers[num_callback].append({'time':recieved_time, 'data':all_data})

        # look at callback 0 to verify speed
        if num_callback == 0:
            self.test_pub2.publish("bleh")        
            finish = rospy.Time.now().to_sec()
            print('gaze processing time: \t', img.header.seq, 1/(finish-self.last_process_time))
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
        self.audio_buffer.append({'time':recieved_time, 'data':all_data})

    def direction_callback(self, msg):
        recieved_time = rospy.Time.now()

        direction = msg.data
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
        
        # these are a deque of size 180 of format (time, (data1, data2, ...))
        video_times = []
        video_data = []
        for item in self.data_buffers[0]: # using only buffer 1
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
            if len(buffer) < 180:
                print("Buffer is not full with 6 seconds of information yet...", len(buffer))
                return CheckBiteTimingResponse(False)

        # if len(self.face_buffer1) < 180 or len(self.face_buffer2) < 180 or len(self.face_buffer3) < 180:
            # print("Buffer is not full with 6 seconds of information yet...")
            # return CheckBiteTimingResponse(False)

            
        aligned_data = self.align_data()

        # Call model with what we have in our buffer
        # trigger = evenly_spaced_trigger(timestamp)
        trigger = 1
        # Later, introduce switching behaviors depending on number of bites

        if trigger == 1:
            self.last_bite_time = rospy.Time.now()

            self.feeding_in_progress = True
            rospy.set_param("/social_dining_study/timingPerceptionOn", False)

            # delete all the subs. let us guarantee a call to this after feeding is done to reinitialize
            self.delete_data_subs()

            return CheckBiteTimingResponse(True)
        return CheckBiteTimingResponse(False)

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
