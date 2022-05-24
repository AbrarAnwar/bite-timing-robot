#!/usr/bin/env python

# import modules
import sys
import cv2
import rospy
import argparse
import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from bite_timing_robot.msg import Frame, Person, BodyPart, Pixel
from sensor_msgs.msg import Image, CameraInfo
import os

from sensor_msgs.msg import CompressedImage

# Import Openpose (Ubuntu)
rospy.init_node('ros_openpose')
# py_openpose_path = rospy.get_param("~py_openpose_path")

# py_openpose_path = os.path.expanduser('/usr/local/python')
py_openpose_path = os.path.expanduser('/home/abrar/openpose/build/python')
# py_openpose_path = os.path.expanduser('/home/abrar/installs/openpose/build/python/')

try:
    # If you run `make install` (default path is `/usr/local/python` for Ubuntu)
    sys.path.append(py_openpose_path)
    from openpose import pyopenpose as op
except ImportError as e:
    rospy.logerr('OpenPose library could not be found. '
                 'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


OPENPOSE1POINT7_OR_HIGHER = 'VectorDatum' in op.__dict__


class rosOpenPose:
    def __init__(self, frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic, op_wrapper, display):

        # self.pub1 = rospy.Publisher('/camera1/' + pub_topic, Frame, queue_size=10)
        # self.pub2 = rospy.Publisher('/camera2/' + pub_topic, Frame, queue_size=10)
        # self.pub3 = rospy.Publisher('/camera3/' + pub_topic, Frame, queue_size=10)
        self.pub = rospy.Publisher(pub_topic, Frame, queue_size=10)
        self.synced_image_pub = rospy.Publisher('/time_synced/' + color_topic, CompressedImage, queue_size=30*6)

        # self.synced_image_pub1 = rospy.Publisher('/camera1/' + color_topic + '/time_synced', CompressedImage, queue_size=10)
        # self.synced_image_pub2 = rospy.Publisher('/camera2/' + color_topic + '/time_synced', CompressedImage, queue_size=10)
        # self.synced_image_pub3 = rospy.Publisher('/camera3/' + color_topic + '/time_synced', CompressedImage, queue_size=10)

        self.frame_id = frame_id
        self.no_depth = no_depth

        self.bridge = CvBridge()

        self.op_wrapper = op_wrapper

        self.display = display
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
        encoding = rospy.wait_for_message(depth_topic, CompressedImage).format
        self.mm_to_m = 0.001 if "16UC1" in encoding else 1.

        # Function wrappers for OpenPose version discrepancies
        if OPENPOSE1POINT7_OR_HIGHER:
            self.emplaceAndPop = lambda datum: self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
            # self.detect = lambda kp: kp is not None
            self.detect = lambda kp: kp.ndim > 1
        else:
            self.emplaceAndPop = lambda datum: self.op_wrapper.emplaceAndPop([datum])
            self.detect = lambda kp: kp.shape != ()

        # image_sub = message_filters.Subscriber(color_topic, CompressedImage)
        # depth_sub = message_filters.Subscriber(depth_topic, CompressedImage)
        self.face = True

        # subs = [message_filters.Subscriber('/camera1/' + color_topic, CompressedImage),
        #         message_filters.Subscriber('/camera2/' + color_topic, CompressedImage),
        #         message_filters.Subscriber('/camera3/' + color_topic, CompressedImage),
        #         message_filters.Subscriber('/camera1/' + depth_topic, CompressedImage),
        #         message_filters.Subscriber('/camera2/' + depth_topic, CompressedImage),
        #         message_filters.Subscriber('/camera3/' + depth_topic, CompressedImage)]
        print(color_topic, depth_topic)
        subs = [message_filters.Subscriber(color_topic, CompressedImage),
                message_filters.Subscriber(depth_topic, CompressedImage)]
        print('registered')
        self.ts = message_filters.ApproximateTimeSynchronizer(subs, 1, .1)
        self.ts.registerCallback(self.callback)

        """ OpenPose skeleton dictionary
        {0, "Nose"}, {13, "LKnee"}
        {1, "Neck"}, {14, "LAnkle"}
        {2, "RShoulder"}, {15, "REye"}
        {3, "RElbow"}, {16, "LEye"}
        {4, "RWrist"}, {17, "REar"}
        {5, "LShoulder"}, {18, "LEar"}
        {6, "LElbow"}, {19, "LBigToe"}
        {7, "LWrist"}, {20, "LSmallToe"}
        {8, "MidHip"}, {21, "LHeel"}
        {9, "RHip"}, {22, "RBigToe"}
        {10, "RKnee"}, {23, "RSmallToe"}
        {11, "RAnkle"}, {24, "RHeel"}
        {12, "LHip"}, {25, "Background"}
        """

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
                if v < depth.shape[0] and u < depth.shape[1] and v > 0 and u > 0:
                    XYZ[i, j, 2] = depth[v, u]

        XYZ[:, :, 2] *= self.mm_to_m  # convert to meters

        # Compute 3D coordinates in vectorized way
        Z = XYZ[:, :, 2]
        XYZ[:, :, 0] = (Z / self.fx) * (U - self.cx)
        XYZ[:, :, 1] = (Z / self.fy) * (V - self.cy)
        return XYZ

    def processOpenPose(self, ros_image, ros_depth, timestamp, pub, synced_image_pub):
        # Construct a frame with current time !before! pushing to OpenPose
        fr = Frame()
        fr.header.frame_id = self.frame_id
        fr.header.stamp = timestamp

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

        if self.face:
            face_kp = datum.getFaceKeypoints()
            if self.detect(face_kp):
                num_persons = face_kp.shape[0]
                face_part_count = face_kp.shape[1]
            else:
                num_persons = 0
                body_part_count = 0
            
            fr.persons = [Person() for _ in range(num_persons)]

            if num_persons != 0:
                f_XYZ = self.compute_3D_vectorized(face_kp, depth)
                # Process face
                for person in range(num_persons):
                    fr.persons[person].bodyParts = [BodyPart() for _ in range(0)]
                    fr.persons[person].leftHandParts = [BodyPart() for _ in range(0)]
                    fr.persons[person].rightHandParts = [BodyPart() for _ in range(0)]
                    fr.persons[person].face = [BodyPart() for _ in range(face_part_count)]
                
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
            print('bleh', face_kp)

        else:

            # pose_kp = datum.poseKeypoints
            # lhand_kp = datum.handKeypoints[0]
            # rhand_kp = datum.handKeypoints[1]
            pose_kp = datum.getPoseKeypoints()
            lhand_kp = datum.getLeftHandKeypoints()
            rhand_kp = datum.getRightHandKeypoints()

            # print(pose_kp)
            # Set number of people detected
            if self.detect(pose_kp):
                num_persons = pose_kp.shape[0]
                body_part_count = pose_kp.shape[1]
            else:
                num_persons = 0
                body_part_count = 0
            face_part_count = 0 # always zero for this case

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
                    fr.persons[person].face = [BodyPart() for _ in range(face_part_count)]

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

        pub.publish(fr)
        
        # change ros_image timestamp to match the frame timestamp
        ros_image.header.stamp = timestamp
        synced_image_pub.publish(ros_image)
        return fr


    def callback(self, ros_image, ros_depth):
        # make sure we should be publishing in the first place!
        try:
            feature_processing_on = rospy.get_param("/feeding/biteTiming/featureProcessingOn")
        except:
            feature_processing_on = False
            print("/feeding/biteTiming/featureProcessingOn rosparam not set yet...")
        if not feature_processing_on:
            return 

        timestamp = rospy.Time.now()
        # timestamp = ros_image.header.stamp

        # Process OpenPose
        fr = self.processOpenPose(ros_image, ros_depth, timestamp, self.pub, self.synced_image_pub)

        # print(1/(rospy.Time.now()-timestamp).to_sec())


    # def callback(self, ros_image1, ros_image2, ros_image3, ros_depth1, ros_depth2, ros_depth3):
    #     # make sure we should be publishing in the first place!
    #     try:
    #         feature_processing_on = rospy.get_param("/feeding/biteTiming/featureProcessingOn")
    #     except:
    #         feature_processing_on = False
    #         print("/feeding/biteTiming/featureProcessingOn rosparam not set yet...")
    #     if not feature_processing_on:
    #         return 

    #     timestamp = rospy.Time.now()
    #     # Process OpenPose
    #     fr = self.processOpenPose(ros_image1, ros_depth1, timestamp, self.pub1, self.synced_image_pub1)
    #     fr = self.processOpenPose(ros_image2, ros_depth2, timestamp, self.pub2, self.synced_image_pub2)
    #     fr = self.processOpenPose(ros_image3, ros_depth3, timestamp, self.pub3, self.synced_image_pub3)



def main():
    # frame_id = rospy.get_param("~frame_id")
    # no_depth = rospy.get_param("~no_depth")
    # pub_topic = rospy.get_param("~pub_topic")
    # color_topic = rospy.get_param("~color_topic")
    # depth_topic = rospy.get_param("~depth_topic")
    # cam_info_topic = rospy.get_param("~cam_info_topic")

    frame_id = 'camera_color_optical_frame'
    no_depth = False
    pub_topic = 'openpose'
    color_topic = "color/image_raw/compressed"
    depth_topic = "aligned_depth_to_color/image_raw/compressed"
    cam_info_topic = "/camera1/color/camera_info"

    try:
        # Flags, refer to include/openpose/flags.hpp for more parameters
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Custom Params
        params = dict()
        # Can manually set params like this as well
        params["model_folder"] = "/home/abrar/openpose/models"
        params['number_people_max'] = 1
        params['tracking'] = 1
        params['render_pose'] = 1
        params['display'] = 0
        params['net_resolution'] = "-1x256"
        # params['hand'] = True
        # params['hand_net_resolution'] = "320x320"


        # if hand:
        # params['body'] = 1
        # params['hand'] = True
        # params['hand_detector'] = 0

        # # if face:
        params['face'] = 1
        params['face_net_resolution'] = "320x320"
        params['face_detector'] = 1
        params['body'] = 0


        # Any more obscure flags can be found through this for loop
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        display = True if 'display' not in params or int(params['display']) > 0 else False

        # Start ros wrapper
        # rop = rosOpenPose(frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic, op_wrapper, display)
        rop1 = rosOpenPose('camera1/'+frame_id, no_depth, 'camera1/'+pub_topic, 'camera1/'+color_topic, 'camera1/'+depth_topic, cam_info_topic, op_wrapper, display)
        rop2 = rosOpenPose('camera2/'+frame_id, no_depth, 'camera2/'+pub_topic, 'camera2/'+color_topic, 'camera2/'+depth_topic, cam_info_topic, op_wrapper, display)
        rop3 = rosOpenPose('camera3/'+frame_id, no_depth, 'camera3/'+pub_topic, 'camera3/'+color_topic, 'camera3/'+depth_topic, cam_info_topic, op_wrapper, display)



        # if display:
        #     while not rospy.is_shutdown():
        #         if rop.frame is not None:
        #             cv2.imshow("Ros OpenPose", rop.frame)
        #             cv2.waitKey(1)
        # else:
        rospy.spin()

    except Exception as e:
        rospy.logerr(e)
        sys.exit(-1)


if __name__ == "__main__":
    main()
