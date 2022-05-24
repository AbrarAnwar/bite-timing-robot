

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from bite_timing_robot.msg import Frame, Person, BodyPart, Pixel, AudioData
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Int32
import os
import sys
import numpy as np

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

        # params['face'] = 1
        # params['face_net_resolution'] = "320x320"
        # params['face_detector'] = 1
        # params['body'] = 1
        params['disable_multi_thread'] = True


        # if not body:
        #     params['face_detector'] = 1
        #     params['body'] = 0


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
