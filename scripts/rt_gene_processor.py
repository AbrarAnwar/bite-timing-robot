from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix

from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import numpy as np
import cv2

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
            # tqdm.write('Could not find faces in the image')
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




