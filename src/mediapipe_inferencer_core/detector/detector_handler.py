import mediapipe as mp
from mediapipe_inferencer_core.result_data import DetectorResults
import cv2
import time
import concurrent.futures
from mediapipe_inferencer_core.detector.landmark_detector import LandmarkDetector


class DetectorHandler:
    def __init__(self, pose: LandmarkDetector = None, hand: LandmarkDetector = None, face: LandmarkDetector = None):
        self.__pose = pose
        self.__hand = hand
        self.__face = face
        self.__detectors = [self.__pose, self.__hand, self.__face]
        self.latest_time_ms = 0

    def inference(self, image):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            return
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)
        # Run detections in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(detector.inference, mp_image, t_ms) for detector in self.__detectors if detector is not None]
            concurrent.futures.wait(futures)
        self.latest_time_ms = t_ms

    @property
    def results(self):
        return DetectorResults(
            self.__pose.results,
            self.__hand.results,
            self.__face.results
        )


def ResultVisualizer(image, results, isFlipped = True):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # TODO: Rewrite this to use MediaPipe.Task results
    mp_holistic = mp.solutions.holistic
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    # Flip the image horizontally for a selfie-view display.
    if isFlipped:
        cv2.imshow('MediaPipe Results', cv2.flip(image, 1))