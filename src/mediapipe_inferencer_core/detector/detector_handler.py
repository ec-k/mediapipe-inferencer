import mediapipe as mp
import cv2
import time
from mediapipe_inferencer_core.data_class.result_data import HolisticResults
from mediapipe_inferencer_core.detector.landmark_detector import LandmarkDetector


class DetectorHandler:
    def __init__(self, pose: LandmarkDetector = None, hand: LandmarkDetector = None, face: LandmarkDetector = None):
        self.__pose = pose
        self.__hand = hand
        self.__face = face
        self.latest_time_ms = 0

    def inference(self, image):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            return
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)
        self.__pose.inference(mp_image, t_ms)
        self.__hand.inference(mp_image, t_ms)
        self.__face.inference(mp_image, t_ms)
        self.latest_time_ms = t_ms

    @property
    def results(self):
        return HolisticResults(
            self.__pose.results,
            self.__hand.results,
            self.__face.results
        )