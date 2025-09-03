import mediapipe as mp
import time
from mediapipe_inferencer_core.data_class.result_data import HolisticResults
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
        for detector in self.__detectors:
            if detector is not None:
                detector.inference(mp_image, t_ms)
        self.latest_time_ms = t_ms

    @property
    def results(self):
        pose_result = self.__pose.results if self.__pose is not None else None
        hand_result = self.__hand.results if self.__hand is not None else None
        face_result = self.__face.results if self.__face is not None else None
        return HolisticResults(
            pose_result,
            hand_result,
            face_result,
            self.latest_time_ms / 1000
        )