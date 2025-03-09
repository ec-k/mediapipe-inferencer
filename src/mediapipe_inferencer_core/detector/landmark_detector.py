from mediapipe.tasks import python as mp_tasks
from abc import ABC, abstractproperty, abstractclassmethod

class LandmarkDetector(ABC):
    @abstractclassmethod
    def inference(self, image, frame_timestamp_ms):
        pass
    @abstractproperty
    def results(self):
        pass

class PoseDetector(LandmarkDetector):
    def __init__(self, model_path, running_mode = mp_tasks.vision.RunningMode.LIVE_STREAM):
        options = None
        if running_mode == mp_tasks.vision.RunningMode.LIVE_STREAM:
            options = mp_tasks.vision.PoseLandmarkerOptions(
                base_options = mp_tasks.BaseOptions(model_asset_buffer = open(model_path, "rb").read()),
                running_mode = running_mode,
                result_callback = self.__save_results)
        else:
            options = mp_tasks.vision.PoseLandmarkerOptions(
                base_options = mp_tasks.BaseOptions(model_asset_buffer = open(model_path, "rb").read()),
                running_mode = running_mode)
        self._running_mode = running_mode
        self.landmarker = mp_tasks.vision.PoseLandmarker.create_from_options(options)
        self.__results = None

    @property
    def results(self):
        return self.__results

    def __save_results(self, results, output_image, timesamp_ms: int):
        self.__results = results

    def inference(self, image, frame_timestamp_ms):
        if self._running_mode == mp_tasks.vision.RunningMode.LIVE_STREAM:
            self.landmarker.detect_async(image, frame_timestamp_ms)
        elif self._running_mode == mp_tasks.vision.RunningMode.VIDEO:
            self.__results = self.landmarker.detect_for_video(image, frame_timestamp_ms)


class HandDetector(LandmarkDetector):
    def __init__(self, model_path, running_mode = mp_tasks.vision.RunningMode.LIVE_STREAM):
        options = None
        if running_mode == mp_tasks.vision.RunningMode.LIVE_STREAM:
            options = mp_tasks.vision.HandLandmarkerOptions(
                base_options = mp_tasks.BaseOptions(model_asset_buffer = open(model_path, "rb").read()),
                num_hands = 2,
                running_mode = running_mode,
                result_callback=self.__save_results)
        else:
            options = mp_tasks.vision.HandLandmarkerOptions(
                base_options = mp_tasks.BaseOptions(model_asset_buffer = open(model_path, "rb").read()),
                num_hands = 2,
                running_mode = running_mode)
        self._running_mode = running_mode
        self.landmarker = mp_tasks.vision.HandLandmarker.create_from_options(options)
        self.__results = None

    @property
    def results(self):
        return self.__results

    def __save_results(self, results, output_image, timesamp_ms: int):
        self.__results = results

    def inference(self, image, frame_timestamp_ms):
        if self._running_mode == mp_tasks.vision.RunningMode.LIVE_STREAM:
            self.landmarker.detect_async(image, frame_timestamp_ms)
        elif self._running_mode == mp_tasks.vision.RunningMode.VIDEO:
            self.__results = self.landmarker.detect_for_video(image, frame_timestamp_ms)


class FaceDetector(LandmarkDetector):
    def __init__(self, model_path, running_mode = mp_tasks.vision.RunningMode.LIVE_STREAM):
        options = None
        if running_mode == mp_tasks.vision.RunningMode.LIVE_STREAM:
            options = mp_tasks.vision.FaceLandmarkerOptions(
                base_options = mp_tasks.BaseOptions(model_asset_buffer = open(model_path, "rb").read()),
                output_face_blendshapes = True,
                output_facial_transformation_matrixes = True,
                num_faces = 1,
                running_mode = running_mode,
                result_callback=self.__save_results)
        else:
            options = mp_tasks.vision.FaceLandmarkerOptions(
                base_options = mp_tasks.BaseOptions(model_asset_buffer = open(model_path, "rb").read()),
                output_face_blendshapes = True,
                output_facial_transformation_matrixes = True,
                num_faces = 1,
                running_mode = running_mode)
        self._running_mode = running_mode
        self.landmarker = mp_tasks.vision.FaceLandmarker.create_from_options(options)
        self.__results = None

    @property
    def results(self):
        return self.__results

    def __save_results(self, results, output_image, timesamp_ms: int):
        self.__results = results

    def inference(self, image, frame_timestamp_ms):
        if self._running_mode == mp_tasks.vision.RunningMode.LIVE_STREAM:
            self.landmarker.detect_async(image, frame_timestamp_ms)
        elif self._running_mode == mp_tasks.vision.RunningMode.VIDEO:
            self.__results = self.landmarker.detect_for_video(image, frame_timestamp_ms)

