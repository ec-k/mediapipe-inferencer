from mediapipe_inferencer_core.data_class.landmark import Landmark
from mediapipe_inferencer_core.packer import pack_to_landmark

class LandmarkResult:
    def __init__(self):
        self.__local = None
        self.__world = None
    @property
    def local(self) -> list[Landmark]:
        return self.__local
    @local.setter
    def local(self, result) -> None:
        self.__local = result
    @property
    def world(self) -> list[Landmark]:
        return self.__world
    @world.setter
    def world(self, result) -> None:
        self.__world = result


class HandResult:
    def __init__(self):
        self.__left_hand = LandmarkResult()
        self.__right_hand = LandmarkResult()

    def update(self, raw_results):
        [left_hand_landmarks, right_hand_landmarks] = pack_to_landmark.extract_hand_landmarks(raw_results)
        if left_hand_landmarks is not None and len(left_hand_landmarks)>0:
            self.__left_hand.local, self.__left_hand.world = pack_to_landmark.pack_hand_landmarks(left_hand_landmarks)
        if right_hand_landmarks is not None and len(right_hand_landmarks)>0:
            self.__right_hand.local, self.__right_hand.world = pack_to_landmark.pack_hand_landmarks(right_hand_landmarks)

    @property
    def left(self) -> LandmarkResult:
        return self.__left_hand
    @property
    def right(self) -> LandmarkResult:
        return self.__right_hand


class FaceResult:
    def __init__(self):
        self.__landmarks = None
        self.__blendshapes = None

    def update(self, raw_results):
        face_results = raw_results
        if face_results is not None:
            if len(face_results.face_landmarks)>0:
                self.__landmarks = pack_to_landmark.pack_landmarks(face_results.face_landmarks[0])
            if len(face_results.face_blendshapes)>0:
                self.__blendshapes = pack_to_landmark.pack_blendshapes(face_results.face_blendshapes[0])

    @property
    def landmarks(self)->list[Landmark]:
        return self.__landmarks
    @landmarks.setter
    def landmarks(self, result:list[Landmark])->None:
        self.__landmarks = result
    @property
    def blendshapes(self)->list[float]:
        return self.__blendshapes


class HolisticResults:
    def __init__(self, raw_pose, raw_hand, raw_face, time_s):
        self.__pose_result = LandmarkResult()
        self.__hand_result = HandResult()
        self.__face_results = FaceResult()
        self.__time_s = time_s
        self.update(raw_pose, raw_hand, raw_face)

    def update(self, raw_pose, raw_hand, raw_face):
        if raw_pose is not None:
            if len(raw_pose.pose_landmarks)>0:
                self.__pose_result.local = pack_to_landmark.pack_landmarks(raw_pose.pose_landmarks[0])
            if len(raw_pose.pose_world_landmarks)>0:
                self.__pose_result.world = pack_to_landmark.pack_landmarks(raw_pose.pose_world_landmarks[0])
        if raw_hand is not None:
            self.__hand_result.update(raw_hand)
        if raw_face is not None:
            self.__face_results.update(raw_face)

    @property
    def pose(self)->LandmarkResult:
        return self.__pose_result
    @property
    def hand(self)->HandResult:
        return self.__hand_result
    @property
    def face(self)->FaceResult:
        return self.__face_results
    @property
    def time(self)->float:
        return self.__time_s