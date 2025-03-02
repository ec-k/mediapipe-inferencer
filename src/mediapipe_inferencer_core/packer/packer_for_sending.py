import proto.holistic_landmarks_pb2 as holistic_lm_pb2
from mediapipe_inferencer_core.data_class.result_data import HolisticResults, LandmarkResult, HandResult, FaceResult
from mediapipe_inferencer_core.data_class.landmark import Landmark

def pack_holistic_landmarks_result(holistic_results: HolisticResults):
    holistic_lm = holistic_lm_pb2.HolisticLandmarks()
    pose = holistic_results.pose
    if (pose is not None) and (pose.world is not None) and (len(pose.world)>0):
        holistic_lm.poseLandmarks.landmarks.extend(pack_landmarks(pose.world))

    left_hand, right_hand = holistic_results.hand.left, holistic_results.hand.right
    if (left_hand is not None) and (left_hand.world is not None) and len(left_hand.world)>0:
        holistic_lm.leftHandLandmarks.landmarks.extend(pack_landmarks(left_hand.world))
    if (right_hand is not None) and (right_hand.world is not None) and len(right_hand.world)>0:
        holistic_lm.rightHandLandmarks.landmarks.extend(pack_landmarks(right_hand.world))

    face = holistic_results.face
    if face is not None:
        if (face.landmarks is not None) and len(face.landmarks)>0:
            holistic_lm.faceResults.landmarks.extend(pack_landmarks(face.landmarks))
        if (face.blendshapes is not None) and len(face.blendshapes)>0:
            holistic_lm.faceResults.blendshapes.scores.extend(pack_blendshapes(face.blendshapes))

    return holistic_lm

def pack_landmarks(pose_landmarks):
    if pose_landmarks is None:
        return []
    return [format_landmark(mp_lm) for mp_lm in pose_landmarks]

def pack_blendshapes(blendshapes: list[float]):
    if blendshapes is None:
        return []
    return [ bl for bl in blendshapes]

def format_landmark(landmark:Landmark):
    packed_landmark = holistic_lm_pb2.Landmark()
    packed_landmark.x = landmark.x
    packed_landmark.y = landmark.y
    packed_landmark.z = landmark.z
    packed_landmark.confidence = landmark.confidence
    return packed_landmark