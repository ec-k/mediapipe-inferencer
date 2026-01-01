from mediapipe_inferencer_core.data_class import Landmark, LandmarkList
import math
import numpy as np

def format_landmark(raw_landmark) -> Landmark:
    return Landmark(raw_landmark.x, raw_landmark.y, raw_landmark.z, raw_landmark.visibility)

def format_landmark_with_confidence(position, confidence) -> Landmark:
    return Landmark(position.x, position.y, position.z, confidence)

def pack_landmarks(raw_landmarks) -> LandmarkList:
    if raw_landmarks is None:
        return LandmarkList([])
    return LandmarkList([format_landmark(lm) for lm in raw_landmarks])

def pack_blendshapes(blendshapes) -> np.ndarray:
    if blendshapes is None:
        return np.array([])
    return np.array([bl.score for bl in blendshapes])

def pack_hand_landmarks(raw_landmarks) -> LandmarkList:
    if raw_landmarks is None:
        return LandmarkList([])
    confidence = raw_landmarks['confidence']
    local = LandmarkList([format_landmark_with_confidence(lm, confidence) for lm in raw_landmarks['local_landmark']])
    world = LandmarkList([format_landmark_with_confidence(lm, confidence) for lm in raw_landmarks['world_landmark']])
    return local, world

def extract_hand_landmarks(hand_results, pose_results):
    """
    Extract hand landmarks prioritizing temporal consistency over detection labels.
    """
    if not (hand_results and hand_results.handedness and pose_results and pose_results.pose_landmarks):
        return [None, None]

    pose_wrists = {'Left': pose_results.pose_landmarks[0][15], 'Right': pose_results.pose_landmarks[0][16]}

    def get_best_hand(label):
        target_pose_wrist = pose_wrists[label]
        candidates = []

        for i, handedness in enumerate(hand_results.handedness):
            if handedness[0].category_name != label:
                continue

            hand_wrist = hand_results.hand_landmarks[i][0]
            dist = l2(hand_wrist, target_pose_wrist)
            candidates.append((dist, i))

        if not candidates:
            return None

        min_dist, best_idx = min(candidates, key=lambda x: x[0])

        if min_dist >= MAX_DISTANCE:
            return None

        return {
            'local_landmark': hand_results.hand_landmarks[best_idx],
            'world_landmark': hand_results.hand_world_landmarks[best_idx],
            'confidence': hand_results.handedness[best_idx][0].score,
        }

    return [get_best_hand('Left'), get_best_hand('Right')]

def l2(p1, p2):
    """Calculate L2 distance (Euclidean distance) between two points."""
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

MAX_DISTANCE = 0.1