from mediapipe_inferencer_core.data_class import Landmark, LandmarkList
import time
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

def extract_hand_landmarks(hand_results):
    left_hand_landmarks = None
    right_hand_landmarks = None

    if hand_results == None:
        return [left_hand_landmarks, right_hand_landmarks]

    num_hands = len(hand_results.handedness)
    if num_hands >= 2:
        category_name_0 = hand_results.handedness[0][0].category_name
        category_name_1 = hand_results.handedness[1][0].category_name

        if category_name_0 != category_name_1:
            # Case: Both Left and Right hands are detected
            left_i, right_i = [0, 1] if category_name_0 == 'Left' else [1, 0]
            left_hand_landmarks = {
                'local_landmark': hand_results.hand_landmarks[left_i],
                'world_landmark': hand_results.hand_world_landmarks[left_i],
                'confidence'    : hand_results.handedness[left_i][0].score,
            }

            right_hand_landmarks = {
                'local_landmark': hand_results.hand_landmarks[right_i],
                'world_landmark': hand_results.hand_world_landmarks[right_i],
                'confidence'    : hand_results.handedness[right_i][0].score,
            }

        else:
            # Case: Duplicate labels detected (e.g., two "Left" hands)
            handedness = category_name_0
            best_i = select_closest_index(handedness, hand_results)

            target_data = {
                'local_landmark': hand_results.hand_landmarks[best_i],
                'world_landmark': hand_results.hand_world_landmarks[best_i],
                'confidence': hand_results.handedness[best_i][0].score
            }

            if handedness == 'Left':
                left_hand_landmarks = target_data
            else:
                right_hand_landmarks = target_data

    elif num_hands >= 1:
        is_left = hand_results.handedness[0][0].category_name == 'Left'
        target_data = {
            'local_landmark': hand_results.hand_landmarks[0],
            'world_landmark': hand_results.hand_world_landmarks[0],
            'confidence': hand_results.handedness[0][0].score
        }
        if is_left:
            left_hand_landmarks = target_data
        else:
            right_hand_landmarks = target_data

    update_hand_cache('Left', left_hand_landmarks)
    update_hand_cache('Right', right_hand_landmarks)

    return [left_hand_landmarks, right_hand_landmarks]


def select_closest_index(handedness, hand_results):
    """
    Select the index closest to the cached landmarks using L1 norm when
    multiple hands share the same label. If no valid cache exists,
    the hand with the higher confidence score is selected.
    """
    prev_data = get_valid_cache(handedness)
    # If no cache is available, select the one with the higher confidence score
    if prev_data is None:
        return 0 if hand_results.handedness[0][0].score >= hand_results.handedness[1][0].score else 1

    def calculate_l1(curr_idx):
        curr_landmarks = hand_results.hand_world_landmarks[curr_idx]
        prev_landmarks = prev_data['world_landmark']
        return sum(abs(c.x - p.x) + abs(c.y - p.y) + abs(c.z - p.z) for c, p in zip(curr_landmarks, prev_landmarks))

    dist0 = calculate_l1(0)
    dist1 = calculate_l1(1)

    return 0 if dist0 <= dist1 else 1


# Global cache for hand landmarks
HAND_CACHE = {
    'Left': {'data': None, 'timestamp': 0},
    'Right': {'data': None, 'timestamp': 0}
}

# Cache expiration threshold in seconds
CACHE_EXPIRATION = 0.5

def update_hand_cache(handedness, hand_landmarks):
    """
    Update the cache only when valid landmark data is provided.
    """
    if hand_landmarks is not None:
        HAND_CACHE[handedness]['data'] = hand_landmarks
        HAND_CACHE[handedness]['timestamp'] = time.time()

def get_valid_cache(handedness):
    """
    Retrieve the cache if it exists and has not expired.
    """
    cache = HAND_CACHE[handedness]
    if cache['data'] is not None and (time.time() - cache['timestamp']) < CACHE_EXPIRATION:
        return cache['data']
    return None
