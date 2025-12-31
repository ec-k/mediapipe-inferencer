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
    """
    Extract hand landmarks prioritizing temporal consistency over detection labels.
    """
    left_out = None
    right_out = None

    if hand_results is None or not hand_results.handedness:
        return [None, None]

    num_detected  = len(hand_results.handedness)
    cache_l = get_valid_cache('Left')
    cache_r = get_valid_cache('Right')
    detected_hands = [
        {
            'local_landmark': hand_results.hand_landmarks[i],
            'world_landmark': hand_results.hand_world_landmarks[i],
            'confidence': hand_results.handedness[i][0].score,
            'label': hand_results.handedness[i][0].category_name
        }
        for i in range(num_detected )
    ]

    if num_detected  >= 2:
        h0, h1 = detected_hands[0], detected_hands[1]

        # Scenario: Both caches are valid - use distance to decide assignment
        if cache_l and cache_r:
            d0_l = calculate_l1_dist(h0['world_landmark'], cache_l['world_landmark'])
            d0_r = calculate_l1_dist(h0['world_landmark'], cache_r['world_landmark'])
            d1_l = calculate_l1_dist(h1['world_landmark'], cache_l['world_landmark'])
            d1_r = calculate_l1_dist(h1['world_landmark'], cache_r['world_landmark'])

            if (d0_l + d1_r) <= (d0_r + d1_l):
                left_out, right_out = h0, h1
            else:
                left_out, right_out = h1, h0

        # Scenario: Only one cache is valid
        elif cache_l or cache_r:
            target_side = 'Left' if cache_l else 'Right'
            ref_pts = cache_l['world_landmark'] if cache_l else cache_r['world_landmark']

            d0 = calculate_l1_dist(h0['world_landmark'], ref_pts)
            d1 = calculate_l1_dist(h1['world_landmark'], ref_pts)

            if target_side == 'Left':
                left_out, right_out = (h0, h1) if d0 <= d1 else (h1, h0)
            else:
                right_out, left_out = (h0, h1) if d0 <= d1 else (h1, h0)

        # Scenario: No valid cache - fallback to detection labels
        else:
            if h0['label'] != h1['label']:
                left_out, right_out = (h0, h1) if h0['label'] == 'Left' else (h1, h0)
            else:
                # Same labels and no cache: use confidence
                left_out, right_out = (h0, h1) if h0['confidence'] >= h1['confidence'] else (h1, h0)
                if h0['label'] == 'Right':
                    left_out, right_out = right_out, left_out

    elif num_detected  == 1:
        h0 = detected_hands[0]

        if cache_l and cache_r:
            d0_l = calculate_l1_dist(h0['world_landmark'], cache_l['world_landmark'])
            d0_r = calculate_l1_dist(h0['world_landmark'], cache_r['world_landmark'])

            closer_to_left = d0_l < d0_r
            min_dist = d0_l if closer_to_left else d0_r

            is_conflict = (closer_to_left and h0['label'] == 'Right') or (not closer_to_left and h0['label'] == 'Left')
            is_too_far = min_dist > MAX_MOVE_DISTANCE

            if is_conflict or is_too_far:
                return [None, None]         # This results is probably misinferenced

            if closer_to_left:
                left_out = h0
            else:
                right_out = h0

        elif cache_l or cache_r:
            active_cache = cache_l if cache_l else cache_r
            active_side = 'Left' if cache_l else 'Right'
            dist = calculate_l1_dist(h0['world_landmark'], active_cache['world_landmark'])

            if h0['label'] == active_side and dist <= MAX_MOVE_DISTANCE:
                if active_side == 'Left': left_out = h0
                else: right_out = h0
            else:
                return [None, None]

        else:
            # Fallback to label
            if h0['label'] == 'Left':
                left_out = h0
            else:
                right_out = h0

    update_hand_cache('Left', left_out)
    update_hand_cache('Right', right_out)

    return [left_out, right_out]

def calculate_l1_dist(landmarks_1, landmarks_2):
    """Calculate L1 distance between two sets of landmarks."""
    return sum(abs(c.x - p.x) + abs(c.y - p.y) + abs(c.z - p.z)
               for c, p in zip(landmarks_1, landmarks_2))

# Global cache for hand landmarks
HAND_CACHE = {
    'Left': {'data': None, 'timestamp': 0},
    'Right': {'data': None, 'timestamp': 0}
}

CACHE_EXPIRATION = 0.2  # Cache expiration threshold in seconds
MAX_MOVE_DISTANCE = 0.25   # Threshold for jitter or sudden jumps

def update_hand_cache(handedness, hand_landmarks):
    """Update the cache only when valid landmark data is provided."""
    if hand_landmarks is not None:
        HAND_CACHE[handedness]['data'] = hand_landmarks
        HAND_CACHE[handedness]['timestamp'] = time.time()

def get_valid_cache(handedness):
    """Retrieve the cache if it exists and has not expired."""
    cache = HAND_CACHE[handedness]
    if cache['data'] is not None and (time.time() - cache['timestamp']) < CACHE_EXPIRATION:
        return cache['data']
    return None
