from mediapipe_inferencer_core.data_class.landmark import Landmark

def format_landmark(raw_landmark) -> Landmark:
    return Landmark(raw_landmark.x, raw_landmark.y, raw_landmark.z, raw_landmark.visibility)

def format_landmark_with_confidence(position, confidence) -> Landmark:
    return Landmark(position.x, position.y, position.z, confidence)

def pack_landmarks(raw_landmarks) -> list[Landmark]:
    if raw_landmarks is None:
        return []
    return [format_landmark(lm) for lm in raw_landmarks]

def pack_blendshapes(blendshapes):
    if blendshapes is None:
        return []
    return [bl.score for bl in blendshapes]

def pack_hand_landmarks(raw_landmarks) -> list[Landmark]:
    if raw_landmarks is None:
        return []
    confidence = raw_landmarks['confidence']
    local = [format_landmark_with_confidence(lm, confidence) for lm in raw_landmarks['local_landmark']]
    world = [format_landmark_with_confidence(lm, confidence) for lm in raw_landmarks['world_landmark']]
    return local, world

def extract_hand_landmarks(hand_results):
    if hand_results == None:
        return [left_hand_landmarks, right_hand_landmarks]

    [left_hand_landmarks, right_hand_landmarks] = [None, None]

    if len(hand_results.handedness)>=2:
        [left_i, right_i] = [0, 1] if hand_results.handedness[0][0].category_name == 'Left' else [1, 0]
        left_hand_landmarks = {}
        left_hand_landmarks['local_landmark'] = hand_results.hand_landmarks[left_i]
        left_hand_landmarks['world_landmark'] = hand_results.hand_world_landmarks[left_i]
        left_hand_landmarks['confidence'] = hand_results.handedness[left_i][0].score

        right_hand_landmarks = {}
        right_hand_landmarks['local_landmark'] = hand_results.hand_landmarks[right_i]
        right_hand_landmarks['world_landmark'] = hand_results.hand_world_landmarks[right_i]
        right_hand_landmarks['confidence'] = hand_results.handedness[right_i][0].score

    if len(hand_results.handedness)>=1:
        isLeft = hand_results.handedness[0][0].category_name == 'Left'
        if isLeft:
            left_hand_landmarks = {}
            left_hand_landmarks['local_landmark'] = hand_results.hand_landmarks[0]
            left_hand_landmarks['world_landmark'] = hand_results.hand_world_landmarks[0]
            left_hand_landmarks['confidence'] = hand_results.handedness[0][0].score
        else:
            right_hand_landmarks = {}
            right_hand_landmarks['local_landmark'] = hand_results.hand_landmarks[0]
            right_hand_landmarks['world_landmark'] = hand_results.hand_world_landmarks[0]
            right_hand_landmarks['confidence'] = hand_results.handedness[0][0].score

    return [left_hand_landmarks, right_hand_landmarks]