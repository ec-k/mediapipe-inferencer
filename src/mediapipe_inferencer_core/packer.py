import proto.holistic_landmarks_pb2 as holistic_lm_pb2

def pack_holistic_landmarks_result(holistic_results):
    holistic_lm = holistic_lm_pb2.HolisticLandmarks()
    pose_landmarks = holistic_results.pose_landmarks
    if (pose_landmarks is not None) and (len(pose_landmarks.pose_world_landmarks)>0):
        holistic_lm.poseLandmarks.landmarks.extend(pack_pose_landmarks(pose_landmarks.pose_world_landmarks[0]))

    [left_hand_landmarks, right_hand_landmarks] = extract_hand_landmarks(holistic_results.hand_landmarks)
    if(left_hand_landmarks is not None and len(left_hand_landmarks)>0):
        holistic_lm.leftHandLandmarks.landmarks.extend(pack_hand_landmarks(left_hand_landmarks))
    if(right_hand_landmarks is not None and len(right_hand_landmarks)>0):
        holistic_lm.rightHandLandmarks.landmarks.extend(pack_hand_landmarks(right_hand_landmarks))

    face_results = holistic_results.face_results
    if(face_results is not None):
        if len(face_results.face_landmarks)>0:
            holistic_lm.faceResults.landmarks.extend(pack_face_landmarks(face_results.face_landmarks[0]))
        if len(face_results.face_blendshapes)>0:
            holistic_lm.faceResults.blendshapes.scores.extend(pack_blendshapes(face_results.face_blendshapes[0]))

    return holistic_lm

def pack_pose_landmarks(pose_landmarks):
    if pose_landmarks is None:
        return []
    formatted_landmarks = [format_landmark(mp_lm) for mp_lm in pose_landmarks]
    return formatted_landmarks

def pack_hand_landmarks(hand_landmarks):
    if hand_landmarks is None:
        return []
    confidence = hand_landmarks['confidence']
    formatted_landmarks = [format_landmark_with_confidence(mp_lm, confidence) for mp_lm in hand_landmarks['landmark']]
    return formatted_landmarks

def pack_face_landmarks(face_landmarks):
    if face_landmarks is None:
        return []
    formatted_landmarks = [format_landmark(mp_lm) for mp_lm in face_landmarks]
    return formatted_landmarks

def pack_blendshapes(blendshapes):
    if blendshapes is None:
        return []
    formatted_blendshapes = [ bl.score for bl in blendshapes]
    return formatted_blendshapes

def format_landmark(mediapipe_landmark):
    landmark = holistic_lm_pb2.Landmark()
    landmark.x = mediapipe_landmark.x
    landmark.y = mediapipe_landmark.y
    landmark.z = mediapipe_landmark.z
    landmark.confidence = mediapipe_landmark.visibility
    return landmark

def format_landmark_with_confidence(landmark_position, confidence):
    landmark = holistic_lm_pb2.Landmark()
    landmark.x = landmark_position.x
    landmark.y = landmark_position.y
    landmark.z = landmark_position.z
    landmark.confidence = confidence
    return landmark

def extract_hand_landmarks(hand_results):
    [left_hand_landmarks, right_hand_landmarks] = [None, None]

    if hand_results == None:
        return [left_hand_landmarks, right_hand_landmarks]

    if len(hand_results.handedness)>=2:
        [left_i, right_i] = [0, 1] if hand_results.handedness[0][0].category_name == 'Left' else [1, 0]
        left_hand_landmarks = {}
        left_hand_landmarks['landmark'] = hand_results.hand_world_landmarks[left_i]
        left_hand_landmarks['confidence'] = hand_results.handedness[left_i][0].score

        right_hand_landmarks = {}
        right_hand_landmarks['landmark'] = hand_results.hand_world_landmarks[right_i]
        right_hand_landmarks['confidence'] = hand_results.handedness[right_i][0].score

    if len(hand_results.handedness)>=1:
        isLeft = hand_results.handedness[0][0].category_name == 'Left'
        if isLeft:
            left_hand_landmarks = {}
            left_hand_landmarks['landmark'] = hand_results.hand_world_landmarks[0]
            left_hand_landmarks['confidence'] = hand_results.handedness[0][0].score
        else:
            right_hand_landmarks = {}
            right_hand_landmarks['landmark'] = hand_results.hand_world_landmarks[0]
            right_hand_landmarks['confidence'] = hand_results.handedness[0][0].score


    return [left_hand_landmarks, right_hand_landmarks]