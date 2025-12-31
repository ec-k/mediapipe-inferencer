import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe_inferencer_core.data_class import LandmarkResult, HandResult, FaceResult
import numpy as np
import cv2


def draw_pose_landmarks_on_image(rgb_image, detection_result:LandmarkResult):
    pose_landmarks = detection_result.local
    if pose_landmarks is None:
        return rgb_image
    annotated_image = np.copy(rgb_image)

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in pose_landmarks.values
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def draw_hand_landmarks_on_image(rgb_image, detection_result:HandResult):
    hands_to_draw = [
        ("Left", detection_result.left.local),
        ("Right", detection_result.right.local)
    ]
    annotated_image = np.copy(rgb_image)
    img_height, img_width, _ = annotated_image.shape

    for label, local_landmarks in hands_to_draw:
        if local_landmarks is None or local_landmarks.values is None:
            continue

        # 1. Draw hand landmarks and connections
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2])
            for lm in local_landmarks.values
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # 2. Draw handedness text
        # Use the wrist landmark (index 0) as the reference point for the text
        wrist = local_landmarks.values[0]
        px_x = int(wrist[0] * img_width)
        px_y = int(wrist[1] * img_height)

        # Draw text slightly above the wrist point
        cv2.putText(
            annotated_image,
            label,
            (px_x, px_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,                    # Font scale
            (91, 249, 255),       # Color: Light Cyan
            2,                    # Thickness
            cv2.LINE_AA
        )

    return annotated_image

def draw_face_landmarks_on_image(rgb_image, detection_result:FaceResult):
    face_landmarks = detection_result.landmarks
    if face_landmarks is None:
        return rgb_image
    annotated_image = np.copy(rgb_image)

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in face_landmarks.values
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_iris_connections_style())

    return annotated_image