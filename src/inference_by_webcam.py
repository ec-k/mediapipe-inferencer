from mediapipe_inferencer_core.network import HolisticPoseSender
from mediapipe_inferencer_core.detector import DetectorHandler, PoseDetector, HandDetector, FaceDetector
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core.image_provider import WebcamImageProvider
from mediapipe_inferencer_core.filter import Gaussian1dFilter, ExponentialSmoothing

import cv2
import time
import copy
from pathlib import Path

if __name__ == "__main__":
    pose_sender = HolisticPoseSender()
    pose_sender.connect()

    root_directory = str(Path(__file__).parent.parent)
    holistic_detector = DetectorHandler(
        pose=PoseDetector(root_directory + "/models/pose_landmarker_full.task"),
        hand=HandDetector(root_directory + "/models/hand_landmarker.task"),
        face=FaceDetector(root_directory + "/models/face_landmarker.task")
    )

    image_provider = WebcamImageProvider(cache_queue_length=2, device_index=0)
    sigma, window_size = 3, 31
    n_pose_landmarks, n_hand_landmarks = 33, 21
    filter = {
        'pose_local':       Gaussian1dFilter(sigma, window_size, n_pose_landmarks),
        'pose_world':       Gaussian1dFilter(sigma, window_size, n_pose_landmarks),
        'left_hand_local':  Gaussian1dFilter(sigma, window_size, n_hand_landmarks),
        'left_hand_world':  Gaussian1dFilter(sigma, window_size, n_hand_landmarks),
        'right_hand_local': Gaussian1dFilter(sigma, window_size, n_hand_landmarks),
        'right_hand_world': Gaussian1dFilter(sigma, window_size, n_hand_landmarks),
        'face_landmark':    ExponentialSmoothing(0.7)
        }
    while image_provider.is_opened:
        # Break in key Ctrl+C pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Inference pose
        image_provider.update()
        image = cv2.cvtColor(image_provider.latest_frame, cv2.COLOR_RGBA2BGR)
        if image is None:
            continue
        holistic_detector.inference(image)

        if holistic_detector.results is None:
            continue
        # Filtering
        results = copy.deepcopy(holistic_detector.results)
        time_s = results.time
        results.pose.local = filter['pose_local'].filter(results.pose.local)
        results.pose.world = filter['pose_world'].filter(results.pose.world)
        results.hand.left.local = filter['left_hand_local'].filter(results.hand.left.local)
        results.hand.left.world = filter['left_hand_world'].filter(results.hand.left.world)
        results.hand.right.local = filter['right_hand_local'].filter(results.hand.right.local)
        results.hand.right.world = filter['right_hand_world'].filter(results.hand.right.world)
        results.face.landmarks = filter['face_landmark'].filter(results.face.landmarks)

        # Send results to solver app
        pose_sender.send_holistic_landmarks(results)

        # Visualize resulted landmarks
        annotated_image = image
        if results.pose is not None:
            annotated_image = visualizer.draw_pose_landmarks_on_image(annotated_image, results.pose)
        if results.hand is not None:
            annotated_image = visualizer.draw_hand_landmarks_on_image(annotated_image, results.hand)
        if results.face is not None:
            annotated_image = visualizer.draw_face_landmarks_on_image(annotated_image, results.face)
        cv2.imshow('MediaPipe Landmarks', cv2.cvtColor(cv2.flip(annotated_image, 1), cv2.COLOR_BGR2RGB))
        time.sleep(1/60)
    image_provider.release_capture()

