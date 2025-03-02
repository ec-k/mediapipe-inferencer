from mediapipe_inferencer_core.network.holistic_pose_sender import HolisticPoseSender
from mediapipe_inferencer_core.detector.detector_handler import DetectorHandler
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core.packer import packer_for_sending
from mediapipe_inferencer_core.image_provider import WebcamImageProvider
from mediapipe_inferencer_core.detector.landmark_detector import PoseDetector, HandDetector, FaceDetector
from mediapipe_inferencer_core.filter.exponential_smoothing import ExponentialSmoothing

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
    filter = {
        'pose_local':ExponentialSmoothing(0.8),
        'pose_world':ExponentialSmoothing(0.8),
        'left_hand_local': ExponentialSmoothing(0.65),
        'left_hand_world': ExponentialSmoothing(0.65),
        'right_hand_local': ExponentialSmoothing(0.65),
        'right_hand_world': ExponentialSmoothing(0.65),
        'face_landmark': ExponentialSmoothing(0.8)
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

        # Filtering
        results = copy.deepcopy(holistic_detector.results)
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

