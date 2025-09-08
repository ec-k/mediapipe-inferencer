from mediapipe_inferencer_core.network import HolisticPoseSender
from mediapipe_inferencer_core.detector import DetectorHandler, PoseDetector, HandDetector, FaceDetector
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core.image_provider import WebcamImageProvider
from mediapipe_inferencer_core.filter import OneEuroFilter

import numpy as np
import cv2
import time
import copy
from pathlib import Path

if __name__ == "__main__":
    pose_sender = HolisticPoseSender("localhost", 9001)
    pose_sender.connect()

    root_directory = str(Path(__file__).parent.parent)
    holistic_detector = DetectorHandler(
        pose=PoseDetector(root_directory + "/models/pose_landmarker_full.task", 0.8),
        hand=HandDetector(root_directory + "/models/hand_landmarker.task", 0.8),
        face=FaceDetector(root_directory + "/models/face_landmarker.task", 0.8)
    )

    image_provider = WebcamImageProvider(cache_queue_length=2, device_index=0)
    min_cutoff, slope, d_min_cutoff = 1.0, 4, 1.0
    filter = {
        'pose_local':       OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'pose_world':       OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'left_hand_local':  OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'left_hand_world':  OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'right_hand_local': OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'right_hand_world': OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'face_landmark':    OneEuroFilter(min_cutoff, slope, d_min_cutoff)
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
        results.pose.local = filter['pose_local'].filter(results.pose.local, time_s)
        results.pose.world = filter['pose_world'].filter(results.pose.world, time_s)
        results.hand.left.local = filter['left_hand_local'].filter(results.hand.left.local, time_s)
        results.hand.left.world = filter['left_hand_world'].filter(results.hand.left.world, time_s)
        results.hand.right.local = filter['right_hand_local'].filter(results.hand.right.local, time_s)
        results.hand.right.world = filter['right_hand_world'].filter(results.hand.right.world, time_s)
        results.face.landmarks = filter['face_landmark'].filter(results.face.landmarks, time_s)

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
        # cv2.imshow('MediaPipe Landmarks', cv2.cvtColor(cv2.flip(annotated_image, 1), cv2.COLOR_BGR2RGB))
        cv2.imshow('MediaPipe Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)) # no-flipping
        time.sleep(1/60)
    image_provider.release_capture()
